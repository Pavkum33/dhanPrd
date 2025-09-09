#!/usr/bin/env python3
"""
scanner.py — Production-ready F&O live resistance-breakout scanner (Dhan WebSocket v2)

Features:
- Automatically downloads Dhan instrument master (detailed CSV) and extracts F&O futures
- Pre-market: fetches historical daily OHLCV for each F&O security and builds history
- Live: connects to Dhan WebSocket v2 with query-string auth; subscribes using v2 RequestCode
- Aggregates ticks into 1-minute candles (volume via cumulative-delta)
- On each 1-minute candle close evaluates Chartink-style breakout conditions
- Alerts: writes to rotating log, saves alert to SQLite, and optionally sends Telegram messages
- Robust reconnect/backoff, binary & JSON incoming parsing safe-fallback, per-batch WS subscriptions
- Extensible for other scan strategies

Configuration and secrets:
- Non-secret config sits in config.json (batch_size, hist_days, etc)
- Secrets must be provided via environment variables (safer) or Docker secrets:
    DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
    TELEGRAM_TOKEN (optional), TELEGRAM_CHAT_ID (optional)

Run:
  python scanner.py --config config.json
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import asyncio
import logging
import argparse
import sqlite3
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

import aiohttp
import requests
import pandas as pd
import websockets

# Optional SDK usage (if installed)
try:
    from dhanhq import DhanContext, dhanhq
    HAS_DHAN_SDK = True
except Exception:
    HAS_DHAN_SDK = False

# ---------- Logging ----------
LOG_FILE = "fno_breakout.log"
logger = logging.getLogger("fno_scanner")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = TimedRotatingFileHandler(LOG_FILE, when="midnight", backupCount=14)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ---------- Defaults (will be overridden by config.json) ----------
DEFAULT_CONFIG = {
    "hist_days": 60,
    "lookback": 50,
    "ema_short": 8,
    "ema_long": 13,
    "volume_factor": 0.5,
    "price_threshold": 50,
    "batch_size": 200,
    "fetch_concurrency": 6,
    "instrument_csv_url": "https://images.dhan.co/api-data/api-scrip-master-detailed.csv",
    "local_instrument_cache": "instruments_cached.csv",
    "sqlite_file": "alerts.db",
    "tele_throttle_sec": 0.6,
    "request_code": 15,
    "ws_base": "wss://api-feed.dhan.co"
}

# ---------- Utility helpers ----------
def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    cfg = DEFAULT_CONFIG.copy()
    if path and os.path.exists(path):
        with open(path, "r") as f:
            j = json.load(f)
            cfg.update(j)
    return cfg

def env_required(name: str) -> Optional[str]:
    """Get required environment variable"""
    v = os.getenv(name)
    if not v:
        logger.warning("Environment variable %s not set.", name)
    return v

# ---------- Instrument loader (auto-fetch & cache) ----------
class InstrumentLoader:
    """Downloads and manages Dhan instrument master data"""
    
    def __init__(self, csv_url: str, local_cache: str):
        self.csv_url = csv_url
        self.local_cache = local_cache
        self.df: Optional[pd.DataFrame] = None
        self.colmap = {}
        self._load_local()

    def _load_local(self):
        """Load cached instrument data if available"""
        if os.path.exists(self.local_cache):
            try:
                self.df = pd.read_csv(self.local_cache)
                self.df.columns = [c.strip() for c in self.df.columns]
                self._resolve_cols()
                logger.info("Loaded instrument cache rows=%d", len(self.df))
            except Exception:
                logger.exception("Failed to load local instrument cache")

    def fetch_and_cache(self) -> bool:
        """Download fresh instrument master from Dhan"""
        try:
            logger.info("Downloading instrument master: %s", self.csv_url)
            r = requests.get(self.csv_url, timeout=30)
            r.raise_for_status()
            tmp = self.local_cache + ".tmp"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, self.local_cache)
            self.df = pd.read_csv(self.local_cache)
            self.df.columns = [c.strip() for c in self.df.columns]
            self._resolve_cols()
            logger.info("Fetched and cached instrument master rows=%d", len(self.df))
            return True
        except Exception:
            logger.exception("Failed to fetch instrument master")
            return False

    def _resolve_cols(self):
        """Map column names for different CSV formats"""
        if self.df is None:
            return
        cols = list(self.df.columns)
        def find(*cands):
            for c in cands:
                if c in cols:
                    return c
            return None
        # Add more column name variations for the detailed CSV
        self.colmap['sid'] = find('SEM_SMST_SECURITY_ID','securityId','SECURITYID','SECURITY_ID','SCRIP_CODE',
                                  'SecurityId','SM_KEY_SYMBOL','SCRIP_CD','SEM_INSTRUMENT_NAME')
        self.colmap['symbol'] = find('SEM_TRADING_SYMBOL','tradingSymbol','SEM_SMST_SECURITY_SYMBOL','symbolName',
                                    'InstrumentName','SYMBOL','TradingSymbol','SYMBOL_NAME','SEM_CUSTOM_SYMBOL')
        self.colmap['segment'] = find('segment','SEM_SEGMENT','EXCHANGE','SEM_EXM_EXCH_ID')
        self.colmap['expiry'] = find('EXPIRY_DATE','EXPIRY')
        self.colmap['lot'] = find('LOT_SIZE','lotSize','SEM_LOT_UNITS')
        logger.debug("Instrument columns: %s", self.colmap)

    def get_futures_df(self) -> pd.DataFrame:
        """Extract F&O futures from instrument master"""
        if self.df is None:
            if not self.fetch_and_cache():
                raise RuntimeError("Instrument master not available")
        df = self.df.copy()
        segc = self.colmap.get('segment')
        if segc and segc in df.columns:
            mask = df[segc].astype(str).str.upper().str.contains("FUT")
            return df[mask].copy()
        # fallback: use expiry column
        expc = self.colmap.get('expiry')
        if expc and expc in df.columns:
            return df[df[expc].notnull()].copy()
        return df.copy()

    def build_sid_map(self, df: pd.DataFrame) -> Dict[str,str]:
        """Build security ID to symbol mapping"""
        sidc = self.colmap.get('sid')
        symc = self.colmap.get('symbol')
        if sidc is None or symc is None:
            logger.error(f"Instrument master missing sid/symbol columns. Available columns: {list(df.columns[:20])}")
            logger.error(f"Column mapping: {self.colmap}")
            # Return empty map instead of crashing
            return {}
        out = {}
        for _, r in df.iterrows():
            try:
                out[str(r[sidc])] = str(r[symc])
            except Exception as e:
                logger.debug(f"Error mapping row: {e}")
                continue
        return out

# ---------- SQLite Alert DB ----------
class AlertDB:
    """Manages alert storage in SQLite database"""
    
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        """Initialize database schema"""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                sid TEXT,
                symbol TEXT,
                message TEXT,
                strategy TEXT DEFAULT 'breakout'
            );
        """)
        self.conn.commit()

    def insert(self, sid: str, symbol: str, message: str, strategy: str = 'breakout'):
        """Insert new alert into database"""
        cur = self.conn.cursor()
        cur.execute("INSERT INTO alerts(ts,sid,symbol,message,strategy) VALUES(?,?,?,?,?)",
                    (datetime.utcnow().isoformat(), sid, symbol, message, strategy))
        self.conn.commit()

    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent alerts"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, ts, sid, symbol, message, strategy 
            FROM alerts 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        return [
            {"id": r[0], "ts": r[1], "sid": r[2], "symbol": r[3], "message": r[4], "strategy": r[5]}
            for r in rows
        ]

# ---------- Telegram notifier (async) ----------
class TelegramNotifier:
    """Sends alerts to Telegram with throttling"""
    
    def __init__(self, token: str, chat_id: str, min_interval: float = 0.6):
        self.token = token
        self.chat_id = chat_id
        self.min_interval = min_interval
        self._last_sent = 0.0

    async def send(self, text: str):
        """Send message to Telegram with rate limiting"""
        # throttle
        wait = max(0.0, (self._last_sent + self.min_interval) - time.time())
        if wait:
            await asyncio.sleep(wait)
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        async with aiohttp.ClientSession() as sess:
            try:
                async with sess.post(url, json={"chat_id": self.chat_id, "text": text}, timeout=10) as resp:
                    tt = await resp.text()
                    if resp.status != 200:
                        logger.warning("Telegram failed: %s %s", resp.status, tt)
                    else:
                        self._last_sent = time.time()
            except Exception:
                logger.exception("Telegram send error")

# ---------- Runtime symbol state ----------
class SymbolState:
    """Tracks per-symbol state for technical analysis"""
    
    def __init__(self, prev_day_vol:int=0, typical_history:Optional[List[float]]=None, 
                 lookback:int=50, ema_short:int=8, ema_long:int=13):
        self.prev_day_volume = int(prev_day_vol or 0)
        self.typical_deque = deque(maxlen=lookback)
        if typical_history:
            for v in typical_history[-lookback:]:
                self.typical_deque.append(float(v))
        self.ema_short = None
        self.ema_long = None
        self.ema_short_len = ema_short
        self.ema_long_len = ema_long
        self.current_candle = None
        self.last_total_volume = 0

    def update_ema(self, close: float):
        """Update exponential moving averages"""
        a_s = 2.0 / (self.ema_short_len + 1)
        a_l = 2.0 / (self.ema_long_len + 1)
        self.ema_short = close if self.ema_short is None else (a_s * close + (1 - a_s) * self.ema_short)
        self.ema_long  = close if self.ema_long is None else (a_l * close + (1 - a_l) * self.ema_long)

    def close_and_evaluate(self, lookback:int, volume_factor:float, price_threshold:float):
        """Close current candle and evaluate breakout conditions"""
        if not self.current_candle:
            return None
        o = float(self.current_candle['open'])
        h = float(self.current_candle['high'])
        l = float(self.current_candle['low'])
        c = float(self.current_candle['close'])
        v = int(self.current_candle['volume'])
        
        typical = (h + l + c) / 3.0
        self.typical_deque.append(typical)
        resistance = math.ceil(max(self.typical_deque)) if self.typical_deque else None
        self.update_ema(c)
        
        # Evaluate breakout conditions
        cond_breakout = (resistance is not None and c > resistance)
        cond_ema = (self.ema_short is not None and self.ema_long is not None and self.ema_short > self.ema_long)
        cond_vol = (v >= self.prev_day_volume * volume_factor) if self.prev_day_volume else True
        cond_price = c > price_threshold
        cond_bull = c > o
        
        passed = all([cond_breakout, cond_ema, cond_vol, cond_price, cond_bull])
        bar = {
            "open": o, "high": h, "low": l, "close": c, "volume": v, 
            "resistance": resistance, "ema_short": self.ema_short, "ema_long": self.ema_long
        }
        self.current_candle = None
        return passed, bar

# ---------- Historical fetch (async) ----------
class HistoricalFetcher:
    """Fetches historical data for pre-market analysis"""
    
    def __init__(self, client_id: str, access_token: str, use_sdk: bool = HAS_DHAN_SDK, concurrency: int = 6):
        self.client_id = client_id
        self.access_token = access_token
        self.use_sdk = use_sdk and HAS_DHAN_SDK
        if self.use_sdk:
            ctx = DhanContext(client_id, access_token)
            self.sdk = dhanhq(ctx)
        self._sem = asyncio.Semaphore(concurrency)

    async def fetch_one(self, sid: str, days: int) -> pd.DataFrame:
        """Fetch historical data for one security"""
        async with self._sem:
            loop = asyncio.get_running_loop()
            if self.use_sdk:
                def sdk_call():
                    try:
                        to_dt = datetime.utcnow().date()
                        from_dt = to_dt - timedelta(days=days + 5)
                        return self.sdk.historical_daily_data(
                            sid, "NSE_FUT", "FUTURE", 
                            from_dt.strftime("%Y-%m-%d"), 
                            to_dt.strftime("%Y-%m-%d")
                        )
                    except Exception as e:
                        logger.exception("SDK historic call failed for sid=%s: %s", sid, e)
                        return {}
                res = await loop.run_in_executor(None, sdk_call)
            else:
                res = {}
            
            data = []
            if isinstance(res, dict):
                data = res.get('data') or res.get('result') or []
            elif isinstance(res, list):
                data = res
            
            if not data:
                return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
            
            df = pd.DataFrame(data)
            # normalize fields
            for candidate in ('timestamp','date','datetime'):
                if candidate in df.columns and 'timestamp' not in df.columns:
                    df['timestamp'] = df[candidate]
                    break
            for c in ['open','high','low','close','volume']:
                if c not in df.columns:
                    df[c] = None
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception:
                pass
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df[['timestamp','open','high','low','close','volume']]

    async def bulk_fetch(self, sids: List[str], days: int, concurrency: int = 6) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple securities"""
        self._sem = asyncio.Semaphore(concurrency)
        tasks = {sid: asyncio.create_task(self.fetch_one(sid, days)) for sid in sids}
        out = {}
        for sid, t in tasks.items():
            try:
                out[sid] = await t
            except Exception:
                logger.exception("Fetch error for %s", sid)
                out[sid] = pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
        return out

# ---------- FNO engine with WebSocket v2 ----------
class FNOEngine:
    """Main engine for F&O scanning with WebSocket connectivity"""
    
    def __init__(self, cfg: Dict[str, Any], client_id: str, access_token: str, 
                 sid_map: Dict[str,str], hist_map: Dict[str,pd.DataFrame]):
        self.cfg = cfg
        self.client_id = client_id
        self.access_token = access_token
        self.sid_map = sid_map
        self.states: Dict[str, SymbolState] = {}
        self.alert_db = AlertDB(cfg.get("sqlite_file", "alerts.db"))
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        self.hist_map = hist_map or {}
        self.batch_size = int(cfg.get("batch_size", 200))
        
        # Initialize SymbolState from history
        for sid, symbol in self.sid_map.items():
            df = self.hist_map.get(sid)
            prev_vol = 0
            typical_hist = []
            if df is not None and not df.empty:
                df2 = df.copy()
                # Coerce numeric types
                for col in ['open','high','close','volume']:
                    if col in df2.columns:
                        df2[col] = pd.to_numeric(df2[col], errors='coerce')
                df2['typ'] = (df2['open'] + df2['high'] + df2['close']) / 3.0
                typical_hist = df2['typ'].dropna().astype(float).tolist()[-cfg['lookback']:]
                prev_vol = int(df2['volume'].fillna(0).astype(float).tolist()[-1]) if len(df2)>0 else 0
            
            self.states[sid] = SymbolState(
                prev_vol, typical_hist, 
                lookback=cfg['lookback'], 
                ema_short=cfg['ema_short'], 
                ema_long=cfg['ema_long']
            )

    def _make_instrument_obj(self, sid: str, instrument_master_df: Optional[pd.DataFrame] = None):
        """Create instrument object for WebSocket subscription"""
        seg = "NSE_FUT"
        if instrument_master_df is not None:
            sid_col = None
            for c in instrument_master_df.columns:
                if 'security' in c.lower() and 'id' in c.lower():
                    sid_col = c
                    break
            if sid_col:
                try:
                    row = instrument_master_df[instrument_master_df[sid_col].astype(str) == str(sid)]
                    if not row.empty:
                        segraw = str(row.iloc[0].get('segment') or row.iloc[0].get('SEM_SEGMENT') or "")
                        seg = ("NSE_FUT" if "FUT" in segraw.upper() else seg)
                except Exception:
                    pass
        return {"ExchangeSegment": seg, "SecurityId": str(sid)}

    async def _send_subscribe(self, ws, instrument_list: List[Dict[str,str]], request_code: int):
        """Send subscription request to WebSocket"""
        msg = {
            "RequestCode": request_code, 
            "InstrumentCount": len(instrument_list), 
            "InstrumentList": instrument_list
        }
        await ws.send(json.dumps(msg))
        logger.info("Subscribed: %d instruments (v2 RequestCode=%d)", len(instrument_list), request_code)

    async def _parse_incoming(self, raw):
        """Robust parsing: handles JSON strings or bytes"""
        data = None
        if isinstance(raw, bytes):
            try:
                s = raw.decode('utf-8')
                data = json.loads(s)
            except Exception:
                logger.debug("Received binary frame len=%d: %s", len(raw), raw[:64].hex())
                return None
        else:
            try:
                data = json.loads(raw)
            except Exception:
                logger.debug("Non-JSON text frame: %s", repr(raw)[:200])
                return None
        return data

    async def _handle_message(self, raw):
        """Process incoming WebSocket message"""
        data = await self._parse_incoming(raw)
        if not data:
            return
        
        # Extract fields from various possible formats
        sid = str(data.get('security_id') or data.get('securityId') or 
                  data.get('ScripCode') or data.get('SecurityId') or "")
        ltp = data.get('ltp') or data.get('LTP') or data.get('lastPrice') or data.get('last_traded_price')
        total_vol = data.get('volumeTradedToday') or data.get('Volume') or data.get('totalVolume') or 0
        
        # Check nested structures
        if not sid or ltp is None:
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, dict):
                        sid = sid or str(v.get('security_id') or v.get('securityId') or v.get('ScripCode') or "")
                        ltp = ltp or v.get('ltp') or v.get('LTP') or v.get('lastPrice')
                        total_vol = total_vol or v.get('volumeTradedToday') or v.get('Volume')
        
        if not sid or ltp is None:
            logger.debug("No tick fields found in message")
            return
        
        try:
            ltp_f = float(ltp)
        except Exception:
            return
        
        total_vol = int(total_vol or 0)
        await self._process_tick(sid, ltp_f, total_vol)

    async def _process_tick(self, sid: str, ltp: float, total_vol: int):
        """Process tick data and aggregate into candles"""
        state = self.states.get(sid)
        if state is None:
            return
        
        # Align to minute
        ts = datetime.utcnow().replace(second=0, microsecond=0)
        
        if state.current_candle is None:
            state.current_candle = {
                "time": ts, "open": ltp, "high": ltp, 
                "low": ltp, "close": ltp, "volume": 0
            }
            state.last_total_volume = total_vol
            return
        
        if ts > state.current_candle['time']:
            # Close previous candle
            result = state.close_and_evaluate(
                self.cfg['lookback'], 
                self.cfg['volume_factor'], 
                self.cfg['price_threshold']
            )
            if result:
                passed, bar = result
                if passed:
                    symbol = self.sid_map.get(sid, sid)
                    text = (f"🔥 BREAKOUT {symbol} (sid={sid}) | "
                           f"Close={bar['close']:.2f} Res={bar['resistance']} "
                           f"Vol={bar['volume']} EMA8={bar['ema_short']:.2f} "
                           f"EMA13={bar['ema_long']:.2f}")
                    await self.alert_queue.put((sid, symbol, text))
                    self.alert_db.insert(sid, symbol, text)
                    logger.info("Queued alert: %s", text)
            
            # Start new candle
            state.current_candle = {
                "time": ts, "open": ltp, "high": ltp, 
                "low": ltp, "close": ltp, "volume": 0
            }
            state.last_total_volume = total_vol
            return
        
        # Update current candle
        c = state.current_candle
        c['high'] = max(c['high'], ltp)
        c['low'] = min(c['low'], ltp)
        c['close'] = ltp
        # Volume delta from cumulative total
        delta = max(0, total_vol - state.last_total_volume)
        c['volume'] = c.get('volume', 0) + delta
        state.last_total_volume = total_vol

    async def _connect_batch(self, instrument_list: List[Dict[str,str]], request_code:int, reconnect_backoff:float = 1.0):
        """Connect WebSocket for a batch of instruments"""
        url = f"{self.cfg['ws_base']}?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"
        backoff = reconnect_backoff
        while True:
            try:
                logger.info("Connecting WS batch (len=%d) url=%s", len(instrument_list), self.cfg['ws_base'])
                async with websockets.connect(url, ping_interval=30, ping_timeout=20, max_size=None) as ws:
                    await self._send_subscribe(ws, instrument_list, request_code)
                    async for raw in ws:
                        asyncio.create_task(self._handle_message(raw))
            except Exception:
                logger.exception("WS batch connection error")
            logger.info("WS batch reconnecting in %.1f s", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)

    async def start(self, instrument_master_df: Optional[pd.DataFrame] = None, request_code:int = 15):
        """Start WebSocket connections for all instruments"""
        sids = list(self.sid_map.keys())
        batches = [sids[i:i+self.batch_size] for i in range(0, len(sids), self.batch_size)]
        tasks = []
        for b in batches:
            inst_list = [self._make_instrument_obj(sid, instrument_master_df) for sid in b]
            tasks.append(asyncio.create_task(self._connect_batch(inst_list, request_code)))
            await asyncio.sleep(0.2)
        return tasks

# ---------- Notifier loop ----------
async def notifier_loop(alert_queue: asyncio.Queue, tg_token: Optional[str], tg_chat: Optional[str], throt: float):
    """Process alert queue and send notifications"""
    tg = TelegramNotifier(tg_token, tg_chat, min_interval=throt) if (tg_token and tg_chat) else None
    while True:
        sid, symbol, text = await alert_queue.get()
        try:
            logger.info("Notifier: %s", text)
            if tg:
                await tg.send(text)
        except Exception:
            logger.exception("Notifier error")
        alert_queue.task_done()

# ---------- Main async entry point ----------
async def main(cfg_path: str):
    """Main entry point for scanner"""
    cfg = load_config(cfg_path)
    
    # Get credentials from environment
    client_id = os.getenv("DHAN_CLIENT_ID")
    access_token = os.getenv("DHAN_ACCESS_TOKEN")
    if not client_id or not access_token:
        logger.error("Missing Dhan credentials. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN env vars.")
        sys.exit(2)

    # Optional Telegram config
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")

    # Load instrument master
    loader = InstrumentLoader(cfg['instrument_csv_url'], cfg['local_instrument_cache'])
    if loader.df is None:
        ok = loader.fetch_and_cache()
        if not ok:
            logger.error("Instrument master unavailable; exiting")
            sys.exit(3)
    
    fno_df = loader.get_futures_df()
    sid_map = loader.build_sid_map(fno_df)
    sids = list(sid_map.keys())
    logger.info("F&O candidates discovered: %d", len(sids))
    
    # Pre-market historical fetch
    hist_days = int(cfg.get('hist_days', 60))
    hf = HistoricalFetcher(client_id, access_token, use_sdk=HAS_DHAN_SDK, concurrency=cfg.get('fetch_concurrency', 6))
    logger.info("Fetching historical data for %d sids (days=%d)", len(sids), hist_days)
    hist_map = await hf.bulk_fetch(sids, hist_days, concurrency=cfg.get('fetch_concurrency', 6))

    # Start engine
    engine = FNOEngine(cfg, client_id, access_token, sid_map, hist_map)
    
    # Start notifier
    asyncio.create_task(notifier_loop(
        engine.alert_queue, telegram_token, telegram_chat, 
        cfg.get('tele_throttle_sec', 0.6)
    ))
    
    # Start WebSocket batches
    tasks = await engine.start(fno_df, request_code=cfg.get('request_code', 15))
    logger.info("Started %d WS tasks", len(tasks))
    
    # Keep running
    while True:
        await asyncio.sleep(1.0)

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="F&O Breakout Scanner for Dhan")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    args = p.parse_args()
    try:
        asyncio.run(main(args.config))
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)