#!/usr/bin/env python3
"""
dhan_fetcher.py - Standalone Dhan historical data fetcher
Extracted from app.py to avoid circular imports
"""

import os
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Optional dhanhq SDK import
try:
    from dhanhq import DhanContext, dhanhq
    HAS_DHAN_SDK = True
    logger.info("dhanhq SDK available")
except ImportError:
    HAS_DHAN_SDK = False
    logger.warning("dhanhq SDK not available - using REST API fallback")

class DhanHistoricalFetcher:
    """Fetches historical data using dhanhq SDK (preferred) or REST API (fallback)"""
    
    def __init__(self, client_id: str, access_token: str, use_sdk: bool = HAS_DHAN_SDK):
        self.client_id = client_id
        self.access_token = access_token
        self.use_sdk = use_sdk and HAS_DHAN_SDK
        self.base_url = "https://api.dhan.co"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': access_token
        }
        self.session = None
        
        # Initialize SDK if available
        if self.use_sdk:
            try:
                # Create DhanContext first, then initialize SDK
                ctx = DhanContext(client_id, access_token)
                self.sdk = dhanhq(ctx)
                logger.info("Initialized dhanhq SDK for historical data")
            except Exception as e:
                logger.warning(f"Failed to initialize dhanhq SDK: {e}, falling back to REST API")
                self.use_sdk = False
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_instruments(self) -> pd.DataFrame:
        """Fetch instrument master from Dhan (use the correct CSV for active F&O)"""
        url = "https://images.dhan.co/api-data/api-scrip-master.csv"  # Use this for active F&O
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    from io import StringIO
                    df = pd.read_csv(StringIO(content))
                    df.columns = [c.strip() for c in df.columns]
                    logger.info(f"Fetched {len(df)} instruments from api-scrip-master.csv")
                    return df
                else:
                    logger.error(f"Failed to fetch instruments: {response.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Error fetching instruments: {e}")
            return pd.DataFrame()
    
    def get_active_fno_futures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for active NSE F&O Futures (current month FUTSTK + FUTIDX) based on user insights"""
        logger.info(f"Filtering active F&O futures from {len(df)} total instruments...")
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Show sample data structure
        if not df.empty:
            sample_data = df.head(3).to_dict('records')
            logger.info(f"Sample instruments: {sample_data}")
        
        # Map column names for different CSV formats
        def find_col(*candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None
        
        # Find the correct column names based on available format
        exch_col = find_col('SEM_EXM_EXCH_ID', 'ExchangeSegment', 'EXCH_ID')
        segment_col = find_col('SEM_SEGMENT', 'Segment', 'SEGMENT')
        instrument_col = find_col('SEM_INSTRUMENT_NAME', 'SEM_EXCH_INSTRUMENT_TYPE', 'InstrumentType', 'INSTRUMENT_TYPE')
        expiry_col = find_col('SEM_EXPIRY_DATE', 'ExpiryDate', 'EXPIRY_DATE')
        sid_col = find_col('SEM_SMST_SECURITY_ID', 'SecurityId', 'SECURITY_ID')
        symbol_col = find_col('SEM_TRADING_SYMBOL', 'TradingSymbol', 'SYMBOL_NAME')
        
        logger.info(f"Column mapping: exch={exch_col}, segment={segment_col}, instrument={instrument_col}, expiry={expiry_col}")
        
        # Step 1: Filter only active NSE F&O Futures
        if not all([exch_col, instrument_col, expiry_col]):
            logger.error(f"Missing required columns for F&O filtering")
            return pd.DataFrame()
        
        # Filter for NSE futures
        nse_mask = df[exch_col].astype(str).str.upper().isin(['NSE', 'NSE_FO', 'NSE_FUT'])
        futstk_mask = df[instrument_col].astype(str).str.upper().isin(['FUTSTK', 'FUTIDX'])
        expiry_mask = df[expiry_col].notnull()
        
        fno_fut = df[nse_mask & futstk_mask & expiry_mask]
        logger.info(f"Found {len(fno_fut)} F&O futures with NSE exchange and valid expiry")
        
        if len(fno_fut) == 0:
            logger.error("No active F&O futures found. Check if CSV contains ExchangeSegment=2 data")
            return pd.DataFrame()
        
        # Step 2: Keep only current month futures (avoid expired / too far contracts)
        logger.info("Filtering for current month futures...")
        try:
            fno_fut[expiry_col] = pd.to_datetime(fno_fut[expiry_col], errors='coerce')
            today = pd.Timestamp.now()
            current_month = today.month
            current_year = today.year
            
            active_futures = fno_fut[
                (fno_fut[expiry_col].dt.month == current_month) & 
                (fno_fut[expiry_col].dt.year == current_year)
            ]
            logger.info(f"Filtered {len(active_futures)} active F&O futures for current month ({current_month}/{current_year})")
            
        except Exception as e:
            logger.warning(f"Error filtering by expiry date: {e}")
            # Fallback: use all futures
            active_futures = fno_fut.copy()
            logger.info(f"Using all {len(active_futures)} F&O futures (expiry filtering failed)")
        
        if len(active_futures) == 0:
            logger.warning(f"No futures found for current month {current_month}/{current_year}")
            # Fallback: Get next month futures
            next_month = (current_month % 12) + 1
            next_year = current_year if next_month > current_month else current_year + 1
            
            active_futures = fno_fut[
                (fno_fut[expiry_col].dt.month == next_month) & 
                (fno_fut[expiry_col].dt.year == next_year)
            ]
            logger.info(f"Fallback: Using {len(active_futures)} futures for next month ({next_month}/{next_year})")
        
        # Step 3: Show breakdown by instrument type and add required columns for extraction
        if len(active_futures) > 0:
            type_breakdown = active_futures[instrument_col].value_counts()
            logger.info(f"Active futures breakdown: {type_breakdown.to_dict()}")
            
            # Show sample symbols
            sample_symbols = active_futures[symbol_col].head(10).tolist()
            logger.info(f"Sample active futures: {sample_symbols}")
            
            # Add standard column mappings for securities extraction
            active_futures = active_futures.copy()
            active_futures['SecurityId'] = active_futures[sid_col]
            active_futures['TradingSymbol'] = active_futures[symbol_col] 
            active_futures['ExchangeSegment'] = 2  # NSE Futures = 2
            active_futures['InstrumentType'] = active_futures[instrument_col]
        
        return active_futures
    
    async def download_csv_if_needed(self, url: str, filepath: str) -> bool:
        """Download CSV file if missing or older than 24h"""
        import os
        import time
        
        if os.path.exists(filepath):
            age = time.time() - os.path.getmtime(filepath)
            if age < 86400:  # < 24h
                logger.info(f"Using cached {os.path.basename(filepath)}")
                return True

        logger.info(f"Downloading {os.path.basename(filepath)}...")
        try:
            # Use public session (no auth needed for CSV downloads)
            async with aiohttp.ClientSession() as public_session:
                async with public_session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(text)
                        logger.info(f"✅ Saved {os.path.basename(filepath)}")
                        return True
                    else:
                        logger.error(f"❌ Failed to download {url}: {response.status}")
                        return False
        except Exception as e:
            logger.exception(f"❌ Error downloading {url}: {e}")
            return False

    async def load_equity_instruments(self) -> dict:
        """Load NSE equity instrument master from CSV and create symbol → securityId mapping
        
        Downloads and caches CSV files from Dhan following the working sample pattern
        """
        import os
        import csv
        
        # CSV URLs and local paths
        eq_url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        data_dir = "data"
        eq_file = os.path.join(data_dir, "nse_eq.csv")
        
        try:
            # Download CSV if needed (auto-caching with 24h refresh)
            if not await self.download_csv_if_needed(eq_url, eq_file):
                logger.error("❌ Failed to download equity CSV")
                return {}
            
            # Parse CSV to create symbol → securityId mapping
            equity_mapping = {}
            with open(eq_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Use column names from your working sample
                    symbol = row.get("SEM_TRADING_SYMBOL")
                    security_id = row.get("SEM_SMST_SECURITY_ID")
                    if symbol and security_id:
                        equity_mapping[symbol.strip()] = security_id.strip()
            
            logger.info(f"✅ Loaded {len(equity_mapping)} equity instruments from CSV for symbol→securityId mapping")
            
            # Debug: Show sample mappings
            sample_symbols = list(equity_mapping.keys())[:5]
            for sym in sample_symbols:
                logger.info(f"Sample mapping: {sym} → {equity_mapping[sym]}")
            
            return equity_mapping
            
        except Exception as e:
            logger.exception(f"❌ Error loading equity instruments from CSV: {e}")
            return {}

    def resolve_segment_and_type(self, symbol: str) -> tuple:
        """Resolve correct exchangeSegment and instrumentType for historical data
        
        Returns (exchangeSegment, instrumentType) tuple based on symbol type
        """
        # List of known index symbols that require NSE_INDEX segment and INDEX type
        index_symbols = {
            "NIFTY", "NIFTY 50", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", 
            "NIFTYNXT50", "NIFTYIT", "NIFTYPHARMA", "NIFTYAUTO", "NIFTYMETAL",
            "NIFTYFMCG", "NIFTYENERGY", "NIFTYPSE", "NIFTYREALTY", "NIFTYMEDIA"
        }
        
        if symbol.upper().strip() in index_symbols:
            return "NSE_INDEX", "INDEX"
        else:
            return "NSE_EQ", "EQUITY"

    async def get_historical_data_for_underlying(self, underlying_symbol: str, security_id: int, days: int = 75, interval: str = "1d") -> pd.DataFrame:
        """Fetch historical data using dhanhq SDK (preferred) or REST API fallback
        
        The /v2/chart/history REST endpoint returns 404, so we use SDK when available
        """
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days + 5)
        
        # Determine if this is an index or equity
        index_symbols = {
            "NIFTY", "NIFTY 50", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", 
            "NIFTYNXT50", "NIFTYIT", "NIFTYPHARMA", "NIFTYAUTO", "NIFTYMETAL"
        }
        
        is_index = underlying_symbol.upper().strip() in index_symbols
        
        if is_index:
            exchange_segment = 12  # NSE_INDEX 
            instrument_type = "INDEX"
        else:
            exchange_segment = 1   # NSE_EQ
            instrument_type = "EQUITY"
        
        logger.info(f"📈 Fetching {underlying_symbol} historical data (securityId={security_id}, segment={exchange_segment}, type={instrument_type})")
        
        # Try SDK first if available
        if self.use_sdk and hasattr(self, 'sdk'):
            try:
                logger.info(f"🔄 Using dhanhq SDK for {underlying_symbol}")
                
                # Try different parameter formats to fix DH-905 error
                logger.info(f"🔍 SDK parameters: security_id={security_id}, segment={exchange_segment}, type={instrument_type}")
                
                # Try multiple parameter combinations
                param_attempts = [
                    # Attempt 1: Original format
                    {
                        "security_id": str(security_id),
                        "exchange_segment": exchange_segment,
                        "instrument_type": instrument_type,
                        "from_date": from_date.strftime("%Y-%m-%d"),
                        "to_date": to_date.strftime("%Y-%m-%d")
                    },
                    # Attempt 2: Integer security_id
                    {
                        "security_id": int(security_id),
                        "exchange_segment": exchange_segment,
                        "instrument_type": instrument_type,
                        "from_date": from_date.strftime("%Y-%m-%d"),
                        "to_date": to_date.strftime("%Y-%m-%d")
                    },
                    # Attempt 3: String exchange_segment (this works for equities)
                    {
                        "security_id": str(security_id),
                        "exchange_segment": "NSE_INDEX" if is_index else "NSE_EQ",
                        "instrument_type": instrument_type,
                        "from_date": from_date.strftime("%Y-%m-%d"),
                        "to_date": to_date.strftime("%Y-%m-%d")
                    }
                ]
                
                # Note: SDK only accepts security_id, not symbol parameter
                # For indices, we need correct security_id values from the instrument mapping
                
                response = None
                for i, params in enumerate(param_attempts, 1):
                    try:
                        # Clean params - remove None values
                        clean_params = {k: v for k, v in params.items() if v is not None}
                        logger.info(f"🔍 SDK attempt {i}/{len(param_attempts)} for {underlying_symbol}: {clean_params}")
                        response = self.sdk.historical_daily_data(**clean_params)
                        
                        if response and isinstance(response, dict) and response.get('status') == 'success':
                            logger.info(f"✅ SDK attempt {i} succeeded for {underlying_symbol}")
                            break
                        elif response and isinstance(response, dict) and response.get('status') == 'failure':
                            error_msg = response.get('remarks', {}).get('error_message', 'Unknown error')
                            logger.warning(f"⚠️ SDK attempt {i} failed for {underlying_symbol}: {error_msg}")
                        else:
                            logger.warning(f"⚠️ SDK attempt {i} returned unexpected response for {underlying_symbol}: {response}")
                    except Exception as e:
                        logger.warning(f"⚠️ SDK attempt {i} exception for {underlying_symbol}: {e}")
                        continue
                
                logger.info(f"🔍 SDK Response for {underlying_symbol}: {response}")
                
                # Handle SDK response (check multiple possible structures)
                if response:
                    # Try different response structures
                    candles = None
                    if isinstance(response, str):
                        logger.warning(f"⚠️ {underlying_symbol}: SDK returned string response: {response[:200]}")
                        return pd.DataFrame()
                    elif isinstance(response, dict):
                        # Check for error status first
                        if response.get('status') == 'failure':
                            error_msg = response.get('remarks', {}).get('error_message', 'Unknown error')
                            logger.error(f"❌ {underlying_symbol}: SDK API error: {error_msg}")
                            return pd.DataFrame()
                        elif response.get('status') == 'success':
                            raw_data = response.get('data', {})
                            # SDK returns data as separate arrays, convert to list of dicts
                            if isinstance(raw_data, dict) and 'open' in raw_data:
                                opens = raw_data.get('open', [])
                                highs = raw_data.get('high', [])
                                lows = raw_data.get('low', [])
                                closes = raw_data.get('close', [])
                                volumes = raw_data.get('volume', [])
                                timestamps = raw_data.get('timestamp', [])
                                
                                # Convert arrays to list of candle dicts
                                candles = []
                                for i in range(len(opens)):
                                    candles.append({
                                        'open': opens[i] if i < len(opens) else 0,
                                        'high': highs[i] if i < len(highs) else 0,
                                        'low': lows[i] if i < len(lows) else 0,
                                        'close': closes[i] if i < len(closes) else 0,
                                        'volume': volumes[i] if i < len(volumes) else 0,
                                        'timestamp': timestamps[i] if i < len(timestamps) else 0
                                    })
                            else:
                                candles = raw_data if isinstance(raw_data, list) else []
                        else:
                            # Try to get data anyway for other status types
                            candles = response.get('data', response.get('candles', []))
                    elif isinstance(response, list):
                        candles = response
                    elif hasattr(response, 'data'):
                        candles = response.data
                    
                    if candles and len(candles) > 0:
                        logger.info(f"✅ {underlying_symbol}: Got {len(candles)} candles via SDK")
                        
                        # Convert SDK result to DataFrame 
                        df_data = []
                        for candle in candles:
                            # Handle timestamp conversion
                            timestamp = candle.get('timestamp', 0)
                            try:
                                if timestamp > 1e10:  # Milliseconds
                                    candle_date = datetime.fromtimestamp(timestamp / 1000)
                                else:  # Seconds
                                    candle_date = datetime.fromtimestamp(timestamp)
                            except:
                                candle_date = datetime.now()
                            
                            df_data.append({
                                'date': candle_date,
                                'open': float(candle.get('open', 0)),
                                'high': float(candle.get('high', 0)), 
                                'low': float(candle.get('low', 0)),
                                'close': float(candle.get('close', 0)),
                                'volume': float(candle.get('volume', 0))
                            })
                        
                        df = pd.DataFrame(df_data)
                        df = df.sort_values('date').reset_index(drop=True)
                        return df
                    else:
                        logger.warning(f"⚠️ {underlying_symbol}: SDK returned no candles. Response type: {type(response)}")
                        return pd.DataFrame()
                else:
                    logger.warning(f"⚠️ {underlying_symbol}: SDK returned None")
                    return pd.DataFrame()
                    
            except Exception as e:
                logger.warning(f"⚠️ {underlying_symbol}: SDK error: {e}, trying alternative endpoints")
        
        # SDK failed or not available - try alternative REST endpoints
        alternative_endpoints = [
            f"{self.base_url}/charts/historical",
            f"{self.base_url}/v2/charts/historical", 
            f"{self.base_url}/historical-data",
            f"{self.base_url}/v1/chart/history"
        ]
        
        params = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrumentType": instrument_type,
            "interval": "1day",
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        for endpoint in alternative_endpoints:
            try:
                logger.info(f"🔍 Trying {underlying_symbol} via {endpoint}")
                async with self.session.get(endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        candles = data.get('data', [])
                        
                        if candles:
                            logger.info(f"✅ {underlying_symbol}: Got {len(candles)} candles from {endpoint}")
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(candles)
                            
                            # Normalize date column
                            if 'timestamp' in df.columns:
                                df['date'] = pd.to_datetime(df['timestamp'])
                            elif 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'])
                            
                            # Ensure required columns exist
                            required_cols = ['open', 'high', 'low', 'close', 'volume']
                            for col in required_cols:
                                if col not in df.columns:
                                    df[col] = 0
                            
                            df = df.sort_values('date').reset_index(drop=True)
                            return df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    else:
                        logger.debug(f"❌ {endpoint}: HTTP {response.status}")
                        
            except Exception as e:
                logger.debug(f"❌ {endpoint}: Exception {e}")
                continue
        
        logger.error(f"❌ {underlying_symbol}: All historical data endpoints failed")
        return pd.DataFrame()
    
    def extract_underlying_symbol(self, future_symbol: str) -> str:
        """Extract underlying symbol from future contract name
        
        Examples:
        - ADANIPORTS-Sep2025-FUT -> ADANIPORTS
        - NIFTY-Sep2025-FUT -> NIFTY
        - RELIANCE-Oct2025-FUT -> RELIANCE
        """
        # Remove common future contract suffixes
        underlying = future_symbol.upper()
        
        # Remove date patterns and FUT suffix
        import re
        # Remove patterns like -Sep2025-FUT, -25SEP-FUT, etc.
        underlying = re.sub(r'-[A-Za-z]{3}\d{4}-FUT$', '', underlying)
        underlying = re.sub(r'-\d{2}[A-Za-z]{3}-FUT$', '', underlying) 
        underlying = re.sub(r'-FUT$', '', underlying)
        
        # Remove any remaining date patterns
        underlying = re.sub(r'\d{4}$', '', underlying)
        underlying = re.sub(r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2,4}$', '', underlying)
        
        logger.info(f"Extracted underlying: {future_symbol} -> {underlying}")
        return underlying.strip()