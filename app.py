#!/usr/bin/env python3
"""
app.py - Web server for F&O Scanner Dashboard
Provides real-time web interface for monitoring scanner activity
"""

import os
import sys
import json
import sqlite3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import asyncio
import pickle
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import logging
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import math
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔑 Global in-memory cache for Railway (replaces file-based cache)
# This solves Railway ephemeral storage issue - no more cache/historical_data.pkl
GLOBAL_SCAN_CACHE = {
    'analyzed_data': {},           # Main analysis results: {security_id: {symbol, analysis, etc}}
    'timestamp': None,             # When cache was last updated  
    'total_symbols': 0,            # Total symbols processed
    'successful_fetches': 0,       # Successful data fetches
    'failed_fetches': 0,           # Failed data fetches  
    'breakouts_found': 0,          # Number of breakout signals
    'last_update': None,           # ISO timestamp of last update
    'narrow_cpr_cache': [],        # Cached narrow CPR results
    'cache_source': 'memory'       # Indicates in-memory vs file cache
}

logger.info(f"🚀 Global in-memory cache initialized for Railway compatibility")

# Optional dhanhq SDK import
try:
    from dhanhq import DhanContext, dhanhq
    HAS_DHAN_SDK = True
    logger.info("dhanhq SDK available")
except ImportError:
    HAS_DHAN_SDK = False
    logger.warning("dhanhq SDK not available - using REST API fallback")

# Import real multi-scan modules
MULTI_SCAN_AVAILABLE = False
cache_manager = None
level_calculator = None

try:
    # Try importing real modules first
    from cache_manager import CacheManager
    from scanners.monthly_levels import MonthlyLevelCalculator
    
    cache_manager = CacheManager()
    level_calculator = MonthlyLevelCalculator(cache_manager)
    MULTI_SCAN_AVAILABLE = True
    logger.info("Real multi-scan modules loaded successfully")
except ImportError as e:
    logger.warning(f"External modules not available ({e}), using Railway-compatible inline versions")
    
    # Railway-compatible inline implementations
    class RailwayCacheManager:
        """Railway-compatible cache manager using SQLite only"""
        def __init__(self):
            self.cache_file = 'cache.db'
            self._init_db()
            logger.info("Railway SQLite cache initialized")
        
        def _init_db(self):
            conn = sqlite3.connect(self.cache_file)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    expires_at REAL
                )
            ''')
            conn.commit()
            conn.close()
        
        def set(self, key, value, expire_hours=24):
            import time
            expires_at = time.time() + (expire_hours * 3600)
            try:
                conn = sqlite3.connect(self.cache_file)
                conn.execute('INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)',
                           (key, pickle.dumps(value), expires_at))
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                logger.error(f"Cache set error: {e}")
                return False
        
        def get(self, key):
            import time
            try:
                conn = sqlite3.connect(self.cache_file)
                cursor = conn.execute('SELECT value, expires_at FROM cache WHERE key = ?', (key,))
                row = cursor.fetchone()
                conn.close()
                
                if row and row[1] > time.time():
                    return pickle.loads(row[0])
                return None
            except Exception as e:
                logger.error(f"Cache get error: {e}")
                return None
        
        def get_stats(self):
            return {
                'redis_available': False,
                'sqlite_available': True,
                'current_backend': 'Railway SQLite'
            }
        
        def get_cache_stats(self):
            """Get cache statistics"""
            return self.get_stats()
        
        def health_check(self):
            """Health check for cache system"""
            return {
                'redis': False,
                'sqlite': True
            }
    
    class RailwayMonthlyLevelCalculator:
        """Railway-compatible monthly level calculator"""
        def __init__(self, cache):
            self.cache = cache
            logger.info("Railway MonthlyLevelCalculator initialized")
        
        def calculate_monthly_cpr(self, high, low, close):
            """Calculate CPR levels using Chartink formulas"""
            pivot = (high + low + close) / 3
            bc = (high + low) / 2
            tc = (pivot - bc) + pivot
            
            cpr_width = abs(tc - bc) / pivot * 100 if pivot > 0 else 0
            
            return {
                'pivot': pivot,
                'bc': bc,
                'tc': tc,
                'cpr_width': cpr_width,
                'is_narrow': cpr_width < 0.5
            }
        
        def calculate_and_cache_symbol_levels(self, symbol, ohlc_data, month):
            """Calculate and cache monthly levels for a symbol"""
            high = ohlc_data.get('high', 0)
            low = ohlc_data.get('low', 0)
            close = ohlc_data.get('close', 0)
            
            # Calculate CPR
            cpr = self.calculate_monthly_cpr(high, low, close)
            
            # Calculate pivot points
            pivot = cpr['pivot']
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            levels = {
                'symbol': symbol,
                'month': month,
                'cpr': cpr,
                'pivots': {
                    'pivot': pivot,
                    'r1': r1, 'r2': r2, 'r3': r3,
                    's1': s1, 's2': s2, 's3': s3
                },
                'source_data': {'high': high, 'low': low, 'close': close},
                'calculated_at': datetime.now().isoformat()
            }
            
            # Cache the levels
            cache_key = f"levels:{symbol}:{month}"
            self.cache.set(cache_key, levels, expire_hours=24*35)  # 35 days
            
            return levels
        
        def get_cached_levels(self, symbol, month):
            """Get cached levels for a symbol"""
            cache_key = f"levels:{symbol}:{month}"
            return self.cache.get(cache_key)
        
        def get_symbols_with_narrow_cpr(self, month=None):
            """Get symbols with narrow CPR from cached historical data"""
            if month is None:
                month = datetime.now().strftime('%Y-%m')
                
            narrow_cpr_symbols = []
            
            # Check if we have historical cache data
            import os
            cache_file = 'cache/historical_data.pkl'
            
            if os.path.exists(cache_file):
                try:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        
                    analyzed_data = cache_data.get('analyzed_data', {})
                    logger.error(f"📊 RAILWAY: Checking {len(analyzed_data)} symbols from historical cache for narrow CPR")
                    print(f"📊 RAILWAY DEBUG: Processing {len(analyzed_data)} symbols from cache")
                    
                    for symbol_data in analyzed_data.values():
                        symbol = symbol_data.get('symbol', '')
                        current_analysis = symbol_data.get('current_analysis', {})
                        
                        # Calculate CPR from current analysis if available
                        if 'close' in current_analysis and 'high' in current_analysis and 'low' in current_analysis:
                            high = current_analysis.get('high', 0)
                            low = current_analysis.get('low', 0)  
                            close = current_analysis.get('close', 0)
                            
                            if high > 0 and low > 0 and close > 0:
                                cpr_data = self.calculate_monthly_cpr(high, low, close)
                                
                                if cpr_data['is_narrow']:
                                    narrow_cpr_symbols.append({
                                        'symbol': symbol,
                                        'cpr_width': cpr_data['cpr_width'],
                                        'pivot': cpr_data['pivot'],
                                        'tc': cpr_data['tc'],
                                        'bc': cpr_data['bc'],
                                        'current_price': close
                                    })
                                    
                    logger.error(f"🎯 RAILWAY: Found {len(narrow_cpr_symbols)} symbols with narrow CPR from historical data")
                    print(f"🎯 RAILWAY DEBUG: Narrow CPR results: {len(narrow_cpr_symbols)} symbols")
                    return narrow_cpr_symbols
                    
                except Exception as e:
                    logger.error(f"❌ RAILWAY: Error reading historical cache: {e}")
                    print(f"❌ RAILWAY DEBUG: Cache error: {e}")
            
            # Fallback to demo data if no cache
            logger.error("📭 RAILWAY: No historical cache found, using demo data")
            print("📭 RAILWAY DEBUG: Using demo data fallback")
            return [
                {'symbol': 'TCS', 'cpr_width': 0.416, 'note': 'demo_data'},
                {'symbol': 'HDFCBANK', 'cpr_width': 0.198, 'note': 'demo_data'}, 
                {'symbol': 'INFY', 'cpr_width': 0.000, 'note': 'demo_data'}
            ]
        
        def get_symbols_near_pivot(self, symbols, current_prices, month):
            """Get symbols near monthly pivot (Railway-compatible)"""
            results = []
            for symbol in symbols:
                current_price = current_prices.get(symbol, 0)
                if current_price > 0:
                    # Get cached levels for this symbol
                    cache_key = f"levels:{symbol}:{month}"
                    levels = self.cache.get(cache_key)
                    if levels and 'pivots' in levels:
                        pivot = levels['pivots']['pivot']
                        # Check proximity (1% below to 0.1% above pivot)
                        lower_range = pivot * 0.99
                        upper_range = pivot * 1.001
                        if lower_range <= current_price <= upper_range:
                            results.append({
                                'symbol': symbol,
                                'current_price': current_price,
                                'pivot_level': pivot,
                                'distance_percent': ((current_price - pivot) / pivot) * 100
                            })
            return results
    
    # Initialize Railway-compatible versions
    cache_manager = RailwayCacheManager()
    level_calculator = RailwayMonthlyLevelCalculator(cache_manager)
    MULTI_SCAN_AVAILABLE = True
    logger.info("Railway-compatible multi-scan modules loaded successfully")
    
    # Debug: List available methods
    logger.info(f"Railway cache_manager methods: {[method for method in dir(cache_manager) if not method.startswith('_')]}")
    logger.info(f"Railway level_calculator methods: {[method for method in dir(level_calculator) if not method.startswith('_')]}")
    
    # Debug Flask route registration
    logger.info(f"Flask routes registered: {[rule.rule for rule in app.url_map.iter_rules() if '/api/levels/' in rule.rule]}")
    
    # Force route registration for Railway compatibility
    logger.info("🔄 Forcing route registration for Railway...")
    
    # Register narrow-cpr route manually
    if not any('/api/levels/narrow-cpr' in rule.rule for rule in app.url_map.iter_rules()):
        @app.route('/api/levels/narrow-cpr-railway')
        def railway_narrow_cpr():
            """Railway-specific narrow CPR endpoint"""
            try:
                logger.error("🚂 Railway narrow CPR endpoint called")  # Use ERROR to ensure visibility
                print("🚂 RAILWAY DEBUG: narrow CPR endpoint accessed")  # Also use print for Railway
                
                if not level_calculator:
                    logger.error("❌ Level calculator not available")
                    return jsonify({'error': 'Level calculator not available'}), 500
                
                month = request.args.get('month', datetime.now().strftime('%Y-%m'))
                print(f"🔍 RAILWAY DEBUG: Calling get_symbols_with_narrow_cpr for month {month}")
                narrow_symbols = level_calculator.get_symbols_with_narrow_cpr()
                print(f"🎯 RAILWAY DEBUG: Got {len(narrow_symbols)} narrow CPR symbols")
                logger.error(f"🎯 RAILWAY: Found {len(narrow_symbols)} narrow CPR symbols")
                
                return jsonify({
                    'narrow_cpr_symbols': narrow_symbols,
                    'count': len(narrow_symbols),
                    'month': month,
                    'message': f'Found {len(narrow_symbols)} symbols with narrow CPR'
                })
            except Exception as e:
                logger.error(f"Railway narrow CPR error: {e}")
                return jsonify({'error': str(e)}), 500
                
        logger.info("✅ Railway narrow CPR route registered manually")
    
    # Register test route manually  
    if not any('/api/levels/test' in rule.rule for rule in app.url_map.iter_rules()):
        @app.route('/api/levels/test-railway', methods=['POST'])
        def railway_test_calculation():
            """Railway-specific test calculation endpoint"""
            try:
                logger.info("🚂 Railway test calculation endpoint called")
                if not level_calculator:
                    return jsonify({'error': 'Level calculator not available'}), 500
                    
                data = request.get_json() or {}
                test_ohlc = data.get('ohlc', {'high': 3125.00, 'low': 2875.00, 'close': 3050.00})
                symbol = data.get('symbol', 'TEST_SYMBOL')
                
                levels = level_calculator.calculate_and_cache_symbol_levels(symbol, test_ohlc, 'test-month')
                
                return jsonify({
                    'message': 'Railway test calculation completed',
                    'calculated_levels': levels,
                    'input_data': {'symbol': symbol, 'ohlc': test_ohlc},
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Railway test calculation error: {e}")
                return jsonify({'error': str(e)}), 500
                
        logger.info("✅ Railway test calculation route registered manually")
    
    # Final route check
    railway_routes = [rule.rule for rule in app.url_map.iter_rules() if 'railway' in rule.rule.lower()]
    logger.info(f"✅ Railway-specific routes: {railway_routes}")
except Exception as e:
    logger.error(f"Failed to initialize multi-scan modules: {e}")
    MULTI_SCAN_AVAILABLE = False
    cache_manager = None
    level_calculator = None

# Import DhanHistoricalFetcher from external module if available
try:
    # Temporarily disable external import to fix hanging issue
    # from dhan_fetcher import DhanHistoricalFetcher
    raise ImportError("Using built-in DhanHistoricalFetcher for stability")
except ImportError:
    logger.info("Using built-in DhanHistoricalFetcher")
    # Will use the class defined below

# Historical Data Fetcher Classes
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
            logger.info(f"Starting CSV download from: {url}")
            async with self.session.get(url) as response:
                logger.info(f"Got response with status: {response.status}")
                if response.status == 200:
                    logger.info("Reading response content...")
                    content = await response.text()
                    logger.info(f"Content length: {len(content)} characters")
                    
                    logger.info("Parsing CSV content...")
                    from io import StringIO
                    df = pd.read_csv(StringIO(content))
                    logger.info(f"Initial CSV rows: {len(df)}")
                    
                    logger.info("Cleaning column names...")
                    df.columns = [c.strip() for c in df.columns]
                    logger.info(f"✅ Successfully fetched {len(df)} instruments from api-scrip-master.csv")
                    return df
                else:
                    logger.error(f"Failed to fetch instruments: {response.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.exception(f"❌ Error fetching instruments: {e}")
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

class BreakoutAnalyzer:
    """Analyzes historical data for breakout patterns"""
    
    def __init__(self, lookback: int = 50, ema_short: int = 8, ema_long: int = 13):
        self.lookback = lookback
        self.ema_short = ema_short
        self.ema_long = ema_long
    
    def calculate_technical_indicators(self, df: pd.DataFrame, volume_factor: float = 0.5, price_threshold: float = 50) -> pd.DataFrame:
        """Calculate technical indicators for Chartink-style breakout analysis"""
        if df.empty or len(df) < self.lookback:
            return df
        
        df = df.copy()
        
        # Calculate typical price (Open + High + Close) / 3 - Chartink style
        df['typical_price'] = (df['open'] + df['high'] + df['close']) / 3.0
        
        # Resistance = ceil(max(typical price over lookback period)) - Chartink style
        df['resistance'] = df['typical_price'].rolling(window=self.lookback).max().apply(lambda x: math.ceil(x) if pd.notna(x) else np.nan)
        
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=self.ema_short).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long).mean()
        
        # Calculate previous day volume for volume check
        df['prev_day_volume'] = df['volume'].shift(1)
        
        # Chartink-style breakout conditions
        df['resistance_breakout'] = df['close'] > df['resistance']
        df['ema_crossover'] = df['ema_short'] > df['ema_long']
        df['volume_spike'] = df['volume'] >= (df['prev_day_volume'] * volume_factor)
        df['price_above_threshold'] = df['close'] > price_threshold
        df['bullish_candle'] = df['close'] > df['open']
        
        # Combined breakout signal - ALL conditions must be true
        df['breakout'] = (
            df['resistance_breakout'] & 
            df['ema_crossover'] & 
            df['volume_spike'] & 
            df['price_above_threshold'] & 
            df['bullish_candle']
        )
        
        # Legacy signal column for backward compatibility
        df['signal'] = df['breakout']
        
        return df
    
    def get_current_analysis(self, df: pd.DataFrame) -> Dict:
        """Get current analysis for latest data point with detailed breakout conditions"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        prev_day = df.iloc[-2] if len(df) > 1 else latest
        
        analysis = {
            'symbol': '',
            'date': latest.get('date', ''),
            'open': float(latest.get('open', 0)),
            'high': float(latest.get('high', 0)),
            'low': float(latest.get('low', 0)),
            'close': float(latest.get('close', 0)),
            'volume': int(latest.get('volume', 0)),
            'typical_price': float(latest.get('typical_price', 0)),
            'resistance': float(latest.get('resistance', 0)),
            'ema_short': float(latest.get('ema_short', 0)),
            'ema_long': float(latest.get('ema_long', 0)),
            'prev_day_volume': int(latest.get('prev_day_volume', 0)),
            'change_pct': ((latest.get('close', 0) - prev_day.get('close', 1)) / prev_day.get('close', 1)) * 100,
            
            # Individual breakout conditions
            'resistance_breakout': bool(latest.get('resistance_breakout', False)),
            'ema_crossover': bool(latest.get('ema_crossover', False)),
            'volume_spike': bool(latest.get('volume_spike', False)),
            'price_above_threshold': bool(latest.get('price_above_threshold', False)),
            'bullish_candle': bool(latest.get('bullish_candle', False)),
            
            # Final breakout signal
            'breakout_signal': bool(latest.get('breakout', False)),
            
            # Legacy field
            'signal': bool(latest.get('signal', False))
        }
        
        return analysis

async def fetch_and_analyze_historical_data(fetch_days: int = 75, lookback_period: int = 50, ema_short: int = 8, ema_long: int = 13, volume_factor: float = 0.5, price_threshold: float = 50, max_concurrent: int = 8, max_retries: int = 4):
    """
    Parallelized fetch + analysis of historical data for all F&O underlyings.
    Includes retry logic with exponential backoff for rate-limit and empty response errors.
    """
    import random
    
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    def emit_progress(step, message, current=0, total=0, data=None):
        """Emit progress updates to all connected clients"""
        progress_data = {
            'step': step,
            'message': message,
            'current': current,
            'total': total,
            'progress_percent': int((current / total) * 100) if total > 0 else 0,
            'data': data or {}
        }
        socketio.emit('historical_progress', progress_data)
        logger.info(f"Progress: {message} ({current}/{total})")
    
    try:
        if not client_id or not access_token:
            emit_progress('error', 'Dhan credentials not found in environment variables')
            return {}
        
        emit_progress('starting', f'Starting parallelized historical fetch... (Fetch: {fetch_days} days, Analysis: {lookback_period} days, Concurrent: {max_concurrent}, Retries: {max_retries})')
        
        async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
            # 1. Load instrument master
            emit_progress('instruments', 'Fetching instrument master from Dhan...')
            instruments_df = await fetcher.get_instruments()
            
            if instruments_df.empty:
                emit_progress('error', 'Failed to fetch instruments from Dhan API')
                return {}
            
            emit_progress('filtering', f'Filtering active F&O futures from {len(instruments_df)} total instruments...')
            fno_df = fetcher.get_active_fno_futures(instruments_df)
            
            if fno_df.empty:
                emit_progress('error', 'No active F&O futures found in instrument master')
                return {}
            
            # 2. Load equity mapping
            emit_progress('loading_equity', 'Loading equity instruments for mapping...')
            equity_mapping = await fetcher.load_equity_instruments()
            
            if not equity_mapping:
                emit_progress('error', 'Failed to load equity instruments - cannot resolve securityIds')
                return {}
            
            # 3. Build underlying symbols list (deduplicated)
            underlying_symbols = set()
            for _, row in fno_df.iterrows():
                try:
                    future_symbol = str(row['TradingSymbol'])
                    underlying_symbol = fetcher.extract_underlying_symbol(future_symbol)
                    
                    # Check if we can map to equity
                    security_id = equity_mapping.get(underlying_symbol)
                    if security_id:
                        underlying_symbols.add((underlying_symbol, security_id))
                        
                except Exception as e:
                    logger.warning(f"Error processing F&O row: {e}")
                    continue
            
            underlying_list = list(underlying_symbols)
            total_symbols = len(underlying_list)
            emit_progress('prepared', f'Prepared {total_symbols} unique underlying symbols for parallel fetch', 0, total_symbols)
            
            # 4. Setup parallel processing with retry logic
            analyzer = BreakoutAnalyzer(lookback=lookback_period, ema_short=ema_short, ema_long=ema_long)
            semaphore = asyncio.Semaphore(max_concurrent)
            successful_fetches = 0
            failed_fetches = 0
            
            async def fetch_with_retry(underlying_symbol, security_id):
                """Fetch historical data with retry + exponential backoff"""
                delay = 1
                for attempt in range(1, max_retries + 1):
                    try:
                        # Add jitter to spread load
                        await asyncio.sleep(random.uniform(0.1, 0.4))
                        
                        df = await fetcher.get_historical_data_for_underlying(
                            underlying_symbol, security_id, days=fetch_days
                        )
                        
                        if not df.empty:
                            logger.info(f"✅ {underlying_symbol}: Got {len(df)} days on attempt {attempt}")
                            return df
                        else:
                            logger.warning(f"⚠️ {underlying_symbol}: Empty data (attempt {attempt}/{max_retries})")
                            
                    except Exception as e:
                        logger.error(f"❌ {underlying_symbol}: Error on attempt {attempt}/{max_retries}: {e}")
                    
                    if attempt < max_retries:
                        logger.info(f"🔄 Retrying {underlying_symbol} after {delay}s...")
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff: 1s → 2s → 4s → 8s
                
                logger.error(f"💥 {underlying_symbol}: Failed after {max_retries} attempts")
                return pd.DataFrame()
            
            async def process_symbol(symbol_data, index):
                """Process single symbol with semaphore control"""
                underlying_symbol, security_id = symbol_data
                
                async with semaphore:
                    emit_progress('fetching', f'[{index+1}/{total_symbols}] Fetching {underlying_symbol}...', 
                                index + 1, total_symbols, {
                        'current_symbol': underlying_symbol,
                        'successful': successful_fetches,
                        'failed': failed_fetches
                    })
                    
                    # Fetch with retry logic
                    df = await fetch_with_retry(underlying_symbol, security_id)
                    
                    if df.empty:
                        logger.warning(f"⚠️ {underlying_symbol}: No historical data after {max_retries} attempts")
                        return None
                    
                    # Analyze the data
                    analyzed_df = analyzer.calculate_technical_indicators(df, volume_factor=volume_factor, price_threshold=price_threshold)
                    current_analysis = analyzer.get_current_analysis(analyzed_df)
                    current_analysis['symbol'] = underlying_symbol
                    current_analysis['security_id'] = str(security_id)
                    
                    # Check for breakout
                    is_breakout = current_analysis['breakout_signal']
                    signal_status = "BREAKOUT" if is_breakout else "No Signal"
                    
                    emit_progress('completed_symbol', f'✅ {underlying_symbol}: Close={current_analysis["close"]:.2f}, {signal_status}', 
                                index + 1, total_symbols, {
                        'symbol': underlying_symbol,
                        'close': current_analysis['close'],
                        'signal': is_breakout,
                        'successful': successful_fetches + 1,
                        'failed': failed_fetches
                    })
                    
                    if is_breakout:
                        logger.info(f"🚀 BREAKOUT: {underlying_symbol} Close={current_analysis['close']:.2f}, Resistance={current_analysis['resistance']:.2f}")
                    else:
                        logger.info(f"📊 {underlying_symbol}: No breakout (Close={current_analysis['close']:.2f}, Resistance={current_analysis['resistance']:.2f})")
                    
                    return {
                        'symbol': underlying_symbol,
                        'security_id': str(security_id),
                        'historical_data': analyzed_df,
                        'current_analysis': current_analysis
                    }
            
            # 5. Launch parallel tasks
            emit_progress('parallel_start', f'Starting parallel fetch with {max_concurrent} concurrent connections...')
            
            tasks = [asyncio.create_task(process_symbol(symbol_data, i)) 
                    for i, symbol_data in enumerate(underlying_list)]
            
            # Process in batches to avoid memory issues on Railway
            batch_size = 50  # Process 50 symbols at a time
            analyzed_data = {}
            
            for batch_start in range(0, len(tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
                emit_progress('batch_processing', f'Processing batch {batch_start//batch_size + 1} ({len(batch_tasks)} symbols)...')
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect successful results
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        failed_fetches += 1
                        logger.error(f"Task failed: {result}")
                    elif result is not None:
                        successful_fetches += 1
                        analyzed_data[result['security_id']] = result
                
                # Progress update after batch
                emit_progress('batch_complete', f'Batch {batch_start//batch_size + 1} complete: {successful_fetches} successful, {failed_fetches} failed')
            
            # 6. Final summary
            breakouts_found = sum(1 for data in analyzed_data.values() if data['current_analysis']['breakout_signal'])
            
            emit_progress('summary', f'Parallelized analysis completed: {successful_fetches} successful, {failed_fetches} failed, {breakouts_found} breakouts found', 
                        total_symbols, total_symbols, {
                'total_analyzed': len(analyzed_data),
                'successful': successful_fetches,
                'failed': failed_fetches,
                'breakouts_found': breakouts_found,
                'concurrent_used': max_concurrent,
                'retries_per_symbol': max_retries
            })
            
            # Emit breakout results for multi-scan dashboard
            socketio.emit('breakout_results', {
                'count': breakouts_found,
                'successful': successful_fetches,
                'analyzed': len(analyzed_data),
                'completed_at': datetime.now().isoformat()
            })
            
            # Cache results
            try:
                cache_data = {
                    'analyzed_data': analyzed_data,
                    'timestamp': datetime.now().isoformat(),
                    'total_symbols': total_symbols,
                    'successful_fetches': successful_fetches,
                    'failed_fetches': failed_fetches,
                    'breakouts_found': breakouts_found
                }
                
                os.makedirs('cache', exist_ok=True)
                with open('cache/historical_data.pkl', 'wb') as f:
                    pickle.dump(cache_data, f)
                    
                emit_progress('cached', f'Parallelized results cached: {len(analyzed_data)} symbols saved')
                logger.info(f"✅ Cached {len(analyzed_data)} analyzed symbols with parallelized approach")
            except Exception as cache_error:
                logger.warning(f"Failed to cache data: {cache_error}")
            
            return analyzed_data
            
    except Exception as e:
        emit_progress('error', f'Critical error in parallelized fetch: {str(e)}')
        logger.exception("Critical error in parallelized historical data fetch")
        return {}

# Global state
scanner_state = {
    'running': False,
    'connected_clients': 0,
    'active_symbols': 0,
    'last_update': None,
    'alerts': [],
    'scanner_data': [],
    'historical_data': {},
    'historical_analysis_running': False
}

# Database connection
def get_db_connection():
    """Get database connection for alerts"""
    conn = sqlite3.connect('alerts.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database if not exists"""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            sid TEXT,
            symbol TEXT,
            message TEXT,
            strategy TEXT DEFAULT 'breakout'
        );
    """)
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get scanner status"""
    return jsonify({
        'running': scanner_state['running'],
        'connected_clients': scanner_state['connected_clients'],
        'active_symbols': scanner_state['active_symbols'],
        'last_update': scanner_state['last_update']
    })

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts from database"""
    limit = request.args.get('limit', 100, type=int)
    conn = get_db_connection()
    alerts = conn.execute(
        'SELECT * FROM alerts ORDER BY id DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    
    return jsonify([dict(alert) for alert in alerts])

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        config = request.json
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    scanner_state['connected_clients'] += 1
    emit('connected', {'data': 'Connected to scanner server'})
    
    # Send initial data
    emit('stats', {
        'activeSymbols': scanner_state['active_symbols'],
        'totalAlerts': len(scanner_state['alerts'])
    })
    
    # Send recent alerts
    if scanner_state['alerts']:
        for alert in scanner_state['alerts'][-10:]:
            emit('alert', alert)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    scanner_state['connected_clients'] -= 1

@socketio.on('start_scanner')
def handle_start_scanner():
    """Start the scanner"""
    if not scanner_state['running']:
        scanner_state['running'] = True
        emit('scanner_status', {'running': True}, broadcast=True)
        # Here you would trigger the actual scanner start
        # For now, we'll simulate with mock data
        start_mock_scanner()

@socketio.on('stop_scanner')
def handle_stop_scanner():
    """Stop the scanner"""
    if scanner_state['running']:
        scanner_state['running'] = False
        emit('scanner_status', {'running': False}, broadcast=True)

@socketio.on('refresh_data')
def handle_refresh_data():
    """Refresh scanner data"""
    emit('scanner_data', scanner_state['scanner_data'])

@socketio.on('update_filters')
def handle_update_filters(filters):
    """Update scanner filters"""
    # Here you would apply filters to the scanner
    print(f"Updating filters: {filters}")
    emit('filters_updated', {'success': True})

@socketio.on('get_symbol_details')
def handle_get_symbol_details(symbol):
    """Get detailed information for a symbol"""
    # Mock data for demonstration
    details = {
        'symbol': symbol,
        'ltp': 1234.56,
        'volume': 123456,
        'resistance': 1250.00,
        'ema8': 1230.00,
        'ema13': 1225.00
    }
    return details

@socketio.on('update_settings')
def handle_update_settings(settings):
    """Update scanner settings"""
    print(f"Updating settings: {settings}")
    emit('settings_updated', {'success': True})

# Mock scanner for demonstration
def start_mock_scanner():
    """Simulate scanner activity with mock data"""
    def scanner_loop():
        symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'RELIANCE', 
                   'TCS', 'HDFC', 'INFY', 'ICICIBANK', 'SBIN']
        
        while scanner_state['running']:
            # Generate mock data
            mock_data = []
            for symbol in symbols:
                import random
                base_price = random.uniform(1000, 5000)
                mock_data.append({
                    'symbol': symbol,
                    'ltp': base_price,
                    'change': random.uniform(-5, 5),
                    'volume': random.randint(10000, 1000000),
                    'resistance': base_price * 1.02,
                    'ema8': base_price * 0.99,
                    'ema13': base_price * 0.98,
                    'signal': 'BREAKOUT' if random.random() > 0.8 else None
                })
            
            scanner_state['scanner_data'] = mock_data
            scanner_state['active_symbols'] = len(symbols)
            scanner_state['last_update'] = datetime.now().isoformat()
            
            # Emit data to all connected clients
            socketio.emit('scanner_data', mock_data)
            socketio.emit('stats', {
                'activeSymbols': len(symbols),
                'totalAlerts': len(scanner_state['alerts'])
            })
            
            # Simulate random alerts
            if random.random() > 0.9:
                alert = {
                    'symbol': random.choice(symbols),
                    'message': 'Breakout detected!',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'breakout'
                }
                scanner_state['alerts'].append(alert)
                socketio.emit('alert', alert)
                
                # Save to database
                conn = get_db_connection()
                conn.execute(
                    'INSERT INTO alerts (ts, sid, symbol, message) VALUES (?, ?, ?, ?)',
                    (alert['timestamp'], '0', alert['symbol'], alert['message'])
                )
                conn.commit()
                conn.close()
            
            time.sleep(5)  # Update every 5 seconds
    
    # Start scanner in background thread
    scanner_thread = threading.Thread(target=scanner_loop)
    scanner_thread.daemon = True
    scanner_thread.start()

# Scanner integration
def integrate_with_scanner():
    """
    Integration point with the actual scanner.py
    This function would be called by scanner.py to send real data
    """
    def send_scanner_update(data):
        """Send scanner data update to all connected clients"""
        scanner_state['scanner_data'] = data
        socketio.emit('scanner_data', data)
    
    def send_alert(alert):
        """Send alert to all connected clients"""
        scanner_state['alerts'].append(alert)
        socketio.emit('alert', alert)
    
    return send_scanner_update, send_alert

# API for scanner.py to send data
@app.route('/api/scanner/update', methods=['POST'])
def scanner_update():
    """Receive updates from scanner.py"""
    data = request.json
    
    if data.get('type') == 'data':
        scanner_state['scanner_data'] = data.get('data', [])
        socketio.emit('scanner_data', scanner_state['scanner_data'])
    
    elif data.get('type') == 'alert':
        alert = data.get('alert')
        scanner_state['alerts'].append(alert)
        socketio.emit('alert', alert)
    
    elif data.get('type') == 'stats':
        stats = data.get('stats')
        scanner_state.update(stats)
        socketio.emit('stats', stats)
    
    return jsonify({'success': True})

@app.route('/api/historical/fetch', methods=['POST'])
def fetch_historical():
    """Fetch historical data for analysis"""
    if scanner_state['historical_analysis_running']:
        return jsonify({'error': 'Historical analysis already running'}), 400
    
    # Get configuration parameters from request
    config = request.json or {}
    fetch_days = config.get('fetch_days', 75)
    lookback_period = config.get('lookback_period', 50)
    ema_short = config.get('ema_short', 8)
    ema_long = config.get('ema_long', 13)
    volume_factor = config.get('volume_factor', 0.5)
    price_threshold = config.get('price_threshold', 50)
    
    def run_historical_fetch():
        scanner_state['historical_analysis_running'] = True
        try:
            # Run async function in thread with configuration
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            historical_data = loop.run_until_complete(fetch_and_analyze_historical_data(
                fetch_days=fetch_days,
                lookback_period=lookback_period, 
                ema_short=ema_short,
                ema_long=ema_long,
                volume_factor=volume_factor,
                price_threshold=price_threshold
            ))
            
            # Update scanner state
            scanner_state['historical_data'] = historical_data
            scanner_state['active_symbols'] = len(historical_data)
            
            # Convert to scanner data format
            scanner_data = []
            for sec_id, data in historical_data.items():
                analysis = data['current_analysis']
                scanner_data.append({
                    'symbol': analysis['symbol'],
                    'ltp': analysis['close'],
                    'change': analysis['change_pct'],
                    'volume': analysis['volume'],
                    'resistance': analysis['resistance'],
                    'ema8': analysis['ema_short'],
                    'ema13': analysis['ema_long'],
                    'signal': 'BREAKOUT' if analysis['breakout_signal'] else None
                })
            
            scanner_state['scanner_data'] = scanner_data
            scanner_state['last_update'] = datetime.now().isoformat()
            
            # Emit updates to all clients
            socketio.emit('scanner_data', scanner_data)
            socketio.emit('stats', {
                'activeSymbols': len(historical_data),
                'totalAlerts': len(scanner_state['alerts'])
            })
            
            # Generate alerts for breakout signals
            for data in historical_data.values():
                analysis = data['current_analysis']
                if analysis['breakout_signal']:
                    alert = {
                        'symbol': analysis['symbol'],
                        'message': f"Historical Breakout - Close: {analysis['close']:.2f}, Resistance: {analysis['resistance']:.2f}",
                        'timestamp': datetime.now().isoformat(),
                        'type': 'breakout'
                    }
                    scanner_state['alerts'].append(alert)
                    socketio.emit('alert', alert)
            
            loop.close()
            
        except Exception as e:
            print(f"Historical fetch error: {e}")
        finally:
            scanner_state['historical_analysis_running'] = False
    
    # Start in background thread
    thread = threading.Thread(target=run_historical_fetch, daemon=True)
    thread.start()
    
    return jsonify({
        'message': 'Historical data fetch started',
        'config': {
            'fetch_days': fetch_days,
            'lookback_period': lookback_period,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'volume_factor': volume_factor,
            'price_threshold': price_threshold
        }
    })

@app.route('/api/historical/status')
def historical_status():
    """Get historical data fetch status"""
    return jsonify({
        'running': scanner_state['historical_analysis_running'],
        'symbols_count': len(scanner_state['historical_data']),
        'last_update': scanner_state['last_update']
    })

@app.route('/api/historical/data')
def get_historical_data():
    """Get current historical data"""
    return jsonify({
        'data': scanner_state['scanner_data'],
        'symbols_count': len(scanner_state['historical_data']),
        'last_update': scanner_state['last_update']
    })

@app.route('/api/analysis/config', methods=['GET'])
def get_analysis_config():
    """Get current analysis configuration defaults"""
    return jsonify({
        'fetch_days': 75,
        'lookback_period': 50,
        'ema_short': 8,
        'ema_long': 13,
        'volume_factor': 0.5,
        'price_threshold': 50,
        'description': {
            'fetch_days': 'Calendar days of data to fetch from API',
            'lookback_period': 'Days used for resistance calculation', 
            'ema_short': 'Short EMA period for crossover',
            'ema_long': 'Long EMA period for crossover',
            'volume_factor': 'Minimum volume multiplier vs previous day',
            'price_threshold': 'Minimum stock price filter'
        }
    })

@app.route('/api/cache/status')
def cache_status():
    """Check cache system status"""
    try:
        if not MULTI_SCAN_AVAILABLE or not cache_manager:
            return jsonify({
                'error': 'Multi-scan modules not available',
                'message': 'Cache manager not loaded',
                'cache_available': False
            }), 500
            
        # Get cache system status
        stats = cache_manager.get_cache_stats()
        health = cache_manager.health_check()
        
        # Also check historical data cache file
        cache_file = 'cache/historical_data.pkl'
        file_status = {'cache_exists': False}
        
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            cache_hours = cache_age / 3600
            file_size = os.path.getsize(cache_file)
            
            # Try to read cache content
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                symbols_count = len(cache_data.get('analyzed_data', {}))
                timestamp = cache_data.get('timestamp', 'Unknown')
                breakouts = sum(1 for data in cache_data.get('analyzed_data', {}).values() 
                              if data.get('current_analysis', {}).get('breakout_signal', False))
            except:
                symbols_count = 0
                timestamp = 'Unknown'
                breakouts = 0
            
            file_status = {
                'cache_exists': True,
                'age_hours': round(cache_hours, 1),
                'file_size_mb': round(file_size / (1024*1024), 2),
                'symbols_cached': symbols_count,
                'cache_timestamp': timestamp,
                'breakouts_cached': breakouts,
                'is_fresh': cache_hours < 24,
                'status': 'fresh' if cache_hours < 24 else 'expired'
            }
        
        return jsonify({
            'cache_system': {
                'available': True,
                'stats': stats,
                'health': health
            },
            'historical_file_cache': file_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/instruments')
def debug_instruments():
    """Debug endpoint to check instrument master structure"""
    try:
        import requests
        url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = [c.strip() for c in df.columns]
        
        # Get sample data
        sample_rows = df.head(5).to_dict('records') if len(df) > 0 else []
        
        return jsonify({
            'total_instruments': len(df),
            'columns': list(df.columns),
            'sample_data': sample_rows,
            'unique_segments': df[df.columns[0]].unique()[:10].tolist() if len(df.columns) > 0 else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/files')
def debug_files():
    """Debug endpoint to check what files exist on Railway"""
    try:
        import os
        import glob
        
        # Check current directory
        cwd = os.getcwd()
        
        # List key files
        files_check = {
            'cache_manager.py': os.path.exists('cache_manager.py'),
            'scanners/': os.path.exists('scanners'),
            'scanners/__init__.py': os.path.exists('scanners/__init__.py'),
            'scanners/monthly_levels.py': os.path.exists('scanners/monthly_levels.py'),
            'current_directory': cwd,
            'directory_contents': os.listdir('.'),
            'scanners_contents': os.listdir('scanners') if os.path.exists('scanners') else 'NOT_FOUND'
        }
        
        # Try import test
        try:
            import cache_manager
            files_check['cache_manager_import'] = 'SUCCESS'
        except Exception as e:
            files_check['cache_manager_import'] = f'FAILED: {str(e)}'
            
        try:
            from scanners.monthly_levels import MonthlyLevelCalculator
            files_check['monthly_levels_import'] = 'SUCCESS'
        except Exception as e:
            files_check['monthly_levels_import'] = f'FAILED: {str(e)}'
        
        return jsonify(files_check)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/fetch-test')
def debug_fetch_test():
    """Debug endpoint to test simplified historical fetch and log each step"""
    try:
        import asyncio
        from datetime import datetime
        
        debug_logs = []
        
        def log_step(message):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] {message}"
            debug_logs.append(log_entry)
            logger.info(log_entry)
        
        async def test_fetch():
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            
            log_step("🔍 Starting simplified fetch test")
            log_step(f"Environment: {'Railway' if os.getenv('PORT') else 'Local'}")
            log_step(f"Credentials: {'Available' if client_id and access_token else 'Missing'}")
            
            if not client_id or not access_token:
                log_step("❌ STOPPED: No DHAN credentials")
                return debug_logs
            
            log_step("✅ STEP 1: Credentials OK")
            
            try:
                # Test DhanHistoricalFetcher initialization
                log_step("🔄 STEP 2: Creating DhanHistoricalFetcher...")
                async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
                    log_step("✅ STEP 2: DhanHistoricalFetcher created successfully")
                    
                    # Test instrument fetching
                    log_step("🔄 STEP 3: Fetching instrument master...")
                    instruments_df = await fetcher.get_instruments()
                    
                    if instruments_df.empty:
                        log_step("❌ STEP 3: Instrument fetch FAILED")
                        return debug_logs
                    
                    log_step(f"✅ STEP 3: Got {len(instruments_df)} instruments")
                    
                    # Test F&O filtering
                    log_step("🔄 STEP 4: Filtering F&O futures...")
                    fno_df = fetcher.get_active_fno_futures(instruments_df)
                    
                    if fno_df.empty:
                        log_step("❌ STEP 4: F&O filtering FAILED")
                        return debug_logs
                        
                    log_step(f"✅ STEP 4: Got {len(fno_df)} F&O futures")
                    
                    # Test equity mapping
                    log_step("🔄 STEP 5: Loading equity instruments...")
                    equity_mapping = await fetcher.load_equity_instruments()
                    
                    if not equity_mapping:
                        log_step("❌ STEP 5: Equity mapping FAILED")
                        return debug_logs
                        
                    log_step(f"✅ STEP 5: Got {len(equity_mapping)} equity mappings")
                    
                    # Test single historical data fetch
                    log_step("🔄 STEP 6: Testing single historical data fetch...")
                    
                    # Try RELIANCE as a safe test
                    test_symbol = "RELIANCE"
                    security_id = equity_mapping.get(test_symbol)
                    
                    if not security_id:
                        log_step(f"❌ STEP 6: No securityId for {test_symbol}")
                        # Try first available symbol
                        test_symbol = list(equity_mapping.keys())[0]
                        security_id = equity_mapping[test_symbol]
                        log_step(f"🔄 STEP 6: Trying {test_symbol} instead (securityId: {security_id})")
                    
                    log_step(f"🔄 STEP 6: Fetching {test_symbol} historical data (securityId: {security_id})...")
                    
                    # Add timeout to prevent hanging
                    try:
                        hist_df = await asyncio.wait_for(
                            fetcher.get_historical_data_for_underlying(test_symbol, security_id, days=10),
                            timeout=30.0  # 30 second timeout
                        )
                        
                        if hist_df.empty:
                            log_step(f"❌ STEP 6: No historical data for {test_symbol}")
                        else:
                            log_step(f"✅ STEP 6: Got {len(hist_df)} days of data for {test_symbol}")
                            log_step(f"📊 Sample: Close prices {hist_df['close'].tail(3).tolist()}")
                            
                    except asyncio.TimeoutError:
                        log_step(f"⏰ STEP 6: TIMEOUT after 30s fetching {test_symbol}")
                    except Exception as e:
                        log_step(f"❌ STEP 6: ERROR fetching {test_symbol}: {str(e)}")
                    
                    log_step("🎉 FETCH TEST COMPLETED")
                    
            except Exception as e:
                log_step(f"💥 CRITICAL ERROR: {str(e)}")
                import traceback
                log_step(f"📍 Traceback: {traceback.format_exc()}")
            
            return debug_logs
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_logs = loop.run_until_complete(test_fetch())
        loop.close()
        
        return jsonify({
            'test_completed': True,
            'environment': 'Railway' if os.getenv('PORT') else 'Local',
            'timestamp': datetime.now().isoformat(),
            'total_steps': len(result_logs),
            'logs': result_logs
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'test_completed': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/debug/equity-mapping')
def debug_equity_mapping():
    """Debug endpoint to test equity instrument loading"""
    try:
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not client_id or not access_token:
            return jsonify({'error': 'Dhan credentials not configured'}), 400
        
        async def test_equity_loading():
            async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
                equity_mapping = await fetcher.load_equity_instruments()
                return equity_mapping
        
        # Run the async function
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        equity_mapping = loop.run_until_complete(test_equity_loading())
        loop.close()
        
        # Sample some results
        sample_mappings = dict(list(equity_mapping.items())[:10]) if equity_mapping else {}
        
        return jsonify({
            'total_mappings': len(equity_mapping),
            'sample_mappings': sample_mappings,
            'status': 'success' if equity_mapping else 'failed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Monthly Levels API Endpoints
@app.route('/api/levels/calculate', methods=['POST'])
def calculate_monthly_levels():
    """Manually trigger monthly level calculation for all symbols"""
    try:
        if not MULTI_SCAN_AVAILABLE:
            return jsonify({
                'error': 'Multi-scan modules not available',
                'message': 'Cache manager or monthly levels calculator not loaded'
            }), 500
            
        # Check for credentials
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not client_id or not access_token:
            return jsonify({
                'error': 'DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables required',
                'message': 'Set real Dhan credentials to calculate levels with live data'
            }), 400
        
        # Run the calculation in background thread
        def run_calculation():
            import asyncio
            job = PremarketJob()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(job.calculate_monthly_levels_for_all_symbols())
            loop.close()
            return result
        
        # Start calculation in thread to avoid blocking
        calculation_thread = threading.Thread(target=run_calculation, daemon=True)
        calculation_thread.start()
        
        return jsonify({
            'message': 'Monthly level calculation started with real Dhan data',
            'status': 'running',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting level calculation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/<symbol>')
def get_symbol_levels(symbol):
    """Get cached monthly levels for a specific symbol"""
    try:
        if not MULTI_SCAN_AVAILABLE or not level_calculator:
            return jsonify({
                'error': 'Multi-scan modules not available',
                'message': 'Cache manager or monthly levels calculator not loaded'
            }), 500
        
        # Get current month by default, or from query param
        month = request.args.get('month', datetime.now().strftime('%Y-%m'))
        
        levels = level_calculator.get_cached_levels(symbol.upper(), month)
        
        if levels:
            return jsonify({
                'symbol': symbol.upper(),
                'month': month,
                'levels': levels,
                'retrieved_at': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': f'No cached levels found for {symbol.upper()} in {month}',
                'message': 'Run level calculation first to populate cache with real data',
                'symbol': symbol.upper(),
                'month': month
            }), 404
            
    except Exception as e:
        logger.error(f"Error retrieving levels for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/narrow-cpr')
def get_narrow_cpr_symbols():
    """Get all symbols with narrow CPR for current month"""
    try:
        logger.info(f"🔍 Narrow CPR endpoint called - MULTI_SCAN_AVAILABLE: {MULTI_SCAN_AVAILABLE}")
        logger.info(f"🔍 cache_manager available: {cache_manager is not None}")
        logger.info(f"🔍 level_calculator available: {level_calculator is not None}")
        
        if not MULTI_SCAN_AVAILABLE or not cache_manager:
            logger.error(f"❌ Multi-scan not available: MULTI_SCAN_AVAILABLE={MULTI_SCAN_AVAILABLE}, cache_manager={cache_manager is not None}")
            return jsonify({
                'error': 'Multi-scan modules not available',
                'message': 'Cache manager or monthly levels calculator not loaded'
            }), 500
        
        # Get month from query param
        month = request.args.get('month', datetime.now().strftime('%Y-%m'))
        
        # Get scan results from cache
        scan_cache_key = f"scan_results:{month}"
        scan_results = cache_manager.get(scan_cache_key)
        
        if scan_results and 'narrow_cpr' in scan_results:
            results = scan_results['narrow_cpr']
            
            # Emit WebSocket event for real-time updates
            socketio.emit('cpr_results', {
                'results': results,
                'count': len(results),
                'month': month,
                'last_updated': scan_results.get('last_updated')
            })
            
            return jsonify({
                'month': month,
                'narrow_cpr_symbols': results,
                'total_symbols': scan_results.get('total_symbols', 0),
                'count': len(results),
                'last_updated': scan_results.get('last_updated'),
                'retrieved_at': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'message': 'No narrow CPR scan results found. Run level calculation with real Dhan credentials first.',
                'month': month,
                'narrow_cpr_symbols': [],
                'count': 0,
                'help': 'Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables and call POST /api/levels/calculate'
            }), 404
            
    except Exception as e:
        logger.error(f"Error retrieving narrow CPR symbols: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/near-pivot', methods=['POST'])
def get_symbols_near_pivot():
    """Get symbols currently trading near monthly pivot"""
    try:
        if not MULTI_SCAN_AVAILABLE or not cache_manager or not level_calculator:
            return jsonify({
                'error': 'Multi-scan modules not available',
                'message': 'Cache manager or monthly levels calculator not loaded'
            }), 500
        
        # Get current prices from request body
        data = request.get_json() or {}
        current_prices = data.get('current_prices', {})
        symbols = data.get('symbols', list(current_prices.keys()))
        
        if not current_prices:
            return jsonify({'error': 'current_prices required in request body'}), 400
        
        month = request.args.get('month', datetime.now().strftime('%Y-%m'))
        
        # Get symbols near pivot
        near_pivot_symbols = level_calculator.get_symbols_near_pivot(symbols, current_prices, month)
        
        # Emit WebSocket event for real-time updates
        socketio.emit('pivot_results', {
            'results': near_pivot_symbols,
            'count': len(near_pivot_symbols),
            'month': month,
            'input_symbols': len(symbols)
        })
        
        return jsonify({
            'month': month,
            'near_pivot_symbols': near_pivot_symbols,
            'count': len(near_pivot_symbols),
            'input_symbols': len(symbols),
            'retrieved_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking symbols near pivot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/premarket-summary')
def get_premarket_summary():
    """Get latest pre-market job execution summary"""
    try:
        if not MULTI_SCAN_AVAILABLE:
            return jsonify({
                'error': 'Multi-scan modules not available',
                'message': 'Premarket job not available'
            }), 500
        
        from premarket_job import PremarketJob
        job = PremarketJob()
        
        # Get latest results
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(job.get_latest_results())
        loop.close()
        
        # Add health check
        health = loop.run_until_complete(job.health_check())
        results['health'] = health
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting premarket summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/test', methods=['POST'])
def test_level_calculation():
    """Test monthly level calculation with sample data"""
    try:
        if not MULTI_SCAN_AVAILABLE or not level_calculator:
            return jsonify({
                'error': 'Multi-scan modules not available',
                'message': 'Monthly levels calculator not loaded'
            }), 500
        
        # Get test data from request
        data = request.get_json() or {}
        
        # Default test data if none provided
        test_ohlc = data.get('ohlc', {
            'high': 3125.00,
            'low': 2875.00,
            'close': 3050.00,
            'open': 2890.50
        })
        
        symbol = data.get('symbol', 'TEST_SYMBOL')
        
        # Calculate levels
        levels = level_calculator.calculate_and_cache_symbol_levels(
            symbol,
            test_ohlc,
            'test-month'
        )
        
        return jsonify({
            'message': 'Test calculation completed',
            'input_data': {
                'symbol': symbol,
                'ohlc': test_ohlc
            },
            'calculated_levels': levels,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in test calculation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/websocket/live-market-test', methods=['POST'])
def live_market_websocket_test():
    """
    🚀 Live Market WebSocket Test Endpoint for Railway
    Tests real-time WebSocket functionality with RELIANCE stock data
    """
    try:
        logger.info("🚀 Starting live market WebSocket test...")
        
        # Check credentials
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not client_id or not access_token:
            # Emit demo WebSocket event
            socketio.emit('live_market_test', {
                'step': 'credentials_missing',
                'message': 'Demo mode: No credentials found',
                'status': 'demo',
                'timestamp': datetime.now().isoformat()
            })
            return jsonify({
                'status': 'demo_mode',
                'message': 'WebSocket test running in demo mode - check browser console for events',
                'websocket_event': 'live_market_test emitted'
            })
        
        # Get request parameters
        data = request.get_json() or {}
        symbol = data.get('symbol', 'RELIANCE')
        test_duration = data.get('duration', 30)  # seconds
        
        def run_live_test():
            """Background task to run live market test with real data"""
            try:
                # Import SDK
                from dhanhq import DhanContext, dhanhq

                # Initialize SDK
                ctx = DhanContext(client_id, access_token)
                dhan = dhanhq(ctx)
                
                # Emit start event
                socketio.emit('live_market_test', {
                    'step': 'starting',
                    'message': f'Starting live test for {symbol} (Duration: {test_duration}s)',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Get equity instruments for symbol resolution
                logger.info(f"🔍 Resolving {symbol} security ID...")
                instruments_df = pd.read_csv('data/nse_eq.csv')
                
                # Find RELIANCE security ID
                reliance_row = instruments_df[instruments_df['SEM_TRADING_SYMBOL'] == symbol]
                if reliance_row.empty:
                    # Try alternative column names
                    reliance_row = instruments_df[instruments_df['SYMBOL'] == symbol] if 'SYMBOL' in instruments_df.columns else pd.DataFrame()
                
                if not reliance_row.empty:
                    security_id = str(reliance_row.iloc[0]['SEM_SMST_SECURITY_ID'] if 'SEM_SMST_SECURITY_ID' in reliance_row.columns 
                                    else reliance_row.iloc[0]['SECURITY_ID'] if 'SECURITY_ID' in reliance_row.columns else '1333')
                    
                    # Emit security ID found
                    socketio.emit('live_market_test', {
                        'step': 'symbol_resolved',
                        'message': f'{symbol} resolved to Security ID: {security_id}',
                        'security_id': security_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Fetch live historical data (last 10 days)
                    from_date = datetime.now() - timedelta(days=10)
                    to_date = datetime.now()
                    
                    socketio.emit('live_market_test', {
                        'step': 'fetching_data',
                        'message': f'Fetching {symbol} data for last 10 days...',
                        'from_date': from_date.strftime('%Y-%m-%d'),
                        'to_date': to_date.strftime('%Y-%m-%d'),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # SDK call
                    response = dhan.historical_daily_data(
                        security_id=security_id,
                        exchange_segment="NSE_EQ",
                        instrument_type="EQUITY",
                        from_date=from_date.strftime("%Y-%m-%d"),
                        to_date=to_date.strftime("%Y-%m-%d")
                    )
                    
                    if response['status'] == 'success' and response['data']:
                        data = response['data']
                        latest_close = data['close'][-1] if data['close'] else 0
                        latest_volume = data['volume'][-1] if data['volume'] else 0
                        data_points = len(data['close']) if data['close'] else 0
                        
                        # Calculate simple metrics
                        avg_close = sum(data['close'][-5:]) / 5 if len(data['close']) >= 5 else latest_close
                        price_change_pct = ((latest_close - avg_close) / avg_close * 100) if avg_close > 0 else 0
                        
                        # Emit success with real data
                        socketio.emit('live_market_test', {
                            'step': 'data_success',
                            'message': f'✅ {symbol} live data fetched successfully!',
                            'data': {
                                'symbol': symbol,
                                'latest_close': latest_close,
                                'latest_volume': latest_volume,
                                'data_points': data_points,
                                'avg_5day_close': round(avg_close, 2),
                                'price_change_pct': round(price_change_pct, 2)
                            },
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Simulate real-time updates for test duration
                        import time
                        updates_sent = 0
                        start_time = time.time()
                        
                        while time.time() - start_time < test_duration and updates_sent < 10:
                            time.sleep(3)  # Update every 3 seconds
                            updates_sent += 1
                            
                            # Simulate price movement (±0.5%)
                            import random
                            simulated_price = latest_close * (1 + random.uniform(-0.005, 0.005))
                            
                            socketio.emit('live_market_test', {
                                'step': 'live_update',
                                'message': f'Live update #{updates_sent} for {symbol}',
                                'data': {
                                    'symbol': symbol,
                                    'simulated_price': round(simulated_price, 2),
                                    'original_close': latest_close,
                                    'update_number': updates_sent
                                },
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # Final completion event
                        socketio.emit('live_market_test', {
                            'step': 'completed',
                            'message': f'🎉 Live market WebSocket test completed! ({updates_sent} updates sent)',
                            'summary': {
                                'symbol': symbol,
                                'total_updates': updates_sent,
                                'duration': round(time.time() - start_time, 1),
                                'websocket_status': 'working',
                                'data_status': 'live'
                            },
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        # Emit data fetch error
                        socketio.emit('live_market_test', {
                            'step': 'data_error',
                            'message': f'❌ Failed to fetch {symbol} data: {response}',
                            'error': response,
                            'timestamp': datetime.now().isoformat()
                        })
                else:
                    # Symbol not found
                    socketio.emit('live_market_test', {
                        'step': 'symbol_error',
                        'message': f'❌ {symbol} not found in equity instruments',
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Live test error: {e}")
                socketio.emit('live_market_test', {
                    'step': 'error',
                    'message': f'❌ Live market test error: {str(e)}',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Start background test
        threading.Thread(target=run_live_test, daemon=True).start()
        
        return jsonify({
            'status': 'test_started',
            'message': f'Live market WebSocket test started for {symbol}',
            'websocket_events': 'live_market_test',
            'duration': f'{test_duration} seconds',
            'instructions': 'Monitor browser console or WebSocket client for real-time events'
        })
        
    except Exception as e:
        logger.error(f"Error starting live market test: {e}")
        return jsonify({'error': str(e)}), 500

def run_scanner_background():
    """Run the scanner in a separate thread"""
    try:
        # Check for credentials
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if client_id and access_token:
            print("Starting live scanner with Dhan credentials...")
            # Import and run scanner
            from scanner import main as scanner_main
            asyncio.run(scanner_main('config.json'))
        else:
            print("No Dhan credentials found - running in demo mode only")
            # Keep thread alive but don't run scanner
            while True:
                time.sleep(60)
                
    except Exception as e:
        print(f"Scanner error: {e}", file=sys.stderr)

def load_cached_data():
    """Load previously cached historical data if available"""
    try:
        cache_file = 'cache/historical_data.pkl'
        if os.path.exists(cache_file):
            # Check cache age
            cache_age = time.time() - os.path.getmtime(cache_file)
            cache_hours = cache_age / 3600
            
            if cache_hours < 24:  # Use cache if less than 24 hours old
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                analyzed_data = cache_data.get('analyzed_data', {})
                timestamp = cache_data.get('timestamp', 'Unknown')
                
                if analyzed_data:
                    # Update scanner state with cached data
                    scanner_state['historical_data'] = analyzed_data
                    scanner_state['active_symbols'] = len(analyzed_data)
                    
                    # Convert to scanner data format
                    scanner_data = []
                    for sec_id, data in analyzed_data.items():
                        analysis = data['current_analysis']
                        scanner_data.append({
                            'symbol': analysis['symbol'],
                            'ltp': analysis['close'],
                            'change': analysis['change_pct'],
                            'volume': analysis['volume'],
                            'resistance': analysis['resistance'],
                            'ema8': analysis['ema_short'],
                            'ema13': analysis['ema_long'],
                            'signal': 'BREAKOUT' if analysis['breakout_signal'] else None
                        })
                    
                    scanner_state['scanner_data'] = scanner_data
                    scanner_state['last_update'] = timestamp
                    
                    print(f"✅ Loaded {len(analyzed_data)} symbols from cache (age: {cache_hours:.1f} hours)")
                    return True
            else:
                print(f"⚠️ Cache expired ({cache_hours:.1f} hours old) - will fetch fresh data")
                
    except Exception as e:
        print(f"⚠️ Failed to load cached data: {e}")
    
    return False

def init_app():
    """Initialize the application"""
    # Initialize database
    init_db()
    
    # Load cached historical data if available
    load_cached_data()
    
    # Start scanner in background thread if credentials are available
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    # Disable background scanner for Railway to prevent resource conflicts with UI fetch
    railway_mode = os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('PORT')  # Railway indicators
    
    if client_id and access_token and not railway_mode:
        print("Starting background scanner thread...")
        scanner_thread = threading.Thread(target=run_scanner_background, daemon=True)
        scanner_thread.start()
        time.sleep(2)  # Give scanner time to initialize
    else:
        if railway_mode:
            print("Railway mode: Background scanner disabled - use UI 'Fetch Historical' button")
        else:
            print("Running in demo mode - add DHAN credentials for live scanning")

# Initialize when module loads
init_app()

if __name__ == '__main__':
    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    host = '0.0.0.0'  # Railway requires binding to 0.0.0.0
    
    print(f"Starting F&O Scanner Dashboard on {host}:{port}")
    
    # Use allow_unsafe_werkzeug for development/Railway
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)