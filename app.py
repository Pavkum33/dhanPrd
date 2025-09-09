#!/usr/bin/env python3
"""
app.py - Web server for F&O Scanner Dashboard
Provides real-time web interface for monitoring scanner activity
"""

import os
import json
import sqlite3
import asyncio
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

# Optional dhanhq SDK import
try:
    from dhanhq import DhanContext, dhanhq
    HAS_DHAN_SDK = True
    logger.info("dhanhq SDK available")
except ImportError:
    HAS_DHAN_SDK = False
    logger.warning("dhanhq SDK not available - using REST API fallback")

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
        """Fetch instrument master from Dhan"""
        url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    from io import StringIO
                    df = pd.read_csv(StringIO(content))
                    df.columns = [c.strip() for c in df.columns]
                    logger.info(f"Fetched {len(df)} instruments")
                    return df
                else:
                    logger.error(f"Failed to fetch instruments: {response.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Error fetching instruments: {e}")
            return pd.DataFrame()
    
    def get_futstk_instruments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for FUTSTK (stock futures) only - excludes indices, currencies, commodities"""
        logger.info(f"Filtering FUTSTK instruments from {len(df)} total instruments...")
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Resolve column mappings
        colmap = {}
        cols = list(df.columns)
        
        def find_col(*candidates):
            for c in candidates:
                if c in cols:
                    return c
            return None
        
        colmap['sid'] = find_col('SECURITY_ID','securityId','SEM_SMST_SECURITY_ID','SCRIP_CODE','SECURITYID')
        colmap['symbol'] = find_col('SYMBOL_NAME','tradingSymbol','SEM_SMST_SECURITY_SYMBOL','symbolName','InstrumentName','SYMBOL')
        colmap['segment'] = find_col('SEGMENT','segment','SEM_SEGMENT','EXCHANGE','SEM_EXM_EXCH_ID')
        colmap['expiry'] = find_col('EXPIRY_DATE','EXPIRY','SEM_EXPIRY_DATE')
        colmap['instrument'] = find_col('INSTRUMENT_TYPE','instrumentType','SEM_INSTRUMENT_NAME','INSTRUMENT')
        
        logger.info(f"Column mapping: {colmap}")
        
        # Debug: Show sample data structure
        if not df.empty:
            sample_data = df.head(5).to_dict('records')
            logger.info(f"Sample instruments: {sample_data}")
        
        # Method 1: Check for INSTRUMENT_TYPE column first (Dhan format)
        inst_col = colmap.get('instrument')
        exp_col = colmap.get('expiry')
        
        if inst_col and inst_col in df.columns:
            logger.info(f"Found instrument type column: {inst_col}")
            
            # Show unique instrument types available
            unique_types = df[inst_col].value_counts()
            logger.info(f"Available instrument types: {unique_types.to_dict()}")
            
            # Look for FUTSTK (stock futures) specifically
            futstk_mask = df[inst_col].astype(str).str.upper() == "FUTSTK"
            fut_df = df[futstk_mask].copy()
            logger.info(f"Found {len(fut_df)} FUTSTK instruments")
            
            # If no FUTSTK found, try broader FUT* pattern
            if len(fut_df) == 0:
                fut_mask = df[inst_col].astype(str).str.upper().str.contains("FUT", na=False)
                fut_df = df[fut_mask].copy()
                logger.info(f"Found {len(fut_df)} instruments with FUT* pattern")
                
        else:
            # Fallback: Filter by segment containing FUT
            seg_col = colmap.get('segment')
            
            if seg_col and seg_col in df.columns:
                logger.info(f"Filtering by segment column: {seg_col}")
                # Show unique segments available
                unique_segments = df[seg_col].value_counts()
                logger.info(f"Available segments: {unique_segments.to_dict()}")
                
                # Look for futures in segment
                fut_mask = df[seg_col].astype(str).str.upper().str.contains("FUT", na=False)
                fut_df = df[fut_mask].copy()
                logger.info(f"Found {len(fut_df)} instruments with FUT in segment")
            else:
                # Final fallback: use expiry column presence
                logger.info("No instrument type or segment column found, using expiry column")
                if exp_col and exp_col in df.columns:
                    fut_df = df[df[exp_col].notnull()].copy()
                    logger.info(f"Found {len(fut_df)} instruments with expiry dates")
                else:
                    logger.warning("No instrument type, segment or expiry columns found")
                    return pd.DataFrame()
        
        # Method 2: If we have INSTRUMENT_TYPE, filter specifically for FUTSTK
        if inst_col and inst_col in fut_df.columns:
            if len(fut_df) > 0:
                # Check if we got FUTSTK specifically
                futstk_only = fut_df[fut_df[inst_col].astype(str).str.upper() == "FUTSTK"].copy()
                if len(futstk_only) > 0:
                    logger.info(f"Using {len(futstk_only)} FUTSTK instruments")
                    futstk_df = futstk_only
                else:
                    # If no FUTSTK, filter out currencies and commodities from FUT* results
                    logger.info("No FUTSTK found, filtering FUT* instruments")
                    sym_col = colmap.get('symbol')
                    if sym_col and sym_col in fut_df.columns:
                        # Check what types of futures we have
                        sample_symbols = fut_df[sym_col].head(10).tolist()
                        logger.info(f"Sample FUT symbols: {sample_symbols}")
                        
                        # Filter out currencies (FUTCUR pattern) and commodities
                        exclude_patterns = [
                            'USD', 'EUR', 'GBP', 'JPY', 'INR', 'USDINR', 'EURINR', 'GBPINR', 'JPYINR',  # Currencies
                            'GOLD', 'SILVER', 'CRUDE', 'COPPER', 'ZINC', 'NICKEL', 'ALUMINIUM',  # Commodities
                            'COTTON', 'NATURALGAS', 'MENTHAOIL'  # Other commodities
                        ]
                        
                        # Also check if INSTRUMENT_TYPE contains FUTCUR to exclude currencies
                        if 'FUTCUR' in fut_df[inst_col].values:
                            fut_df = fut_df[fut_df[inst_col].astype(str).str.upper() != "FUTCUR"].copy()
                            logger.info(f"After excluding FUTCUR: {len(fut_df)} instruments")
                        
                        pattern_str = '|'.join(exclude_patterns)
                        exclude_mask = fut_df[sym_col].astype(str).str.upper().str.contains(pattern_str, na=False)
                        futstk_df = fut_df[~exclude_mask].copy()
                        logger.info(f"After excluding currencies/commodities: {len(futstk_df)} potential stock futures")
                    else:
                        futstk_df = fut_df.copy()
                        logger.warning("No symbol column found for filtering")
            else:
                futstk_df = fut_df.copy()
        else:
            # Fallback filtering for non-stock instruments
            sym_col = colmap.get('symbol')
            if sym_col and sym_col in fut_df.columns:
                logger.info(f"Filtering out non-stock instruments using symbol column: {sym_col}")
                # Exclude index futures, currency futures, commodity futures
                exclude_patterns = [
                    'INDEX', 'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX',  # Indices
                    'USD', 'EUR', 'GBP', 'JPY', 'USDINR', 'EURINR', 'GBPINR', 'JPYINR',  # Currencies
                    'GOLD', 'SILVER', 'CRUDE', 'COPPER', 'ZINC', 'NICKEL', 'ALUMINIUM',  # Commodities
                    'COTTON', 'NATURALGAS', 'MENTHAOIL'  # Other commodities
                ]
                
                pattern_str = '|'.join(exclude_patterns)
                exclude_mask = fut_df[sym_col].astype(str).str.upper().str.contains(pattern_str, na=False)
                futstk_df = fut_df[~exclude_mask].copy()
                logger.info(f"After excluding indices/currencies/commodities: {len(futstk_df)} FUTSTK instruments")
            else:
                futstk_df = fut_df.copy()
                logger.warning("No symbol column found for filtering, keeping all futures")
        
        # Method 3: Select nearest expiry for each underlying  
        sym_col = colmap.get('symbol')  # Ensure sym_col is defined for Method 3
        if exp_col and exp_col in futstk_df.columns and sym_col and sym_col in futstk_df.columns:
            logger.info("Selecting nearest expiry for each underlying stock")
            try:
                futstk_df[exp_col] = pd.to_datetime(futstk_df[exp_col], errors='coerce')
                today = pd.Timestamp.now().normalize()
                
                # Only keep futures that haven't expired
                futstk_df = futstk_df[futstk_df[exp_col] >= today]
                
                # Group by underlying symbol (remove expiry suffix to group)
                def get_underlying(symbol):
                    """Extract underlying symbol by removing expiry suffix"""
                    symbol_str = str(symbol).upper()
                    # Remove common date patterns and month codes
                    import re
                    # Remove patterns like 25JAN, 25FEB, 2025, etc.
                    clean = re.sub(r'\d{2}[A-Z]{3}|\d{4}|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC', '', symbol_str)
                    # Remove trailing numbers
                    clean = re.sub(r'\d+$', '', clean).strip()
                    return clean
                
                futstk_df['underlying'] = futstk_df[sym_col].apply(get_underlying)
                
                # Select nearest expiry for each underlying
                futstk_df = futstk_df.sort_values(exp_col).groupby('underlying').first().reset_index(drop=True)
                logger.info(f"After selecting nearest expiry: {len(futstk_df)} unique FUTSTK contracts")
                
            except Exception as e:
                logger.warning(f"Error processing expiry dates: {e}")
        
        # Final validation - ensure we have required columns
        if not futstk_df.empty:
            sid_col = colmap.get('sid')
            if sid_col and sid_col in futstk_df.columns:
                # Remove any rows with missing security IDs
                futstk_df = futstk_df[futstk_df[sid_col].notnull()].copy()
                logger.info(f"Final FUTSTK result: {len(futstk_df)} instruments")
                
                if len(futstk_df) > 0:
                    # Log sample of selected instruments (use safe symbol column reference)
                    sample_symbols = futstk_df[sym_col].head(10).tolist() if sym_col and sym_col in futstk_df.columns else []
                    logger.info(f"Sample FUTSTK symbols: {sample_symbols}")
                    return futstk_df
            
        logger.error("No FUTSTK instruments found in instrument master")
        return pd.DataFrame()
    
    async def get_historical_data(self, security_id: str, days: int = 60) -> pd.DataFrame:
        """Fetch historical daily data for a security using SDK or REST API"""
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days + 5)
        
        logger.info(f"Fetching historical data: securityId={security_id}, exchangeSegment=2, instrumentType=FUTSTK")
        
        # Method 1: Try dhanhq SDK first (more reliable)
        if self.use_sdk and hasattr(self, 'sdk'):
            try:
                loop = asyncio.get_running_loop()
                
                def sdk_call():
                    try:
                        # SDK call for historical daily data - use correct parameters for FUTSTK
                        result = self.sdk.historical_daily_data(
                            securityId=int(security_id),  # Must be numeric
                            exchangeSegment=2,            # NSE Futures = 2 (not string)
                            instrumentType="FUTSTK",      # Stock futures (not generic "FUTURE")
                            fromDate=from_date.strftime("%Y-%m-%d"),
                            toDate=to_date.strftime("%Y-%m-%d")
                        )
                        return result
                    except Exception as e:
                        logger.warning(f"SDK historical call failed for {security_id}: {e}")
                        return {}
                
                # Execute SDK call in thread pool
                result = await loop.run_in_executor(None, sdk_call)
                
                # Process SDK response
                if result:
                    data = result.get('data') or result.get('result') or []
                    if isinstance(result, list):
                        data = result
                    
                    if data:
                        df = pd.DataFrame(data)
                        
                        # Normalize date column
                        for date_col in ['timestamp', 'date', 'datetime']:
                            if date_col in df.columns and 'date' not in df.columns:
                                df['date'] = df[date_col]
                                break
                        
                        # Ensure required columns exist
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in required_cols:
                            if col not in df.columns:
                                df[col] = 0
                        
                        if 'date' in df.columns:
                            try:
                                df['date'] = pd.to_datetime(df['date'])
                                df = df.sort_values('date').reset_index(drop=True)
                                logger.info(f"SDK: Fetched {len(df)} historical records for {security_id}")
                                return df[['date', 'open', 'high', 'low', 'close', 'volume']]
                            except Exception as e:
                                logger.warning(f"Error processing SDK data for {security_id}: {e}")
                
            except Exception as e:
                logger.warning(f"SDK historical fetch failed for {security_id}: {e}")
        
        # Method 2: Fallback to REST API
        url = f"{self.base_url}/charts/historical"
        payload = {
            "securityId": int(security_id),  # Must be numeric
            "exchangeSegment": 2,            # NSE Futures = 2 
            "instrument": "FUTSTK",          # Stock futures
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])
                        
                        # Normalize date column
                        if 'timestamp' in df.columns:
                            df['date'] = pd.to_datetime(df['timestamp'])
                        elif 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                        
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in required_cols:
                            if col not in df.columns:
                                df[col] = 0
                        
                        df = df.sort_values('date').reset_index(drop=True)
                        logger.info(f"REST: Fetched {len(df)} historical records for {security_id}")
                        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
                else:
                    logger.warning(f"REST API error {response.status} for {security_id}")
                    
            return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Error fetching historical data for {security_id}: {e}")
            return pd.DataFrame()

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

async def fetch_and_analyze_historical_data():
    """Main function to fetch and analyze historical data with progress updates"""
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
        
        emit_progress('starting', 'Starting historical data fetch...')
        
        async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
            emit_progress('instruments', 'Fetching instrument master from Dhan...')
            instruments_df = await fetcher.get_instruments()
            
            if instruments_df.empty:
                emit_progress('error', 'Failed to fetch instruments from Dhan API')
                return {}
            
            emit_progress('filtering', f'Filtering F&O instruments from {len(instruments_df)} total instruments...')
            fno_df = fetcher.get_futstk_instruments(instruments_df)
            
            if fno_df.empty:
                emit_progress('error', 'No F&O instruments found in instrument master')
                return {}
            
            # Prepare securities list (limit to first 15 for stability)
            securities = []
            
            # Define column mappings for FUTSTK data extraction
            def find_col(*candidates):
                for c in candidates:
                    if c in fno_df.columns:
                        return c
                return None
            
            sid_col = find_col('SECURITY_ID','securityId','SEM_SMST_SECURITY_ID','SCRIP_CODE','SECURITYID')
            sym_col = find_col('SYMBOL_NAME','tradingSymbol','SEM_SMST_SECURITY_SYMBOL','symbolName','InstrumentName','SYMBOL')
            underlying_col = find_col('UNDERLYING_SYMBOL','underlying','underlyingSymbol')
            
            logger.info(f"Using columns: sid={sid_col}, symbol={sym_col}, underlying={underlying_col}")
            
            for _, row in fno_df.head(15).iterrows():
                if sid_col and sym_col and sid_col in row.index and sym_col in row.index:
                    security_id = str(row[sid_col])
                    symbol = str(row[sym_col])
                    underlying = str(row[underlying_col]) if underlying_col and underlying_col in row.index else symbol
                    
                    # Debug: Show what we're extracting
                    logger.info(f"Extracted: securityId={security_id}, symbol={symbol}, underlying={underlying}")
                    
                    securities.append({
                        'security_id': security_id,
                        'symbol': underlying,  # Use underlying symbol for display (NYKAA vs NYKAFUT)
                        'trading_symbol': symbol  # Keep trading symbol for reference
                    })
                else:
                    logger.warning(f"Missing required columns in row: {list(row.index)}")
            
            total_securities = len(securities)
            emit_progress('prepared', f'Prepared {total_securities} F&O securities for analysis', 0, total_securities)
            
            # Fetch historical data with rate limiting
            analyzed_data = {}
            # Configuration parameters for Chartink-style analysis
            lookback_period = 50
            ema_short = 8
            ema_long = 13
            volume_factor = 0.5
            price_threshold = 50
            
            analyzer = BreakoutAnalyzer(lookback=lookback_period, ema_short=ema_short, ema_long=ema_long)
            successful_fetches = 0
            failed_fetches = 0
            
            for i, sec_info in enumerate(securities):
                try:
                    current_symbol = sec_info['symbol']
                    emit_progress('fetching', f'Fetching historical data for {current_symbol}...', i + 1, total_securities, {
                        'current_symbol': current_symbol,
                        'successful': successful_fetches,
                        'failed': failed_fetches
                    })
                    
                    await asyncio.sleep(0.3)  # Rate limiting
                    df = await fetcher.get_historical_data(sec_info['security_id'])
                    
                    if not df.empty:
                        emit_progress('analyzing', f'Analyzing {current_symbol} ({len(df)} days of data)...', i + 1, total_securities)
                        
                        analyzed_df = analyzer.calculate_technical_indicators(df, volume_factor=volume_factor, price_threshold=price_threshold)
                        current_analysis = analyzer.get_current_analysis(analyzed_df)
                        current_analysis['symbol'] = current_symbol
                        current_analysis['security_id'] = sec_info['security_id']
                        
                        analyzed_data[sec_info['security_id']] = {
                            'symbol': current_symbol,
                            'historical_data': analyzed_df,
                            'current_analysis': current_analysis
                        }
                        
                        successful_fetches += 1
                        
                        # Check for breakout signal
                        signal_status = "BREAKOUT" if current_analysis['breakout_signal'] else "No Signal"
                        emit_progress('completed_symbol', f'✅ {current_symbol}: Close={current_analysis["close"]:.2f}, {signal_status}', 
                                    i + 1, total_securities, {
                            'symbol': current_symbol,
                            'close': current_analysis['close'],
                            'signal': current_analysis['breakout_signal'],
                            'successful': successful_fetches,
                            'failed': failed_fetches
                        })
                    else:
                        failed_fetches += 1
                        emit_progress('failed_symbol', f'❌ {current_symbol}: No historical data available', 
                                    i + 1, total_securities, {
                            'symbol': current_symbol,
                            'successful': successful_fetches,
                            'failed': failed_fetches
                        })
                    
                except Exception as e:
                    failed_fetches += 1
                    emit_progress('error_symbol', f'❌ {current_symbol}: Error - {str(e)[:50]}...', 
                                i + 1, total_securities, {
                        'symbol': current_symbol,
                        'error': str(e),
                        'successful': successful_fetches,
                        'failed': failed_fetches
                    })
            
            emit_progress('summary', f'Historical analysis completed: {successful_fetches} successful, {failed_fetches} failed', 
                        total_securities, total_securities, {
                'total_analyzed': len(analyzed_data),
                'successful': successful_fetches,
                'failed': failed_fetches,
                'breakouts_found': sum(1 for data in analyzed_data.values() if data['current_analysis']['breakout_signal'])
            })
            
            return analyzed_data
            
    except Exception as e:
        emit_progress('error', f'Critical error: {str(e)}')
        logger.exception("Critical error in historical data fetch")
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
    
    def run_historical_fetch():
        scanner_state['historical_analysis_running'] = True
        try:
            # Run async function in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            historical_data = loop.run_until_complete(fetch_and_analyze_historical_data())
            
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
    
    return jsonify({'message': 'Historical data fetch started'})

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

def init_app():
    """Initialize the application"""
    # Initialize database
    init_db()
    
    # Start scanner in background thread if credentials are available
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if client_id and access_token:
        print("Starting background scanner thread...")
        scanner_thread = threading.Thread(target=run_scanner_background, daemon=True)
        scanner_thread.start()
        time.sleep(2)  # Give scanner time to initialize
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