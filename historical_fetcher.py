#!/usr/bin/env python3
"""
historical_fetcher.py - Fetch historical data via Dhan REST API for closed market analysis
"""

import os
import json
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DhanHistoricalFetcher:
    """Fetches historical data from Dhan REST API"""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api.dhan.co"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': access_token
        }
        self.session = None
    
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
                    # Save to local file for caching
                    with open('instruments_cached.csv', 'w') as f:
                        f.write(content)
                    
                    # Parse CSV
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
    
    def get_fno_instruments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter F&O instruments"""
        # Find segment column
        seg_col = None
        for col in df.columns:
            if 'segment' in col.lower():
                seg_col = col
                break
        
        if seg_col:
            fno_df = df[df[seg_col].astype(str).str.contains('FUT', na=False)].copy()
        else:
            # Fallback: use expiry column if available
            exp_cols = [col for col in df.columns if 'expiry' in col.lower()]
            if exp_cols:
                fno_df = df[df[exp_cols[0]].notnull()].copy()
            else:
                # Take first 50 instruments as fallback
                fno_df = df.head(50).copy()
        
        logger.info(f"Found {len(fno_df)} F&O instruments")
        return fno_df
    
    async def get_historical_data(self, security_id: str, instrument_type: str = "FUTURE", 
                                exchange: str = "NSE", days: int = 60) -> pd.DataFrame:
        """Fetch historical daily data for a security"""
        
        # Calculate date range
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days + 5)  # Add buffer
        
        url = f"{self.base_url}/charts/historical"
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange,
            "instrument": instrument_type,
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])
                        # Normalize column names
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
                        logger.info(f"Fetched {len(df)} days for security {security_id}")
                        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    else:
                        logger.warning(f"No data returned for security {security_id}")
                        return pd.DataFrame()
                else:
                    logger.warning(f"API call failed for {security_id}: {response.status}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.exception(f"Error fetching data for {security_id}: {e}")
            return pd.DataFrame()
    
    async def bulk_fetch_historical(self, securities: List[Dict], days: int = 60, 
                                  concurrency: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple securities with concurrency control"""
        
        semaphore = asyncio.Semaphore(concurrency)
        results = {}
        
        async def fetch_one(sec_info):
            async with semaphore:
                await asyncio.sleep(0.1)  # Rate limiting
                df = await self.get_historical_data(
                    sec_info['security_id'], 
                    sec_info.get('instrument_type', 'FUTURE'),
                    sec_info.get('exchange', 'NSE'),
                    days
                )
                return sec_info['security_id'], df
        
        # Create tasks
        tasks = [fetch_one(sec) for sec in securities]
        
        # Execute with progress tracking
        completed = 0
        for coro in asyncio.as_completed(tasks):
            try:
                sec_id, df = await coro
                results[sec_id] = df
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(tasks)} securities")
            except Exception as e:
                logger.exception(f"Error in bulk fetch: {e}")
        
        logger.info(f"Completed historical data fetch for {len(results)} securities")
        return results

class BreakoutAnalyzer:
    """Analyzes historical data for breakout patterns"""
    
    def __init__(self, lookback: int = 50, ema_short: int = 8, ema_long: int = 13):
        self.lookback = lookback
        self.ema_short = ema_short
        self.ema_long = ema_long
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for breakout analysis"""
        if df.empty or len(df) < self.lookback:
            return df
        
        df = df.copy()
        
        # Calculate Typical Price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        
        # Calculate rolling resistance (max of typical price over lookback period)
        df['resistance'] = df['typical_price'].rolling(window=self.lookback).max()
        
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=self.ema_short).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long).mean()
        
        # Calculate breakout signals
        df['breakout'] = (df['close'] > df['resistance']) & (df['ema_short'] > df['ema_long'])
        
        # Volume analysis (compare to 20-day average)
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_avg'] * 1.5)
        
        # Final breakout signal
        df['signal'] = df['breakout'] & df['volume_spike'] & (df['close'] > df['open'])
        
        return df
    
    def get_current_analysis(self, df: pd.DataFrame) -> Dict:
        """Get current analysis for latest data point"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        prev_day = df.iloc[-2] if len(df) > 1 else latest
        
        return {
            'symbol': '',
            'date': latest.get('date', ''),
            'close': float(latest.get('close', 0)),
            'resistance': float(latest.get('resistance', 0)),
            'ema_short': float(latest.get('ema_short', 0)),
            'ema_long': float(latest.get('ema_long', 0)),
            'volume': int(latest.get('volume', 0)),
            'prev_day_volume': int(prev_day.get('volume', 0)),
            'breakout_signal': bool(latest.get('signal', False)),
            'change_pct': ((latest.get('close', 0) - prev_day.get('close', 1)) / prev_day.get('close', 1)) * 100
        }

async def fetch_and_analyze_historical_data():
    """Main function to fetch and analyze historical data"""
    
    # Get credentials from environment
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if not client_id or not access_token:
        logger.error("Dhan credentials not found in environment variables")
        return {}
    
    logger.info("Starting historical data fetch...")
    
    async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
        # Get instruments
        instruments_df = await fetcher.get_instruments()
        if instruments_df.empty:
            logger.error("Failed to fetch instruments")
            return {}
        
        # Get F&O instruments
        fno_df = fetcher.get_fno_instruments(instruments_df)
        
        # Prepare securities list (limit to first 20 for testing)
        securities = []
        for _, row in fno_df.head(20).iterrows():
            security_id = None
            symbol = None
            
            # Find security ID column
            for col in row.index:
                if 'security' in col.lower() and 'id' in col.lower():
                    security_id = str(row[col])
                    break
            
            # Find symbol column
            for col in row.index:
                if any(x in col.lower() for x in ['symbol', 'name', 'trading']):
                    symbol = str(row[col])
                    break
            
            if security_id and symbol:
                securities.append({
                    'security_id': security_id,
                    'symbol': symbol,
                    'instrument_type': 'FUTURE',
                    'exchange': 'NSE'
                })
        
        logger.info(f"Fetching historical data for {len(securities)} securities...")
        
        # Fetch historical data
        historical_data = await fetcher.bulk_fetch_historical(securities, days=60)
        
        # Analyze data
        analyzer = BreakoutAnalyzer()
        analyzed_data = {}
        
        for i, (sec_id, df) in enumerate(historical_data.items()):
            if not df.empty:
                # Add symbol info
                symbol = next((s['symbol'] for s in securities if s['security_id'] == sec_id), sec_id)
                
                # Calculate indicators
                analyzed_df = analyzer.calculate_technical_indicators(df)
                
                # Get current analysis
                current_analysis = analyzer.get_current_analysis(analyzed_df)
                current_analysis['symbol'] = symbol
                current_analysis['security_id'] = sec_id
                
                analyzed_data[sec_id] = {
                    'symbol': symbol,
                    'historical_data': analyzed_df,
                    'current_analysis': current_analysis
                }
                
                logger.info(f"Analyzed {symbol}: Close={current_analysis['close']:.2f}, "
                           f"Resistance={current_analysis['resistance']:.2f}, "
                           f"Signal={current_analysis['breakout_signal']}")
        
        logger.info(f"Historical analysis completed for {len(analyzed_data)} securities")
        return analyzed_data

if __name__ == "__main__":
    # Run the historical data fetch
    result = asyncio.run(fetch_and_analyze_historical_data())
    print(f"Fetched and analyzed {len(result)} securities")