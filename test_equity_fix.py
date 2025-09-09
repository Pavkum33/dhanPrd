#!/usr/bin/env python3
"""
Test script to validate the equity loading fix using CSV download pattern
Exactly matches the working sample pattern
"""

import asyncio
import aiohttp
import logging
import os
import csv
import time
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("equity_test")

BASE_URL = "https://api.dhan.co"
DATA_DIR = "data"
EQ_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
EQ_FILE = os.path.join(DATA_DIR, "nse_eq.csv")

class EquityTester:
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': access_token
        }
    
    async def download_csv_if_needed(self, url: str, filepath: str) -> bool:
        """Download CSV file if missing or older than 24h"""
        if os.path.exists(filepath):
            age = time.time() - os.path.getmtime(filepath)
            if age < 86400:  # < 24h
                logger.info(f"Using cached {os.path.basename(filepath)}")
                return True

        logger.info(f"Downloading {os.path.basename(filepath)}...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
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
    
    async def test_equity_loading(self):
        """Test loading NSE equity instruments from CSV"""
        try:
            # Download CSV if needed
            if not await self.download_csv_if_needed(EQ_URL, EQ_FILE):
                logger.error("❌ Failed to download equity CSV")
                return {}
            
            # Parse CSV to create symbol → securityId mapping
            equity_mapping = {}
            with open(EQ_FILE, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get("SEM_TRADING_SYMBOL")
                    security_id = row.get("SEM_SMST_SECURITY_ID")
                    if symbol and security_id:
                        equity_mapping[symbol.strip()] = security_id.strip()
            
            logger.info(f"✅ Loaded {len(equity_mapping)} equity instruments from CSV")
            
            # Show sample mappings
            sample_symbols = list(equity_mapping.keys())[:5]
            for sym in sample_symbols:
                logger.info(f"Sample: {sym} → {equity_mapping[sym]}")
            
            return equity_mapping
            
        except Exception as e:
            logger.exception(f"❌ Error loading equity instruments from CSV: {e}")
            return {}
    
    async def test_historical_fetch(self, equity_mapping):
        """Test historical data fetching for sample symbols"""
        test_symbols = ['RELIANCE', 'NIFTY 50', 'BANKNIFTY', 'TCS', 'INFY']
        success, fail = 0, 0
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for symbol in test_symbols:
                security_id = equity_mapping.get(symbol)
                if not security_id:
                    logger.warning(f"❌ {symbol}: No securityId found in equity mapping")
                    fail += 1
                    continue
                
                # Test historical data fetch
                url = f"{BASE_URL}/v2/chart/history"
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=30)
                
                params = {
                    "securityId": str(security_id),
                    "exchangeSegment": "NSE_EQ",
                    "instrument": "EQUITY",
                    "interval": "1d",
                    "fromDate": start_date.strftime("%Y-%m-%d"),
                    "toDate": end_date.strftime("%Y-%m-%d")
                }
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            candles = data.get("data", [])
                            if candles:
                                logger.info(f"✅ {symbol}: Got {len(candles)} candles (securityId: {security_id})")
                                success += 1
                            else:
                                logger.warning(f"❌ {symbol}: No historical data in response")
                                fail += 1
                        else:
                            logger.warning(f"❌ {symbol}: Historical API error {response.status}")
                            fail += 1
                except Exception as e:
                    logger.warning(f"❌ {symbol}: Historical fetch error: {e}")
                    fail += 1
                
                # Add small delay to respect rate limits
                await asyncio.sleep(0.2)
        
        logger.info(f"Historical test completed: {success} successful, {fail} failed")
        return success, fail

async def main():
    """Main test function"""
    logger.info("Starting equity loading and historical data test...")
    
    # Get credentials
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if not client_id or not access_token:
        logger.error("❌ Missing Dhan credentials. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN env vars.")
        return
    
    # Test equity loading
    tester = EquityTester(client_id, access_token)
    equity_mapping = await tester.test_equity_loading()
    
    if not equity_mapping:
        logger.error("❌ Equity loading failed - cannot proceed with historical test")
        return
    
    # Test historical data fetching
    success, fail = await tester.test_historical_fetch(equity_mapping)
    
    if success > 0:
        logger.info(f"🎉 Fix validated! {success} symbols successfully fetched historical data")
    else:
        logger.error(f"❌ Fix validation failed. All {fail} symbols failed to fetch historical data")

if __name__ == "__main__":
    asyncio.run(main())