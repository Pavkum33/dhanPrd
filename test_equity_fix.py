#!/usr/bin/env python3
"""
Test script to validate the equity loading fix
Mimics the working sample pattern for debugging
"""

import asyncio
import aiohttp
import logging
import os
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("equity_test")

BASE_URL = "https://api.dhan.co"

class EquityTester:
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': access_token
        }
    
    async def test_equity_loading(self):
        """Test loading NSE equity instruments"""
        url = f"{BASE_URL}/v2/instruments/NSE_EQ.json"
        
        # Use public session without auth headers (following working sample)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        instruments = await response.json()
                        logger.info(f"✅ Fetched {len(instruments)} instruments from NSE_EQ.json")
                        
                        # Create symbol → securityId mapping
                        equity_mapping = {}
                        for inst in instruments:
                            if inst.get("instrumentType") in ["EQUITY", "INDEX"]:
                                symbol = inst.get("symbol")
                                security_id = inst.get("securityId")
                                if symbol and security_id:
                                    equity_mapping[symbol] = security_id
                        
                        logger.info(f"✅ Created {len(equity_mapping)} symbol→securityId mappings")
                        
                        # Show sample mappings
                        sample_symbols = list(equity_mapping.keys())[:5]
                        for sym in sample_symbols:
                            logger.info(f"Sample: {sym} → {equity_mapping[sym]}")
                        
                        return equity_mapping
                    else:
                        logger.error(f"❌ Failed to fetch NSE_EQ.json: {response.status}")
                        return {}
            except Exception as e:
                logger.exception(f"❌ Error loading equity instruments: {e}")
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