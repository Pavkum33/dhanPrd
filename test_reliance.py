#!/usr/bin/env python3
"""
Direct test for RELIANCE historical data to validate the fix
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
logger = logging.getLogger("reliance_test")

BASE_URL = "https://api.dhan.co"
DATA_DIR = "data"
EQ_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
EQ_FILE = os.path.join(DATA_DIR, "nse_eq.csv")

async def download_csv_if_needed(url: str, filepath: str) -> bool:
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

async def load_equity_mapping():
    """Load equity instruments and find RELIANCE securityId"""
    try:
        # Download CSV if needed
        if not await download_csv_if_needed(EQ_URL, EQ_FILE):
            logger.error("❌ Failed to download equity CSV")
            return {}
        
        # Parse CSV to find RELIANCE
        with open(EQ_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get("SEM_TRADING_SYMBOL", "").strip()
                security_id = row.get("SEM_SMST_SECURITY_ID", "").strip()
                
                if symbol == "RELIANCE" and security_id:
                    logger.info(f"✅ Found RELIANCE: securityId={security_id}")
                    return {"RELIANCE": security_id}
        
        logger.error("❌ RELIANCE not found in equity CSV")
        return {}
        
    except Exception as e:
        logger.exception(f"❌ Error loading equity mapping: {e}")
        return {}

async def test_reliance_historical(security_id: str, access_token: str):
    """Test RELIANCE historical data with correct parameters"""
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json', 
        'access-token': access_token
    }
    
    # Test parameters for equity stock
    exchange_segment = "NSE_EQ"
    instrument_type = "EQUITY"
    
    url = f"{BASE_URL}/v2/chart/history"
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    params = {
        "securityId": security_id,
        "exchangeSegment": exchange_segment,
        "instrumentType": instrument_type,
        "interval": "1d",
        "fromDate": start_date.strftime("%Y-%m-%d"),
        "toDate": end_date.strftime("%Y-%m-%d")
    }
    
    logger.info(f"Testing RELIANCE: securityId={security_id}, segment={exchange_segment}, type={instrument_type}")
    
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = data.get("data", [])
                    
                    if candles:
                        logger.info(f"🎉 SUCCESS! RELIANCE: Got {len(candles)} candles")
                        logger.info(f"Sample candle: {candles[0] if candles else 'None'}")
                        return True
                    else:
                        logger.warning(f"⚠️ RELIANCE: API returned 200 but no candles in data")
                        logger.info(f"Full response: {data}")
                        return False
                else:
                    logger.error(f"❌ RELIANCE: HTTP {response.status}")
                    response_text = await response.text()
                    logger.error(f"Response: {response_text}")
                    return False
                    
        except Exception as e:
            logger.exception(f"❌ RELIANCE: Exception during API call: {e}")
            return False

async def main():
    """Main test function"""
    logger.info("🧪 Testing RELIANCE historical data specifically...")
    
    # Get credentials
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if not client_id or not access_token:
        logger.error("❌ Missing Dhan credentials. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN env vars.")
        return
    
    # Load equity mapping to get RELIANCE securityId
    equity_mapping = await load_equity_mapping()
    reliance_security_id = equity_mapping.get('RELIANCE')
    
    if not reliance_security_id:
        logger.error("❌ Could not find RELIANCE securityId")
        return
    
    # Test RELIANCE historical data
    success = await test_reliance_historical(reliance_security_id, access_token)
    
    if success:
        logger.info("🎉 RELIANCE test PASSED! Historical data API is working correctly.")
    else:
        logger.error("❌ RELIANCE test FAILED! Need to investigate further.")

if __name__ == "__main__":
    asyncio.run(main())