#!/usr/bin/env python3
"""
Test monthly CPR/Pivot calculation with real Dhan data
Shows how we aggregate DAILY data to get MONTHLY levels
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Test monthly calculation from daily data"""

    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')

    logger.info(f"Client ID: {client_id}")
    logger.info(f"Access Token: {access_token[:20]}..." if access_token else "No token")

    if not client_id or not access_token:
        logger.error("Credentials not found in .env file")
        return

    from dhan_fetcher import DhanHistoricalFetcher
    from scanners.monthly_levels import MonthlyLevelCalculator
    from cache_manager import CacheManager

    # Initialize
    cache = CacheManager()
    calculator = MonthlyLevelCalculator(cache)

    async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
        # Load instruments
        logger.info("Loading instruments...")
        instruments_df = await fetcher.get_instruments()

        if instruments_df.empty:
            logger.error("No instruments loaded")
            return

        # Load equity mappings
        logger.info("Loading equity mappings...")
        equity_mapping = await fetcher.load_equity_instruments()

        # Test with RELIANCE
        test_symbol = "RELIANCE"

        if test_symbol not in equity_mapping:
            # Try first available symbol
            test_symbol = list(equity_mapping.keys())[0]
            logger.info(f"Using {test_symbol} instead")

        security_id = int(equity_mapping[test_symbol])
        logger.info(f"\n=== Testing {test_symbol} (security_id={security_id}) ===")

        # Get previous month's date range
        today = datetime.now()
        first_day_current = today.replace(day=1)
        last_day_previous = first_day_current - timedelta(days=1)
        first_day_previous = last_day_previous.replace(day=1)

        logger.info(f"Previous month: {first_day_previous.strftime('%B %Y')}")
        logger.info(f"Date range: {first_day_previous.date()} to {last_day_previous.date()}")

        # Calculate days to fetch
        days_to_fetch = (today - first_day_previous).days + 10

        # Fetch DAILY data
        logger.info(f"Fetching {days_to_fetch} days of DAILY data from Dhan API...")
        daily_df = await fetcher.get_historical_data_for_underlying(
            test_symbol,
            security_id,
            days=days_to_fetch
        )

        if daily_df is None or daily_df.empty:
            logger.error("No daily data received")
            return

        logger.info(f"✅ Received {len(daily_df)} days of daily data")

        # Convert date column
        daily_df['date'] = pd.to_datetime(daily_df['date'])

        # Filter for previous month
        month_data = daily_df[
            (daily_df['date'] >= first_day_previous) &
            (daily_df['date'] <= last_day_previous)
        ]

        if month_data.empty:
            logger.error("No data for previous month")
            return

        logger.info(f"✅ Filtered to {len(month_data)} days for {first_day_previous.strftime('%B %Y')}")

        # AGGREGATE daily data to get monthly OHLC
        monthly_ohlc = {
            'high': month_data['high'].max(),      # Highest high of the month
            'low': month_data['low'].min(),        # Lowest low of the month
            'close': month_data.iloc[-1]['close'], # Last day's close
            'open': month_data.iloc[0]['open']     # First day's open
        }

        logger.info(f"\n📊 MONTHLY OHLC (aggregated from {len(month_data)} daily candles):")
        logger.info(f"  Open:  {monthly_ohlc['open']:.2f} (first day)")
        logger.info(f"  High:  {monthly_ohlc['high']:.2f} (month high)")
        logger.info(f"  Low:   {monthly_ohlc['low']:.2f} (month low)")
        logger.info(f"  Close: {monthly_ohlc['close']:.2f} (last day)")

        # Calculate CPR and Pivots
        cpr = calculator.calculate_monthly_cpr(
            monthly_ohlc['high'],
            monthly_ohlc['low'],
            monthly_ohlc['close']
        )

        pivots = calculator.calculate_monthly_pivots(
            monthly_ohlc['high'],
            monthly_ohlc['low'],
            monthly_ohlc['close']
        )

        logger.info(f"\n📈 MONTHLY CPR LEVELS (for {today.strftime('%B %Y')} trading):")
        logger.info(f"  TC (Top Central):     {cpr['tc']:.2f}")
        logger.info(f"  Pivot:                {cpr['pivot']:.2f}")
        logger.info(f"  BC (Bottom Central):  {cpr['bc']:.2f}")
        logger.info(f"  CPR Width:            {cpr['width']:.2f}")
        logger.info(f"  CPR Width %:          {cpr['width_percent']:.3f}%")
        logger.info(f"  Is Narrow CPR:        {'✅ YES' if cpr['is_narrow'] else '❌ NO'}")

        logger.info(f"\n📊 MONTHLY PIVOT LEVELS:")
        logger.info(f"  R3: {pivots['r3']:.2f}")
        logger.info(f"  R2: {pivots['r2']:.2f}")
        logger.info(f"  R1: {pivots['r1']:.2f}")
        logger.info(f"  ---PIVOT: {pivots['pivot']:.2f}---")
        logger.info(f"  S1: {pivots['s1']:.2f}")
        logger.info(f"  S2: {pivots['s2']:.2f}")
        logger.info(f"  S3: {pivots['s3']:.2f}")

        # Show some daily data for verification
        logger.info(f"\n📅 Sample daily data from {test_symbol}:")
        for i, row in month_data.head(3).iterrows():
            logger.info(f"  {row['date'].strftime('%Y-%m-%d')}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}")
        logger.info("  ...")
        for i, row in month_data.tail(3).iterrows():
            logger.info(f"  {row['date'].strftime('%Y-%m-%d')}: O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}")

        logger.info("\n✅ Successfully calculated monthly levels from daily Dhan data!")
        logger.info("💡 Key Point: Dhan provides DAILY data, we aggregate it to get MONTHLY levels")

if __name__ == "__main__":
    asyncio.run(main())