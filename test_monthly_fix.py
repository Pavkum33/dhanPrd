#!/usr/bin/env python3
"""
Test the fixed monthly pivot/CPR calculation
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_monthly_calculation():
    """Test monthly calculation with fixed function signature"""

    # Check credentials
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')

    if not client_id or not access_token:
        logger.warning("No credentials found - using demo data")
        # Create demo data for testing
        demo_data = {
            'RELIANCE': {
                'security_id': '25',
                'high': 3200,
                'low': 2900,
                'close': 3050
            },
            'TCS': {
                'security_id': '3456',
                'high': 4200,
                'low': 3900,
                'close': 4100
            }
        }

        # Test with demo data
        from scanners.monthly_levels import MonthlyLevelCalculator
        calculator = MonthlyLevelCalculator()

        for symbol, data in demo_data.items():
            logger.info(f"\n=== Testing {symbol} ===")

            # Calculate CPR
            cpr = calculator.calculate_monthly_cpr(
                data['high'], data['low'], data['close']
            )

            # Calculate Pivots
            pivots = calculator.calculate_monthly_pivots(
                data['high'], data['low'], data['close']
            )

            logger.info(f"CPR Results for {symbol}:")
            logger.info(f"  TC: {cpr['tc']}")
            logger.info(f"  Pivot: {cpr['pivot']}")
            logger.info(f"  BC: {cpr['bc']}")
            logger.info(f"  Width %: {cpr['width_percent']}")
            logger.info(f"  Is Narrow: {cpr['is_narrow']}")

            logger.info(f"Pivot Levels for {symbol}:")
            logger.info(f"  R3: {pivots['r3']}")
            logger.info(f"  R2: {pivots['r2']}")
            logger.info(f"  R1: {pivots['r1']}")
            logger.info(f"  Pivot: {pivots['pivot']}")
            logger.info(f"  S1: {pivots['s1']}")
            logger.info(f"  S2: {pivots['s2']}")
            logger.info(f"  S3: {pivots['s3']}")

        return

    # Test with real data
    logger.info("Testing with real Dhan data...")

    from dhan_fetcher import DhanHistoricalFetcher
    from scanners.monthly_levels import MonthlyLevelCalculator
    from cache_manager import CacheManager

    # Initialize components
    cache = CacheManager()
    calculator = MonthlyLevelCalculator(cache)

    async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
        # Load instruments
        logger.info("Loading instruments...")
        instruments_df = await fetcher.get_instruments()

        if instruments_df.empty:
            logger.error("No instruments loaded")
            return

        # Get active futures
        active_futures = fetcher.get_active_fno_futures(instruments_df)
        logger.info(f"Found {len(active_futures)} active futures")

        # Load equity mappings
        logger.info("Loading equity mappings...")
        equity_mapping = await fetcher.load_equity_instruments()
        logger.info(f"Loaded {len(equity_mapping)} equity mappings")

        # Test with a sample symbol
        test_symbol = "RELIANCE"

        if test_symbol not in equity_mapping:
            logger.error(f"{test_symbol} not found in equity mappings")
            # Try another symbol
            test_symbol = list(equity_mapping.keys())[0] if equity_mapping else None
            if not test_symbol:
                logger.error("No symbols available for testing")
                return
            logger.info(f"Using {test_symbol} instead")

        security_id = int(equity_mapping[test_symbol])
        logger.info(f"Testing {test_symbol} (security_id={security_id})")

        # Calculate date range for previous month
        today = datetime.now()
        first_day_current = today.replace(day=1)
        last_day_previous = first_day_current - timedelta(days=1)
        first_day_previous = last_day_previous.replace(day=1)

        logger.info(f"Fetching data for {first_day_previous.date()} to {last_day_previous.date()}")

        # Calculate days to fetch
        days_to_fetch = (today - first_day_previous).days + 5

        # Fetch historical data with FIXED function signature
        logger.info(f"Fetching {days_to_fetch} days of historical data...")
        historical_df = await fetcher.get_historical_data_for_underlying(
            test_symbol,
            security_id,  # Now passing security_id correctly
            days=days_to_fetch  # Now passing days correctly
        )

        if historical_df is None or historical_df.empty:
            logger.error(f"No historical data received for {test_symbol}")
            return

        logger.info(f"Received {len(historical_df)} days of data")

        # Filter for previous month
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        month_data = historical_df[
            (historical_df['date'] >= first_day_previous) &
            (historical_df['date'] <= last_day_previous)
        ]

        if month_data.empty:
            logger.error(f"No data for previous month")
            return

        logger.info(f"Filtered to {len(month_data)} days for previous month")

        # Calculate monthly OHLC
        monthly_ohlc = {
            'high': month_data['high'].max(),
            'low': month_data['low'].min(),
            'close': month_data.iloc[-1]['close'],
            'open': month_data.iloc[0]['open']
        }

        logger.info(f"Monthly OHLC: {monthly_ohlc}")

        # Calculate and display levels
        current_month = today.strftime('%Y-%m')
        levels = calculator.calculate_and_cache_symbol_levels(
            test_symbol,
            monthly_ohlc,
            current_month
        )

        logger.info(f"\n=== Monthly Levels for {test_symbol} (Month: {current_month}) ===")
        logger.info(f"Source Data:")
        logger.info(f"  High: {monthly_ohlc['high']}")
        logger.info(f"  Low: {monthly_ohlc['low']}")
        logger.info(f"  Close: {monthly_ohlc['close']}")

        cpr = levels['cpr']
        logger.info(f"\nCPR Levels:")
        logger.info(f"  TC: {cpr['tc']}")
        logger.info(f"  Pivot: {cpr['pivot']}")
        logger.info(f"  BC: {cpr['bc']}")
        logger.info(f"  Width %: {cpr['width_percent']}%")
        logger.info(f"  Is Narrow: {'YES' if cpr['is_narrow'] else 'NO'}")
        logger.info(f"  Trend: {cpr['trend']}")

        pivots = levels['pivots']
        logger.info(f"\nPivot Levels:")
        logger.info(f"  R3: {pivots['r3']}")
        logger.info(f"  R2: {pivots['r2']}")
        logger.info(f"  R1: {pivots['r1']}")
        logger.info(f"  Pivot: {pivots['pivot']}")
        logger.info(f"  S1: {pivots['s1']}")
        logger.info(f"  S2: {pivots['s2']}")
        logger.info(f"  S3: {pivots['s3']}")

        logger.info("\n✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_monthly_calculation())