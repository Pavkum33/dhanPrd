#!/usr/bin/env python3
"""
Test the daily EOD updater with a few symbols
"""

import asyncio
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import os
import logging
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_updater():
    """Test with a few symbols"""

    from daily_eod_updater import DailyEODUpdater
    from dhan_fetcher import DhanHistoricalFetcher

    updater = DailyEODUpdater()

    # Test with just 3 symbols
    test_symbols = ['RELIANCE', 'TCS', 'INFY']

    logger.info("="*60)
    logger.info("TESTING DAILY EOD UPDATER")
    logger.info("="*60)

    async with DhanHistoricalFetcher(updater.client_id, updater.access_token) as fetcher:
        # Load equity mappings
        equity_mapping = await fetcher.load_equity_instruments()

        for symbol in test_symbols:
            if symbol not in equity_mapping:
                logger.warning(f"{symbol} not found in equity mappings")
                continue

            security_id = equity_mapping[symbol]
            logger.info(f"\nUpdating {symbol} (security_id={security_id})")

            # Fetch last 30 days of data
            records = await updater.update_symbol_data(symbol, security_id, days_back=30)
            logger.info(f"{symbol}: Added {records} records")

    # Check what we have now
    updater.cursor.execute("""
        SELECT symbol, COUNT(*) as days, MIN(date), MAX(date)
        FROM eod_data
        WHERE symbol IN %s
        GROUP BY symbol
    """, (tuple(test_symbols),))

    results = updater.cursor.fetchall()

    logger.info("\n" + "="*60)
    logger.info("DATA STORED IN DATABASE:")
    logger.info("="*60)

    for symbol, days, min_date, max_date in results:
        logger.info(f"{symbol}: {days} days from {min_date} to {max_date}")

    # Now calculate monthly OHLC from the daily data
    logger.info("\n" + "="*60)
    logger.info("CALCULATING MONTHLY OHLC FROM DAILY DATA:")
    logger.info("="*60)

    for symbol in test_symbols:
        updater.cursor.execute("""
            SELECT
                MIN(date) as first_date,
                MAX(date) as last_date,
                (SELECT open FROM eod_data WHERE symbol = %s ORDER BY date ASC LIMIT 1) as month_open,
                MAX(high) as month_high,
                MIN(low) as month_low,
                (SELECT close FROM eod_data WHERE symbol = %s ORDER BY date DESC LIMIT 1) as month_close,
                COUNT(*) as days_count
            FROM eod_data
            WHERE symbol = %s
        """, (symbol, symbol, symbol))

        result = updater.cursor.fetchone()
        if result:
            first_date, last_date, m_open, m_high, m_low, m_close, days = result
            logger.info(f"\n{symbol} Monthly OHLC (from {days} days of daily data):")
            logger.info(f"  Date range: {first_date} to {last_date}")
            logger.info(f"  Open:  {m_open:.2f} (first day)")
            logger.info(f"  High:  {m_high:.2f} (highest of {days} days)")
            logger.info(f"  Low:   {m_low:.2f} (lowest of {days} days)")
            logger.info(f"  Close: {m_close:.2f} (last day)")

            # Calculate CPR from this monthly data
            from scanners.monthly_levels import MonthlyLevelCalculator
            calculator = MonthlyLevelCalculator()

            cpr = calculator.calculate_monthly_cpr(float(m_high), float(m_low), float(m_close))
            logger.info(f"  CPR Width: {cpr['width_percent']:.3f}%")
            logger.info(f"  Is Narrow: {'YES' if cpr['is_narrow'] else 'NO'}")

    updater.close()

    logger.info("\n" + "="*60)
    logger.info("TEST COMPLETE - Daily data stored and monthly OHLC calculated")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(test_updater())