#!/usr/bin/env python3
"""
Daily EOD Data Updater - Runs every day to store EOD data in PostgreSQL
This should be scheduled to run daily after market close (e.g., 4:00 PM IST)
"""

import asyncio
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging
import pandas as pd

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DailyEODUpdater:
    """
    Daily EOD data updater that fetches and stores data for all F&O symbols
    Maintains complete historical data in PostgreSQL
    """

    def __init__(self):
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')

        # PostgreSQL connection
        self.conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='dhan_scanner_prod',
            user='postgres',
            password='India@123'
        )
        self.cursor = self.conn.cursor()

    def get_last_updated_date(self, symbol):
        """Get the last date we have data for this symbol"""
        self.cursor.execute("""
            SELECT MAX(date) FROM eod_data WHERE symbol = %s
        """, (symbol,))
        result = self.cursor.fetchone()
        return result[0] if result[0] else None

    def get_missing_dates(self, symbol, start_date=None):
        """Get list of missing trading dates for a symbol"""
        if not start_date:
            # Default to last 60 days if no data exists
            start_date = datetime.now().date() - timedelta(days=60)

        end_date = datetime.now().date()

        # Get all dates we have
        self.cursor.execute("""
            SELECT DISTINCT date FROM eod_data
            WHERE symbol = %s AND date >= %s
            ORDER BY date
        """, (symbol, start_date))

        existing_dates = set(row[0] for row in self.cursor.fetchall())

        # Generate all weekdays (potential trading days)
        all_dates = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                all_dates.append(current)
            current += timedelta(days=1)

        # Find missing dates (excluding holidays - we'll skip those during fetch)
        missing_dates = [d for d in all_dates if d not in existing_dates]
        return missing_dates

    async def update_symbol_data(self, symbol, security_id, days_back=60):
        """Update EOD data for a single symbol"""
        try:
            from dhan_fetcher import DhanHistoricalFetcher

            # Check what data we already have
            last_date = self.get_last_updated_date(symbol)

            if last_date:
                # Calculate days since last update
                days_missing = (datetime.now().date() - last_date).days
                if days_missing <= 0:
                    logger.info(f"{symbol}: Already up to date (last: {last_date})")
                    return 0
                days_to_fetch = min(days_missing + 5, days_back)  # Add buffer
            else:
                # No data exists, fetch full history
                days_to_fetch = days_back
                logger.info(f"{symbol}: No existing data, fetching {days_to_fetch} days")

            async with DhanHistoricalFetcher(self.client_id, self.access_token) as fetcher:
                # Fetch historical data
                df = await fetcher.get_historical_data_for_underlying(
                    symbol,
                    int(security_id),
                    days=days_to_fetch
                )

                if df is None or df.empty:
                    logger.warning(f"{symbol}: No data received from API")
                    return 0

                # Convert date column
                df['date'] = pd.to_datetime(df['date'])

                # Insert/Update data
                records_inserted = 0
                for _, row in df.iterrows():
                    # Use INSERT ... ON CONFLICT to handle duplicates
                    self.cursor.execute("""
                        INSERT INTO eod_data
                        (symbol, date, open, high, low, close, volume, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (symbol, date)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            created_at = NOW()
                    """, (
                        symbol,
                        row['date'].date(),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row.get('volume', 0))
                    ))
                    records_inserted += 1

                self.conn.commit()
                logger.info(f"{symbol}: Inserted/Updated {records_inserted} records")
                return records_inserted

        except Exception as e:
            logger.error(f"{symbol}: Error updating data - {e}")
            self.conn.rollback()
            return 0

    async def update_all_symbols(self):
        """Update EOD data for all F&O symbols"""
        from dhan_fetcher import DhanHistoricalFetcher

        logger.info("="*60)
        logger.info("DAILY EOD UPDATE STARTED")
        logger.info(f"Time: {datetime.now()}")
        logger.info("="*60)

        async with DhanHistoricalFetcher(self.client_id, self.access_token) as fetcher:
            # Get all F&O instruments
            instruments_df = await fetcher.get_instruments()
            active_futures = fetcher.get_active_fno_futures(instruments_df)

            # Load equity mappings
            equity_mapping = await fetcher.load_equity_instruments()

            # Extract unique underlying symbols
            underlying_symbols = set()
            for _, row in active_futures.iterrows():
                symbol_col = 'SEM_TRADING_SYMBOL' if 'SEM_TRADING_SYMBOL' in row else 'SYMBOL_NAME'
                underlying = fetcher.extract_underlying_symbol(row[symbol_col])
                if underlying and underlying in equity_mapping:
                    underlying_symbols.add(underlying)

            logger.info(f"Found {len(underlying_symbols)} F&O underlying symbols to update")

            # Update each symbol
            total_records = 0
            symbols_updated = 0

            for i, symbol in enumerate(underlying_symbols, 1):
                if symbol not in equity_mapping:
                    continue

                security_id = equity_mapping[symbol]
                logger.info(f"[{i}/{len(underlying_symbols)}] Updating {symbol}...")

                records = await self.update_symbol_data(symbol, security_id)
                total_records += records
                if records > 0:
                    symbols_updated += 1

                # Small delay to avoid API rate limits
                await asyncio.sleep(0.1)

            # Update monthly aggregates
            self.update_monthly_aggregates()

            # Log summary to job_logs table
            self.cursor.execute("""
                INSERT INTO job_logs
                (job_name, job_type, status, started_at, completed_at,
                 records_processed, records_success, records_failed, errors)
                VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s, %s)
            """, (
                'daily_eod_update',
                'EOD_UPDATE',
                'SUCCESS',
                datetime.now(),
                total_records,
                symbols_updated,
                len(underlying_symbols) - symbols_updated,
                None
            ))
            self.conn.commit()

            logger.info("="*60)
            logger.info("DAILY EOD UPDATE COMPLETED")
            logger.info(f"Symbols updated: {symbols_updated}/{len(underlying_symbols)}")
            logger.info(f"Total records: {total_records}")
            logger.info("="*60)

    def update_monthly_aggregates(self):
        """Calculate and store monthly OHLC from daily data"""
        logger.info("Updating monthly aggregates...")

        # Get current and previous month
        today = datetime.now()
        current_month = today.strftime('%Y-%m')

        # Calculate monthly aggregates for current and previous month
        for month_offset in [0, -1]:
            target_date = today.replace(day=1) + timedelta(days=month_offset*30)
            month_str = target_date.strftime('%Y-%m')

            # Get first and last day of month
            first_day = target_date.replace(day=1)
            if target_date.month == 12:
                last_day = target_date.replace(year=target_date.year+1, month=1, day=1) - timedelta(days=1)
            else:
                last_day = target_date.replace(month=target_date.month+1, day=1) - timedelta(days=1)

            logger.info(f"Calculating monthly aggregates for {month_str}")

            # Calculate monthly OHLC for each symbol
            self.cursor.execute("""
                INSERT INTO monthly_ohlc (symbol, month, open, high, low, close, volume, days_count, created_at)
                SELECT
                    symbol,
                    %s as month,
                    (SELECT open FROM eod_data e2
                     WHERE e2.symbol = e.symbol
                     AND e2.date >= %s AND e2.date <= %s
                     ORDER BY date ASC LIMIT 1) as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    (SELECT close FROM eod_data e3
                     WHERE e3.symbol = e.symbol
                     AND e3.date >= %s AND e3.date <= %s
                     ORDER BY date DESC LIMIT 1) as close,
                    SUM(volume) as volume,
                    COUNT(*) as days_count,
                    NOW() as created_at
                FROM eod_data e
                WHERE date >= %s AND date <= %s
                GROUP BY symbol
                HAVING COUNT(*) >= 10  -- Only if we have at least 10 days of data
                ON CONFLICT (symbol, month)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    days_count = EXCLUDED.days_count,
                    updated_at = NOW()
            """, (month_str, first_day, last_day, first_day, last_day, first_day, last_day))

            affected_rows = self.cursor.rowcount
            self.conn.commit()
            logger.info(f"Updated {affected_rows} monthly aggregates for {month_str}")

    def check_data_quality(self):
        """Check data quality and completeness"""
        logger.info("Checking data quality...")

        # Check for gaps in data
        self.cursor.execute("""
            SELECT symbol, MIN(date) as first_date, MAX(date) as last_date,
                   COUNT(*) as total_days,
                   MAX(date) - MIN(date) + 1 as expected_days
            FROM eod_data
            GROUP BY symbol
            HAVING COUNT(*) < (MAX(date) - MIN(date) + 1) * 0.7  -- Less than 70% of expected days
            ORDER BY symbol
        """)

        gaps = self.cursor.fetchall()
        if gaps:
            logger.warning(f"Found {len(gaps)} symbols with data gaps:")
            for row in gaps[:10]:  # Show first 10
                logger.warning(f"  {row[0]}: {row[2]}/{row[4]} days ({row[1]} to {row[2]})")
        else:
            logger.info("No significant data gaps found")

        # Check latest data freshness
        self.cursor.execute("""
            SELECT COUNT(*) as stale_count
            FROM (
                SELECT symbol, MAX(date) as last_date
                FROM eod_data
                GROUP BY symbol
                HAVING MAX(date) < CURRENT_DATE - INTERVAL '3 days'
            ) s
        """)

        stale_count = self.cursor.fetchone()[0]
        if stale_count > 0:
            logger.warning(f"{stale_count} symbols have stale data (>3 days old)")
        else:
            logger.info("All symbols have recent data")

    def close(self):
        """Close database connection"""
        self.conn.close()

async def main():
    """Main function to run daily update"""
    updater = DailyEODUpdater()

    try:
        # Run the daily update
        await updater.update_all_symbols()

        # Check data quality
        updater.check_data_quality()

    finally:
        updater.close()

if __name__ == "__main__":
    asyncio.run(main())