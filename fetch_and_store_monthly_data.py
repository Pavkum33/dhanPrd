#!/usr/bin/env python3
"""
Fetch complete monthly OHLC data from Dhan API and store in PostgreSQL
"""

import asyncio
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_and_store_monthly_data():
    """Fetch full month data from Dhan and store in PostgreSQL"""

    from dhan_fetcher import DhanHistoricalFetcher

    # Dhan credentials
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')

    if not client_id or not access_token:
        logger.error("Dhan credentials not found!")
        return

    # PostgreSQL connection
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='dhan_scanner_prod',
        user='postgres',
        password='India@123'
    )
    cursor = conn.cursor()

    logger.info("Connected to PostgreSQL database: dhan_scanner_prod")

    # Calculate date range for previous month
    today = datetime.now()
    first_day_current = today.replace(day=1)
    last_day_previous = first_day_current - timedelta(days=1)
    first_day_previous = last_day_previous.replace(day=1)

    logger.info(f"Fetching data for: {first_day_previous.strftime('%B %Y')}")
    logger.info(f"Date range: {first_day_previous.date()} to {last_day_previous.date()}")

    async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
        # Load instruments
        logger.info("Loading F&O instruments...")
        instruments_df = await fetcher.get_instruments()

        if instruments_df.empty:
            logger.error("No instruments loaded")
            return

        # Get active F&O futures
        active_futures = fetcher.get_active_fno_futures(instruments_df)
        logger.info(f"Found {len(active_futures)} active F&O futures")

        # Load equity mappings
        logger.info("Loading equity mappings...")
        equity_mapping = await fetcher.load_equity_instruments()
        logger.info(f"Loaded {len(equity_mapping)} equity mappings")

        # Extract unique underlying symbols
        underlying_symbols = set()
        for _, row in active_futures.iterrows():
            symbol_col = 'SEM_TRADING_SYMBOL' if 'SEM_TRADING_SYMBOL' in row else 'SYMBOL_NAME'
            underlying = fetcher.extract_underlying_symbol(row[symbol_col])
            if underlying and underlying in equity_mapping:
                underlying_symbols.add(underlying)

        logger.info(f"Processing {len(underlying_symbols)} underlying symbols")

        # Test with first 5 symbols for now
        test_symbols = list(underlying_symbols)[:5]

        total_days_fetched = 0
        symbols_processed = 0

        for symbol in test_symbols:
            try:
                security_id = int(equity_mapping[symbol])
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {symbol} (security_id={security_id})")

                # Calculate days to fetch (previous month + buffer)
                days_to_fetch = (today - first_day_previous).days + 5

                # Fetch daily data
                logger.info(f"Fetching {days_to_fetch} days of daily data...")
                daily_df = await fetcher.get_historical_data_for_underlying(
                    symbol,
                    security_id,
                    days=days_to_fetch
                )

                if daily_df is None or daily_df.empty:
                    logger.warning(f"No data received for {symbol}")
                    continue

                # Convert date column
                daily_df['date'] = pd.to_datetime(daily_df['date'])

                # Filter for previous month
                month_data = daily_df[
                    (daily_df['date'] >= first_day_previous) &
                    (daily_df['date'] <= last_day_previous)
                ]

                if month_data.empty:
                    logger.warning(f"No data for previous month for {symbol}")
                    continue

                logger.info(f"Got {len(month_data)} days of data for {symbol}")

                # Clear existing data for this symbol and month
                cursor.execute("""
                    DELETE FROM eod_data
                    WHERE symbol = %s
                    AND date >= %s
                    AND date <= %s
                """, (symbol, first_day_previous, last_day_previous))

                # Insert daily data into PostgreSQL
                for _, row in month_data.iterrows():
                    cursor.execute("""
                        INSERT INTO eod_data
                        (symbol, date, open, high, low, close, volume, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    """, (
                        symbol,
                        row['date'].date(),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row.get('volume', 0))
                    ))

                total_days_fetched += len(month_data)

                # Calculate monthly aggregates
                monthly_open = month_data.iloc[0]['open']
                monthly_high = month_data['high'].max()
                monthly_low = month_data['low'].min()
                monthly_close = month_data.iloc[-1]['close']

                logger.info(f"Monthly OHLC for {symbol}:")
                logger.info(f"  Open:  {monthly_open:.2f} (first day: {month_data.iloc[0]['date'].date()})")
                logger.info(f"  High:  {monthly_high:.2f} (highest of {len(month_data)} days)")
                logger.info(f"  Low:   {monthly_low:.2f} (lowest of {len(month_data)} days)")
                logger.info(f"  Close: {monthly_close:.2f} (last day: {month_data.iloc[-1]['date'].date()})")

                # Store monthly aggregates in a separate table (if exists)
                cursor.execute("""
                    INSERT INTO monthly_ohlc
                    (symbol, month, open, high, low, close, days_count, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol, month) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        days_count = EXCLUDED.days_count,
                        created_at = NOW()
                """, (
                    symbol,
                    first_day_previous.strftime('%Y-%m'),
                    float(monthly_open),
                    float(monthly_high),
                    float(monthly_low),
                    float(monthly_close),
                    len(month_data)
                ))

                symbols_processed += 1

                # Commit after each symbol
                conn.commit()

                # Small delay to avoid API rate limits
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                conn.rollback()
                continue

        logger.info("\n" + "="*60)
        logger.info(f"SUMMARY:")
        logger.info(f"  Symbols processed: {symbols_processed}")
        logger.info(f"  Total daily records stored: {total_days_fetched}")
        logger.info(f"  Month: {first_day_previous.strftime('%B %Y')}")
        logger.info("="*60)

        # Verify stored data
        cursor.execute("""
            SELECT symbol, COUNT(*) as days,
                   MIN(date) as first_date,
                   MAX(date) as last_date,
                   MIN(low) as month_low,
                   MAX(high) as month_high
            FROM eod_data
            WHERE date >= %s AND date <= %s
            GROUP BY symbol
            ORDER BY symbol
        """, (first_day_previous, last_day_previous))

        results = cursor.fetchall()
        logger.info("\nStored data verification:")
        for row in results:
            logger.info(f"  {row[0]}: {row[1]} days ({row[2]} to {row[3]}) L={row[4]:.2f} H={row[5]:.2f}")

    conn.close()
    logger.info("\nDatabase connection closed.")

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues
    asyncio.run(fetch_and_store_monthly_data())