#!/usr/bin/env python3
"""
Fetch 80 days of historical data for ALL F&O symbols
Then calculate proper monthly CPR/Pivot levels
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_all_fno_data():
    """Fetch 80 days of data for all F&O symbols"""

    from dhan_fetcher import DhanHistoricalFetcher
    from scanners.monthly_levels import MonthlyLevelCalculator

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

    logger.info("="*80)
    logger.info("FETCHING 80 DAYS DATA FOR ALL F&O SYMBOLS")
    logger.info("="*80)

    async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
        # Get all F&O instruments
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

        logger.info(f"Processing {len(underlying_symbols)} unique F&O underlying symbols")

        # Clear old data first
        logger.info("Clearing old data...")
        cursor.execute("TRUNCATE TABLE eod_data")
        cursor.execute("TRUNCATE TABLE monthly_ohlc")
        conn.commit()

        total_records = 0
        symbols_processed = 0
        failed_symbols = []

        # Process each symbol
        for i, symbol in enumerate(underlying_symbols, 1):
            try:
                if symbol not in equity_mapping:
                    logger.warning(f"[{i}/{len(underlying_symbols)}] {symbol}: No security ID mapping")
                    continue

                security_id = int(equity_mapping[symbol])
                logger.info(f"[{i}/{len(underlying_symbols)}] Fetching {symbol} (security_id={security_id})...")

                # Fetch 80 days of data
                df = await fetcher.get_historical_data_for_underlying(
                    symbol,
                    security_id,
                    days=80
                )

                if df is None or df.empty:
                    logger.warning(f"{symbol}: No data received")
                    failed_symbols.append(symbol)
                    continue

                # Convert date column
                df['date'] = pd.to_datetime(df['date'])

                # Insert into database
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO eod_data
                        (symbol, date, open, high, low, close, volume, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (symbol, date) DO NOTHING
                    """, (
                        symbol,
                        row['date'].date(),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row.get('volume', 0))
                    ))

                total_records += len(df)
                symbols_processed += 1

                # Commit every 10 symbols
                if symbols_processed % 10 == 0:
                    conn.commit()
                    logger.info(f"Progress: {symbols_processed}/{len(underlying_symbols)} symbols, {total_records} records")

                # Small delay to avoid rate limits
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"{symbol}: Error - {e}")
                failed_symbols.append(symbol)
                continue

        # Final commit
        conn.commit()

        logger.info("="*80)
        logger.info(f"FETCH COMPLETE:")
        logger.info(f"  Symbols processed: {symbols_processed}")
        logger.info(f"  Total records: {total_records}")
        logger.info(f"  Failed symbols: {len(failed_symbols)}")
        logger.info("="*80)

        # Now calculate monthly OHLC for each month
        logger.info("\nCalculating monthly OHLC from daily data...")

        # Get distinct months in the data
        cursor.execute("""
            SELECT DISTINCT DATE_TRUNC('month', date) as month
            FROM eod_data
            ORDER BY month
        """)

        months = cursor.fetchall()
        logger.info(f"Found data for {len(months)} months")

        for month_date, in months:
            month_str = month_date.strftime('%Y-%m')
            logger.info(f"\nProcessing month: {month_str}")

            # Calculate monthly OHLC for all symbols
            cursor.execute("""
                INSERT INTO monthly_ohlc (symbol, month, open, high, low, close, volume, days_count, created_at)
                SELECT
                    symbol,
                    %s as month,
                    (SELECT open FROM eod_data e2
                     WHERE e2.symbol = e.symbol
                     AND DATE_TRUNC('month', e2.date) = %s
                     ORDER BY date ASC LIMIT 1) as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    (SELECT close FROM eod_data e3
                     WHERE e3.symbol = e.symbol
                     AND DATE_TRUNC('month', e3.date) = %s
                     ORDER BY date DESC LIMIT 1) as close,
                    SUM(volume) as volume,
                    COUNT(*) as days_count,
                    NOW() as created_at
                FROM eod_data e
                WHERE DATE_TRUNC('month', date) = %s
                GROUP BY symbol
                HAVING COUNT(*) >= 10
                ON CONFLICT (symbol, month) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    days_count = EXCLUDED.days_count,
                    updated_at = NOW()
            """, (month_str, month_date, month_date, month_date))

            affected = cursor.rowcount
            conn.commit()
            logger.info(f"  Calculated monthly OHLC for {affected} symbols in {month_str}")

        # Calculate CPR/Pivots for the latest complete month
        logger.info("\n" + "="*80)
        logger.info("CALCULATING MONTHLY CPR/PIVOT LEVELS")
        logger.info("="*80)

        # Get the latest complete month's data
        cursor.execute("""
            SELECT symbol, month, open, high, low, close, days_count
            FROM monthly_ohlc
            WHERE days_count >= 15
            ORDER BY month DESC, symbol
            LIMIT 500
        """)

        monthly_data = cursor.fetchall()

        if monthly_data:
            calculator = MonthlyLevelCalculator()
            narrow_cpr_count = 0

            current_month = None
            for symbol, month, m_open, m_high, m_low, m_close, days_count in monthly_data[:20]:  # Show first 20
                if current_month != month:
                    current_month = month
                    logger.info(f"\nMonth: {month}")
                    logger.info("-"*40)

                # Calculate CPR
                cpr = calculator.calculate_monthly_cpr(float(m_high), float(m_low), float(m_close))

                # Calculate Pivots
                pivots = calculator.calculate_monthly_pivots(float(m_high), float(m_low), float(m_close))

                if cpr['is_narrow']:
                    narrow_cpr_count += 1
                    marker = "[NARROW]"
                else:
                    marker = ""

                logger.info(f"{symbol}: {days_count} days, CPR Width: {cpr['width_percent']:.3f}% {marker}")

                # Store in pivot_levels table
                next_month = datetime.strptime(month, '%Y-%m').replace(day=1) + timedelta(days=32)
                next_month = next_month.replace(day=1)

                cursor.execute("""
                    INSERT INTO pivot_levels
                    (symbol, timeframe, calculation_date, prev_high, prev_low, prev_close,
                     pivot, bc, tc, cpr_width, r1, r2, r3, s1, s2, s3, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol, timeframe, calculation_date) DO UPDATE SET
                        prev_high = EXCLUDED.prev_high,
                        prev_low = EXCLUDED.prev_low,
                        prev_close = EXCLUDED.prev_close,
                        pivot = EXCLUDED.pivot,
                        bc = EXCLUDED.bc,
                        tc = EXCLUDED.tc,
                        cpr_width = EXCLUDED.cpr_width,
                        r1 = EXCLUDED.r1,
                        r2 = EXCLUDED.r2,
                        r3 = EXCLUDED.r3,
                        s1 = EXCLUDED.s1,
                        s2 = EXCLUDED.s2,
                        s3 = EXCLUDED.s3,
                        created_at = NOW()
                """, (
                    symbol, 'monthly', next_month,
                    float(m_high), float(m_low), float(m_close),
                    cpr['pivot'], cpr['bc'], cpr['tc'], cpr['width'],
                    pivots['r1'], pivots['r2'], pivots['r3'],
                    pivots['s1'], pivots['s2'], pivots['s3']
                ))

            conn.commit()

            # Summary statistics
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT symbol) as total_symbols,
                    COUNT(DISTINCT date) as total_days,
                    MIN(date) as first_date,
                    MAX(date) as last_date
                FROM eod_data
            """)

            stats = cursor.fetchone()

            logger.info("\n" + "="*80)
            logger.info("FINAL STATISTICS:")
            logger.info("="*80)
            logger.info(f"Total symbols with data: {stats[0]}")
            logger.info(f"Total trading days: {stats[1]}")
            logger.info(f"Date range: {stats[2]} to {stats[3]}")

            cursor.execute("""
                SELECT COUNT(*) FROM monthly_ohlc WHERE days_count >= 15
            """)

            reliable_count = cursor.fetchone()[0]
            logger.info(f"Reliable monthly calculations (>=15 days): {reliable_count}")
            logger.info(f"Narrow CPR stocks found: {narrow_cpr_count}")

        else:
            logger.error("No monthly data available for CPR calculation")

    conn.close()
    logger.info("\nDatabase connection closed.")

if __name__ == "__main__":
    asyncio.run(fetch_all_fno_data())