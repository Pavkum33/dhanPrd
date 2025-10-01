#!/usr/bin/env python3
"""
Test PostgreSQL database with monthly pivot/CPR calculations
"""

import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import calculator
from scanners.monthly_levels import MonthlyLevelCalculator

def main():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='dhan_scanner_prod',
        user='postgres',
        password='India@123'
    )

    cursor = conn.cursor()

    print("=" * 80)
    print("TESTING POSTGRESQL DATABASE WITH MONTHLY CALCULATIONS")
    print("=" * 80)

    # Get September 2025 data for monthly calculation
    cursor.execute("""
        SELECT symbol,
               MIN(open) as first_open,
               MAX(high) as month_high,
               MIN(low) as month_low,
               MAX(close) as last_close
        FROM eod_data
        WHERE date >= '2025-09-01' AND date <= '2025-09-30'
        GROUP BY symbol
        LIMIT 10
    """)

    monthly_data = cursor.fetchall()

    if not monthly_data:
        print("[ERROR] No data found for September 2025")
        return

    print(f"\n[OK] Found {len(monthly_data)} symbols with September 2025 data")

    # Initialize calculator
    calculator = MonthlyLevelCalculator()

    # Calculate for each symbol
    for symbol, first_open, month_high, month_low, last_close in monthly_data:
        print(f"\n{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"September 2025 Monthly OHLC:")
        print(f"  Open:  {first_open:.2f} (first day)")
        print(f"  High:  {month_high:.2f} (month high)")
        print(f"  Low:   {month_low:.2f} (month low)")
        print(f"  Close: {last_close:.2f} (last day)")

        # Calculate CPR
        cpr = calculator.calculate_monthly_cpr(
            float(month_high),
            float(month_low),
            float(last_close)
        )

        # Calculate Pivots
        pivots = calculator.calculate_monthly_pivots(
            float(month_high),
            float(month_low),
            float(last_close)
        )

        print(f"\nOctober 2025 CPR Levels:")
        print(f"  TC:     {cpr['tc']:.2f}")
        print(f"  Pivot:  {cpr['pivot']:.2f}")
        print(f"  BC:     {cpr['bc']:.2f}")
        print(f"  Width%: {cpr['width_percent']:.3f}%")
        print(f"  Narrow: {'YES' if cpr['is_narrow'] else 'NO'}")

        print(f"\nOctober 2025 Support/Resistance:")
        print(f"  R3: {pivots['r3']:.2f}")
        print(f"  R2: {pivots['r2']:.2f}")
        print(f"  R1: {pivots['r1']:.2f}")
        print(f"  --PIVOT: {pivots['pivot']:.2f}--")
        print(f"  S1: {pivots['s1']:.2f}")
        print(f"  S2: {pivots['s2']:.2f}")
        print(f"  S3: {pivots['s3']:.2f}")

        # Store in pivot_levels table
        cursor.execute("""
            INSERT INTO pivot_levels
            (symbol, timeframe, calculation_date, prev_high, prev_low, prev_close,
             pivot, bc, tc, cpr_width, r1, r2, r3, s1, s2, s3, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (symbol, timeframe, calculation_date)
            DO UPDATE SET
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
            symbol, 'monthly', '2025-10-01',
            float(month_high), float(month_low), float(last_close),
            cpr['pivot'], cpr['bc'], cpr['tc'], cpr['width'],
            pivots['r1'], pivots['r2'], pivots['r3'],
            pivots['s1'], pivots['s2'], pivots['s3']
        ))

        # Store narrow CPR stocks
        if cpr['is_narrow']:
            cursor.execute("""
                INSERT INTO narrow_cpr_stocks
                (symbol, timeframe, calculation_date, pivot, bc, tc, cpr_width, r1, s1)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timeframe, calculation_date)
                DO UPDATE SET
                    pivot = EXCLUDED.pivot,
                    bc = EXCLUDED.bc,
                    tc = EXCLUDED.tc,
                    cpr_width = EXCLUDED.cpr_width,
                    r1 = EXCLUDED.r1,
                    s1 = EXCLUDED.s1
            """, (
                symbol, 'monthly', '2025-10-01',
                cpr['pivot'], cpr['bc'], cpr['tc'], cpr['width_percent'],
                pivots['r1'], pivots['s1']
            ))

    # Commit changes
    conn.commit()

    print("\n" + "=" * 80)
    print("[SUCCESS] STORED MONTHLY LEVELS IN POSTGRESQL")
    print("=" * 80)

    # Verify stored data
    cursor.execute("""
        SELECT COUNT(*) FROM pivot_levels
        WHERE timeframe = 'monthly' AND calculation_date = '2025-10-01'
    """)
    count = cursor.fetchone()[0]
    print(f"\nStored {count} monthly pivot levels for October 2025")

    cursor.execute("""
        SELECT COUNT(*) FROM narrow_cpr_stocks
        WHERE timeframe = 'monthly' AND calculation_date = '2025-10-01'
    """)
    narrow_count = cursor.fetchone()[0]
    print(f"Found {narrow_count} narrow CPR stocks for October 2025")

    # Show narrow CPR stocks
    if narrow_count > 0:
        cursor.execute("""
            SELECT symbol, cpr_width
            FROM narrow_cpr_stocks
            WHERE timeframe = 'monthly' AND calculation_date = '2025-10-01'
            ORDER BY cpr_width ASC
            LIMIT 5
        """)
        narrow_stocks = cursor.fetchall()
        print(f"\nTop Narrow CPR Stocks:")
        for symbol, width in narrow_stocks:
            print(f"  {symbol}: {width:.3f}%")

    conn.close()

if __name__ == "__main__":
    main()