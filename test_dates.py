#!/usr/bin/env python3
"""
Test script to understand why we get 44 days instead of 50
"""

from datetime import datetime, timedelta
import pandas as pd

def analyze_trading_days():
    """Analyze the actual trading days vs calendar days"""
    
    # Current date calculation (same as in app.py)
    to_date = datetime.now().date()
    
    # Test different day parameters including new default
    for request_days in [50, 60, 75, 90]:
        from_date = to_date - timedelta(days=request_days + 5)
        
        print(f"\n=== Requesting {request_days} days ===")
        print(f"From Date: {from_date}")
        print(f"To Date: {to_date}")
        print(f"Calendar Days: {(to_date - from_date).days}")
        
        # Calculate trading days (exclude weekends)
        current_date = from_date
        trading_days = 0
        weekend_days = 0
        
        while current_date <= to_date:
            # Monday=0, Sunday=6
            if current_date.weekday() < 5:  # Mon-Fri
                trading_days += 1
            else:
                weekend_days += 1
            current_date += timedelta(days=1)
        
        print(f"Trading Days (Mon-Fri): {trading_days}")
        print(f"Weekend Days: {weekend_days}")
        
        # Account for typical market holidays in India (rough estimate)
        estimated_holidays = int(trading_days * 0.08)  # ~8% holiday rate
        estimated_actual_trading_days = trading_days - estimated_holidays
        
        print(f"Estimated Holidays: {estimated_holidays}")
        print(f"Estimated Actual Trading Days: {estimated_actual_trading_days}")

def check_september_holidays():
    """Check specific holidays in September 2025"""
    print("\n=== September 2025 Market Holidays ===")
    
    # Common Indian market holidays that might affect data
    holidays_2025 = [
        ("2025-08-15", "Independence Day"),
        ("2025-08-19", "Parsi New Year"),
        ("2025-09-07", "Ganesh Chaturthi"),
        ("2025-09-16", "Id-e-Milad"),
        ("2025-10-02", "Gandhi Jayanti"),
        ("2025-10-21", "Dussehra"),
        ("2025-11-01", "Diwali Balipratipada"),
        ("2025-11-13", "Bhai Dooj"),
    ]
    
    from_date = datetime.now().date() - timedelta(days=65)
    to_date = datetime.now().date()
    
    relevant_holidays = []
    for holiday_date, name in holidays_2025:
        holiday = datetime.strptime(holiday_date, "%Y-%m-%d").date()
        if from_date <= holiday <= to_date:
            relevant_holidays.append((holiday, name))
    
    print(f"Holidays in period {from_date} to {to_date}:")
    for holiday, name in relevant_holidays:
        print(f"  {holiday} - {name}")
    
    return len(relevant_holidays)

if __name__ == "__main__":
    print("HISTORICAL DATA DATE ANALYSIS")
    print("=" * 50)
    
    analyze_trading_days()
    holiday_count = check_september_holidays()
    
    print(f"\nSUMMARY:")
    print(f"   • OLD CONFIG: 60 calendar days + 5 buffer = 65 calendar days -> ~44 trading days")
    print(f"   • NEW CONFIG: 75 calendar days + 5 buffer = 80 calendar days -> ~57 trading days") 
    print(f"   • Weekend exclusions: ~16-20 days")  
    print(f"   • Market holidays: ~{holiday_count} days")
    print(f"   • With 75-day config: MORE DATA for better resistance calculations!")
    print(f"   • Analysis still uses 50-day lookback for breakout logic")