#!/usr/bin/env python3
"""
Populate Test Data - Create scan results cache for UI testing
This bypasses the API routing issue and directly populates cache with test data
"""

from cache_manager import CacheManager
from scanners.monthly_levels import MonthlyLevelCalculator
from datetime import datetime

def populate_scan_results_cache():
    """Populate cache with scan results for testing multi-scan UI"""
    
    print("=" * 60)
    print("POPULATING TEST DATA FOR MULTI-SCAN UI")
    print("=" * 60)
    
    try:
        # Initialize components
        cache = CacheManager()
        calculator = MonthlyLevelCalculator(cache)
        
        month = '2024-12'
        
        # Create test OHLC data for multiple stocks
        test_stocks = {
            'TCS': {'high': 4150.00, 'low': 3850.00, 'close': 4025.00, 'open': 3900.00},
            'HDFCBANK': {'high': 1720.00, 'low': 1650.00, 'close': 1680.00, 'open': 1660.00},
            'INFY': {'high': 1890.00, 'low': 1820.00, 'close': 1855.00, 'open': 1835.00},
            'RELIANCE': {'high': 3125.00, 'low': 2875.00, 'close': 3050.00, 'open': 2890.50},
            'BAJFINANCE': {'high': 7850.00, 'low': 7650.00, 'close': 7720.00, 'open': 7700.00},
            'WIPRO': {'high': 445.00, 'low': 430.00, 'close': 440.00, 'open': 435.00},
            'TECHM': {'high': 1680.00, 'low': 1620.00, 'close': 1650.00, 'open': 1640.00},
            'LT': {'high': 3650.00, 'low': 3500.00, 'close': 3580.00, 'open': 3520.00},
            'MARUTI': {'high': 11200.00, 'low': 10800.00, 'close': 11000.00, 'open': 10900.00},
            'ASIANPAINT': {'high': 3250.00, 'low': 3100.00, 'close': 3180.00, 'open': 3120.00}
        }
        
        print(f"Calculating levels for {len(test_stocks)} stocks...")
        
        # Calculate levels for all test stocks
        calculated_levels = {}
        narrow_cpr_stocks = []
        
        for symbol, ohlc in test_stocks.items():
            print(f"Processing {symbol}...")
            
            # Calculate and cache levels
            levels = calculator.calculate_and_cache_symbol_levels(symbol, ohlc, month)
            calculated_levels[symbol] = levels
            
            # Check if narrow CPR
            if levels['cpr']['is_narrow']:
                narrow_cpr_stocks.append({
                    'symbol': symbol,
                    'cpr_width_percent': levels['cpr']['width_percent'],
                    'pivot': levels['cpr']['pivot'],
                    'volume_above_avg': symbol in ['TCS', 'RELIANCE'],  # Simulate volume data
                    'change': 1.5 if symbol == 'TCS' else -0.8  # Simulate price change
                })
        
        # Create comprehensive scan results cache
        scan_results = {
            'narrow_cpr': narrow_cpr_stocks,
            'total_symbols': len(test_stocks),
            'last_updated': datetime.now().isoformat(),
            'calculation_time': '2.3 seconds',
            'success_count': len(test_stocks),
            'failed_count': 0
        }
        
        # Store scan results in cache
        scan_cache_key = f"scan_results:{month}"
        cache.set(scan_cache_key, scan_results, expiry_hours=24)
        
        print(f"\n[RESULTS]")
        print(f"Total stocks processed: {len(test_stocks)}")
        print(f"Narrow CPR stocks found: {len(narrow_cpr_stocks)}")
        for stock in narrow_cpr_stocks:
            print(f"  {stock['symbol']}: {stock['cpr_width_percent']:.3f}%")
        
        # Test pivot proximity with simulated current prices
        current_prices = {
            'TCS': 4010.00,      # Very close to pivot (4008.33)
            'HDFCBANK': 1683.50, # Close to pivot (1685.00)  
            'RELIANCE': 3020.00, # Close to pivot (3016.67)
            'INFY': 1853.00,     # Very close to pivot (1855.00)
            'WIPRO': 439.50      # Close to pivot  
        }
        
        symbols = list(current_prices.keys())
        near_pivot = calculator.get_symbols_near_pivot(symbols, current_prices, month)
        
        # Add pivot proximity to scan results
        scan_results['near_pivot'] = near_pivot
        cache.set(scan_cache_key, scan_results, expiry_hours=24)
        
        print(f"\nNear pivot stocks found: {len(near_pivot)}")
        for stock in near_pivot:
            print(f"  {stock['symbol']}: {stock['proximity_percent']:.3f}% from pivot")
        
        print(f"\n[CACHE STATUS]")
        stats = cache.get_cache_stats()
        print(f"Cache backend: {stats['current_backend']}")
        print(f"Active entries: {stats['sqlite_active_entries']}")
        
        print(f"\n[SUCCESS] Test data populated successfully!")
        print(f"Multi-scan UI now has data to display:")
        print(f"  - Narrow CPR: {len(narrow_cpr_stocks)} stocks") 
        print(f"  - Near Pivot: {len(near_pivot)} stocks")
        print(f"  - Cache key: {scan_cache_key}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to populate test data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = populate_scan_results_cache()
    if success:
        print(f"\n{'='*60}")
        print("✅ MULTI-SCAN UI READY FOR TESTING!")
        print("Navigate to Multi-Scan tab to see the populated data")
        print("='*60")
    else:
        print(f"\n{'='*60}")
        print("❌ Data population failed")
        print("='*60")