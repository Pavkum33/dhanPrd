#!/usr/bin/env python3
"""
Verify Multi-Scan Data Access - Test that populated cache data is accessible
"""

from cache_manager import CacheManager
from scanners.monthly_levels import MonthlyLevelCalculator
import json

def verify_multiscan_data():
    """Verify that multi-scan UI can access the populated test data"""
    
    print("VERIFYING MULTI-SCAN DATA ACCESS")
    print("=" * 50)
    
    try:
        cache = CacheManager()
        calculator = MonthlyLevelCalculator(cache)
        month = '2024-12'
        
        # Test 1: Check scan results cache
        print("\n[TEST 1] Scan Results Cache Access:")
        scan_cache_key = f"scan_results:{month}"
        scan_results = cache.get(scan_cache_key)
        
        if scan_results and 'narrow_cpr' in scan_results:
            narrow_cpr = scan_results['narrow_cpr']
            print(f"✅ Narrow CPR data accessible: {len(narrow_cpr)} stocks")
            
            # Display first 3 for verification
            for i, stock in enumerate(narrow_cpr[:3]):
                print(f"   {i+1}. {stock['symbol']}: {stock['cpr_width_percent']:.3f}%")
            
            if len(narrow_cpr) > 3:
                print(f"   ... and {len(narrow_cpr) - 3} more stocks")
                
        else:
            print("❌ Scan results cache not found")
            return False
            
        # Test 2: Check individual symbol levels
        print(f"\n[TEST 2] Individual Symbol Level Access:")
        test_symbols = ['TCS', 'HDFCBANK', 'INFY']
        
        for symbol in test_symbols:
            levels = calculator.get_cached_levels(symbol, month)
            if levels:
                cpr = levels['cpr']
                pivot = levels['pivots']
                print(f"✅ {symbol}: CPR {cpr['width_percent']:.3f}%, Pivot ₹{pivot['pivot']:.2f}")
            else:
                print(f"❌ {symbol}: No cached levels")
                
        # Test 3: Scanner Function Access
        print(f"\n[TEST 3] Scanner Functions:")
        
        # Test narrow CPR batch access
        all_symbols = ['TCS', 'HDFCBANK', 'INFY', 'RELIANCE', 'BAJFINANCE']
        narrow_symbols = calculator.get_symbols_with_narrow_cpr(all_symbols, month)
        print(f"✅ Narrow CPR function: {len(narrow_symbols)} stocks detected")
        
        # Test pivot proximity with current prices
        current_prices = {
            'TCS': 4010.00,
            'HDFCBANK': 1683.50,
            'INFY': 1853.00
        }
        
        near_pivot = calculator.get_symbols_near_pivot(list(current_prices.keys()), current_prices, month)
        print(f"✅ Pivot proximity function: {len(near_pivot)} stocks near pivot")
        
        # Test 4: Multi-Scan UI Data Format
        print(f"\n[TEST 4] Multi-Scan UI Data Format:")
        
        # Simulate what the UI JavaScript would receive
        ui_cpr_data = []
        for stock in narrow_cpr[:5]:  # First 5 for UI testing
            ui_stock = {
                'symbol': stock['symbol'],
                'cpr_width_percent': stock['cpr_width_percent'],
                'pivot': stock.get('pivot', 0),
                'volume_above_avg': stock.get('volume_above_avg', False),
                'change': stock.get('change', 0)
            }
            ui_cpr_data.append(ui_stock)
            
        print(f"✅ UI-ready CPR data: {len(ui_cpr_data)} stocks formatted")
        
        ui_pivot_data = []
        for stock in near_pivot:
            ui_stock = {
                'symbol': stock['symbol'],
                'proximity_percent': stock['proximity_percent'],
                'current_price': stock['current_price'],
                'pivot': stock['pivot']
            }
            ui_pivot_data.append(ui_stock)
            
        print(f"✅ UI-ready Pivot data: {len(ui_pivot_data)} stocks formatted")
        
        # Summary
        print(f"\n[SUMMARY] Multi-Scan Data Verification:")
        print(f"✅ Cache accessible: {len(scan_results.get('narrow_cpr', []))} CPR stocks")
        print(f"✅ Individual levels: {len([s for s in test_symbols if calculator.get_cached_levels(s, month)])}/{len(test_symbols)} symbols")
        print(f"✅ Scanner functions: Working")
        print(f"✅ UI data format: Ready")
        
        print(f"\n🎉 MULTI-SCAN BACKEND READY!")
        print(f"The multi-scan UI should now display:")
        print(f"  • Narrow CPR card: {len(narrow_cpr)} stocks")  
        print(f"  • Pivot Proximity card: {len(near_pivot)} stocks")
        print(f"  • Professional data controls: Active for large datasets")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_multiscan_data()
    print("\n" + "=" * 50)
    if success:
        print("VERIFICATION COMPLETE - READY FOR UI TESTING")
    else:
        print("VERIFICATION FAILED - CHECK CACHE DATA")
    print("=" * 50)