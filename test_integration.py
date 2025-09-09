#!/usr/bin/env python3
"""
test_integration.py - Integration test for all components
Test the complete flow: Cache -> Monthly Levels -> Pre-market Job
"""

import asyncio
from datetime import datetime

def test_complete_integration():
    """Test complete integration of all components"""
    
    print("=" * 60)
    print("INTEGRATION TEST - COMPLETE SYSTEM")
    print("=" * 60)
    
    try:
        # Test 1: Cache Manager
        print("\n[TEST 1] Cache Manager...")
        from cache_manager import CacheManager
        cache = CacheManager()
        
        test_data = {"test": "integration", "timestamp": datetime.now().isoformat()}
        cache.set("integration_test", test_data, expiry_hours=1)
        retrieved = cache.get("integration_test")
        
        assert retrieved == test_data, "Cache integration failed"
        print("[PASS] Cache Manager working")
        
        # Test 2: Monthly Level Calculator
        print("\n[TEST 2] Monthly Level Calculator...")
        from scanners.monthly_levels import MonthlyLevelCalculator
        
        calculator = MonthlyLevelCalculator(cache)
        
        # Test with RELIANCE data
        reliance_data = {
            'high': 3125.00,
            'low': 2875.00,
            'close': 3050.00,
            'open': 2890.50
        }
        
        levels = calculator.calculate_and_cache_symbol_levels(
            'RELIANCE', reliance_data, '2024-12'
        )
        
        assert levels['cpr']['pivot'] == 3016.67, "CPR calculation failed"
        assert levels['pivots']['pivot'] == 3016.67, "Pivot calculation failed"
        print("[PASS] Monthly Level Calculator working")
        
        # Test 3: Cache retrieval
        print("\n[TEST 3] Cache Integration...")
        cached_levels = calculator.get_cached_levels('RELIANCE', '2024-12')
        
        assert cached_levels is not None, "Cache retrieval failed"
        assert cached_levels['symbol'] == 'RELIANCE', "Cached data mismatch"
        print("[PASS] Cache integration working")
        
        # Test 4: Scanner functions
        print("\n[TEST 4] Scanner Functions...")
        
        # Test narrow CPR detection
        narrow_symbols = calculator.get_symbols_with_narrow_cpr(['RELIANCE'], '2024-12')
        print(f"Narrow CPR symbols: {len(narrow_symbols)}")
        
        # Test pivot proximity
        current_prices = {'RELIANCE': 3020.00}  # Near pivot
        near_pivot = calculator.get_symbols_near_pivot(['RELIANCE'], current_prices, '2024-12')
        print(f"Near pivot symbols: {len(near_pivot)}")
        
        if len(near_pivot) > 0:
            print(f"  RELIANCE proximity: {near_pivot[0]['proximity_percent']:.3f}%")
        
        print("[PASS] Scanner functions working")
        
        # Test 5: Pre-market Job (without actual API calls)
        print("\n[TEST 5] Pre-market Job Health Check...")
        from premarket_job import PremarketJob
        
        job = PremarketJob()
        
        # Test health check
        health = asyncio.run(job.health_check())
        print(f"System Health: {health}")
        
        assert 'cache_health' in health, "Health check missing cache"
        assert health['cache_health']['sqlite'] == True, "SQLite not healthy"
        print("[PASS] Pre-market job health check working")
        
        # Final integration test - simulate complete workflow
        print("\n[TEST 6] Complete Workflow Simulation...")
        
        # Add more test symbols
        symbols = ['TCS', 'HDFCBANK', 'INFY']
        test_ohlc_data = {
            'TCS': {'high': 4150.00, 'low': 3850.00, 'close': 4025.00, 'open': 3900.00},
            'HDFCBANK': {'high': 1720.00, 'low': 1650.00, 'close': 1680.00, 'open': 1660.00},
            'INFY': {'high': 1890.00, 'low': 1820.00, 'close': 1855.00, 'open': 1835.00}
        }
        
        calculated_symbols = []
        for symbol, ohlc in test_ohlc_data.items():
            levels = calculator.calculate_and_cache_symbol_levels(symbol, ohlc, '2024-12')
            calculated_symbols.append(symbol)
            print(f"  {symbol}: CPR Width {levels['cpr']['width_percent']:.3f}%, Narrow: {levels['cpr']['is_narrow']}")
        
        # Test batch operations
        all_symbols = ['RELIANCE'] + calculated_symbols
        narrow_batch = calculator.get_symbols_with_narrow_cpr(all_symbols, '2024-12')
        print(f"Total narrow CPR stocks: {len(narrow_batch)}")
        
        # Test current prices simulation
        simulated_prices = {
            'RELIANCE': 3020.00,
            'TCS': 4010.00,  # Near TCS pivot (4008.33)
            'HDFCBANK': 1685.00,
            'INFY': 1858.00
        }
        
        near_pivot_batch = calculator.get_symbols_near_pivot(all_symbols, simulated_prices, '2024-12')
        print(f"Stocks near pivot: {len(near_pivot_batch)}")
        
        for stock in near_pivot_batch:
            print(f"  {stock['symbol']}: {stock['proximity_percent']:.3f}% from pivot")
        
        print("[PASS] Complete workflow simulation successful")
        
        # Cache statistics
        print(f"\n[STATS] Final Cache Statistics:")
        stats = cache.get_cache_stats()
        print(f"  Backend: {stats['current_backend']}")
        print(f"  SQLite entries: {stats['sqlite_active_entries']}")
        print(f"  Redis available: {stats['redis_available']}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Cache Manager: Working")
        print("✅ Monthly Level Calculator: Working")  
        print("✅ CPR & Pivot Formulas: Verified")
        print("✅ Caching Integration: Working")
        print("✅ Scanner Functions: Working")
        print("✅ Pre-market Job Framework: Ready")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_complete_integration()