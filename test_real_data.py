#!/usr/bin/env python3
"""
test_real_data.py - Test script to populate cache with real Dhan data
Bypasses environment variable issues by setting credentials directly
"""

import os
import asyncio
from cache_manager import CacheManager
from scanners.monthly_levels import MonthlyLevelCalculator
from dhan_fetcher import DhanHistoricalFetcher

async def test_real_data_population():
    """Test populating cache with real Dhan data"""
    
    # Set credentials directly (bypass environment variable issues)
    client_id = "1106283829"
    access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU5OTk1NTg4LCJpYXQiOjE3NTc0MDM1ODgsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA2MjgzODI5In0.YSQW8r4VKTLAEeLfz3AnlvsL-4nPc_hi50DNJSGv8J9Mp9UVTHavYNvGK4cKlGSRqPH-pGNehP10FmQ5qE9ONA"
    
    print("TESTING Real Dhan API Connection...")
    print(f"Client ID: {client_id}")
    print(f"Token: {access_token[:50]}...")
    
    # Initialize cache and calculator
    cache = CacheManager()
    calculator = MonthlyLevelCalculator(cache)
    
    # Test with a few well-known symbols
    test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ADANIPORTS']
    
    print(f"\nTesting with symbols: {test_symbols}")
    
    try:
        # Initialize fetcher
        async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
            
            print("\n1. Testing API connection...")
            
            # Load equity mapping
            print("2. Loading equity instruments...")
            equity_mapping = await fetcher.load_equity_instruments()
            print(f"   Loaded {len(equity_mapping)} equity mappings")
            
            if len(equity_mapping) == 0:
                print("❌ No equity mappings loaded - API may be failing")
                return
            
            # Show sample mappings
            sample_symbols = list(equity_mapping.keys())[:5]
            for sym in sample_symbols:
                print(f"   {sym} -> {equity_mapping[sym]}")
            
            print("\n3. Testing monthly level calculation for sample symbols...")
            
            successful_calculations = 0
            
            for symbol in test_symbols:
                try:
                    if symbol not in equity_mapping:
                        print(f"   WARNING {symbol}: Not found in equity mapping")
                        continue
                    
                    security_id = int(equity_mapping[symbol])
                    print(f"   FETCHING {symbol}: Getting historical data (security_id={security_id})...")
                    
                    # Fetch 30 days of data for monthly calculation
                    historical_df = await fetcher.get_historical_data_for_underlying(
                        symbol, security_id, days=30
                    )
                    
                    if historical_df.empty:
                        print(f"   ERROR {symbol}: No historical data received")
                        continue
                    
                    print(f"   SUCCESS {symbol}: Got {len(historical_df)} days of data")
                    
                    # Calculate monthly OHLC from the data
                    monthly_ohlc = {
                        'high': historical_df['high'].max(),
                        'low': historical_df['low'].min(), 
                        'close': historical_df['close'].iloc[-1],  # Last close
                        'open': historical_df['open'].iloc[0]      # First open
                    }
                    
                    print(f"   DATA {symbol}: H={monthly_ohlc['high']:.2f}, L={monthly_ohlc['low']:.2f}, C={monthly_ohlc['close']:.2f}")
                    
                    # Calculate and cache levels
                    levels = calculator.calculate_and_cache_symbol_levels(
                        symbol, monthly_ohlc
                    )
                    
                    # Show CPR analysis
                    cpr = levels['cpr']
                    print(f"   CPR {symbol}: Width = {cpr['width_percent']:.4f}%, Narrow = {cpr['is_narrow']}")
                    
                    if cpr['is_narrow']:
                        print(f"   ALERT {symbol}: NARROW CPR DETECTED! Width = {cpr['width_percent']:.4f}%")
                    
                    successful_calculations += 1
                    
                    # Small delay to avoid overwhelming API
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"   ❌ {symbol}: Error - {e}")
                    continue
            
            print(f"\n4. ✅ Successfully calculated levels for {successful_calculations}/{len(test_symbols)} symbols")
            
            if successful_calculations > 0:
                print("\n5. Testing cache retrieval...")
                
                # Test narrow CPR detection
                narrow_cpr_results = calculator.get_symbols_with_narrow_cpr(test_symbols)
                print(f"   🔍 Found {len(narrow_cpr_results)} symbols with narrow CPR:")
                
                for result in narrow_cpr_results:
                    print(f"   📊 {result['symbol']}: {result['cpr_width_percent']:.4f}% width, {result['trend']} trend")
                
                # Cache scan results (like premarket job does)
                scan_results = {
                    'narrow_cpr': [{
                        'symbol': r['symbol'],
                        'cpr_width_percent': r['cpr_width_percent'],
                        'breakout_level': r['breakout_level'],
                        'trend': r['trend']
                    } for r in narrow_cpr_results],
                    'total_symbols': len(test_symbols),
                    'last_updated': f"{asyncio.datetime.datetime.now().isoformat()}"
                }
                
                scan_cache_key = f"scan_results:2025-09"  # Current month
                cache.set(scan_cache_key, scan_results, expiry_hours=24*7)
                
                print(f"   💾 Cached scan results: {len(narrow_cpr_results)} narrow CPR symbols")
                
                print(f"\n🎉 SUCCESS: Multi-scan system populated with real Dhan data!")
                print(f"   - Cache contains levels for {successful_calculations} symbols")
                print(f"   - Found {len(narrow_cpr_results)} narrow CPR opportunities")
                print(f"   - Scan results cached for API access")
                
                return True
            else:
                print(f"\n❌ FAILED: Could not populate any data")
                return False
                
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_real_data_population())
    
    if result:
        print(f"\n✅ MULTI-SCAN SYSTEM READY WITH REAL DATA!")
        print(f"   API endpoints should now return real market data instead of 404 errors")
    else:
        print(f"\n❌ FAILED TO POPULATE REAL DATA") 