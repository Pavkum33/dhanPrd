#!/usr/bin/env python3
"""
test_api_endpoints.py - Test the new monthly levels API endpoints
"""

import requests
import json
from datetime import datetime

# Base URL for local testing
BASE_URL = "http://localhost:5000"

def test_api_endpoints():
    """Test all new API endpoints"""
    
    print("=" * 60)
    print("TESTING MONTHLY LEVELS API ENDPOINTS")
    print("=" * 60)
    
    # Test 1: Test calculation endpoint
    print("\n[TEST 1] Testing /api/levels/test endpoint...")
    test_data = {
        "symbol": "RELIANCE_TEST",
        "ohlc": {
            "high": 3125.00,
            "low": 2875.00, 
            "close": 3050.00,
            "open": 2890.50
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/levels/test", json=test_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[PASS] Test calculation successful")
            print(f"Symbol: {result['input_data']['symbol']}")
            print(f"CPR Pivot: {result['calculated_levels']['cpr']['pivot']}")
            print(f"CPR Narrow: {result['calculated_levels']['cpr']['is_narrow']}")
            print(f"Pivot Point: {result['calculated_levels']['pivots']['pivot']}")
        else:
            print(f"[FAIL] Test calculation failed: {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Test calculation error: {e}")
    
    # Test 2: Retrieve symbol levels
    print(f"\n[TEST 2] Testing /api/levels/<symbol> endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/levels/RELIANCE_TEST?month=test-month")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[PASS] Symbol levels retrieved")
            print(f"Symbol: {result['symbol']}")
            print(f"Month: {result['month']}")
            print(f"CPR Width %: {result['levels']['cpr']['width_percent']}%")
        else:
            print(f"[INFO] No cached levels found (expected for first run): {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Symbol levels error: {e}")
    
    # Test 3: Near pivot detection
    print(f"\n[TEST 3] Testing /api/levels/near-pivot endpoint...")
    pivot_test_data = {
        "current_prices": {
            "RELIANCE": 3020.00,  # Near the pivot (3016.67)
            "TCS": 4100.00,
            "HDFCBANK": 1680.00
        },
        "symbols": ["RELIANCE", "TCS", "HDFCBANK"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/levels/near-pivot", json=pivot_test_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[PASS] Near pivot check completed")
            print(f"Symbols checked: {result['input_symbols']}")
            print(f"Near pivot count: {result['count']}")
            
            for symbol in result['near_pivot_symbols']:
                print(f"  {symbol['symbol']}: {symbol['current_price']} (Pivot: {symbol['pivot']}, Distance: {symbol['proximity_percent']:.2f}%)")
        else:
            print(f"[INFO] Near pivot check result: {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Near pivot error: {e}")
    
    # Test 4: Narrow CPR symbols  
    print(f"\n[TEST 4] Testing /api/levels/narrow-cpr endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/levels/narrow-cpr")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[PASS] Narrow CPR symbols retrieved")
            print(f"Month: {result['month']}")
            print(f"Narrow CPR count: {result['count']}")
            
            for symbol in result['narrow_cpr_symbols'][:5]:  # Show first 5
                print(f"  {symbol['symbol']}: {symbol['cpr_width_percent']}% width")
        else:
            print(f"[INFO] No narrow CPR data found (run calculation first): {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Narrow CPR error: {e}")
    
    # Test 5: Pre-market summary
    print(f"\n[TEST 5] Testing /api/levels/premarket-summary endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/levels/premarket-summary")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[PASS] Pre-market summary retrieved")
            
            if result.get('summary'):
                summary = result['summary']
                print(f"Job ID: {summary.get('job_id', 'N/A')}")
                print(f"Total symbols: {summary.get('total_symbols', 0)}")
                print(f"Successful: {summary.get('successful', 0)}")
                print(f"Failed: {summary.get('failed', 0)}")
            else:
                print("[INFO] No pre-market summary available yet")
                
            health = result.get('health', {})
            print(f"System Health:")
            print(f"  Cache: {health.get('cache_health', {})}")
            print(f"  Credentials: {health.get('credentials_available', False)}")
        else:
            print(f"[INFO] Pre-market summary: {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Pre-market summary error: {e}")
    
    # Test 6: Manual calculation trigger (optional - comment out if not needed)
    print(f"\n[TEST 6] Testing /api/levels/calculate endpoint (OPTIONAL)...")
    try:
        # Uncomment to test - will start actual calculation
        # response = requests.post(f"{BASE_URL}/api/levels/calculate")
        # print(f"Status: {response.status_code}")
        # 
        # if response.status_code == 200:
        #     result = response.json()
        #     print(f"[PASS] Manual calculation started")
        #     print(f"Message: {result['message']}")
        #     print(f"Status: {result['status']}")
        # else:
        #     print(f"[FAIL] Manual calculation failed: {response.text}")
        
        print("[SKIPPED] Manual calculation test (requires Dhan credentials)")
    
    except Exception as e:
        print(f"[ERROR] Manual calculation error: {e}")
    
    print("\n" + "=" * 60)
    print("[COMPLETE] API endpoint testing finished")
    print("=" * 60)

def check_app_running():
    """Check if Flask app is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            print(f"[INFO] Flask app is running - API status OK")
            return True
        else:
            print(f"[ERROR] Flask app returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Flask app not reachable: {e}")
        print("[INFO] Make sure Flask app is running with: python app.py")
        return False

if __name__ == "__main__":
    print("Checking if Flask app is running...")
    
    if check_app_running():
        test_api_endpoints()
    else:
        print("Cannot test - Flask app not running")
        print("Start the app with: python app.py")