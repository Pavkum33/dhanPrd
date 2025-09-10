#!/usr/bin/env python3
"""
Test Railway HTTP endpoint for live market test (without WebSocket)
This verifies the endpoint works even if WebSocket connection fails
"""
import requests
import time
import json

RAILWAY_URL = "https://dhanscanpoc.up.railway.app"

def test_railway_http_endpoint():
    """Test just the HTTP endpoint on Railway"""
    try:
        print("Testing Railway HTTP endpoint...")
        
        # Test basic status first
        print("1. Checking Railway status...")
        status_response = requests.get(f'{RAILWAY_URL}/api/status', timeout=10)
        print(f"   Status: {status_response.status_code}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   Active symbols: {status_data.get('active_symbols', 0)}")
            print(f"   Connected clients: {status_data.get('connected_clients', 0)}")
        
        # Test the live market endpoint
        print("\n2. Testing live market endpoint...")
        response = requests.post(f'{RAILWAY_URL}/api/websocket/live-market-test', 
                               json={
                                   'symbol': 'RELIANCE',
                                   'duration': 10  # Short test
                               },
                               headers={'Content-Type': 'application/json'},
                               timeout=15)
        
        print(f"   HTTP Response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Message: {data.get('message', 'No message')}")
            print(f"   WebSocket Events: {data.get('websocket_events', 'None')}")
            print(f"   Duration: {data.get('duration', 'Unknown')}")
            
            if data.get('status') == 'test_started':
                print("   SUCCESS: Live market test started on Railway!")
                return True
            elif data.get('status') == 'demo_mode':
                print("   SUCCESS: Running in demo mode (no credentials)")
                return True
            else:
                print(f"   Unexpected status: {data.get('status')}")
                return False
        else:
            print(f"   ERROR: {response.text}")
            return False
            
    except Exception as e:
        print(f"Railway HTTP test failed: {e}")
        return False

def test_railway_dashboard():
    """Test if Railway dashboard is accessible"""
    try:
        print("\n3. Testing Railway dashboard...")
        response = requests.get(f'{RAILWAY_URL}/', timeout=10)
        print(f"   Dashboard: {response.status_code}")
        
        if response.status_code == 200:
            print("   SUCCESS: Dashboard accessible")
            return True
        else:
            print("   ERROR: Dashboard not accessible")
            return False
            
    except Exception as e:
        print(f"Dashboard test failed: {e}")
        return False

if __name__ == "__main__":
    print("Railway HTTP Live Market Test\n")
    
    endpoint_success = test_railway_http_endpoint()
    dashboard_success = test_railway_dashboard()
    
    print(f"\nTest Results:")
    print(f"  HTTP Endpoint: {'PASSED' if endpoint_success else 'FAILED'}")
    print(f"  Dashboard: {'PASSED' if dashboard_success else 'FAILED'}")
    
    overall_success = endpoint_success and dashboard_success
    print(f"\nOverall Railway Status: {'WORKING' if overall_success else 'ISSUES DETECTED'}")
    
    if overall_success:
        print("\nNext step: Test WebSocket in browser at:")
        print(f"{RAILWAY_URL}")
        print("Click 'Live Market Test' button to see WebSocket events in action!")
    else:
        print("\nCheck Railway deployment logs for issues")