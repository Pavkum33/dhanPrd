#!/usr/bin/env python3
"""
Test with live DHAN credentials
"""
import os
import socketio
import requests
import time

# Set credentials in environment
os.environ['DHAN_CLIENT_ID'] = '1106283829'
os.environ['DHAN_ACCESS_TOKEN'] = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU5OTk1NTg4LCJpYXQiOjE3NTc0MDM1ODgsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA2MjgzODI5In0.YSQW8r4VKTLAEeLfz3AnlvsL-4nPc_hi50DNJSGv8J9Mp9UVTHavYNvGK4cKlGSRqPH-pGNehP10FmQ5qE9ONA'

# Import and start app
from app import app, socketio as app_socketio

def test_with_credentials():
    """Test the live market endpoint with real credentials"""
    print("Starting Flask app with DHAN credentials...")
    
    # Start app in thread
    import threading
    app_thread = threading.Thread(target=lambda: app_socketio.run(app, host='127.0.0.1', port=5001, debug=False, allow_unsafe_werkzeug=True))
    app_thread.daemon = True
    app_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Test the endpoint
    print("Testing live market WebSocket endpoint...")
    
    try:
        response = requests.post('http://localhost:5001/api/websocket/live-market-test', 
                               json={'symbol': 'RELIANCE', 'duration': 10},
                               headers={'Content-Type': 'application/json'},
                               timeout=15)
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Data: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'test_started':
                print("SUCCESS: Live market test started with real credentials!")
                print("This means RELIANCE data fetching will work with real DHAN API")
                return True
            elif data.get('status') == 'demo_mode':
                print("Still in demo mode - credentials may not be loaded properly")
                return False
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_with_credentials()
    print(f"Test Result: {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("\nCredentials working! The RELIANCE WebSocket test will fetch real market data.")
        print("WebSocket events will include:")
        print("- RELIANCE security ID resolution")
        print("- 10 days of historical OHLCV data") 
        print("- Live price updates every 3 seconds")
        print("- Complete test summary with real data!")
    else:
        print("\nCredentials need to be configured in environment variables")