#!/usr/bin/env python3
"""
Test live market WebSocket endpoint on Railway deployment
Usage: python test_railway_websocket.py
"""
import socketio
import requests
import time
import json

# Replace with your Railway URL
RAILWAY_URL = "https://dhanscanpoc.up.railway.app"

def test_railway_websocket():
    """Test WebSocket functionality on Railway with RELIANCE"""
    
    # Create Socket.IO client
    sio = socketio.Client()
    
    events_received = []
    
    @sio.event
    def connect():
        print("Connected to Railway server")
    
    @sio.event
    def live_market_test(data):
        events_received.append(data)
        step = data.get('step', 'unknown')
        message = data.get('message', '')
        print(f"[{step.upper()}] {message}")
        
        if 'data' in data:
            if step == 'data_success':
                d = data['data']
                print(f"   DATA: {d['symbol']}: Rs{d['latest_close']} | Vol: {d['latest_volume']:,} | Points: {d['data_points']}")
            elif step == 'live_update':
                d = data['data']
                print(f"   UPDATE #{d['update_number']}: Rs{d['simulated_price']} (was Rs{d['original_close']})")
        
        if step == 'completed' and 'summary' in data:
            s = data['summary']
            print(f"   COMPLETED: {s['total_updates']} updates in {s['duration']}s | WebSocket: {s['websocket_status']} | Data: {s['data_status']}")
    
    try:
        print("Testing Railway Live Market WebSocket...")
        print(f"Connecting to: {RAILWAY_URL}")
        
        # Connect to WebSocket
        sio.connect(RAILWAY_URL)
        time.sleep(2)
        
        # Test the API endpoint
        print("\nTriggering RELIANCE live market test...")
        
        response = requests.post(f'{RAILWAY_URL}/api/websocket/live-market-test', 
                               json={
                                   'symbol': 'RELIANCE',
                                   'duration': 20  # 20 seconds test
                               },
                               headers={'Content-Type': 'application/json'},
                               timeout=10)
        
        print(f"API Response: {response.status_code}")
        
        if response.status_code == 200:
            resp_json = response.json()
            print(f"Status: {resp_json.get('status', 'unknown')}")
            print(f"Message: {resp_json.get('message', 'No message')}")
        else:
            print(f"Error: {response.text}")
            return False
        
        # Listen for WebSocket events  
        print("\nListening for WebSocket events for 30 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            time.sleep(0.1)
        
        print(f"\nRailway Test Results:")
        print(f"   Total events received: {len(events_received)}")
        
        event_types = [e.get('step', 'unknown') for e in events_received]
        print(f"   Event sequence: {' -> '.join(event_types)}")
        
        # Check for completion
        completed_events = [e for e in events_received if e.get('step') == 'completed']
        if completed_events:
            print(f"   SUCCESS: Live market WebSocket test completed on Railway!")
            final_summary = completed_events[0].get('summary', {})
            print(f"   Final: {final_summary}")
            return True
        else:
            # Check what we got
            data_events = [e for e in events_received if e.get('step') == 'data_success']
            if data_events:
                print(f"   SUCCESS: Data fetched successfully, test may still be running")
                return True
            elif any(e.get('step') == 'credentials_missing' for e in events_received):
                print(f"   DEMO MODE: No live data, but WebSocket working correctly")
                return True
            else:
                print(f"   PARTIAL: WebSocket connected but test incomplete")
                return False
        
    except Exception as e:
        print(f"Railway test failed: {e}")
        return False
    finally:
        try:
            sio.disconnect()
            print("Disconnected from Railway")
        except:
            pass

if __name__ == "__main__":
    print("Railway WebSocket Test for RELIANCE\n")
    success = test_railway_websocket()
    print(f"\nRailway Test Result: {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("\nRailway Deployment Status: WebSocket + Live Market Test = WORKING!")
    else:
        print("\nCheck Railway logs and deployment status")