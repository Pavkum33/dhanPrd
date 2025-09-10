#!/usr/bin/env python3
"""
Test live market WebSocket endpoint locally
"""
import socketio
import requests
import time
import json

def test_live_market_websocket_endpoint():
    """Test the new live market WebSocket endpoint"""
    
    # Create Socket.IO client
    sio = socketio.Client()
    
    events_received = []
    
    @sio.event
    def connect():
        print("Connected to server")
    
    @sio.event
    def live_market_test(data):
        events_received.append(data)
        step = data.get('step', 'unknown')
        message = data.get('message', '')
        print(f"[{step.upper()}] {message}")
        
        if 'data' in data:
            print(f"   Data: {json.dumps(data['data'], indent=2)}")
        
        if step == 'completed':
            print(f"   Summary: {json.dumps(data['summary'], indent=2)}")
    
    try:
        print("Testing Live Market WebSocket Endpoint...")
        
        # Connect to WebSocket
        sio.connect('http://localhost:5000')
        time.sleep(1)
        
        # Test the API endpoint
        print("\nTriggering live market test for RELIANCE...")
        
        response = requests.post('http://localhost:5000/api/websocket/live-market-test', 
                               json={
                                   'symbol': 'RELIANCE',
                                   'duration': 15  # 15 seconds test
                               },
                               headers={'Content-Type': 'application/json'})
        
        print(f"API Response: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Listen for WebSocket events
        print("\nListening for WebSocket events for 20 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 20:
            time.sleep(0.1)
        
        print(f"\nTest Results:")
        print(f"   Total events received: {len(events_received)}")
        print(f"   Event types: {[e.get('step', 'unknown') for e in events_received]}")
        
        # Show summary
        completed_events = [e for e in events_received if e.get('step') == 'completed']
        if completed_events:
            print(f"   Test completed successfully!")
            print(f"   Final summary: {completed_events[0].get('summary', {})}")
        else:
            print(f"   Test may still be running or had issues")
        
        return len(events_received) > 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        try:
            sio.disconnect()
            print("Disconnected")
        except:
            pass

if __name__ == "__main__":
    print("Testing Live Market WebSocket Endpoint\n")
    success = test_live_market_websocket_endpoint()
    print(f"\nTest Result: {'PASSED' if success else 'FAILED'}")