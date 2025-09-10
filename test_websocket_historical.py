#!/usr/bin/env python3
"""
Test WebSocket functionality during historical fetch
"""
import socketio
import time
import requests
import threading

# Create a Socket.IO client
sio = socketio.Client()

def test_historical_websocket():
    """Test WebSocket events during historical fetch"""
    events_received = []
    
    # Set up event handlers for historical progress
    @sio.event
    def connect():
        events_received.append("connected")
        print("Connected to server")
    
    @sio.event
    def connected(data):
        events_received.append("server_connected")
        print(f"Server connected: {data}")
    
    @sio.event
    def historical_progress(data):
        events_received.append("historical_progress")
        step = data.get('step', 'unknown')
        message = data.get('message', 'No message')
        current = data.get('current', 0)
        total = data.get('total', 0)
        print(f"Historical Progress [{step}]: {message} ({current}/{total})")
        
    @sio.event
    def breakout_results(data):
        events_received.append("breakout_results")
        print(f"Breakout results: {data}")
    
    @sio.event
    def cpr_results(data):
        events_received.append("cpr_results") 
        print(f"CPR results: {data}")
        
    @sio.event
    def pivot_results(data):
        events_received.append("pivot_results")
        print(f"Pivot results: {data}")
    
    try:
        print("Testing WebSocket during historical fetch...")
        
        # Connect to the server
        sio.connect('http://localhost:5000')
        print("WebSocket connected!")
        
        # Wait a moment to ensure connection is stable
        time.sleep(1)
        
        # Trigger historical fetch via API
        print("\nTriggering historical fetch...")
        response = requests.post('http://localhost:5000/api/historical/fetch', 
                               json={'fetch_days': 10},
                               headers={'Content-Type': 'application/json'})
        print(f"API Response: {response.json()}")
        
        # Listen for progress events for 15 seconds
        print("\nListening for WebSocket events for 15 seconds...")
        start_time = time.time()
        while time.time() - start_time < 15:
            time.sleep(0.1)  # Small sleep to prevent busy waiting
            
        print(f"\nTotal events received: {len(events_received)}")
        print(f"Event types: {list(set(events_received))}")
        
        # Test multi-scan endpoints that should work
        print("\nTesting multi-scan WebSocket integration...")
        
        # Test narrow CPR endpoint
        try:
            cpr_response = requests.get('http://localhost:5000/api/levels/narrow-cpr-railway')
            print(f"Narrow CPR endpoint: {cpr_response.status_code}")
        except Exception as e:
            print(f"Narrow CPR endpoint failed: {e}")
            
        time.sleep(2)  # Wait for potential WebSocket events
        
        print("WebSocket historical test completed!")
        return len(events_received) > 0
        
    except Exception as e:
        print(f"WebSocket historical test failed: {e}")
        return False
    finally:
        try:
            sio.disconnect()
            print("WebSocket disconnected")
        except:
            pass

if __name__ == "__main__":
    success = test_historical_websocket()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")