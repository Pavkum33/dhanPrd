#!/usr/bin/env python3
"""
Test WebSocket functionality of the dashboard
"""
import socketio
import time

# Create a Socket.IO client
sio = socketio.Client()

def test_websocket_connection():
    """Test basic WebSocket connection and events"""
    events_received = []
    
    # Set up event handlers
    @sio.event
    def connect():
        events_received.append("connected")
        print("Connected event received")
    
    @sio.event
    def connected(data):
        events_received.append("server_connected")
        print(f"Server connected event: {data}")
    
    @sio.event
    def stats(data):
        events_received.append("stats")
        print(f"Stats event: {data}")
    
    @sio.event
    def scanner_data(data):
        events_received.append("scanner_data")
        print(f"Scanner data event: {len(data)} items" if isinstance(data, list) else f"Scanner data: {data}")
        
    @sio.event
    def historical_progress(data):
        events_received.append("historical_progress")
        print(f"Historical progress: {data}")
    
    try:
        print("Testing WebSocket connection...")
        
        # Connect to the server
        sio.connect('http://localhost:5000')
        print("WebSocket connected successfully!")
        
        # Wait for initial events
        time.sleep(2)
        
        # Test emitting events
        print("\nTesting WebSocket events...")
        sio.emit('refresh_data')
        time.sleep(1)
        
        sio.emit('start_scanner')
        time.sleep(1)
        
        sio.emit('stop_scanner')
        time.sleep(1)
        
        # Summary
        print(f"\nEvents received: {events_received}")
        print("WebSocket functionality test completed!")
        
        return True
        
    except Exception as e:
        print(f"WebSocket test failed: {e}")
        return False
    finally:
        try:
            sio.disconnect()
            print("WebSocket disconnected")
        except:
            pass

if __name__ == "__main__":
    test_websocket_connection()