#!/usr/bin/env python3
"""
Simple WebSocket test to verify events are being received
"""
import socketio
import requests
import time

sio = socketio.Client()

@sio.event
def connect():
    print("Connected to server")

@sio.event
def live_market_test(data):
    print(f"WEBSOCKET EVENT: {data}")

try:
    print("Connecting to WebSocket...")
    sio.connect('http://localhost:5000')
    time.sleep(1)
    
    print("Triggering test...")
    response = requests.post('http://localhost:5000/api/websocket/live-market-test', 
                           json={'symbol': 'RELIANCE', 'duration': 5})
    print(f"API Response: {response.json()}")
    
    print("Waiting for WebSocket events...")
    time.sleep(8)
    
except Exception as e:
    print(f"Error: {e}")
finally:
    sio.disconnect()