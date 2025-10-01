#!/usr/bin/env python3
"""
Simple WebSocket test to identify data structure
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from dhanhq import DhanContext, MarketFeed
import time
import json

load_dotenv()

client_id = os.getenv('DHAN_CLIENT_ID')
access_token = os.getenv('DHAN_ACCESS_TOKEN')

print("="*60)
print("SIMPLE WEBSOCKET DATA STRUCTURE TEST")
print("="*60)

# Create context
dhan_context = DhanContext(client_id, access_token)

# Subscribe to just a few instruments
instruments = [
    (MarketFeed.NSE, "2885", MarketFeed.Full),   # RELIANCE
    (MarketFeed.NSE, "11536", MarketFeed.Full),  # TCS
    (MarketFeed.NSE, "1594", MarketFeed.Full),   # INFY
]

print(f"Subscribing to {len(instruments)} instruments...")

# Create MarketFeed
feed = MarketFeed(dhan_context, instruments, version="v2")

# Track data
packet_count = 0
symbol_map = {
    "2885": "RELIANCE",
    "11536": "TCS",
    "1594": "INFY",
}

print("\nListening for data (10 seconds)...")
print("-"*60)

start = time.time()
while time.time() - start < 10:
    try:
        data = feed.get_data()

        if data:
            packet_count += 1

            # Print first 5 packets in detail
            if packet_count <= 5:
                print(f"\nPacket #{packet_count}:")
                print(f"  Type: {type(data)}")

                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
                    print(f"  Full data: {json.dumps(data, indent=2)}")
                elif isinstance(data, (list, tuple)):
                    print(f"  Length: {len(data)}")
                    print(f"  Data: {data}")
                else:
                    print(f"  Data: {data}")

                # Try to identify the symbol
                if isinstance(data, dict):
                    # Look for LTP to identify by price
                    ltp = data.get('LTP', data.get('ltp', 0))
                    print(f"  LTP: {ltp}")

                    # Identify by price range
                    if 2800 < ltp < 3000:
                        print(f"  -> Likely RELIANCE")
                    elif 3500 < ltp < 4000:
                        print(f"  -> Likely TCS")
                    elif 1300 < ltp < 1600:
                        print(f"  -> Likely INFY")
            else:
                # Just count after first 5
                if packet_count % 10 == 0:
                    print(f"Received {packet_count} packets...")

        time.sleep(0.1)

    except Exception as e:
        if "get_data" not in str(e):
            print(f"Error: {e}")

print("\n" + "="*60)
print(f"Total packets received: {packet_count}")
print("="*60)

if packet_count > 0:
    print("\n✅ WebSocket is receiving data!")
    print("Data structure has been logged above.")
else:
    print("\n❌ No data received - check market hours")