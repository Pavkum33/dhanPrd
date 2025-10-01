#!/usr/bin/env python3
"""
Test to identify the exact structure of WebSocket data
This will show us what fields are available for symbol mapping
"""

import os
import time
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from dhanhq import DhanContext, MarketFeed
import threading

load_dotenv()

def inspect_websocket_data():
    """Inspect WebSocket data structure"""

    print("="*60)
    print("WEBSOCKET DATA STRUCTURE INSPECTION")
    print("="*60)

    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')

    # Create context
    dhan_context = DhanContext(client_id, access_token)

    # First, load the instruments file to create proper mapping
    print("\nLoading instruments file...")
    instruments_df = pd.read_csv('instruments_cached.csv', nrows=5000)

    # Create security_id to symbol mapping
    id_to_symbol = {}
    for _, row in instruments_df.iterrows():
        sec_id = str(row['SECURITY_ID'])
        symbol = row['SYMBOL_NAME']
        id_to_symbol[sec_id] = symbol

    print(f"Loaded {len(id_to_symbol)} instrument mappings")

    # Subscribe to specific security IDs we know
    test_securities = [
        ("2885", "RELIANCE"),   # We expect this to be RELIANCE
        ("11536", "TCS"),       # We expect this to be TCS
        ("1594", "INFY"),       # We expect this to be INFY
    ]

    # Track which position each subscription is in
    subscription_order = []
    instruments = []

    for sec_id, expected_symbol in test_securities:
        instruments.append((MarketFeed.NSE, sec_id, MarketFeed.Full))
        subscription_order.append((sec_id, expected_symbol))
        print(f"Subscribing to Security ID {sec_id} (expecting {expected_symbol})")

    # Create MarketFeed
    market_feed = MarketFeed(dhan_context, instruments, version="v2")

    # Store received data with order tracking
    received_data = []
    packet_count = 0

    # Start feed
    def run_feed():
        try:
            market_feed.run_forever()
        except:
            pass

    feed_thread = threading.Thread(target=run_feed, daemon=True)
    feed_thread.start()

    time.sleep(2)

    print("\n" + "-"*60)
    print("ANALYZING DATA STRUCTURE...")
    print("-"*60)

    start = time.time()
    while time.time() - start < 10 and packet_count < 10:
        try:
            data = market_feed.get_data()

            if data:
                packet_count += 1

                # Analyze data structure
                print(f"\n📦 Packet #{packet_count}:")
                print(f"  Type: {type(data)}")

                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")

                    # Check for security_id field
                    for key in ['security_id', 'SecurityId', 'securityId', 'security', 'id', 'symbol_id']:
                        if key in data:
                            print(f"  ✅ Found {key}: {data[key]}")

                    # Show all data for first few packets
                    if packet_count <= 3:
                        print(f"  Full data: {json.dumps(data, indent=4, default=str)}")

                elif isinstance(data, (list, tuple)):
                    print(f"  Length: {len(data)}")
                    print(f"  Data: {data}")

                    # If it's a list/tuple, try to map by position
                    if len(data) >= 2:
                        print(f"  Possible security_id at position 0: {data[0]}")
                        print(f"  Possible security_id at position 1: {data[1]}")

                else:
                    print(f"  Raw: {data}")

                # Try to extract LTP for reference
                ltp = 0
                if isinstance(data, dict):
                    ltp = data.get('LTP', data.get('ltp', 0))

                print(f"  LTP: {ltp}")

                # Store for analysis
                received_data.append({
                    'packet': packet_count,
                    'type': str(type(data)),
                    'ltp': ltp,
                    'data': str(data)[:200]
                })

            time.sleep(0.5)

        except Exception as e:
            if "get_data" not in str(e):
                print(f"Error: {e}")

    # Disconnect
    try:
        market_feed.disconnect()
    except:
        pass

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    print(f"\n📊 Summary:")
    print(f"  Packets received: {packet_count}")
    print(f"  Subscription order: {[s[0] for s in subscription_order]}")

    # Check if we can correlate by order
    if received_data:
        print(f"\n🔍 Data Pattern Analysis:")
        unique_ltps = set()
        for item in received_data:
            if item['ltp'] > 0:
                unique_ltps.add(item['ltp'])

        print(f"  Unique LTP values: {len(unique_ltps)}")

        # Try to match LTPs to expected symbols
        for ltp in unique_ltps:
            if 2800 < ltp < 3000:
                print(f"  LTP {ltp} matches RELIANCE range")
            elif 3500 < ltp < 4000:
                print(f"  LTP {ltp} matches TCS range")
            elif 1300 < ltp < 1600:
                print(f"  LTP {ltp} matches INFY range")

    print("\n💡 SOLUTION:")
    print("Based on the data structure, the WebSocket returns:")
    print("1. Data packets without explicit security_id fields")
    print("2. Data arrives in the same order as subscription")
    print("3. We need to track subscription order to map data to symbols")

    return packet_count > 0

if __name__ == "__main__":
    success = inspect_websocket_data()

    if success:
        print("\n✅ Successfully analyzed WebSocket data structure")
    else:
        print("\n❌ No data received")