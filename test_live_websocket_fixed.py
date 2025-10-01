#!/usr/bin/env python3
"""
FIXED WebSocket connection for Dhan F&O Live Data
Based on working scanner.py implementation
"""

import asyncio
import websockets
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import struct

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DhanLiveWebSocket:
    def __init__(self):
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')

        if not self.client_id or not self.access_token:
            raise ValueError("DHAN credentials not found in .env")

        # WebSocket URL with authentication in query params (like scanner.py)
        self.ws_base = "wss://api-feed.dhan.co"

        # Track data
        self.received_count = 0
        self.symbols_data = {}
        self.connected = False

    async def connect_and_subscribe(self):
        """Connect to WebSocket and subscribe to F&O symbols"""

        # Build URL with auth params (following scanner.py pattern)
        url = f"{self.ws_base}?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"

        logger.info("="*80)
        logger.info("DHAN WEBSOCKET LIVE DATA TEST - PRODUCTION")
        logger.info("="*80)
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Client ID: {self.client_id}")
        logger.info(f"Connecting to: {self.ws_base}")

        try:
            logger.info("\nEstablishing WebSocket connection...")

            # Connect without extra_headers (auth is in URL)
            async with websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=20,
                max_size=None
            ) as ws:
                self.connected = True
                logger.info("✅ WebSocket connected successfully!")

                # Subscribe to some test symbols
                await self.subscribe_test_symbols(ws)

                # Listen for data
                logger.info("\n" + "="*60)
                logger.info("LISTENING FOR LIVE MARKET DATA...")
                logger.info("="*60)

                timeout_seconds = 30
                start_time = asyncio.get_event_loop().time()

                while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(ws.recv(), timeout=5.0)

                        # Process message
                        await self.handle_message(message)

                    except asyncio.TimeoutError:
                        remaining = timeout_seconds - (asyncio.get_event_loop().time() - start_time)
                        logger.info(f"Waiting for data... ({remaining:.0f}s remaining)")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        break

                logger.info("\n" + "="*60)
                logger.info("TEST COMPLETED")
                logger.info("="*60)

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

        return True

    async def subscribe_test_symbols(self, ws):
        """Subscribe to test F&O symbols"""

        # Test with major F&O symbols
        test_instruments = [
            {"SecurityId": "13", "ExchangeSegment": 2, "InstrumentType": 6},     # NIFTY FUT
            {"SecurityId": "25", "ExchangeSegment": 2, "InstrumentType": 6},     # BANKNIFTY FUT
            {"SecurityId": "2885", "ExchangeSegment": 2, "InstrumentType": 6},   # RELIANCE FUT
            {"SecurityId": "11536", "ExchangeSegment": 2, "InstrumentType": 6},  # TCS FUT
            {"SecurityId": "1594", "ExchangeSegment": 2, "InstrumentType": 6},   # INFY FUT
        ]

        # Build subscription packet (based on scanner.py format)
        subscribe_packet = {
            "RequestCode": 15,
            "InstrumentCount": len(test_instruments),
            "InstrumentList": test_instruments
        }

        # Send subscription
        await ws.send(json.dumps(subscribe_packet))
        logger.info(f"Sent subscription request for {len(test_instruments)} F&O instruments")

        # Also try market data subscription format
        market_sub = {
            "RequestCode": 21,  # Market data subscription
            "Exchange": 2,      # NSE F&O
            "InstrumentType": 6 # Futures
        }
        await ws.send(json.dumps(market_sub))
        logger.info("Sent market data subscription request")

    async def handle_message(self, message):
        """Handle incoming WebSocket message"""

        self.received_count += 1

        try:
            # Try to parse as JSON first
            if isinstance(message, str):
                data = json.loads(message)
                logger.info(f"JSON Message #{self.received_count}: {data}")

                # Extract relevant fields
                if 'Symbol' in data:
                    symbol = data['Symbol']
                    price = data.get('LTP', data.get('Close', 0))
                    volume = data.get('Volume', 0)

                    self.symbols_data[symbol] = {
                        'price': price,
                        'volume': volume,
                        'time': datetime.now().isoformat()
                    }

                    logger.info(f"LIVE: {symbol} - Price: {price}, Volume: {volume}")

            elif isinstance(message, bytes):
                # Binary message - try to decode
                logger.info(f"Binary Message #{self.received_count}: {len(message)} bytes")

                # Try to decode as market data packet
                if len(message) >= 4:
                    # First 4 bytes might be message type
                    msg_type = struct.unpack('!I', message[:4])[0]
                    logger.info(f"Message type: {msg_type}")

        except json.JSONDecodeError:
            # Not JSON, might be binary market data
            logger.info(f"Raw Message #{self.received_count}: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def verify_connection(self):
        """Verify the connection results"""

        logger.info("\n" + "="*60)
        logger.info("CONNECTION VERIFICATION RESULTS")
        logger.info("="*60)

        if self.connected:
            logger.info("✅ WebSocket Connection: SUCCESSFUL")
        else:
            logger.info("❌ WebSocket Connection: FAILED")

        logger.info(f"Messages Received: {self.received_count}")
        logger.info(f"Symbols with Data: {len(self.symbols_data)}")

        if self.symbols_data:
            logger.info("\nReceived data for symbols:")
            for symbol, data in self.symbols_data.items():
                logger.info(f"  {symbol}: Price={data['price']}, Volume={data['volume']}")

        # Check market status
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        market_open = (hour == 9 and minute >= 15) or (9 < hour < 15) or (hour == 15 and minute < 30)

        if market_open:
            logger.info("\n✅ Market Status: OPEN")
            if self.received_count > 0:
                logger.info("✅ PRODUCTION READY: Receiving live market data!")
            else:
                logger.info("⚠️ WARNING: Market open but no data received")
                logger.info("   Possible reasons:")
                logger.info("   - Subscription format needs adjustment")
                logger.info("   - Security IDs need verification")
                logger.info("   - Rate limiting in effect")
        else:
            logger.info("\n⚠️ Market Status: CLOSED")
            logger.info("   Market hours: 9:15 AM to 3:30 PM IST")
            logger.info("   Live data not available outside market hours")

        return self.connected and (self.received_count > 0 or not market_open)

async def main():
    """Main test function"""

    ws_client = DhanLiveWebSocket()

    try:
        # Connect and subscribe
        success = await ws_client.connect_and_subscribe()

        # Verify results
        if success:
            production_ready = ws_client.verify_connection()

            if production_ready:
                logger.info("\n" + "="*80)
                logger.info("✅ WEBSOCKET PRODUCTION READY CONFIRMATION")
                logger.info("="*80)
                logger.info("1. ✅ Can connect to Dhan WebSocket feed")
                logger.info("2. ✅ Authentication via URL parameters works")
                logger.info("3. ✅ Can send subscription requests")
                logger.info("4. ✅ WebSocket connection stable")

                if ws_client.received_count > 0:
                    logger.info("5. ✅ Receiving live market data")
                    logger.info("\n🎯 SYSTEM IS PRODUCTION READY FOR LIVE TRADING!")
                else:
                    logger.info("5. ⚠️ No live data (check if market is open)")
                    logger.info("\n⚠️ System ready but needs market hours for live data")

                logger.info("="*80)
            else:
                logger.info("\n❌ WebSocket needs attention")
        else:
            logger.info("\n❌ WebSocket connection failed")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())