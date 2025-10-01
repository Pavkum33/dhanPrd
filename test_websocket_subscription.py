#!/usr/bin/env python3
"""
Test WebSocket subscription for Dhan Live Market Data
Simplified test to verify WebSocket connectivity and data reception
"""

import asyncio
import websockets
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import struct
import pandas as pd

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DhanWebSocketTest:
    def __init__(self):
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')

        if not self.client_id or not self.access_token:
            raise ValueError("DHAN credentials not found")

        self.ws_url = f"wss://api-feed.dhan.co?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"
        self.received_count = 0
        self.symbols_data = {}

    async def test_connection(self):
        """Test WebSocket connection and subscription"""

        logger.info("="*80)
        logger.info("DHAN WEBSOCKET SUBSCRIPTION TEST")
        logger.info("="*80)
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Client ID: {self.client_id}")

        try:
            # Load F&O instruments to get proper security IDs
            instruments_df = await self.load_fno_instruments()

            logger.info("\nConnecting to WebSocket...")

            async with websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=20,
                max_size=None
            ) as ws:
                logger.info("✅ Connected successfully!")

                # Try different subscription formats
                await self.test_subscriptions(ws, instruments_df)

                # Listen for data
                logger.info("\nListening for data (30 seconds)...")

                timeout = 30
                start_time = asyncio.get_event_loop().time()

                while (asyncio.get_event_loop().time() - start_time) < timeout:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        await self.handle_message(message)

                    except asyncio.TimeoutError:
                        remaining = timeout - (asyncio.get_event_loop().time() - start_time)
                        if remaining > 0:
                            logger.info(f"Waiting... ({remaining:.0f}s remaining)")
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        break

                logger.info("\nTest completed!")

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

        # Show results
        self.show_results()
        return self.received_count > 0

    async def load_fno_instruments(self):
        """Load F&O instruments from CSV"""

        csv_path = 'api-scrip-master.csv'

        if not os.path.exists(csv_path):
            logger.warning("CSV not found, using hardcoded security IDs")
            return None

        try:
            df = pd.read_csv(csv_path)

            # Find F&O columns
            columns = df.columns.tolist()

            # Filter F&O futures
            if 'SEM_INSTRUMENT_NAME' in columns:
                fno_df = df[
                    (df['SEM_EXM_EXCH_ID'].str.upper().isin(['NSE_FO'])) &
                    (df['SEM_INSTRUMENT_NAME'].str.upper().isin(['FUTSTK', 'FUTIDX']))
                ]
                logger.info(f"Found {len(fno_df)} F&O instruments")
                return fno_df
            else:
                logger.warning("Column format not recognized")
                return None

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None

    async def test_subscriptions(self, ws, instruments_df):
        """Test different subscription formats"""

        # Format 1: Direct subscription with known security IDs
        logger.info("\n1. Testing direct subscription...")

        # Use hardcoded security IDs that we know work
        instruments = [
            {"SecurityId": "35000", "ExchangeSegment": 2, "InstrumentType": 6},  # NIFTY
            {"SecurityId": "35001", "ExchangeSegment": 2, "InstrumentType": 6},  # BANKNIFTY
            {"SecurityId": "45825", "ExchangeSegment": 2, "InstrumentType": 6},  # RELIANCE
            {"SecurityId": "46376", "ExchangeSegment": 2, "InstrumentType": 6},  # TCS
            {"SecurityId": "34817", "ExchangeSegment": 2, "InstrumentType": 6},  # INFY
        ]

        subscribe_packet = {
            "RequestCode": 15,
            "InstrumentCount": len(instruments),
            "InstrumentList": instruments
        }

        await ws.send(json.dumps(subscribe_packet))
        logger.info(f"Sent subscription for {len(instruments)} instruments")

        # Format 2: Try market data request
        await asyncio.sleep(0.5)

        logger.info("\n2. Testing market data request...")

        market_request = {
            "RequestCode": 21,
            "Exchange": 2,  # NSE F&O
            "InstrumentType": 6  # Futures
        }

        await ws.send(json.dumps(market_request))
        logger.info("Sent market data request")

        # Format 3: Try ticker subscription (different format)
        await asyncio.sleep(0.5)

        logger.info("\n3. Testing ticker subscription...")

        ticker_sub = {
            "RequestCode": 11,  # Ticker subscription
            "symbols": ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
        }

        await ws.send(json.dumps(ticker_sub))
        logger.info("Sent ticker subscription")

    async def handle_message(self, message):
        """Handle incoming WebSocket message"""

        self.received_count += 1

        try:
            if isinstance(message, str):
                # JSON message
                data = json.loads(message)
                logger.info(f"[{self.received_count}] JSON: {json.dumps(data, indent=2)}")

                # Extract data if available
                if 'symbol' in data or 'Symbol' in data:
                    symbol = data.get('symbol', data.get('Symbol'))
                    ltp = data.get('LTP', data.get('ltp', 0))
                    volume = data.get('Volume', data.get('volume', 0))

                    self.symbols_data[symbol] = {
                        'ltp': ltp,
                        'volume': volume,
                        'time': datetime.now().isoformat()
                    }

                    logger.info(f"LIVE: {symbol} @ {ltp} (Vol: {volume})")

            elif isinstance(message, bytes):
                # Binary message
                logger.info(f"[{self.received_count}] Binary: {len(message)} bytes")

                # Try to decode binary format
                if len(message) >= 8:
                    # Try to unpack as different formats
                    try:
                        # Format 1: Message type + data
                        msg_type = struct.unpack('!H', message[:2])[0]
                        logger.info(f"  Message type: {msg_type}")

                        # Check if it's ticker data (common format)
                        if msg_type in [1, 2, 3, 4, 5]:  # Different tick types
                            self.parse_tick_data(message)

                    except Exception as e:
                        logger.debug(f"  Binary decode error: {e}")

        except json.JSONDecodeError:
            logger.info(f"[{self.received_count}] Raw: {message[:100] if len(str(message)) > 100 else message}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    def parse_tick_data(self, data):
        """Parse binary tick data"""

        try:
            # Common tick data format
            # 2 bytes: packet type
            # 4 bytes: security id
            # 4 bytes: LTP (as integer, divide by 100)
            # 4 bytes: volume

            if len(data) >= 14:
                packet_type = struct.unpack('!H', data[0:2])[0]
                security_id = struct.unpack('!I', data[2:6])[0]
                ltp_int = struct.unpack('!I', data[6:10])[0]
                volume = struct.unpack('!I', data[10:14])[0]

                ltp = ltp_int / 100.0  # Convert to decimal

                logger.info(f"  TICK: SecurityID={security_id}, LTP={ltp}, Vol={volume}")

                self.symbols_data[f"ID_{security_id}"] = {
                    'ltp': ltp,
                    'volume': volume,
                    'time': datetime.now().isoformat()
                }

        except Exception as e:
            logger.debug(f"  Tick parse error: {e}")

    def show_results(self):
        """Show test results"""

        logger.info("\n" + "="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)

        logger.info(f"Messages received: {self.received_count}")
        logger.info(f"Symbols with data: {len(self.symbols_data)}")

        if self.symbols_data:
            logger.info("\nReceived data:")
            for symbol, data in list(self.symbols_data.items())[:10]:
                logger.info(f"  {symbol}: LTP={data['ltp']}, Vol={data['volume']}")

        # Check market status
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        market_open = (hour == 9 and minute >= 15) or (9 < hour < 15) or (hour == 15 and minute < 30)

        logger.info(f"\nMarket Status: {'OPEN' if market_open else 'CLOSED'}")

        if self.received_count > 0:
            logger.info("\n✅ WebSocket is working and receiving data!")
        elif market_open:
            logger.info("\n⚠️ WebSocket connected but no data received")
            logger.info("   Possible issues:")
            logger.info("   - Incorrect security IDs")
            logger.info("   - Subscription format needs adjustment")
            logger.info("   - Rate limiting")
        else:
            logger.info("\n⚠️ Market is closed - no live data available")
            logger.info("   Test during market hours for live data")

async def main():
    """Main test function"""

    tester = DhanWebSocketTest()
    success = await tester.test_connection()

    if success:
        logger.info("\n✅ WebSocket subscription test SUCCESSFUL!")
    else:
        logger.info("\n❌ WebSocket subscription test needs attention")

if __name__ == "__main__":
    asyncio.run(main())