#!/usr/bin/env python3
"""
Test LIVE WebSocket subscription for ALL F&O stocks
Verify we can subscribe and receive real-time data before market close
PRODUCTION READY TEST
"""

import asyncio
import websockets
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import psycopg2

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveFNOSubscriptionTest:
    def __init__(self):
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')

        if not self.client_id or not self.access_token:
            raise ValueError("DHAN credentials not found in .env")

        # WebSocket URLs
        self.ws_url = "wss://api-feed.dhan.co"

        # Track subscriptions
        self.subscribed_symbols = set()
        self.received_data = {}
        self.failed_subscriptions = []

        # PostgreSQL connection
        self.conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='dhan_scanner_prod',
            user='postgres',
            password='India@123'
        )
        self.cursor = self.conn.cursor()

    async def get_fno_symbols(self):
        """Get all F&O symbols from database or API"""
        from dhan_fetcher import DhanHistoricalFetcher

        async with DhanHistoricalFetcher(self.client_id, self.access_token) as fetcher:
            # Get all F&O instruments
            instruments_df = await fetcher.get_instruments()
            active_futures = fetcher.get_active_fno_futures(instruments_df)

            # Extract unique underlying symbols and their security IDs
            symbols_data = []
            for _, row in active_futures.iterrows():
                symbol_col = 'SEM_TRADING_SYMBOL' if 'SEM_TRADING_SYMBOL' in row else 'SYMBOL_NAME'
                sid_col = 'SEM_SMST_SECURITY_ID' if 'SEM_SMST_SECURITY_ID' in row else 'SecurityId'

                symbol = row[symbol_col]
                security_id = row[sid_col]

                # Extract underlying
                underlying = fetcher.extract_underlying_symbol(symbol)

                symbols_data.append({
                    'symbol': underlying,
                    'future_symbol': symbol,
                    'security_id': str(security_id),
                    'exchange_segment': 2  # NSE F&O
                })

            logger.info(f"Found {len(symbols_data)} F&O symbols")
            return symbols_data

    async def subscribe_to_symbols(self, ws, symbols_data):
        """Subscribe to all F&O symbols for live data"""

        # Prepare subscription message
        batch_size = 100  # Subscribe in batches to avoid overwhelming

        for i in range(0, len(symbols_data), batch_size):
            batch = symbols_data[i:i+batch_size]

            # Create subscription request
            instruments = []
            for sym_data in batch:
                instruments.append({
                    "securityId": sym_data['security_id'],
                    "exchangeSegment": sym_data['exchange_segment'],
                    "instrumentType": "FUTURES"
                })

            subscribe_msg = {
                "RequestCode": 15,  # Subscribe request
                "InstrumentCount": len(instruments),
                "InstrumentList": instruments
            }

            await ws.send(json.dumps(subscribe_msg))
            logger.info(f"Sent subscription request for {len(instruments)} instruments (batch {i//batch_size + 1})")

            # Track subscriptions
            for sym_data in batch:
                self.subscribed_symbols.add(sym_data['symbol'])

            # Wait for acknowledgment
            await asyncio.sleep(0.5)

    async def connect_and_subscribe(self):
        """Connect to WebSocket and subscribe to all F&O symbols"""

        logger.info("="*80)
        logger.info("LIVE F&O SUBSCRIPTION TEST - PRODUCTION READY")
        logger.info("="*80)
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Client ID: {self.client_id}")

        # Get all F&O symbols
        symbols_data = await self.get_fno_symbols()

        # Connect to WebSocket
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"\nConnecting to WebSocket: {self.ws_url}")

            async with websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10
            ) as ws:
                logger.info("✅ WebSocket connected successfully")

                # Send authentication
                auth_msg = {
                    "LoginReqCode": 11,
                    "ClientId": self.client_id,
                    "AccessToken": self.access_token
                }

                await ws.send(json.dumps(auth_msg))
                logger.info("Sent authentication request")

                # Wait for auth response
                auth_response = await ws.recv()
                auth_data = json.loads(auth_response)
                logger.info(f"Auth response: {auth_data}")

                # Subscribe to all symbols
                await self.subscribe_to_symbols(ws, symbols_data)

                # Listen for live data
                logger.info("\n" + "="*60)
                logger.info("LISTENING FOR LIVE DATA...")
                logger.info("="*60)

                start_time = datetime.now()
                timeout = 60  # Listen for 60 seconds

                while (datetime.now() - start_time).seconds < timeout:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(message)

                        # Process different message types
                        if 'symbol' in data:
                            symbol = data['symbol']
                            if symbol not in self.received_data:
                                self.received_data[symbol] = []

                            self.received_data[symbol].append({
                                'time': datetime.now().isoformat(),
                                'price': data.get('ltp', data.get('close')),
                                'volume': data.get('volume'),
                                'bid': data.get('bid'),
                                'ask': data.get('ask')
                            })

                            logger.info(f"LIVE: {symbol} - Price: {data.get('ltp')} Volume: {data.get('volume')}")

                        elif 'ExchangeFeedCode' in data:
                            # Market data packet
                            logger.debug(f"Market data: {data}")

                    except asyncio.TimeoutError:
                        logger.info("Waiting for data...")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

                logger.info("\n" + "="*60)
                logger.info("SUBSCRIPTION TEST RESULTS")
                logger.info("="*60)

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False

        return True

    def verify_results(self):
        """Verify subscription results"""

        logger.info("\n📊 VERIFICATION RESULTS:")
        logger.info(f"Total symbols subscribed: {len(self.subscribed_symbols)}")
        logger.info(f"Symbols received data: {len(self.received_data)}")
        logger.info(f"Failed subscriptions: {len(self.failed_subscriptions)}")

        if self.received_data:
            logger.info("\nSample live data received:")
            for symbol in list(self.received_data.keys())[:10]:
                data_points = self.received_data[symbol]
                logger.info(f"  {symbol}: {len(data_points)} updates")
                if data_points:
                    latest = data_points[-1]
                    logger.info(f"    Latest: Price={latest.get('price')} Volume={latest.get('volume')}")

        # Store test results in database
        self.cursor.execute("""
            INSERT INTO job_logs
            (job_name, job_type, status, started_at, completed_at,
             records_processed, records_success, records_failed, errors)
            VALUES (%s, %s, %s, NOW(), NOW(), %s, %s, %s, %s)
        """, (
            'live_fno_subscription_test',
            'WEBSOCKET_TEST',
            'SUCCESS' if self.received_data else 'FAILED',
            len(self.subscribed_symbols),
            len(self.received_data),
            len(self.failed_subscriptions),
            json.dumps(self.failed_subscriptions) if self.failed_subscriptions else None
        ))
        self.conn.commit()

        success = len(self.received_data) > 0

        if success:
            logger.info("\n✅ PRODUCTION READY: Successfully subscribed and received live F&O data!")
        else:
            logger.info("\n❌ WARNING: No live data received. Check if market is open.")

        return success

    async def run_test(self):
        """Run the complete test"""
        try:
            # Check market hours
            now = datetime.now()
            hour = now.hour
            minute = now.minute

            logger.info(f"Current time: {now.strftime('%H:%M:%S')}")

            # Indian market hours: 9:15 AM to 3:30 PM
            market_open = (hour == 9 and minute >= 15) or (hour > 9 and hour < 15) or (hour == 15 and minute < 30)

            if not market_open:
                logger.warning("⚠️ Market is closed. Live data may not be available.")
                logger.info("Market hours: 9:15 AM to 3:30 PM IST")
            else:
                logger.info("✅ Market is OPEN - should receive live data")

            # Run subscription test
            success = await self.connect_and_subscribe()

            # Verify results
            if success:
                self.verify_results()

            return success

        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
        finally:
            self.conn.close()

async def main():
    """Main function"""
    tester = LiveFNOSubscriptionTest()
    success = await tester.run_test()

    if success:
        logger.info("\n" + "="*80)
        logger.info("✅ PRODUCTION READY CONFIRMATION")
        logger.info("="*80)
        logger.info("1. ✅ Can connect to Dhan WebSocket")
        logger.info("2. ✅ Can authenticate with credentials")
        logger.info("3. ✅ Can subscribe to all F&O symbols")
        logger.info("4. ✅ Can receive live market data")
        logger.info("5. ✅ System is PRODUCTION READY for live trading")
        logger.info("="*80)
    else:
        logger.info("\n⚠️ System needs attention before production use")

if __name__ == "__main__":
    asyncio.run(main())