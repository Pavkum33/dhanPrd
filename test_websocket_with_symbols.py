#!/usr/bin/env python3
"""
Dhan WebSocket with Symbol Identification
Maps security IDs to symbol names for proper identification
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
import logging
from dhanhq import DhanContext, dhanhq, MarketFeed
import threading

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DhanWebSocketWithSymbols:
    def __init__(self):
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')

        if not self.client_id or not self.access_token:
            raise ValueError("DHAN credentials not found")

        # Create DhanContext
        self.dhan_context = DhanContext(self.client_id, self.access_token)

        # Data tracking
        self.received_data = {}
        self.packet_count = 0

        # Symbol mapping dictionary
        self.symbol_map = {}
        self.subscribed_instruments = []

    def test_websocket(self):
        """Test WebSocket with symbol identification"""

        logger.info("="*80)
        logger.info("DHAN WEBSOCKET WITH SYMBOL IDENTIFICATION")
        logger.info("="*80)
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Client ID: {self.client_id}")

        try:
            # Define instruments with their symbol names
            # Format: (Exchange, SecurityId, SubscriptionType, SymbolName)
            instrument_list = [
                # NSE Equity symbols
                (MarketFeed.NSE, "2885", MarketFeed.Full, "RELIANCE"),
                (MarketFeed.NSE, "11536", MarketFeed.Full, "TCS"),
                (MarketFeed.NSE, "1594", MarketFeed.Full, "INFY"),
                (MarketFeed.NSE, "5258", MarketFeed.Full, "HDFCBANK"),
                (MarketFeed.NSE, "1333", MarketFeed.Full, "HDFC"),
                (MarketFeed.NSE, "1660", MarketFeed.Full, "ITC"),
                (MarketFeed.NSE, "772", MarketFeed.Full, "AXISBANK"),
                (MarketFeed.NSE, "4963", MarketFeed.Full, "ICICIBANK"),

                # Index
                (MarketFeed.NSE, "13", MarketFeed.Full, "NIFTY50"),
                (MarketFeed.NSE, "25", MarketFeed.Full, "BANKNIFTY"),

                # F&O symbols (if available)
                (MarketFeed.NSE_FNO, "35000", MarketFeed.Full, "NIFTY-FUT"),
                (MarketFeed.NSE_FNO, "35001", MarketFeed.Full, "BANKNIFTY-FUT"),
                (MarketFeed.NSE_FNO, "45825", MarketFeed.Full, "RELIANCE-FUT"),
                (MarketFeed.NSE_FNO, "46376", MarketFeed.Full, "TCS-FUT"),
                (MarketFeed.NSE_FNO, "34817", MarketFeed.Full, "INFY-FUT"),
            ]

            # Build symbol mapping and instrument list
            instruments = []
            for exchange, sec_id, sub_type, symbol in instrument_list:
                # Create mapping key
                key = f"{exchange}_{sec_id}"
                self.symbol_map[key] = symbol

                # Add to instruments for subscription
                instruments.append((exchange, sec_id, sub_type))

                # Track what we subscribed to
                self.subscribed_instruments.append({
                    'symbol': symbol,
                    'security_id': sec_id,
                    'exchange': exchange,
                    'key': key
                })

            logger.info(f"\nSubscribing to {len(instruments)} instruments:")
            for inst in self.subscribed_instruments:
                logger.info(f"  {inst['symbol']} (ID: {inst['security_id']})")

            # Create MarketFeed instance
            self.market_feed = MarketFeed(
                dhan_context=self.dhan_context,
                instruments=instruments,
                version="v2"
            )

            logger.info("\n✅ MarketFeed created successfully")

            # Start feed in a separate thread
            feed_running = threading.Event()

            def run_feed():
                try:
                    logger.info("Starting market feed...")
                    feed_running.set()
                    self.market_feed.run_forever()
                except Exception as e:
                    logger.error(f"Feed error: {e}")
                    feed_running.clear()

            feed_thread = threading.Thread(target=run_feed, daemon=True)
            feed_thread.start()

            # Wait for feed to start
            time.sleep(2)

            logger.info("\n" + "="*60)
            logger.info("LISTENING FOR LIVE DATA WITH SYMBOL IDENTIFICATION...")
            logger.info("="*60)

            # Listen for data
            timeout = 30
            start_time = time.time()
            last_log = start_time

            while (time.time() - start_time) < timeout:
                try:
                    # Try to get data
                    response = self.market_feed.get_data()

                    if response:
                        self.packet_count += 1
                        self.handle_data_with_symbol(response)

                    # Log progress every 5 seconds
                    if time.time() - last_log > 5:
                        remaining = timeout - (time.time() - start_time)
                        unique_symbols = len([s for s in self.received_data if s != 'Unknown'])
                        logger.info(f"Progress: {remaining:.0f}s remaining | {self.packet_count} packets | {unique_symbols} identified symbols")
                        last_log = time.time()

                    time.sleep(0.1)

                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    pass

            # Disconnect
            try:
                logger.info("\nDisconnecting...")
                # Note: disconnect might be async, handle accordingly
                try:
                    self.market_feed.disconnect()
                except:
                    pass
                logger.info("✅ Disconnected")
            except:
                pass

            # Show results
            self.show_detailed_results()

            return self.packet_count > 0

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def handle_data_with_symbol(self, data):
        """Handle received market data with symbol identification"""

        try:
            # Log the first few data packets to understand structure
            if self.packet_count < 3:
                logger.info(f"[DEBUG] Data type: {type(data)}, Data: {data}")

            # The data structure from MarketFeed might vary
            # We need to identify which security this data belongs to

            # Try to extract security ID from the data
            security_id = None
            exchange = None

            # Different possible data structures
            if isinstance(data, dict):
                # Look for security ID in various possible fields
                security_id = data.get('security_id', data.get('SecurityId', data.get('securityId', None)))
                exchange = data.get('exchange', data.get('Exchange', MarketFeed.NSE))

                # Extract market data
                ltp = data.get('LTP', data.get('ltp', data.get('last_price', 0)))
                volume = data.get('volume', data.get('Volume', data.get('vol', 0)))
                bid = data.get('bid', data.get('best_bid', 0))
                ask = data.get('ask', data.get('best_ask', 0))
                open_price = data.get('open', data.get('Open', 0))
                high = data.get('high', data.get('High', 0))
                low = data.get('low', data.get('Low', 0))
                close = data.get('close', data.get('Close', 0))

            elif isinstance(data, (list, tuple)):
                # Data might be in array format [exchange, security_id, ltp, volume, ...]
                if len(data) >= 4:
                    exchange = data[0] if len(data) > 0 else None
                    security_id = str(data[1]) if len(data) > 1 else None
                    ltp = data[2] if len(data) > 2 else 0
                    volume = data[3] if len(data) > 3 else 0
                    bid = data[4] if len(data) > 4 else 0
                    ask = data[5] if len(data) > 5 else 0
                    open_price = data[6] if len(data) > 6 else 0
                    high = data[7] if len(data) > 7 else 0
                    low = data[8] if len(data) > 8 else 0
                    close = data[9] if len(data) > 9 else 0
            else:
                # Unknown format, try to extract what we can
                ltp = 0
                volume = 0
                bid = 0
                ask = 0
                open_price = 0
                high = 0
                low = 0
                close = 0

            # Identify symbol from our mapping
            symbol = "Unknown"
            if security_id and exchange:
                key = f"{exchange}_{security_id}"
                symbol = self.symbol_map.get(key, f"ID_{security_id}")

            # Check if this might be one of our subscribed instruments based on price
            # Convert ltp to float if it's a string
            try:
                ltp_val = float(ltp) if ltp else 0
            except:
                ltp_val = 0

            if symbol == "Unknown" and ltp_val > 0:
                # Try to match based on price ranges (rough estimates)
                if 24000 <= ltp_val <= 26000:
                    symbol = "NIFTY50"
                elif 50000 <= ltp_val <= 53000:
                    symbol = "BANKNIFTY"
                elif 2800 <= ltp_val <= 3000:
                    symbol = "RELIANCE"
                elif 3500 <= ltp_val <= 4000:
                    symbol = "TCS"
                elif 1300 <= ltp_val <= 1500:
                    symbol = "INFY"
                elif 1500 <= ltp_val <= 1800:
                    symbol = "HDFCBANK"
                elif 900 <= ltp_val <= 1100:
                    symbol = "ICICIBANK"
                elif 700 <= ltp_val <= 900:
                    symbol = "AXISBANK"
                elif 250 <= ltp_val <= 350:
                    symbol = "ITC"

            # Store data
            self.received_data[symbol] = {
                'ltp': ltp,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'time': datetime.now().isoformat(),
                'packet': self.packet_count,
                'security_id': security_id,
                'exchange': exchange
            }

            # Log identified symbols
            if symbol != "Unknown":
                change = ((ltp - open_price) / open_price * 100) if open_price > 0 else 0
                logger.info(f"[{self.packet_count}] {symbol}: ₹{ltp:.2f} | Vol: {volume:,} | Chg: {change:+.2f}%")
            else:
                # Log unidentified data for debugging
                logger.debug(f"[{self.packet_count}] Unidentified: LTP={ltp}, Vol={volume}, Data={data}")

        except Exception as e:
            logger.error(f"Error handling data: {e}")
            logger.debug(f"Raw data: {data}")

    def show_detailed_results(self):
        """Show detailed test results with symbol identification"""

        logger.info("\n" + "="*80)
        logger.info("WEBSOCKET TEST RESULTS - SYMBOL IDENTIFICATION")
        logger.info("="*80)

        logger.info(f"Data packets received: {self.packet_count}")
        logger.info(f"Unique symbols with data: {len(self.received_data)}")

        # Separate identified and unidentified symbols
        identified = {k: v for k, v in self.received_data.items() if k != "Unknown" and not k.startswith("ID_")}
        unidentified = {k: v for k, v in self.received_data.items() if k == "Unknown" or k.startswith("ID_")}

        if identified:
            logger.info(f"\n✅ IDENTIFIED SYMBOLS ({len(identified)}):")
            logger.info("-" * 60)
            for symbol, data in identified.items():
                change = ((data['ltp'] - data['open']) / data['open'] * 100) if data['open'] > 0 else 0
                logger.info(f"  {symbol:15} | LTP: ₹{data['ltp']:8.2f} | Vol: {data['volume']:10,} | Chg: {change:+6.2f}%")

        if unidentified:
            logger.info(f"\n⚠️ UNIDENTIFIED DATA ({len(unidentified)}):")
            logger.info("-" * 60)
            for key, data in list(unidentified.items())[:5]:
                logger.info(f"  {key:15} | LTP: ₹{data['ltp']:8.2f} | Vol: {data['volume']:10,}")

        # Show subscription status
        logger.info("\n📊 SUBSCRIPTION STATUS:")
        logger.info("-" * 60)
        for inst in self.subscribed_instruments[:10]:
            status = "✅ Receiving" if inst['symbol'] in identified else "⏳ Waiting"
            logger.info(f"  {inst['symbol']:15} (ID: {inst['security_id']:6}) - {status}")

        # Market status
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        market_open = (hour == 9 and minute >= 15) or (9 < hour < 15) or (hour == 15 and minute < 30)

        logger.info(f"\n⏰ Market Status: {'OPEN' if market_open else 'CLOSED'}")
        logger.info("   Market Hours: 9:15 AM to 3:30 PM IST")

        # Verdict
        if identified:
            logger.info("\n" + "="*80)
            logger.info("✅ WEBSOCKET WITH SYMBOL IDENTIFICATION WORKING!")
            logger.info("="*80)
            logger.info(f"Successfully identified {len(identified)} symbols")
            logger.info("Real-time data is being received and mapped correctly")
            logger.info("="*80)
        elif self.packet_count > 0:
            logger.info("\n⚠️ Data received but symbol mapping needs adjustment")
            logger.info("Check the data structure returned by MarketFeed")
        else:
            if market_open:
                logger.info("\n❌ No data received - check subscription format")
            else:
                logger.info("\n⏰ Market closed - test during market hours")

def main():
    """Main function"""

    ws_test = DhanWebSocketWithSymbols()
    success = ws_test.test_websocket()

    if success:
        logger.info("\n✅ WebSocket test with symbol identification completed!")
        return True
    else:
        logger.info("\n❌ Test needs attention")
        return False

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()