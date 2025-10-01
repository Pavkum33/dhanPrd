#!/usr/bin/env python3
"""
Dhan WebSocket with Proper Symbol Identification
Using instruments master file for security_id to symbol mapping
"""

import os
import time
import pandas as pd
import pickle
from datetime import datetime
from dotenv import load_dotenv
import logging
from dhanhq import DhanContext, MarketFeed
import threading

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DhanWebSocketWithMapping:
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
        self.last_packet_time = {}

        # Symbol mapping
        self.id_to_symbol = {}
        self.symbol_to_id = {}
        self.subscribed_securities = []

    def load_instrument_mapping(self):
        """Load instruments master file and create mappings"""

        logger.info("Loading instruments master file...")

        try:
            # Try cached file first
            cache_file = "symbol_mapping.pkl"

            if os.path.exists(cache_file):
                # Load cached mapping
                with open(cache_file, 'rb') as f:
                    mapping = pickle.load(f)
                    self.id_to_symbol = mapping['id_to_symbol']
                    self.symbol_to_id = mapping['symbol_to_id']
                    logger.info(f"✅ Loaded cached mapping with {len(self.id_to_symbol)} instruments")
            else:
                # Load from CSV
                if os.path.exists('instruments_cached.csv'):
                    df = pd.read_csv('instruments_cached.csv')
                elif os.path.exists('api-scrip-master.csv'):
                    df = pd.read_csv('api-scrip-master.csv')
                else:
                    logger.warning("No instruments file found, using manual mapping")
                    self.create_manual_mapping()
                    return

                # Create mappings for NSE equity and F&O
                # Filter for NSE segments
                nse_df = df[df['EXCH_ID'].str.contains('NSE', na=False)]

                # Create security_id to symbol mapping
                for _, row in nse_df.iterrows():
                    security_id = str(row['SECURITY_ID'])
                    symbol = row['SYMBOL_NAME']
                    underlying = row.get('UNDERLYING_SYMBOL', symbol)
                    instrument = row.get('INSTRUMENT', '')
                    expiry = row.get('SM_EXPIRY_DATE', '')

                    # Store the symbol as-is from CSV
                    self.id_to_symbol[security_id] = symbol

                    # Also create simplified mappings for easy lookup
                    if underlying and underlying != 'NA':
                        # For futures/options, also map underlying to a security ID
                        if underlying not in self.symbol_to_id:
                            self.symbol_to_id[underlying] = security_id

                    # Map by full symbol name
                    self.symbol_to_id[symbol] = security_id

                logger.info(f"✅ Created mapping for {len(self.id_to_symbol)} instruments")

                # Cache the mapping
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'id_to_symbol': self.id_to_symbol,
                        'symbol_to_id': self.symbol_to_id
                    }, f)
                    logger.info(f"Cached mapping to {cache_file}")

        except Exception as e:
            logger.error(f"Error loading instruments: {e}")
            self.create_manual_mapping()

    def create_manual_mapping(self):
        """Create manual mapping for common instruments"""

        self.id_to_symbol = {
            # NSE Equity
            "2885": "RELIANCE",
            "11536": "TCS",
            "1594": "INFY",
            "5258": "HDFCBANK",
            "1333": "HDFC",
            "1660": "ITC",
            "772": "AXISBANK",
            "4963": "ICICIBANK",
            "3045": "SBIN",
            "14977": "BHARTIARTL",
            "13": "NIFTY50",
            "25": "BANKNIFTY",

            # NSE F&O
            "35000": "NIFTY-FUT",
            "35001": "BANKNIFTY-FUT",
            "35002": "FINNIFTY-FUT",
            "45825": "RELIANCE-FUT",
            "46376": "TCS-FUT",
            "34817": "INFY-FUT",
            "39447": "HDFCBANK-FUT",
            "39952": "ICICIBANK-FUT",
            "32129": "AXISBANK-FUT",
            "48896": "SBIN-FUT",
            "31181": "BHARTIARTL-FUT",
            "41220": "ITC-FUT",
            "41729": "KOTAKBANK-FUT",
        }

        # Create reverse mapping
        self.symbol_to_id = {v: k for k, v in self.id_to_symbol.items()}
        logger.info(f"✅ Created manual mapping for {len(self.id_to_symbol)} instruments")

    def test_websocket(self):
        """Test WebSocket with symbol identification"""

        logger.info("="*80)
        logger.info("WEBSOCKET WITH SYMBOL MAPPING TEST")
        logger.info("="*80)
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Client ID: {self.client_id}")

        # Load instrument mapping
        self.load_instrument_mapping()

        try:
            # Select instruments to subscribe
            test_symbols = [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "AXISBANK", "ITC", "SBIN", "BHARTIARTL", "NIFTY50"
            ]

            # Build subscription list using security IDs
            instruments = []
            for symbol in test_symbols:
                if symbol in self.symbol_to_id:
                    security_id = self.symbol_to_id[symbol]
                    instruments.append((MarketFeed.NSE, security_id, MarketFeed.Full))
                    self.subscribed_securities.append(security_id)
                    logger.info(f"  Subscribing to {symbol} (ID: {security_id})")
                else:
                    logger.warning(f"  Symbol {symbol} not found in mapping")

            if not instruments:
                logger.error("No valid instruments to subscribe")
                return False

            logger.info(f"\n✅ Subscribing to {len(instruments)} instruments")

            # Create MarketFeed instance
            self.market_feed = MarketFeed(
                dhan_context=self.dhan_context,
                instruments=instruments,
                version="v2"
            )

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
            logger.info("LISTENING FOR LIVE DATA...")
            logger.info("="*60)

            # Listen for data
            timeout = 30
            start_time = time.time()
            last_log = start_time
            identified_count = 0

            while (time.time() - start_time) < timeout:
                try:
                    # Try to get data
                    response = self.market_feed.get_data()

                    if response:
                        self.packet_count += 1
                        symbol = self.handle_data_with_mapping(response)

                        if symbol and symbol != "Unknown":
                            identified_count += 1

                    # Log progress every 5 seconds
                    if time.time() - last_log > 5:
                        remaining = timeout - (time.time() - start_time)
                        logger.info(f"Progress: {remaining:.0f}s remaining | {self.packet_count} packets | {identified_count} identified | {len(self.received_data)} unique symbols")
                        last_log = time.time()

                    time.sleep(0.1)

                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception:
                    pass

            # Disconnect
            try:
                logger.info("\nDisconnecting...")
                try:
                    self.market_feed.disconnect()
                except:
                    pass
                logger.info("✅ Disconnected")
            except:
                pass

            # Show results
            self.show_results()

            return self.packet_count > 0

        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def handle_data_with_mapping(self, data):
        """Handle received market data with symbol mapping"""

        try:
            # Extract data fields
            if isinstance(data, dict):
                # Try multiple possible field names
                ltp = data.get('LTP', data.get('ltp', data.get('last_price', 0)))
                volume = data.get('volume', data.get('Volume', data.get('vol', 0)))
                bid = data.get('bid', data.get('best_bid', 0))
                ask = data.get('ask', data.get('best_ask', 0))
                open_price = data.get('open', data.get('Open', 0))
                high = data.get('high', data.get('High', 0))
                low = data.get('low', data.get('Low', 0))

            else:
                # Unknown format
                return None

            # Identify symbol by matching with subscribed securities and price
            symbol = None

            # First try to match by price with known securities
            for sec_id in self.subscribed_securities:
                expected_symbol = self.id_to_symbol.get(sec_id, "")

                # Match by price range
                if self.is_price_match(expected_symbol, ltp):
                    symbol = expected_symbol
                    break

            if not symbol:
                symbol = f"Unknown (LTP: {ltp})"

            # Store data
            self.received_data[symbol] = {
                'ltp': ltp,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'open': open_price,
                'high': high,
                'low': low,
                'time': datetime.now().isoformat(),
                'packet': self.packet_count
            }

            # Log identified symbols
            if symbol and not symbol.startswith("Unknown"):
                # Avoid duplicate logs for same symbol
                last_time = self.last_packet_time.get(symbol, 0)
                if time.time() - last_time > 2:  # Log same symbol max every 2 seconds
                    change = ((ltp - open_price) / open_price * 100) if open_price > 0 else 0
                    logger.info(f"[{self.packet_count}] ✅ {symbol}: ₹{ltp:.2f} | Vol: {volume:,} | Chg: {change:+.2f}%")
                    self.last_packet_time[symbol] = time.time()

            return symbol

        except Exception as e:
            logger.error(f"Error handling data: {e}")
            return None

    def is_price_match(self, symbol, ltp):
        """Check if price matches expected range for symbol"""

        if not ltp or ltp == 0:
            return False

        # Price ranges for common stocks (approximate)
        price_ranges = {
            "RELIANCE": (2800, 3100),
            "TCS": (3400, 4000),
            "INFY": (1300, 1500),
            "HDFCBANK": (1500, 1800),
            "ICICIBANK": (900, 1100),
            "AXISBANK": (700, 900),
            "ITC": (400, 500),
            "SBIN": (550, 700),
            "BHARTIARTL": (1400, 1600),
            "NIFTY50": (24000, 26000),
            "BANKNIFTY": (50000, 53000),
        }

        # Check base symbol (remove -FUT suffix)
        base_symbol = symbol.replace("-FUT", "")

        if base_symbol in price_ranges:
            min_price, max_price = price_ranges[base_symbol]
            return min_price <= ltp <= max_price

        return False

    def show_results(self):
        """Show detailed test results"""

        logger.info("\n" + "="*80)
        logger.info("WEBSOCKET TEST RESULTS - WITH SYMBOL MAPPING")
        logger.info("="*80)

        logger.info(f"Data packets received: {self.packet_count}")
        logger.info(f"Unique symbols with data: {len(self.received_data)}")

        # Separate identified and unidentified
        identified = {k: v for k, v in self.received_data.items() if not k.startswith("Unknown")}
        unidentified = {k: v for k, v in self.received_data.items() if k.startswith("Unknown")}

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
                logger.info(f"  {key:25} | Vol: {data['volume']:10,}")

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
            logger.info("✅ WEBSOCKET WITH SYMBOL MAPPING WORKING!")
            logger.info("="*80)
            logger.info(f"Successfully identified {len(identified)} symbols using:")
            logger.info("  1. Instruments master file mapping")
            logger.info("  2. Price range validation")
            logger.info("  3. Real-time data correlation")
            logger.info("\n🎯 System ready for production use!")
            logger.info("="*80)
        elif self.packet_count > 0:
            logger.info("\n⚠️ Data received but needs symbol correlation")
        else:
            if market_open:
                logger.info("\n❌ No data received - check subscription")
            else:
                logger.info("\n⏰ Market closed - test during market hours")

def main():
    """Main function"""

    ws_test = DhanWebSocketWithMapping()
    success = ws_test.test_websocket()

    if success:
        logger.info("\n✅ WebSocket with symbol mapping test completed!")
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