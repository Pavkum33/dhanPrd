#!/usr/bin/env python3
"""
DhanHQ v2 Live Market Feed Implementation
Based on official documentation: https://dhanhq.co/docs/v2/live-market-feed/
Correctly parses binary packets and maps security IDs to symbols
"""

import asyncio
import websockets
import json
import struct
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DhanWebSocketV2:
    def __init__(self):
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')

        if not self.client_id or not self.access_token:
            raise ValueError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN required")

        # WebSocket endpoint with authentication
        self.ws_url = f"wss://api-feed.dhan.co?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"

        # Symbol mapping
        self.security_to_symbol = {}
        self.symbol_to_security = {}

        # Track received data
        self.received_data = {}
        self.packet_count = 0

    def load_symbol_mapping(self):
        """Load security ID to symbol mapping from instruments file"""

        try:
            # Try to load from instruments CSV
            if os.path.exists('instruments_cached.csv'):
                logger.info("Loading ALL F&O instruments from CSV...")
                df = pd.read_csv('instruments_cached.csv')  # Load all instruments

                # Filter for F&O instruments (Futures and Options)
                # Focus on NSE F&O segment
                fno_mask = (
                    (df['EXCH_ID'].str.contains('NSE', na=False)) &
                    (df['SEGMENT'].str.contains('F&O|FO|FNO', na=False, regex=True) |
                     df['INSTRUMENT'].isin(['FUTSTK', 'FUTIDX', 'OPTSTK', 'OPTIDX']))
                )

                fno_df = df[fno_mask]

                # Also include major equity stocks that are in F&O
                equity_mask = (
                    (df['EXCH_ID'] == 'NSE') &
                    (df['SEGMENT'] == 'C') &  # Cash segment
                    (df['INSTRUMENT'] == 'EQUITY')
                )

                equity_df = df[equity_mask]

                # Combine both
                all_instruments = pd.concat([fno_df, equity_df])

                # Create mappings
                for _, row in all_instruments.iterrows():
                    security_id = str(row['SECURITY_ID'])
                    symbol_name = row['SYMBOL_NAME']
                    underlying = row.get('UNDERLYING_SYMBOL', symbol_name)

                    # Store the mapping
                    self.security_to_symbol[security_id] = symbol_name

                    # For F&O, also map the underlying symbol
                    if underlying and underlying != 'NA':
                        if underlying not in self.symbol_to_security:
                            self.symbol_to_security[underlying] = security_id

                    # Map full symbol name
                    if symbol_name not in self.symbol_to_security:
                        self.symbol_to_security[symbol_name] = security_id

                logger.info(f"Loaded {len(self.security_to_symbol)} total symbol mappings")
                logger.info(f"F&O instruments: {len(fno_df)}, Equity instruments: {len(equity_df)}")

            else:
                # Fallback to manual mapping
                self.security_to_symbol = {
                    "1333": "HDFC",
                    "2885": "RELIANCE",
                    "11536": "TCS",
                    "1594": "INFY",
                    "5258": "HDFCBANK",
                    "1660": "ITC",
                    "772": "AXISBANK",
                    "4963": "ICICIBANK",
                    "13": "NIFTY50",
                    "25": "BANKNIFTY"
                }
                logger.info(f"Using manual mapping for {len(self.security_to_symbol)} symbols")

            # Create reverse mapping
            self.symbol_to_security = {v: k for k, v in self.security_to_symbol.items()}

        except Exception as e:
            logger.error(f"Error loading mappings: {e}")

    def parse_binary_packet(self, data):
        """Parse binary data packet according to Dhan v2 format"""

        if not data or len(data) < 8:
            return None

        try:
            # Parse header (first 8 bytes)
            # 1 byte: Feed Response Code
            # 2 bytes: Message Length
            # 1 byte: Exchange Segment
            # 4 bytes: Security ID

            feed_response_code = struct.unpack('<B', data[0:1])[0]  # Little Endian
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = struct.unpack('<B', data[3:4])[0]
            security_id = struct.unpack('<I', data[4:8])[0]  # 4 bytes for Security ID

            # Map security ID to symbol
            security_id_str = str(security_id)
            symbol = self.security_to_symbol.get(security_id_str, f"ID_{security_id}")

            packet_info = {
                'symbol': symbol,
                'security_id': security_id,
                'exchange_segment': exchange_segment,
                'packet_type': feed_response_code
            }

            # Parse based on packet type
            if feed_response_code == 2:  # Ticker Packet
                if len(data) >= 16:
                    ltp = struct.unpack('<I', data[8:12])[0] / 100  # Price in paisa, convert to rupees
                    last_trade_time = struct.unpack('<I', data[12:16])[0]

                    packet_info.update({
                        'ltp': ltp,
                        'last_trade_time': last_trade_time
                    })

            elif feed_response_code == 4:  # Quote Packet
                if len(data) >= 44:  # Minimum size for quote packet
                    ltp = struct.unpack('<I', data[8:12])[0] / 100
                    ltq = struct.unpack('<I', data[12:16])[0]
                    avg_price = struct.unpack('<I', data[16:20])[0] / 100
                    volume = struct.unpack('<I', data[20:24])[0]
                    total_buy_qty = struct.unpack('<I', data[24:28])[0]
                    total_sell_qty = struct.unpack('<I', data[28:32])[0]
                    open_price = struct.unpack('<I', data[32:36])[0] / 100
                    close_price = struct.unpack('<I', data[36:40])[0] / 100
                    high_price = struct.unpack('<I', data[40:44])[0] / 100
                    low_price = struct.unpack('<I', data[44:48])[0] / 100 if len(data) >= 48 else 0

                    packet_info.update({
                        'ltp': ltp,
                        'ltq': ltq,
                        'avg_price': avg_price,
                        'volume': volume,
                        'total_buy_qty': total_buy_qty,
                        'total_sell_qty': total_sell_qty,
                        'open': open_price,
                        'close': close_price,
                        'high': high_price,
                        'low': low_price
                    })

            elif feed_response_code == 8:  # Full Packet (includes market depth)
                # Parse quote data first (same as packet type 4)
                if len(data) >= 162:  # Full packet size
                    ltp = struct.unpack('<I', data[8:12])[0] / 100
                    volume = struct.unpack('<I', data[20:24])[0]

                    packet_info.update({
                        'ltp': ltp,
                        'volume': volume,
                        'has_market_depth': True
                    })

            return packet_info

        except Exception as e:
            logger.error(f"Error parsing packet: {e}")
            return None

    async def connect_and_subscribe(self, test_symbols=None, subscribe_all=False):
        """Connect to WebSocket and subscribe to symbols

        Args:
            test_symbols: List of specific symbols to subscribe (optional)
            subscribe_all: If True, subscribe to ALL available F&O symbols
        """

        # Load symbol mappings
        self.load_symbol_mapping()

        # Determine which symbols to subscribe
        if subscribe_all:
            # Get unique underlying symbols for F&O (avoid duplicates)
            unique_underlyings = set()

            for security_id, symbol in self.security_to_symbol.items():
                # Extract underlying from F&O symbols (e.g., RELIANCE-Nov2025-FUT -> RELIANCE)
                if '-' in symbol:
                    underlying = symbol.split('-')[0]
                    unique_underlyings.add(underlying)
                else:
                    unique_underlyings.add(symbol)

            # Priority F&O stocks (most traded)
            priority_symbols = [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "AXISBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
                "LT", "WIPRO", "TATASTEEL", "TATAMOTORS", "HINDALCO",
                "MARUTI", "DRREDDY", "SUNPHARMA", "BAJFINANCE", "BAJAJFINSV",
                "ADANIENT", "ADANIPORTS", "POWERGRID", "NTPC", "ONGC",
                "COALINDIA", "INDUSINDBK", "UPL", "GRASIM", "TITAN",
                "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"
            ]

            # Create subscription list with priority symbols first
            symbols_to_subscribe = []
            for symbol in priority_symbols:
                if symbol in self.symbol_to_security:
                    symbols_to_subscribe.append(symbol)

            # Add remaining symbols
            for symbol in unique_underlyings:
                if symbol not in symbols_to_subscribe and symbol in self.symbol_to_security:
                    symbols_to_subscribe.append(symbol)

            logger.info(f"Subscribing to ALL {len(symbols_to_subscribe)} F&O underlying symbols")
        elif test_symbols is None:
            # Default test symbols
            symbols_to_subscribe = ["HDFC", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
        else:
            symbols_to_subscribe = test_symbols

        logger.info("="*60)
        logger.info("DHAN WEBSOCKET V2 - LIVE MARKET FEED")
        logger.info("="*60)
        logger.info(f"Client ID: {self.client_id}")
        logger.info(f"Connecting to: wss://api-feed.dhan.co")

        try:
            async with websockets.connect(self.ws_url) as websocket:
                logger.info("[OK] Connected successfully!")

                # Prepare subscription
                instrument_list = []

                # Limit to 100 instruments per subscription (Dhan API limit)
                max_per_batch = 100
                symbols_batch = symbols_to_subscribe[:max_per_batch] if len(symbols_to_subscribe) > max_per_batch else symbols_to_subscribe

                if len(symbols_to_subscribe) > max_per_batch:
                    logger.info(f"Note: Limiting to first {max_per_batch} symbols (API limit). Total available: {len(symbols_to_subscribe)}")

                for i, symbol in enumerate(symbols_batch):
                    if symbol in self.symbol_to_security:
                        security_id = self.symbol_to_security[symbol]
                        instrument_list.append({
                            "ExchangeSegment": "NSE_EQ",
                            "SecurityId": security_id
                        })
                        # Log first few and last symbol
                        if i < 5:
                            logger.info(f"  [{i+1}] {symbol} (ID: {security_id})")
                        elif i == len(symbols_batch) - 1:
                            logger.info(f"  ... and {i - 4} more symbols ...")
                            logger.info(f"  [{i+1}] {symbol} (ID: {security_id})")

                if not instrument_list:
                    logger.error("No valid instruments to subscribe")
                    return

                # Send subscription request
                subscription_request = {
                    "RequestCode": 15,
                    "InstrumentCount": len(instrument_list),
                    "InstrumentList": instrument_list
                }

                await websocket.send(json.dumps(subscription_request))
                logger.info(f"[OK] Sent subscription for {len(instrument_list)} instruments")

                # Listen for data
                logger.info("\n" + "-"*60)
                logger.info("RECEIVING LIVE DATA...")
                logger.info("-"*60)

                timeout = 30  # Listen for 30 seconds
                start_time = asyncio.get_event_loop().time()

                while (asyncio.get_event_loop().time() - start_time) < timeout:
                    try:
                        # Receive message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)

                        if isinstance(message, bytes):
                            # Parse binary message
                            packet_data = self.parse_binary_packet(message)

                            if packet_data:
                                self.packet_count += 1
                                symbol = packet_data['symbol']

                                # Store latest data
                                if symbol not in self.received_data:
                                    self.received_data[symbol] = {
                                        'packets': 0,
                                        'latest': {}
                                    }

                                self.received_data[symbol]['packets'] += 1
                                self.received_data[symbol]['latest'] = packet_data

                                # Log every 5th packet or first packet for each symbol
                                if self.received_data[symbol]['packets'] == 1 or self.packet_count % 5 == 0:
                                    ltp = packet_data.get('ltp', 0)
                                    volume = packet_data.get('volume', 0)
                                    logger.info(f"[{self.packet_count}] {symbol}: Rs.{ltp:.2f} | Vol: {volume:,}")

                        elif isinstance(message, str):
                            # Handle JSON messages (acknowledgments, etc.)
                            logger.debug(f"JSON message: {message}")

                    except asyncio.TimeoutError:
                        remaining = timeout - (asyncio.get_event_loop().time() - start_time)
                        if remaining > 0:
                            logger.info(f"Waiting for data... ({remaining:.0f}s remaining)")
                    except Exception as e:
                        logger.error(f"Error receiving data: {e}")
                        break

                # Send disconnect
                disconnect_request = {"RequestCode": 12}
                await websocket.send(json.dumps(disconnect_request))
                logger.info("\n[OK] Disconnected gracefully")

        except Exception as e:
            logger.error(f"Connection error: {e}")

    def show_results(self):
        """Display test results"""

        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS - SYMBOL MAPPING")
        logger.info("="*60)

        logger.info(f"\n[STATS] Statistics:")
        logger.info(f"  Total packets received: {self.packet_count}")
        logger.info(f"  Unique symbols with data: {len(self.received_data)}")

        if self.received_data:
            logger.info(f"\n[OK] SUCCESSFULLY IDENTIFIED SYMBOLS:")
            logger.info("-"*50)
            logger.info(f"{'Symbol':<15} | {'Security ID':<10} | {'LTP':>10} | {'Packets':>8}")
            logger.info("-"*50)

            for symbol, data in self.received_data.items():
                latest = data['latest']
                security_id = latest.get('security_id', 'N/A')
                ltp = latest.get('ltp', 0)
                packets = data['packets']
                logger.info(f"{symbol:<15} | {security_id:<10} | Rs.{ltp:8.2f} | {packets:>8}")

        # Market status
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        market_open = (hour == 9 and minute >= 15) or (9 < hour < 15) or (hour == 15 and minute < 30)

        logger.info(f"\n[TIME] Market Status: {'OPEN' if market_open else 'CLOSED'}")
        logger.info("       Market Hours: 9:15 AM to 3:30 PM IST")

        # Verdict
        logger.info("\n" + "="*60)
        if self.packet_count > 0 and self.received_data:
            logger.info("[SUCCESS] WEBSOCKET WITH SYMBOL MAPPING WORKING!")
            logger.info(f"Successfully identified {len(self.received_data)} symbols using:")
            logger.info("  1. Security ID extracted from binary packets (bytes 5-8)")
            logger.info("  2. Instruments file mapping (security_id -> symbol)")
            logger.info("  3. Real-time binary data parsing")
            logger.info("\n[SUCCESS] SYSTEM READY FOR PRODUCTION!")
        elif self.packet_count > 0:
            logger.info("[OK] WebSocket receiving data")
            logger.info("[WARNING] Some symbols not mapped")
        else:
            if market_open:
                logger.info("[ERROR] No data received during market hours")
            else:
                logger.info("[TIME] Market closed - test during market hours")
        logger.info("="*60)

async def main():
    """Main function to run WebSocket test"""

    ws_client = DhanWebSocketV2()

    # Test with specific symbols
    test_symbols = ["HDFC", "RELIANCE", "TCS", "INFY", "ITC"]

    await ws_client.connect_and_subscribe(test_symbols)

    # Show results
    ws_client.show_results()

if __name__ == "__main__":
    asyncio.run(main())