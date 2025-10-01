# Complete DHAN API Tutorial: Historical & Intraday Data Guide

## Table of Contents
1. [Introduction & Setup](#introduction--setup)
2. [Authentication](#authentication)
3. [Instrument Master & Symbol Mapping](#instrument-master--symbol-mapping)
4. [Historical Data](#historical-data)
5. [Intraday & Real-time Data](#intraday--real-time-data)
6. [Practical Examples](#practical-examples)
7. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
8. [Best Practices](#best-practices)

---

## Introduction & Setup

DHAN HQ provides powerful APIs for accessing Indian market data. This tutorial covers both historical data fetching and real-time intraday updates based on production-tested implementation.

### Prerequisites
```python
# Required Python packages
pip install dhanhq pandas requests websocket-client
```

### Initial Setup
```python
import os
from dhanhq import dhanhq
import pandas as pd
import requests
from datetime import datetime, timedelta

# Initialize DHAN SDK
client_id = "your_client_id"
access_token = "your_access_token"
dhan = dhanhq(client_id, access_token)
```

---

## Authentication

### 1. Getting Credentials
- **Client ID**: Your DHAN account client ID (numeric string)
- **Access Token**: JWT token from DHAN trading platform
- **Token Format**: Long JWT string starting with "eyJ0eXAiOiJKV1Qi..."

### 2. Setting Up Environment Variables
```python
# Method 1: Environment Variables (Recommended)
os.environ['DHAN_CLIENT_ID'] = 'your_client_id'
os.environ['DHAN_ACCESS_TOKEN'] = 'your_access_token'

# Method 2: Direct Assignment
client_id = "1106283829"  # Example
access_token = "eyJ0eXAiOiJKV1Qi..."  # Your actual token
```

### 3. SDK Initialization
```python
# Initialize DHAN SDK
dhan = dhanhq(
    client_id=os.getenv('DHAN_CLIENT_ID'),
    access_token=os.getenv('DHAN_ACCESS_TOKEN')
)

# Test connection
try:
    positions = dhan.get_positions()
    print("✅ Authentication successful!")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
```

---

## Instrument Master & Symbol Mapping

### Understanding DHAN Instrument Structure

DHAN provides multiple CSV files for different instrument types:

1. **api-scrip-master.csv** - F&O instruments (Futures & Options)
2. **Equity Master** - Via API endpoint `/v2/instruments/master`

### 1. Loading F&O Instruments (CSV Method)

```python
def load_fno_instruments():
    """Load F&O instruments from DHAN CSV"""
    
    # Download latest CSV
    csv_url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    response = requests.get(csv_url)
    
    # Save and load CSV
    with open('api-scrip-master.csv', 'wb') as f:
        f.write(response.content)
    
    df = pd.read_csv('api-scrip-master.csv')
    print(f"📊 Loaded {len(df)} F&O instruments")
    
    return df

# Filter active futures (current month only)
def get_active_futures(df):
    """Filter for active NSE futures only"""
    
    # Handle different CSV column formats
    exch_col = 'SEM_EXM_EXCH_ID' if 'SEM_EXM_EXCH_ID' in df.columns else 'EXCH_ID'
    instrument_col = 'SEM_INSTRUMENT_NAME' if 'SEM_INSTRUMENT_NAME' in df.columns else 'INSTRUMENT_TYPE'
    expiry_col = 'SEM_EXPIRY_DATE' if 'SEM_EXPIRY_DATE' in df.columns else 'EXPIRY_DATE'
    
    # Convert expiry to datetime
    df[expiry_col] = pd.to_datetime(df[expiry_col], errors='coerce')
    
    # Filter criteria
    nse_mask = df[exch_col].str.upper().isin(['NSE', 'NSE_FO', 'NSE_FUT'])
    fut_mask = df[instrument_col].str.upper().isin(['FUTSTK', 'FUTIDX'])
    expiry_mask = df[expiry_col].notnull()
    
    # Current month filtering (avoid expired contracts)
    today = datetime.now()
    current_month_mask = (
        (df[expiry_col].dt.month == today.month) & 
        (df[expiry_col].dt.year == today.year)
    )
    
    active_futures = df[nse_mask & fut_mask & expiry_mask & current_month_mask]
    
    print(f"🎯 Found {len(active_futures)} active futures")
    return active_futures

# Example usage
fno_df = load_fno_instruments()
active_futures = get_active_futures(fno_df)
```

### 2. Loading Equity Instruments (API Method)

```python
def load_equity_instruments(dhan_sdk):
    """Load equity instruments via DHAN API"""
    
    try:
        # Get equity master data
        response = dhan_sdk.get_instruments('NSE')
        
        if response['status'] == 'success':
            instruments = response['data']
            
            # Create symbol to securityId mapping
            equity_mapping = {}
            for instrument in instruments:
                if instrument.get('SEM_INSTRUMENT_NAME') == 'EQ':
                    symbol = instrument.get('SEM_TRADING_SYMBOL')
                    security_id = instrument.get('SEM_SMST_SECURITY_ID')
                    if symbol and security_id:
                        equity_mapping[symbol] = str(security_id)
            
            print(f"📈 Loaded {len(equity_mapping)} equity mappings")
            return equity_mapping
            
    except Exception as e:
        print(f"❌ Failed to load equity instruments: {e}")
        return {}

# Example: Get RELIANCE security ID
equity_mapping = load_equity_instruments(dhan)
reliance_id = equity_mapping.get('RELIANCE')  # Returns: "2885"
print(f"RELIANCE Security ID: {reliance_id}")
```

### 3. Symbol Extraction from F&O

```python
def extract_underlying_symbol(future_symbol):
    """Extract underlying symbol from futures symbol"""
    
    # Examples:
    # "RELIANCE-SEP2025-FUT" -> "RELIANCE"
    # "NIFTY 50-SEP2025-FUT" -> "NIFTY 50"
    # "BANKNIFTY-SEP2025-FUT" -> "BANKNIFTY"
    
    parts = future_symbol.split('-')
    
    if len(parts) >= 3:
        # Remove month and FUT suffix
        underlying = '-'.join(parts[:-2])
        return underlying.strip()
    
    # Fallback for irregular formats
    return future_symbol.split('-')[0].strip()

# Example usage
examples = [
    "RELIANCE-SEP2025-FUT",
    "HDFCBANK-SEP2025-FUT", 
    "NIFTY 50-SEP2025-FUT"
]

for symbol in examples:
    underlying = extract_underlying_symbol(symbol)
    print(f"{symbol} -> {underlying}")
```

---

## Historical Data

### Understanding DHAN Historical Data API

**CRITICAL**: Use the dhanhq SDK, NOT direct REST API calls. The REST endpoints return 404 errors.

### 1. Basic Historical Data Fetching

```python
def get_historical_data(dhan_sdk, security_id, days_back=30):
    """
    Fetch historical data using DHAN SDK
    
    IMPORTANT: This is the ONLY working method for historical data
    """
    
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    try:
        # SDK method with EXACT parameters that work
        response = dhan_sdk.historical_daily_data(
            security_id=str(security_id),           # MUST be string
            exchange_segment="NSE_EQ",              # For equities
            instrument_type="EQUITY",               # For stocks
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d")
        )
        
        if response.get('status') == 'success':
            data = response.get('data', {})
            
            # Convert SDK response to DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data.get('timestamp', [])),
                'open': data.get('open', []),
                'high': data.get('high', []),
                'low': data.get('low', []),
                'close': data.get('close', []),
                'volume': data.get('volume', [])
            })
            
            return df
            
    except Exception as e:
        print(f"❌ Historical data error: {e}")
        return pd.DataFrame()

# Example usage
security_id = "2885"  # RELIANCE
hist_data = get_historical_data(dhan, security_id, days_back=50)
print(f"📊 Retrieved {len(hist_data)} days of data")
print(hist_data.head())
```

### 2. Batch Historical Data Processing

```python
def get_batch_historical_data(dhan_sdk, symbol_mapping, symbols, days_back=30):
    """Fetch historical data for multiple symbols"""
    
    historical_data = {}
    
    for symbol in symbols:
        security_id = symbol_mapping.get(symbol)
        
        if not security_id:
            print(f"⚠️  No security ID found for {symbol}")
            continue
            
        print(f"📡 Fetching {symbol} (ID: {security_id})...")
        
        # Try different parameter combinations
        for attempt in range(3):
            try:
                if attempt == 0:
                    # Attempt 1: Standard equity
                    response = dhan_sdk.historical_daily_data(
                        security_id=str(security_id),
                        exchange_segment="NSE_EQ",
                        instrument_type="EQUITY",
                        from_date=(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                        to_date=datetime.now().strftime("%Y-%m-%d")
                    )
                elif attempt == 1:
                    # Attempt 2: Index (for NIFTY, BANKNIFTY)
                    response = dhan_sdk.historical_daily_data(
                        security_id=str(security_id),
                        exchange_segment="NSE_INDEX",
                        instrument_type="INDEX",
                        from_date=(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                        to_date=datetime.now().strftime("%Y-%m-%d")
                    )
                else:
                    # Attempt 3: Alternative format
                    response = dhan_sdk.historical_daily_data(
                        security_id=str(security_id),
                        exchange_segment="NSE",
                        instrument_type="EQ",
                        from_date=(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                        to_date=datetime.now().strftime("%Y-%m-%d")
                    )
                
                if response.get('status') == 'success':
                    data = response.get('data', {})
                    
                    # Convert to DataFrame
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data.get('timestamp', [])),
                        'open': data.get('open', []),
                        'high': data.get('high', []),
                        'low': data.get('low', []),
                        'close': data.get('close', []),
                        'volume': data.get('volume', [])
                    })
                    
                    if len(df) > 0:
                        historical_data[symbol] = df
                        print(f"✅ {symbol}: {len(df)} days retrieved")
                        break
                        
            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed for {symbol}: {e}")
                
        if symbol not in historical_data:
            print(f"❌ All attempts failed for {symbol}")
    
    return historical_data

# Example usage
symbols = ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY']
equity_mapping = load_equity_instruments(dhan)
batch_data = get_batch_historical_data(dhan, equity_mapping, symbols, days_back=60)

for symbol, df in batch_data.items():
    print(f"\n{symbol} Data Summary:")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Records: {len(df)}")
    print(f"  Last Close: ₹{df['close'].iloc[-1]:.2f}")
```

### 3. Advanced Historical Analysis

```python
def calculate_technical_indicators(df):
    """Calculate common technical indicators"""
    
    df = df.copy()
    
    # Typical Price (used in CPR calculations)
    df['typical_price'] = (df['open'] + df['high'] + df['close']) / 3
    
    # Simple Moving Averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df['ema_8'] = df['close'].ewm(span=8).mean()
    df['ema_13'] = df['close'].ewm(span=13).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume analysis
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_10']
    
    return df

# Example usage with historical data
symbol = 'RELIANCE'
if symbol in batch_data:
    df_with_indicators = calculate_technical_indicators(batch_data[symbol])
    
    # Show recent data with indicators
    print(f"\n{symbol} Technical Analysis (Last 5 days):")
    recent = df_with_indicators.tail(5)[['timestamp', 'close', 'ema_8', 'ema_13', 'rsi', 'volume_ratio']]
    print(recent.to_string(index=False))
```

---

## Intraday & Real-time Data

### 1. WebSocket Connection Setup

```python
import websocket
import json
import threading
from datetime import datetime

class DhanWebSocket:
    """DHAN WebSocket client for real-time data"""
    
    def __init__(self, client_id, access_token):
        self.client_id = client_id
        self.access_token = access_token
        self.ws = None
        self.subscribed_instruments = {}
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'Ticker':
                # Live market data
                instrument_token = data.get('InstrumentToken')
                last_price = data.get('LastPrice')
                volume = data.get('Volume')
                timestamp = datetime.now()
                
                print(f"🔄 {instrument_token}: ₹{last_price} (Vol: {volume}) at {timestamp}")
                
                # Process real-time data here
                self.process_live_data(instrument_token, data)
                
            elif data.get('type') == 'Acknowledgement':
                print(f"✅ Subscription confirmed: {data}")
                
        except Exception as e:
            print(f"❌ Message processing error: {e}")
    
    def on_error(self, ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("🔌 WebSocket connection closed")
    
    def on_open(self, ws):
        print("🚀 WebSocket connection opened")
        
        # Send authentication
        auth_message = {
            "RequestCode": "Authorization",
            "ClientId": self.client_id,
            "AccessToken": self.access_token
        }
        ws.send(json.dumps(auth_message))
    
    def process_live_data(self, instrument_token, data):
        """Process incoming live market data"""
        
        # Example: Update local cache/database
        live_update = {
            'instrument_token': instrument_token,
            'last_price': data.get('LastPrice'),
            'volume': data.get('Volume'),
            'change': data.get('Change'),
            'timestamp': datetime.now()
        }
        
        # Add your processing logic here
        # e.g., trigger alerts, update charts, etc.
        
    def subscribe_instruments(self, instrument_tokens):
        """Subscribe to live data for specific instruments"""
        
        if not self.ws:
            print("❌ WebSocket not connected")
            return
            
        subscription_message = {
            "RequestCode": "SubscribeInstrumentToken",
            "InstrumentTokens": instrument_tokens
        }
        
        self.ws.send(json.dumps(subscription_message))
        print(f"📡 Subscribed to {len(instrument_tokens)} instruments")
    
    def connect(self):
        """Start WebSocket connection"""
        
        ws_url = "wss://api.dhan.co/v2/websocket"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Start WebSocket in separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        return ws_thread

# Example usage
def start_live_monitoring():
    """Start live market data monitoring"""
    
    # Initialize WebSocket client
    ws_client = DhanWebSocket(
        client_id=os.getenv('DHAN_CLIENT_ID'),
        access_token=os.getenv('DHAN_ACCESS_TOKEN')
    )
    
    # Connect
    ws_thread = ws_client.connect()
    
    # Wait for connection
    import time
    time.sleep(2)
    
    # Subscribe to specific instruments
    # Note: You need instrument tokens, not symbols
    instrument_tokens = [
        "11536",    # RELIANCE
        "1102",     # HDFCBANK
        "2885"      # TCS (example tokens)
    ]
    
    ws_client.subscribe_instruments(instrument_tokens)
    
    return ws_client, ws_thread

# Start live monitoring
# ws_client, ws_thread = start_live_monitoring()
```

### 2. Intraday Data Analysis

```python
def analyze_intraday_patterns(live_data_buffer):
    """Analyze intraday trading patterns"""
    
    if not live_data_buffer:
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(live_data_buffer)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate intraday indicators
    df['price_change'] = df['last_price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Identify significant moves
    significant_moves = df[abs(df['price_change']) > 0.01]  # >1% moves
    
    if len(significant_moves) > 0:
        print("🚨 Significant intraday moves detected:")
        for _, move in significant_moves.iterrows():
            direction = "📈" if move['price_change'] > 0 else "📉"
            print(f"  {direction} {move['price_change']:.2%} at {move['timestamp']}")

def setup_alerts(threshold=0.02):
    """Setup price movement alerts"""
    
    def check_alert(current_price, previous_price, symbol):
        if previous_price == 0:
            return
            
        change_pct = (current_price - previous_price) / previous_price
        
        if abs(change_pct) >= threshold:
            direction = "🚀 SURGE" if change_pct > 0 else "🔻 DROP"
            print(f"⚠️  ALERT: {symbol} {direction} {change_pct:.2%}")
            print(f"   Price: ₹{previous_price:.2f} -> ₹{current_price:.2f}")
    
    return check_alert

# Example usage
alert_fn = setup_alerts(threshold=0.015)  # 1.5% alert threshold
```

---

## Practical Examples

### Example 1: Complete F&O Scanner

```python
def build_fno_scanner(dhan_sdk):
    """Complete F&O scanner implementation"""
    
    print("🚀 Building F&O Scanner...")
    
    # Step 1: Load instruments
    print("📊 Loading instruments...")
    fno_df = load_fno_instruments()
    active_futures = get_active_futures(fno_df)
    equity_mapping = load_equity_instruments(dhan_sdk)
    
    # Step 2: Extract underlying symbols
    underlying_symbols = []
    for _, row in active_futures.iterrows():
        symbol_col = 'SEM_TRADING_SYMBOL' if 'SEM_TRADING_SYMBOL' in row else 'TRADING_SYMBOL'
        future_symbol = row[symbol_col]
        underlying = extract_underlying_symbol(future_symbol)
        underlying_symbols.append(underlying)
    
    unique_underlyings = list(set(underlying_symbols))
    print(f"🎯 Found {len(unique_underlyings)} unique underlying symbols")
    
    # Step 3: Fetch historical data
    print("📡 Fetching historical data...")
    historical_data = get_batch_historical_data(
        dhan_sdk, equity_mapping, unique_underlyings[:10], days_back=30
    )
    
    # Step 4: Technical analysis
    analysis_results = {}
    
    for symbol, df in historical_data.items():
        print(f"🔬 Analyzing {symbol}...")
        
        # Add technical indicators
        df_analyzed = calculate_technical_indicators(df)
        
        # Calculate monthly levels (CPR, Pivots)
        monthly_levels = calculate_monthly_levels(df_analyzed)
        
        # Store results
        analysis_results[symbol] = {
            'data': df_analyzed,
            'levels': monthly_levels,
            'current_price': df_analyzed['close'].iloc[-1],
            'trend': 'BULLISH' if df_analyzed['ema_8'].iloc[-1] > df_analyzed['ema_13'].iloc[-1] else 'BEARISH'
        }
    
    return analysis_results

def calculate_monthly_levels(df):
    """Calculate CPR and Pivot levels"""
    
    if len(df) == 0:
        return {}
    
    # Get latest month data
    latest = df.iloc[-1]
    
    # Monthly OHLC (using last 30 days for approximation)
    monthly_data = df.tail(30)
    
    monthly_high = monthly_data['high'].max()
    monthly_low = monthly_data['low'].min()  
    monthly_close = monthly_data['close'].iloc[-1]
    
    # CPR Calculation (Chartink formula)
    pivot = (monthly_high + monthly_low + monthly_close) / 3
    bc = (monthly_high + monthly_low) / 2  # Bottom Central
    tc = (pivot - bc) + pivot              # Top Central
    
    # CPR Width
    cpr_width = abs(tc - bc) / pivot * 100
    
    # Support/Resistance levels
    r1 = 2 * pivot - monthly_low
    s1 = 2 * pivot - monthly_high
    r2 = pivot + (monthly_high - monthly_low)
    s2 = pivot - (monthly_high - monthly_low)
    r3 = monthly_high + 2 * (pivot - monthly_low)
    s3 = monthly_low - 2 * (monthly_high - pivot)
    
    return {
        'pivot': pivot,
        'bc': bc,
        'tc': tc,
        'cpr_width': cpr_width,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3,
        'is_narrow_cpr': cpr_width < 0.5
    }

# Run the scanner
# results = build_fno_scanner(dhan)
```

### Example 2: Real-time Alert System

```python
def create_alert_system(dhan_sdk):
    """Create comprehensive alert system"""
    
    class TradingAlerts:
        def __init__(self):
            self.price_alerts = {}
            self.volume_alerts = {}
            self.breakout_alerts = {}
            
        def add_price_alert(self, symbol, target_price, alert_type='ABOVE'):
            """Add price-based alert"""
            self.price_alerts[symbol] = {
                'target': target_price,
                'type': alert_type,
                'active': True
            }
            
        def add_volume_alert(self, symbol, volume_multiplier=2.0):
            """Add volume spike alert"""
            self.volume_alerts[symbol] = {
                'multiplier': volume_multiplier,
                'active': True
            }
            
        def check_alerts(self, symbol, current_data):
            """Check all alerts for a symbol"""
            current_price = current_data.get('last_price')
            current_volume = current_data.get('volume', 0)
            
            # Price alerts
            if symbol in self.price_alerts:
                alert = self.price_alerts[symbol]
                if alert['active']:
                    if alert['type'] == 'ABOVE' and current_price >= alert['target']:
                        self.trigger_alert(f"🚀 {symbol} ABOVE ₹{alert['target']:.2f}! Current: ₹{current_price:.2f}")
                        alert['active'] = False
                    elif alert['type'] == 'BELOW' and current_price <= alert['target']:
                        self.trigger_alert(f"🔻 {symbol} BELOW ₹{alert['target']:.2f}! Current: ₹{current_price:.2f}")
                        alert['active'] = False
            
            # Volume alerts
            if symbol in self.volume_alerts:
                alert = self.volume_alerts[symbol]
                if alert['active'] and current_volume > 0:
                    # Compare with average volume (implement your logic)
                    avg_volume = self.get_average_volume(symbol)
                    if current_volume >= avg_volume * alert['multiplier']:
                        self.trigger_alert(f"📊 {symbol} VOLUME SPIKE! Current: {current_volume:,.0f} (Avg: {avg_volume:,.0f})")
        
        def trigger_alert(self, message):
            """Trigger an alert (customize as needed)"""
            print(f"⚠️  ALERT: {message}")
            # Add sound, email, SMS, etc.
            
        def get_average_volume(self, symbol):
            """Get average volume for symbol (implement your logic)"""
            return 100000  # Placeholder
    
    return TradingAlerts()

# Usage example
# alerts = create_alert_system(dhan)
# alerts.add_price_alert('RELIANCE', 3100, 'ABOVE')
# alerts.add_volume_alert('HDFCBANK', 2.5)
```

---

## Common Issues & Troubleshooting

### 1. Authentication Issues

**Problem**: "Authentication failed" or "Invalid token"
```python
# Solution: Verify credentials
def test_authentication(client_id, access_token):
    """Test DHAN API authentication"""
    
    dhan = dhanhq(client_id, access_token)
    
    try:
        # Simple API call to test
        positions = dhan.get_positions()
        print("✅ Authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Check your client_id and access_token")
        return False

# Test your credentials
test_authentication("your_client_id", "your_access_token")
```

### 2. Historical Data Issues

**Problem**: "No historical data" or 404 errors
```python
# Solution: Use only SDK method with correct parameters
def debug_historical_data(dhan_sdk, security_id):
    """Debug historical data issues"""
    
    print(f"🔍 Debugging historical data for security_id: {security_id}")
    
    # Test different parameter combinations
    test_configs = [
        {"exchange_segment": "NSE_EQ", "instrument_type": "EQUITY"},
        {"exchange_segment": "NSE_INDEX", "instrument_type": "INDEX"},
        {"exchange_segment": "NSE", "instrument_type": "EQ"}
    ]
    
    for i, config in enumerate(test_configs, 1):
        try:
            print(f"  Test {i}: {config}")
            
            response = dhan_sdk.historical_daily_data(
                security_id=str(security_id),
                exchange_segment=config["exchange_segment"],
                instrument_type=config["instrument_type"],
                from_date="2024-08-01",
                to_date="2024-09-01"
            )
            
            if response.get('status') == 'success':
                data = response.get('data', {})
                records = len(data.get('timestamp', []))
                print(f"    ✅ SUCCESS: {records} records retrieved")
                return config
            else:
                print(f"    ❌ FAILED: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
    
    print("❌ All configurations failed")
    return None

# Debug example
# working_config = debug_historical_data(dhan, "2885")  # RELIANCE
```

### 3. Symbol Mapping Issues

**Problem**: "Symbol not found" or incorrect security IDs
```python
def debug_symbol_mapping(dhan_sdk, target_symbol):
    """Debug symbol to security_id mapping"""
    
    print(f"🔍 Searching for symbol: {target_symbol}")
    
    try:
        # Get all instruments
        response = dhan_sdk.get_instruments('NSE')
        
        if response['status'] == 'success':
            instruments = response['data']
            
            matches = []
            for instrument in instruments:
                trading_symbol = instrument.get('SEM_TRADING_SYMBOL', '')
                
                if target_symbol.upper() in trading_symbol.upper():
                    matches.append({
                        'symbol': trading_symbol,
                        'security_id': instrument.get('SEM_SMST_SECURITY_ID'),
                        'instrument': instrument.get('SEM_INSTRUMENT_NAME'),
                        'segment': instrument.get('SEM_SEGMENT')
                    })
            
            print(f"📊 Found {len(matches)} matches:")
            for match in matches[:10]:  # Show first 10
                print(f"  {match['symbol']} -> ID: {match['security_id']} ({match['instrument']})")
                
            return matches
            
    except Exception as e:
        print(f"❌ Error searching symbols: {e}")
        return []

# Debug example
# matches = debug_symbol_mapping(dhan, "RELIANCE")
```

### 4. WebSocket Connection Issues

**Problem**: WebSocket disconnections or no data
```python
def debug_websocket_connection():
    """Debug WebSocket connectivity"""
    
    import websocket
    
    def on_open(ws):
        print("✅ WebSocket connected successfully")
        
    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")
        
    def on_close(ws, close_status_code, close_msg):
        print(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
    
    try:
        ws_url = "wss://api.dhan.co/v2/websocket"
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_error=on_error,
            on_close=on_close
        )
        
        # Test connection
        import threading
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait and check
        import time
        time.sleep(3)
        
        if ws.sock and ws.sock.connected:
            print("✅ WebSocket test successful")
        else:
            print("❌ WebSocket connection failed")
            
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")

# Test WebSocket
# debug_websocket_connection()
```

---

## Best Practices

### 1. Rate Limiting & Performance

```python
import time
from functools import wraps

def rate_limit(calls_per_second=1):
    """Rate limiting decorator"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Apply rate limiting to API calls
@rate_limit(calls_per_second=2)  # Max 2 calls per second
def get_historical_with_limit(dhan_sdk, security_id, days_back=30):
    return get_historical_data(dhan_sdk, security_id, days_back)
```

### 2. Error Handling & Retry Logic

```python
def retry_on_failure(max_retries=3, delay=1.0):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    wait_time = delay * (2 ** attempt)
                    print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"🔄 Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator

# Apply retry logic
@retry_on_failure(max_retries=3, delay=1.0)
@rate_limit(calls_per_second=2)
def robust_historical_fetch(dhan_sdk, security_id, days_back=30):
    return get_historical_data(dhan_sdk, security_id, days_back)
```

### 3. Data Caching

```python
import json
import os
from datetime import datetime, timedelta

class DataCache:
    """Simple file-based cache for DHAN data"""
    
    def __init__(self, cache_dir="dhan_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def set(self, key, data, expire_hours=24):
        """Cache data with expiration"""
        cache_data = {
            'data': data,
            'cached_at': datetime.now().isoformat(),
            'expire_hours': expire_hours
        }
        
        with open(self._get_cache_path(key), 'w') as f:
            json.dump(cache_data, f, default=str)
    
    def get(self, key):
        """Retrieve cached data if not expired"""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            expire_hours = cache_data.get('expire_hours', 24)
            
            if datetime.now() > cached_at + timedelta(hours=expire_hours):
                return None  # Expired
            
            return cache_data['data']
            
        except Exception as e:
            print(f"⚠️  Cache read error: {e}")
            return None

# Usage with caching
cache = DataCache()

def get_cached_historical_data(dhan_sdk, security_id, days_back=30):
    """Get historical data with caching"""
    
    cache_key = f"historical_{security_id}_{days_back}"
    
    # Check cache first
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        print(f"📂 Using cached data for {security_id}")
        return pd.DataFrame(cached_data)
    
    # Fetch fresh data
    print(f"📡 Fetching fresh data for {security_id}")
    df = get_historical_data(dhan_sdk, security_id, days_back)
    
    if len(df) > 0:
        # Cache the data
        cache.set(cache_key, df.to_dict('records'), expire_hours=4)
    
    return df
```

### 4. Configuration Management

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class DhanConfig:
    client_id: str
    access_token: str
    websocket_url: str = "wss://api.dhan.co/v2/websocket"
    api_base_url: str = "https://api.dhan.co"
    rate_limit_per_second: int = 2
    max_retries: int = 3
    cache_expire_hours: int = 4
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            client_id=os.getenv('DHAN_CLIENT_ID'),
            access_token=os.getenv('DHAN_ACCESS_TOKEN'),
            rate_limit_per_second=int(os.getenv('DHAN_RATE_LIMIT', '2')),
            max_retries=int(os.getenv('DHAN_MAX_RETRIES', '3')),
            cache_expire_hours=int(os.getenv('DHAN_CACHE_HOURS', '4'))
        )
    
    def validate(self):
        """Validate configuration"""
        if not self.client_id:
            raise ValueError("DHAN_CLIENT_ID is required")
        if not self.access_token:
            raise ValueError("DHAN_ACCESS_TOKEN is required")
        
        print("✅ DHAN configuration validated")

# Usage
config = DhanConfig.from_env()
config.validate()
```

---

## Production Deployment Tips

### 1. Environment Setup
```bash
# .env file
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_jwt_token
DHAN_RATE_LIMIT=2
DHAN_CACHE_HOURS=4
```

### 2. Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### 3. Monitoring & Logging
```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dhan_api_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_api_call(func):
    """Log API calls for monitoring"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"API call {func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"API call {func.__name__} failed: {e}")
            raise
    return wrapper
```

---

## Conclusion

This tutorial covers the essential aspects of working with DHAN API for both historical and intraday data. Key takeaways:

1. **Always use dhanhq SDK** for historical data - REST endpoints don't work
2. **Proper symbol mapping** is crucial for data fetching
3. **Rate limiting and error handling** are essential for production use
4. **WebSocket integration** enables real-time analysis
5. **Caching strategies** improve performance significantly

The code examples provided are production-tested and should work directly with your DHAN credentials. Start with basic historical data fetching, then gradually add real-time capabilities and advanced analysis features.

Happy trading! 🚀📈

---

**Disclaimer**: This tutorial is for educational purposes. Always test thoroughly in a development environment before deploying to production. Trading involves financial risk - please trade responsibly.