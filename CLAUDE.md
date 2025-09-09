# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready F&O (Futures & Options) scanner for Dhan trading platform with historical data analysis and professional web dashboard. The system operates as dual-mode architecture:

1. **Integrated Web Application** (`app.py`) - Primary mode: Flask-SocketIO web server with built-in historical analysis engine and background scanner thread
2. **Standalone Scanner Engine** (`scanner.py`) - Legacy/reference: Real-time market data processor using Dhan WebSocket v2 for advanced users

## Current Architecture

The system primarily operates as a single Flask application with integrated analysis:

```
F&O Instrument CSV → Active Futures Filter → Extract Underlying Symbols
                                    ↓
NSE Equity Master API → Symbol→SecurityId Mapping → Historical Equity Data
                                    ↓
Breakout Analysis ← Background Thread → Progress Updates → Flask-SocketIO
                                    ↓
Professional Web UI (Dashboard)
```

**Primary Components:**
- **DhanHistoricalFetcher**: Fetches active F&O futures, resolves underlying equity securityIds, retrieves historical data
- **BreakoutAnalyzer**: Implements Chartink-style resistance breakout analysis with 5-condition logic
- **Flask-SocketIO**: Real-time progress updates and WebSocket communication to browser
- **Professional Dashboard**: Dark theme trading interface with multiple tabs and live updates

**Key Insight**: The system analyzes **underlying equity data** to predict **future contract behavior**, since Dhan's API doesn't provide historical data for future contracts directly.

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Primary Mode: Integrated web application with historical analysis
python app.py
# Demo mode on http://localhost:5000 (no credentials needed for testing UI)

# With credentials for live historical data
export DHAN_CLIENT_ID=your_id
export DHAN_ACCESS_TOKEN=your_token
python app.py

# Legacy Mode: Standalone real-time scanner (advanced users)
python scanner.py --config config.json
```

### Production Commands
```bash
# Docker build and run
docker build -t fno-scanner:latest .
docker run -p 5000:5000 -e DHAN_CLIENT_ID=xxx -e DHAN_ACCESS_TOKEN=xxx fno-scanner:latest

# Production deployment (systemd)
sudo systemctl start dhan-scanner
sudo journalctl -u dhan-scanner -f

# Smart startup script (handles both local and container)
./run.sh
```

### Testing Historical Data Analysis
```bash
# Test the core functionality by clicking "Fetch Historical" in the web UI
# Or test API endpoint directly:
curl http://localhost:5000/api/debug/instruments

# Check app health
curl http://localhost:5000/api/status
```

### Local Testing Flow
1. **Demo Mode**: Run `python app.py` without credentials → Web UI loads with demo message
2. **Live Mode**: Set credentials → Click "Fetch Historical" → See real F&O analysis
3. **Debug Mode**: Use `/api/debug/instruments` to inspect instrument filtering logic

### Railway Deployment (Primary)
```bash
# Current production setup uses Railway with GitHub integration
# Procfile: web: python app.py
# railway.json: Nixpacks builder with health checks

# Required environment variables in Railway:
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_access_token

# Deployment process:
git push origin main  # Triggers automatic Railway deployment
```

### Alternative Deployment Methods
```bash
# Docker Compose (local production testing)
docker-compose up -d
docker-compose logs -f

# systemd Service (Linux servers)
sudo cp dhan-scanner.service /etc/systemd/system/
sudo systemctl enable dhan-scanner
sudo systemctl start dhan-scanner
```

## Key Components & Architecture Details

### Core Classes (app.py)

#### DhanHistoricalFetcher
- **Purpose**: Fetches and filters active F&O instruments, retrieves historical data for underlying equities
- **Key Methods**: 
  - `get_active_fno_futures()`: Filters for current month FUTSTK/FUTIDX from Dhan CSV
  - `load_equity_instruments()`: Loads NSE equity master to create symbol → securityId mapping
  - `get_historical_data_for_underlying()`: Fetches underlying equity data using numeric securityIds
  - `extract_underlying_symbol()`: Converts future symbols (RELIANCE-Sep2025-FUT → RELIANCE)
- **Data Sources**: 
  - `api-scrip-master.csv` for active F&O filtering
  - `/v2/instruments/master` API for equity securityId resolution

#### BreakoutAnalyzer  
- **Purpose**: Implements Chartink-style resistance breakout analysis
- **Core Logic** (`calculate_technical_indicators()`):
  - Typical Price: `(open + high + close) / 3`
  - Resistance: `ceil(max(typical_price))` over lookback period  
  - **5-Condition Breakout Logic** (ALL must be true):
    1. `close > resistance` (Resistance breakout)
    2. `EMA8 > EMA13` (EMA crossover)
    3. `volume >= prev_day_volume * volume_factor` (Volume spike)
    4. `close > price_threshold` (Price filter)
    5. `close > open` (Bullish candle)

### Critical Implementation Details

#### F&O Instrument Filtering Logic
Located in `DhanHistoricalFetcher.get_active_fno_futures()`:

```python
# Step 1: Filter active NSE F&O Futures
nse_mask = df[exch_col].str.upper().isin(['NSE', 'NSE_FO', 'NSE_FUT'])
futstk_mask = df[instrument_col].str.upper().isin(['FUTSTK', 'FUTIDX']) 
expiry_mask = df[expiry_col].notnull()

# Step 2: Current month filtering (avoids expired contracts)
current_month = today.month
current_year = today.year
active_futures = fno_fut[
    (fno_fut[expiry_col].dt.month == current_month) & 
    (fno_fut[expiry_col].dt.year == current_year)
]
```

**Why This Matters**: Previous versions had "No historical data" issues because they used:
- Wrong CSV (`-detailed.csv` vs standard CSV)  
- Mixed instrument types (currencies, commodities)
- Expired contracts

Current implementation filters to **active, current-month stock/index futures only**.

#### Historical Data API Parameters
**Critical for avoiding "No historical data" errors**:

```python
# Correct API parameters (app.py:176-182)
result = self.sdk.historical_daily_data(
    securityId=int(security_id),  # MUST be numeric (not string)
    exchangeSegment=2,            # NSE Futures = 2 (not string "NSE_FUT")  
    instrumentType="FUTSTK",      # Stock futures (not generic "FUTURE")
    fromDate=from_date.strftime("%Y-%m-%d"),
    toDate=to_date.strftime("%Y-%m-%d")
)
```

**Common Mistakes**:
- Using symbol strings instead of numeric security IDs
- Wrong exchangeSegment format ("NSE_FUT" vs 2)
- Generic instrumentType ("FUTURE" vs "FUTSTK")

### Web Dashboard Architecture

**Files Structure**:
- `templates/dashboard.html`: Single-page application with multiple tabs
- `static/css/dashboard.css`: Professional dark theme trading interface  
- `static/js/dashboard.js`: Socket.IO client, real-time progress updates

**Key Features**:
- **Scanner Tab**: Main historical analysis interface with "Fetch Historical" button
- **Progress Tracking**: Real-time WebSocket updates during analysis
- **Debug Endpoint**: `/api/debug/instruments` for troubleshooting instrument filtering

## Common Development Patterns

### Adding New Analysis Logic
1. **Extend BreakoutAnalyzer** in `app.py`:
   ```python
   def calculate_new_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
       # Your logic here
       df['new_signal'] = your_calculation
       return df
   ```

2. **Update Progress Tracking**: Use `emit_progress()` function for real-time updates

3. **Add UI Elements**: Extend dashboard.html with new tabs or data displays

### Multi-timeframe Support (Configuration Available)
The codebase includes `scanner_config.json` with predefined symbols and timeframes for extending analysis:
```json
{
  "symbols": [
    {"underlying": "RELIANCE"}, {"underlying": "HDFCBANK"}, {"underlying": "INFY"}, 
    {"underlying": "TCS"}, {"underlying": "NIFTY 50"}, {"underlying": "BANKNIFTY"}
  ],
  "timeframes": ["1d", "15m", "5m"],
  "analysis": {
    "lookback_period": 50, "ema_short": 8, "ema_long": 13,
    "volume_factor": 0.5, "price_threshold": 50
  }
}
```

### Adding New Strategy Classes
The scanner supports extensible strategies by following the pattern in `scanner.py`:
```python
# In scanner.py: FNOEngine.__init__() initializes SymbolState for each security
# Each state tracks: typical_price_history, ema_short/long, prev_day_volume
# Evaluation happens in: close_and_evaluate() with 5-condition breakout logic
```

## Security Notes

- **Never commit credentials**: Use environment variables only
- **Credentials validation**: App runs in demo mode without credentials
- **Safe deployment**: Railway/Docker handle environment variable injection securely

## Performance Characteristics

- **Memory**: ~500MB for 500 symbols
- **CPU**: I/O bound, minimal computation
- **Network**: Handles multiple WebSocket batches with exponential backoff reconnection
- **Database**: SQLite with connection pooling, rotating logs (14-day retention)

## Common Issues & Troubleshooting

### "No F&O instruments found" Error
**Root Cause**: Column mapping mismatch between CSV formats
**Solution**: The `get_active_fno_futures()` method handles multiple CSV column formats:
- Standard CSV: `SEM_EXM_EXCH_ID`, `SEM_INSTRUMENT_NAME`, `SEM_TRADING_SYMBOL`
- Detailed CSV: `EXCH_ID`, `INSTRUMENT_TYPE`, `SYMBOL_NAME`

### "No historical data available" Error  
**Root Cause**: Dhan's historical API requires numeric `securityId`, NOT symbol names
- ❌ Wrong: `"symbol": "SAMMAANCAP"` → No data  
- ✅ Correct: `"securityId": "12345"` → Data available

**Solutions Applied**:
1. **Load NSE equity instrument master** via `/v2/instruments/master`
2. **Create symbol → securityId mapping** (SAMMAANCAP → 12345)
3. **Resolve securityId** for each underlying symbol before API calls
4. **Use numeric securityId** in historical API payload instead of symbol names

**Implementation Pattern** (following user's working sample):
```python
# 1. Load equity master
equity_mapping = await fetcher.load_equity_instruments()

# 2. Resolve securityId  
underlying = "SAMMAANCAP" 
security_id = equity_mapping.get(underlying)  # → 12345

# 3. API call with securityId
payload = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ"}
```

### Missing dhanhq SDK
**Symptom**: "dhanhq SDK not available - using REST API fallback"
**Impact**: System works but uses REST API instead of SDK
**Solution**: `pip install dhanhq` (may require specific package source)

### Railway Deployment
**Current Setup**: Simple Python execution via `Procfile: web: python app.py`
**Requirements**: Set `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` in Railway environment
**Fallback**: App runs in demo mode without credentials

## Deployment Architecture & Files

### Key Configuration Files
- `Procfile`: `web: python app.py` (Railway deployment)
- `railway.json`: Nixpacks builder with health checks and restart policies
- `Dockerfile`: Production container with systemd support
- `dhan-scanner.service`: systemd service file for Linux servers
- `run.sh`: Smart startup script (detects Docker vs local environment)
- `DEPLOYMENT.md`: Complete production deployment guide
- `TESTING.md`: Testing checklist and Railway deployment validation

### Important Development Files  
- `config.json`: Scanner parameters (batch_size, lookback_period, API endpoints)
- `scanner_config.json`: Multi-timeframe symbol configuration
- `requirements.txt`: Python dependencies (aiohttp, websockets, pandas, flask-socketio)

## Extensibility for New Strategies

The architecture supports multiple extension patterns:

1. **Web Dashboard Extensions** (Primary): Extend BreakoutAnalyzer in `app.py` for historical analysis
2. **Real-time Scanner Extensions** (Advanced): Extend SymbolState evaluation in `scanner.py` for live WebSocket data
3. **UI Extensions**: Add new tabs in `templates/dashboard.html` with Socket.IO real-time updates
4. **API Extensions**: Add new endpoints in `app.py` following `/api/historical/fetch` pattern