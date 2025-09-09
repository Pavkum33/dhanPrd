# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready F&O (Futures & Options) scanner for Dhan trading platform with real-time WebSocket connectivity and a professional web dashboard. The system consists of two main components that work together:

1. **Scanner Engine** (`scanner.py`) - Real-time market data processor using Dhan WebSocket v2
2. **Web Dashboard** (`app.py`) - Flask-SocketIO web server providing real-time UI

## Architecture

The system follows a dual-process architecture:

```
Dhan WebSocket API → Scanner Engine → SQLite Database
                                  ↓
Web Dashboard ← Socket.IO ← Background Thread
```

- **scanner.py**: Handles WebSocket connections, processes ticks into 1-minute candles, evaluates breakout conditions
- **app.py**: Serves web UI, manages WebSocket connections to browsers, can run scanner in background thread
- **SQLite Database**: Persistent storage for alerts and historical data
- **Real-time Communication**: Flask-SocketIO for browser updates

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run scanner only (background process)
python scanner.py --config config.json

# Run web dashboard only (with demo data)
python app.py

# Run both together (production mode)
export DHAN_CLIENT_ID=your_id
export DHAN_ACCESS_TOKEN=your_token
python app.py  # Automatically starts scanner in background
```

### Testing
```bash
# Test with small batch size (edit config.json first)
{
  "batch_size": 5,
  "hist_days": 10
}

# Check scanner health
curl http://localhost:5000/api/status

# View logs
tail -f fno_breakout.log
```

### Production Deployment

#### Docker
```bash
# Build and run
docker build -t fno-scanner:latest .
docker-compose up -d

# View logs
docker-compose logs -f
```

#### Railway
```bash
# Deploy uses gunicorn with eventlet worker
gunicorn -c gunicorn.conf.py app:app

# Environment variables required:
# DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
```

## Key Components

### Scanner Engine Classes
- **InstrumentLoader**: Downloads Dhan instrument master, extracts F&O futures automatically
- **HistoricalFetcher**: Fetches pre-market historical data (optional dhanhq SDK support)
- **SymbolState**: Tracks per-symbol technical indicators (EMA, resistance, volume)
- **FNOEngine**: Main WebSocket handler, processes ticks, evaluates breakout conditions

### Technical Analysis Logic
Located in `SymbolState.close_and_evaluate()`:
- Resistance breakout: `close > ceil(max(typical_price_history))`
- EMA crossover: `EMA8 > EMA13`
- Volume spike: `volume >= prev_day_volume * volume_factor`
- Bullish candle: `close > open`

### Configuration System
- **config.json**: Non-secret runtime parameters
- **Environment Variables**: Credentials (DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
- **Key Parameters**: batch_size (WebSocket subscriptions), lookback (resistance calculation), volume_factor

### Web Dashboard Features
- **Real-time Scanner View**: Live F&O data with filters
- **Alert Management**: Historical alerts with export functionality
- **Watchlist**: Symbol tracking and monitoring
- **Settings**: Runtime configuration via UI

## Extensibility for New Strategies

The codebase is designed for easy extension:

1. **Add Strategy Class**: Create new evaluation logic in scanner.py
2. **Modify SymbolState**: Add new technical indicators
3. **Update UI**: Add new tab in templates/dashboard.html
4. **Database Schema**: Extend alerts table with strategy field

## Security Notes

- **Never commit credentials**: Use environment variables only
- **Credentials validation**: App runs in demo mode without credentials
- **Safe deployment**: Railway/Docker handle environment variable injection securely

## Performance Characteristics

- **Memory**: ~500MB for 500 symbols
- **CPU**: I/O bound, minimal computation
- **Network**: Handles multiple WebSocket batches with exponential backoff reconnection
- **Database**: SQLite with connection pooling, rotating logs (14-day retention)

## Common Issues

### WebSocket Connection
- Binary frames logged but handled gracefully
- Automatic reconnection with exponential backoff
- Batch processing prevents rate limiting

### Deployment
- Production uses Gunicorn + Eventlet worker for WebSocket support
- Health check endpoint: `/api/status`
- Graceful degradation: Web UI works without scanner credentials