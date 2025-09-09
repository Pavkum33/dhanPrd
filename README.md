# Dhan F&O Scanner - Professional Trading Scanner

A production-ready F&O (Futures & Options) scanner for Dhan trading platform with real-time WebSocket connectivity, technical analysis, and professional web dashboard.

## Features

### Core Scanner Features
- **Automatic Instrument Discovery**: Downloads and parses Dhan instrument master CSV
- **Real-time WebSocket v2**: Connects to Dhan WebSocket for live market data
- **Technical Analysis**: 
  - Resistance breakout detection using Typical Price
  - EMA crossover (8/13 period)
  - Volume analysis with configurable factors
  - 1-minute candle aggregation
- **Multi-batch Processing**: Handles large symbol lists with configurable batch sizes
- **Robust Reconnection**: Automatic reconnect with exponential backoff
- **Alert System**: SQLite storage + optional Telegram notifications

### Professional Web Dashboard
- **Real-time Updates**: Live data streaming via Socket.IO
- **Dark Theme UI**: Professional trading interface
- **Multiple Views**:
  - Live Scanner with filters
  - Alert History
  - Watchlist Management
  - Settings Configuration
- **Interactive Charts**: Live breakout visualization
- **Export Functionality**: CSV export for alerts

## Security Requirements

### Environment Variables (REQUIRED)
```bash
# Dhan API Credentials (MANDATORY)
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here

# Optional Telegram Alerts
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**NEVER commit credentials to source control!**

## Quick Start

### 1. Local Development

```bash
# Clone repository
git clone <your-repo>
cd dhan_demo

# Create environment file
echo "DHAN_CLIENT_ID=your_id" > .env
echo "DHAN_ACCESS_TOKEN=your_token" >> .env

# Install dependencies
pip install -r requirements.txt

# Run scanner
python scanner.py --config config.json

# In another terminal, run web UI
python app.py

# Open browser to http://localhost:5000
```

### 2. Docker Deployment

```bash
# Build image
docker build -t fno-scanner:latest .

# Run with environment variables
docker run -d \
  --name fno-scanner \
  -p 5000:5000 \
  -e DHAN_CLIENT_ID=$DHAN_CLIENT_ID \
  -e DHAN_ACCESS_TOKEN=$DHAN_ACCESS_TOKEN \
  -e TELEGRAM_TOKEN=$TELEGRAM_TOKEN \
  -e TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/alerts.db:/app/alerts.db \
  fno-scanner:latest
```

### 3. Docker Compose

```bash
# Create .env file with credentials
cat > .env << EOF
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_token
TELEGRAM_TOKEN=optional_bot_token
TELEGRAM_CHAT_ID=optional_chat_id
EOF

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Production Deployment (systemd)

```bash
# Copy files to server
scp -r * ubuntu@your-server:/home/ubuntu/fno-scanner/

# SSH to server
ssh ubuntu@your-server

# Create credentials file (root only)
sudo bash -c 'cat > /etc/default/dhan-scanner << EOF
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_token
TELEGRAM_TOKEN=optional_token
TELEGRAM_CHAT_ID=optional_chat_id
EOF'

sudo chmod 600 /etc/default/dhan-scanner

# Install service
sudo cp dhan-scanner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dhan-scanner
sudo systemctl start dhan-scanner

# Check status
sudo systemctl status dhan-scanner
sudo journalctl -fu dhan-scanner
```

## Configuration (config.json)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hist_days` | 60 | Days of historical data to fetch |
| `lookback` | 50 | Lookback period for resistance calculation |
| `ema_short` | 8 | Short EMA period |
| `ema_long` | 13 | Long EMA period |
| `volume_factor` | 0.5 | Minimum volume multiplier for alerts |
| `price_threshold` | 50 | Minimum price for consideration |
| `batch_size` | 200 | WebSocket subscription batch size |
| `fetch_concurrency` | 6 | Concurrent historical fetch limit |
| `request_code` | 15 | WebSocket v2 request code (15=Ticker) |

## Testing Strategy

### Phase 1: Small Scale Test
```bash
# Edit config.json
{
  "batch_size": 5,
  "hist_days": 10
}

# Run locally
export DHAN_CLIENT_ID=xxx
export DHAN_ACCESS_TOKEN=xxx
python scanner.py --config config.json
```

### Phase 2: Validate Data
- Check logs for WebSocket connection
- Verify instrument master download
- Monitor for binary frame warnings
- Confirm 1-minute candle aggregation

### Phase 3: Scale Up
```bash
# Increase batch_size gradually
"batch_size": 50  # Test
"batch_size": 200 # Production
"batch_size": 400 # If network allows
```

## Extending for Other Scans

The scanner is designed to be extensible. To add new scanning strategies:

1. **Create Strategy Module** (`strategies/your_strategy.py`):
```python
class YourStrategy:
    def evaluate(self, candle_data):
        # Your logic here
        return signal
```

2. **Register in Scanner**:
```python
# In scanner.py
from strategies.your_strategy import YourStrategy
self.strategies['your_strategy'] = YourStrategy()
```

3. **Add UI Tab** (in `dashboard.html`):
```html
<a href="#" class="nav-item" data-tab="your-strategy">
    <span class="nav-icon">📊</span>
    <span class="nav-text">Your Strategy</span>
</a>
```

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Dhan API      │────▶│   Scanner    │────▶│   SQLite    │
│   WebSocket     │     │   (scanner.py)│     │   Database  │
└─────────────────┘     └──────────────┘     └─────────────┘
                               │                      │
                               ▼                      │
                        ┌──────────────┐             │
                        │   Flask App  │◀────────────┘
                        │   (app.py)   │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   Web UI     │
                        │  (Browser)   │
                        └──────────────┘
```

## Troubleshooting

### Binary WebSocket Frames
If you see hex output in logs:
```
Received binary frame len=XXX: 0a1b2c3d...
```
This means Dhan is sending binary-encoded data. The scanner logs but doesn't crash.

### Connection Issues
- Verify credentials are correct
- Check network connectivity to `wss://api-feed.dhan.co`
- Ensure system time is synchronized (for auth)

### No Historical Data
- The scanner works without historical data (SDK optional)
- Install `dhanhq` package for full historical support
- Or implement REST API calls if you have the endpoints

## Performance Tuning

- **batch_size**: Start at 200, increase if stable
- **fetch_concurrency**: Reduce if hitting rate limits
- **Database**: Use SSD for alerts.db in production
- **Memory**: ~500MB for 500 symbols
- **CPU**: Minimal usage, mostly I/O bound

## Support & Monitoring

### Health Check
```bash
curl http://localhost:5000/api/status
```

### Logs
- Scanner: `fno_breakout.log` (rotating daily, 14 day retention)
- Web UI: Standard output
- Systemd: `journalctl -u dhan-scanner`

### Metrics Available
- Active symbols count
- Total alerts generated
- Uptime counter
- Connection status
- Market open/closed indicator

## License

MIT License - See LICENSE file

## Disclaimer

This software is for educational purposes only. Trading involves risk. Always verify signals and use proper risk management.

## Contributing

Pull requests welcome! Please ensure:
1. No hardcoded credentials
2. Tests pass
3. Code follows existing patterns
4. Documentation updated