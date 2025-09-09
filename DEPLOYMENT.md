# Production Deployment Guide

This guide shows how to deploy your Dhan F&O Scanner with Docker and systemd for 24/7 operation.

## 📂 Project Structure

```
dhan_demo/
├── app.py                    # Flask web dashboard with integrated historical analysis
├── scanner.py               # Standalone WebSocket v2 scanner (legacy/reference)
├── config.json              # Scanner configuration
├── requirements.txt         # Python dependencies  
├── Dockerfile              # Production container
├── dhan-scanner.service    # Systemd service file
├── run.sh                  # Startup script
├── static/                 # Web UI assets
├── templates/              # HTML templates
└── DEPLOYMENT.md           # This file
```

## 🚀 Quick Start

### 1. Build Docker Image
```bash
cd dhan_demo
docker build -t fno-scanner:latest .
```

### 2. Test Locally
```bash
# Set your credentials
export DHAN_CLIENT_ID="your_client_id"
export DHAN_ACCESS_TOKEN="your_access_token"

# Run container
docker run --rm -p 5000:5000 \
    -v $(pwd)/config.json:/app/config.json:ro \
    -v $(pwd)/data:/app/data \
    -e DHAN_CLIENT_ID=$DHAN_CLIENT_ID \
    -e DHAN_ACCESS_TOKEN=$DHAN_ACCESS_TOKEN \
    fno-scanner:latest

# Access dashboard at http://localhost:5000
```

### 3. Production Deployment

#### Install on Ubuntu Server
```bash
# 1. Create deployment directory
sudo mkdir -p /opt/dhan_scanner/{data,logs}
sudo cp -r * /opt/dhan_scanner/
cd /opt/dhan_scanner

# 2. Build image
sudo docker build -t fno-scanner:latest .

# 3. Create environment file
sudo tee /etc/default/dhan-scanner << 'EOF'
DHAN_CLIENT_ID=your_actual_client_id
DHAN_ACCESS_TOKEN=your_actual_access_token
TELEGRAM_TOKEN=your_telegram_bot_token_optional
TELEGRAM_CHAT_ID=your_chat_id_optional
EOF

# 4. Install systemd service
sudo cp dhan-scanner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dhan-scanner
sudo systemctl start dhan-scanner

# 5. Check status
sudo systemctl status dhan-scanner
sudo journalctl -u dhan-scanner -f
```

## 🔧 Configuration

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `DHAN_CLIENT_ID` | Yes | Your Dhan client ID |
| `DHAN_ACCESS_TOKEN` | Yes | Your Dhan access token |
| `TELEGRAM_TOKEN` | No | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | No | Telegram chat ID for alerts |

### config.json Parameters
```json
{
  "hist_days": 60,           // Days of historical data to fetch
  "lookback": 50,            // Lookback period for resistance calculation  
  "ema_short": 8,            // Short EMA period
  "ema_long": 13,            // Long EMA period
  "volume_factor": 0.5,      // Minimum volume factor for breakout
  "price_threshold": 50,     // Minimum price threshold
  "batch_size": 200,         // WebSocket subscription batch size
  "request_code": 15         // Dhan WebSocket v2 request code
}
```

## 📊 System Architecture

### Integrated Mode (Primary)
```
Flask App (app.py) --> Historical Analysis --> Web Dashboard
     ↓
Background Thread --> Real-time Scanner --> WebSocket Updates
```

### Standalone Mode (Reference)  
```
scanner.py --> WebSocket v2 --> Alerts --> SQLite + Telegram
```

## 🔍 Monitoring & Logs

### Service Status
```bash
# Check service status
sudo systemctl status dhan-scanner

# View real-time logs
sudo journalctl -u dhan-scanner -f

# Restart service
sudo systemctl restart dhan-scanner
```

### Log Files
- **Service Logs**: `journalctl -u dhan-scanner`
- **Application Logs**: `/opt/dhan_scanner/logs/fno_breakout.log`
- **SQLite Database**: `/opt/dhan_scanner/data/alerts.db`

### Web Dashboard
- **URL**: `http://your-server:5000`
- **Features**: Real-time historical analysis, progress tracking, breakout alerts
- **API Endpoints**: 
  - `/api/status` - Service status
  - `/api/historical/fetch` - Trigger historical analysis
  - `/api/alerts` - Recent alerts

## 🛠️ Troubleshooting

### Common Issues

#### 1. "No F&O instruments found"
```bash
# Check instrument CSV download
curl https://images.dhan.co/api-data/api-scrip-master-detailed.csv | head -5

# Debug API endpoint
curl http://localhost:5000/api/debug/instruments
```

#### 2. "No historical data available"
- Ensure DHAN credentials are valid
- Check if securities have underlying equity mapping
- Verify API rate limits aren't exceeded

#### 3. WebSocket Connection Issues
- Check Dhan API status
- Verify access token hasn't expired
- Review network connectivity

### Performance Tuning
- **Memory**: Default limit 1GB (adjust in systemd service)
- **CPU**: Default 200% quota (2 cores)
- **Batch Size**: Reduce if WebSocket subscriptions fail
- **Concurrency**: Lower `fetch_concurrency` if rate limited

## 🔄 Updates & Maintenance

### Update Application
```bash
cd /opt/dhan_scanner
git pull  # or copy new files
sudo docker build -t fno-scanner:latest .
sudo systemctl restart dhan-scanner
```

### Backup Data
```bash
# Backup alerts database
cp /opt/dhan_scanner/data/alerts.db ~/alerts_backup_$(date +%Y%m%d).db

# Backup configuration  
cp /opt/dhan_scanner/config.json ~/config_backup.json
```

### Log Rotation
Logs rotate automatically (14-day retention). Manual cleanup:
```bash
sudo find /opt/dhan_scanner/logs -name "*.log.*" -mtime +14 -delete
```

## 🚨 Security Notes

- Never commit credentials to version control
- Use environment variables for all secrets
- Run with minimal Docker privileges
- Enable UFW firewall: `sudo ufw allow 5000`
- Consider reverse proxy (nginx) for HTTPS

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances with different F&O segments
- Use load balancer for web dashboard
- Separate WebSocket scanner from web interface

### Vertical Scaling
- Increase batch sizes for more symbols
- Add more CPU/memory resources
- Optimize SQL queries and caching

---

✅ Your Dhan F&O Scanner is now production-ready with:
- 24/7 automatic operation via systemd
- Docker containerization for consistency
- Web dashboard with historical analysis  
- Real-time WebSocket v2 scanning
- SQLite persistence and Telegram alerts
- Comprehensive monitoring and logging