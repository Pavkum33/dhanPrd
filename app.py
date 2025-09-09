#!/usr/bin/env python3
"""
app.py - Web server for F&O Scanner Dashboard
Provides real-time web interface for monitoring scanner activity
"""

import os
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import logging
import aiohttp
import pandas as pd
from typing import Dict, List, Optional
import math
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Historical Data Fetcher Classes
class DhanHistoricalFetcher:
    """Fetches historical data from Dhan REST API"""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api.dhan.co"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': access_token
        }
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_instruments(self) -> pd.DataFrame:
        """Fetch instrument master from Dhan"""
        url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    from io import StringIO
                    df = pd.read_csv(StringIO(content))
                    df.columns = [c.strip() for c in df.columns]
                    logger.info(f"Fetched {len(df)} instruments")
                    return df
                else:
                    logger.error(f"Failed to fetch instruments: {response.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Error fetching instruments: {e}")
            return pd.DataFrame()
    
    def get_fno_instruments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter F&O instruments with improved detection"""
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Try multiple approaches to find F&O instruments
        fno_df = pd.DataFrame()
        
        # Method 1: Look for segment column with FUT
        seg_cols = [col for col in df.columns if any(x in col.lower() for x in ['segment', 'exch', 'market'])]
        if seg_cols:
            seg_col = seg_cols[0]
            logger.info(f"Trying segment column: {seg_col}")
            fno_df = df[df[seg_col].astype(str).str.contains('FUT|FUTURE|NSE_FO|BSE_FO', case=False, na=False)].copy()
            if len(fno_df) > 0:
                logger.info(f"Found {len(fno_df)} F&O instruments using segment filter")
                return fno_df
        
        # Method 2: Look for expiry date column
        exp_cols = [col for col in df.columns if any(x in col.lower() for x in ['expiry', 'expire', 'maturity'])]
        if exp_cols:
            exp_col = exp_cols[0]
            logger.info(f"Trying expiry column: {exp_col}")
            fno_df = df[df[exp_col].notnull()].copy()
            if len(fno_df) > 0:
                logger.info(f"Found {len(fno_df)} instruments with expiry dates")
                return fno_df
        
        # Method 3: Look for instrument type column
        inst_cols = [col for col in df.columns if any(x in col.lower() for x in ['instrument', 'type', 'product'])]
        if inst_cols:
            inst_col = inst_cols[0]
            logger.info(f"Trying instrument column: {inst_col}")
            fno_df = df[df[inst_col].astype(str).str.contains('FUT|FUTURE|OPTION|INDEX', case=False, na=False)].copy()
            if len(fno_df) > 0:
                logger.info(f"Found {len(fno_df)} F&O instruments using instrument type")
                return fno_df
        
        # Method 4: Look for symbol patterns (NIFTY, BANKNIFTY, etc.)
        sym_cols = [col for col in df.columns if any(x in col.lower() for x in ['symbol', 'name', 'trading'])]
        if sym_cols:
            sym_col = sym_cols[0]
            logger.info(f"Trying symbol pattern matching in: {sym_col}")
            # Common F&O patterns
            patterns = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX']
            pattern_str = '|'.join(patterns)
            fno_df = df[df[sym_col].astype(str).str.contains(pattern_str, case=False, na=False)].copy()
            if len(fno_df) > 0:
                logger.info(f"Found {len(fno_df)} instruments using symbol patterns")
                return fno_df
        
        # Method 5: Sample a few popular stocks that likely have F&O
        if sym_cols:
            sym_col = sym_cols[0]
            logger.info(f"Trying popular F&O stocks in: {sym_col}")
            popular_fno = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'ITC', 'SBIN', 'LT', 'HCLTECH', 'AXISBANK', 'MARUTI']
            pattern_str = '|'.join(popular_fno)
            fno_df = df[df[sym_col].astype(str).str.contains(pattern_str, case=False, na=False)].copy()
            if len(fno_df) > 0:
                logger.info(f"Found {len(fno_df)} popular F&O stocks")
                return fno_df
        
        # Method 6: Fallback - take a sample of instruments for testing
        logger.warning("No F&O instruments found, taking sample for testing")
        fno_df = df.sample(min(20, len(df))).copy() if len(df) > 0 else pd.DataFrame()
        
        logger.info(f"Final result: {len(fno_df)} instruments selected")
        return fno_df
    
    async def get_historical_data(self, security_id: str, days: int = 60) -> pd.DataFrame:
        """Fetch historical daily data for a security"""
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days + 5)
        
        url = f"{self.base_url}/charts/historical"
        payload = {
            "securityId": security_id,
            "exchangeSegment": "NSE",
            "instrument": "FUTURE",
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])
                        if 'timestamp' in df.columns:
                            df['date'] = pd.to_datetime(df['timestamp'])
                        elif 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                        
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in required_cols:
                            if col not in df.columns:
                                df[col] = 0
                        
                        df = df.sort_values('date').reset_index(drop=True)
                        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
                return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Error fetching data for {security_id}: {e}")
            return pd.DataFrame()

class BreakoutAnalyzer:
    """Analyzes historical data for breakout patterns"""
    
    def __init__(self, lookback: int = 50, ema_short: int = 8, ema_long: int = 13):
        self.lookback = lookback
        self.ema_short = ema_short
        self.ema_long = ema_long
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for breakout analysis"""
        if df.empty or len(df) < self.lookback:
            return df
        
        df = df.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['resistance'] = df['typical_price'].rolling(window=self.lookback).max()
        df['ema_short'] = df['close'].ewm(span=self.ema_short).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long).mean()
        df['breakout'] = (df['close'] > df['resistance']) & (df['ema_short'] > df['ema_long'])
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_avg'] * 1.5)
        df['signal'] = df['breakout'] & df['volume_spike'] & (df['close'] > df['open'])
        return df
    
    def get_current_analysis(self, df: pd.DataFrame) -> Dict:
        """Get current analysis for latest data point"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        prev_day = df.iloc[-2] if len(df) > 1 else latest
        
        return {
            'symbol': '',
            'date': latest.get('date', ''),
            'close': float(latest.get('close', 0)),
            'resistance': float(latest.get('resistance', 0)),
            'ema_short': float(latest.get('ema_short', 0)),
            'ema_long': float(latest.get('ema_long', 0)),
            'volume': int(latest.get('volume', 0)),
            'prev_day_volume': int(prev_day.get('volume', 0)),
            'breakout_signal': bool(latest.get('signal', False)),
            'change_pct': ((latest.get('close', 0) - prev_day.get('close', 1)) / prev_day.get('close', 1)) * 100
        }

async def fetch_and_analyze_historical_data():
    """Main function to fetch and analyze historical data with progress updates"""
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    def emit_progress(step, message, current=0, total=0, data=None):
        """Emit progress updates to all connected clients"""
        progress_data = {
            'step': step,
            'message': message,
            'current': current,
            'total': total,
            'progress_percent': int((current / total) * 100) if total > 0 else 0,
            'data': data or {}
        }
        socketio.emit('historical_progress', progress_data)
        logger.info(f"Progress: {message} ({current}/{total})")
    
    try:
        if not client_id or not access_token:
            emit_progress('error', 'Dhan credentials not found in environment variables')
            return {}
        
        emit_progress('starting', 'Starting historical data fetch...')
        
        async with DhanHistoricalFetcher(client_id, access_token) as fetcher:
            emit_progress('instruments', 'Fetching instrument master from Dhan...')
            instruments_df = await fetcher.get_instruments()
            
            if instruments_df.empty:
                emit_progress('error', 'Failed to fetch instruments from Dhan API')
                return {}
            
            emit_progress('filtering', f'Filtering F&O instruments from {len(instruments_df)} total instruments...')
            fno_df = fetcher.get_fno_instruments(instruments_df)
            
            if fno_df.empty:
                emit_progress('error', 'No F&O instruments found in instrument master')
                return {}
            
            # Prepare securities list (limit to first 15 for stability)
            securities = []
            for _, row in fno_df.head(15).iterrows():
                security_id = None
                symbol = None
                
                for col in row.index:
                    if 'security' in col.lower() and 'id' in col.lower():
                        security_id = str(row[col])
                        break
                
                for col in row.index:
                    if any(x in col.lower() for x in ['symbol', 'name', 'trading']):
                        symbol = str(row[col])
                        break
                
                if security_id and symbol:
                    securities.append({
                        'security_id': security_id,
                        'symbol': symbol
                    })
            
            total_securities = len(securities)
            emit_progress('prepared', f'Prepared {total_securities} F&O securities for analysis', 0, total_securities)
            
            # Fetch historical data with rate limiting
            analyzed_data = {}
            analyzer = BreakoutAnalyzer()
            successful_fetches = 0
            failed_fetches = 0
            
            for i, sec_info in enumerate(securities):
                try:
                    current_symbol = sec_info['symbol']
                    emit_progress('fetching', f'Fetching historical data for {current_symbol}...', i + 1, total_securities, {
                        'current_symbol': current_symbol,
                        'successful': successful_fetches,
                        'failed': failed_fetches
                    })
                    
                    await asyncio.sleep(0.3)  # Rate limiting
                    df = await fetcher.get_historical_data(sec_info['security_id'])
                    
                    if not df.empty:
                        emit_progress('analyzing', f'Analyzing {current_symbol} ({len(df)} days of data)...', i + 1, total_securities)
                        
                        analyzed_df = analyzer.calculate_technical_indicators(df)
                        current_analysis = analyzer.get_current_analysis(analyzed_df)
                        current_analysis['symbol'] = current_symbol
                        current_analysis['security_id'] = sec_info['security_id']
                        
                        analyzed_data[sec_info['security_id']] = {
                            'symbol': current_symbol,
                            'historical_data': analyzed_df,
                            'current_analysis': current_analysis
                        }
                        
                        successful_fetches += 1
                        
                        # Check for breakout signal
                        signal_status = "BREAKOUT" if current_analysis['breakout_signal'] else "No Signal"
                        emit_progress('completed_symbol', f'✅ {current_symbol}: Close={current_analysis["close"]:.2f}, {signal_status}', 
                                    i + 1, total_securities, {
                            'symbol': current_symbol,
                            'close': current_analysis['close'],
                            'signal': current_analysis['breakout_signal'],
                            'successful': successful_fetches,
                            'failed': failed_fetches
                        })
                    else:
                        failed_fetches += 1
                        emit_progress('failed_symbol', f'❌ {current_symbol}: No historical data available', 
                                    i + 1, total_securities, {
                            'symbol': current_symbol,
                            'successful': successful_fetches,
                            'failed': failed_fetches
                        })
                    
                except Exception as e:
                    failed_fetches += 1
                    emit_progress('error_symbol', f'❌ {current_symbol}: Error - {str(e)[:50]}...', 
                                i + 1, total_securities, {
                        'symbol': current_symbol,
                        'error': str(e),
                        'successful': successful_fetches,
                        'failed': failed_fetches
                    })
            
            emit_progress('summary', f'Historical analysis completed: {successful_fetches} successful, {failed_fetches} failed', 
                        total_securities, total_securities, {
                'total_analyzed': len(analyzed_data),
                'successful': successful_fetches,
                'failed': failed_fetches,
                'breakouts_found': sum(1 for data in analyzed_data.values() if data['current_analysis']['breakout_signal'])
            })
            
            return analyzed_data
            
    except Exception as e:
        emit_progress('error', f'Critical error: {str(e)}')
        logger.exception("Critical error in historical data fetch")
        return {}

# Global state
scanner_state = {
    'running': False,
    'connected_clients': 0,
    'active_symbols': 0,
    'last_update': None,
    'alerts': [],
    'scanner_data': [],
    'historical_data': {},
    'historical_analysis_running': False
}

# Database connection
def get_db_connection():
    """Get database connection for alerts"""
    conn = sqlite3.connect('alerts.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database if not exists"""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            sid TEXT,
            symbol TEXT,
            message TEXT,
            strategy TEXT DEFAULT 'breakout'
        );
    """)
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get scanner status"""
    return jsonify({
        'running': scanner_state['running'],
        'connected_clients': scanner_state['connected_clients'],
        'active_symbols': scanner_state['active_symbols'],
        'last_update': scanner_state['last_update']
    })

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts from database"""
    limit = request.args.get('limit', 100, type=int)
    conn = get_db_connection()
    alerts = conn.execute(
        'SELECT * FROM alerts ORDER BY id DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    
    return jsonify([dict(alert) for alert in alerts])

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        config = request.json
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    scanner_state['connected_clients'] += 1
    emit('connected', {'data': 'Connected to scanner server'})
    
    # Send initial data
    emit('stats', {
        'activeSymbols': scanner_state['active_symbols'],
        'totalAlerts': len(scanner_state['alerts'])
    })
    
    # Send recent alerts
    if scanner_state['alerts']:
        for alert in scanner_state['alerts'][-10:]:
            emit('alert', alert)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    scanner_state['connected_clients'] -= 1

@socketio.on('start_scanner')
def handle_start_scanner():
    """Start the scanner"""
    if not scanner_state['running']:
        scanner_state['running'] = True
        emit('scanner_status', {'running': True}, broadcast=True)
        # Here you would trigger the actual scanner start
        # For now, we'll simulate with mock data
        start_mock_scanner()

@socketio.on('stop_scanner')
def handle_stop_scanner():
    """Stop the scanner"""
    if scanner_state['running']:
        scanner_state['running'] = False
        emit('scanner_status', {'running': False}, broadcast=True)

@socketio.on('refresh_data')
def handle_refresh_data():
    """Refresh scanner data"""
    emit('scanner_data', scanner_state['scanner_data'])

@socketio.on('update_filters')
def handle_update_filters(filters):
    """Update scanner filters"""
    # Here you would apply filters to the scanner
    print(f"Updating filters: {filters}")
    emit('filters_updated', {'success': True})

@socketio.on('get_symbol_details')
def handle_get_symbol_details(symbol):
    """Get detailed information for a symbol"""
    # Mock data for demonstration
    details = {
        'symbol': symbol,
        'ltp': 1234.56,
        'volume': 123456,
        'resistance': 1250.00,
        'ema8': 1230.00,
        'ema13': 1225.00
    }
    return details

@socketio.on('update_settings')
def handle_update_settings(settings):
    """Update scanner settings"""
    print(f"Updating settings: {settings}")
    emit('settings_updated', {'success': True})

# Mock scanner for demonstration
def start_mock_scanner():
    """Simulate scanner activity with mock data"""
    def scanner_loop():
        symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'RELIANCE', 
                   'TCS', 'HDFC', 'INFY', 'ICICIBANK', 'SBIN']
        
        while scanner_state['running']:
            # Generate mock data
            mock_data = []
            for symbol in symbols:
                import random
                base_price = random.uniform(1000, 5000)
                mock_data.append({
                    'symbol': symbol,
                    'ltp': base_price,
                    'change': random.uniform(-5, 5),
                    'volume': random.randint(10000, 1000000),
                    'resistance': base_price * 1.02,
                    'ema8': base_price * 0.99,
                    'ema13': base_price * 0.98,
                    'signal': 'BREAKOUT' if random.random() > 0.8 else None
                })
            
            scanner_state['scanner_data'] = mock_data
            scanner_state['active_symbols'] = len(symbols)
            scanner_state['last_update'] = datetime.now().isoformat()
            
            # Emit data to all connected clients
            socketio.emit('scanner_data', mock_data)
            socketio.emit('stats', {
                'activeSymbols': len(symbols),
                'totalAlerts': len(scanner_state['alerts'])
            })
            
            # Simulate random alerts
            if random.random() > 0.9:
                alert = {
                    'symbol': random.choice(symbols),
                    'message': 'Breakout detected!',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'breakout'
                }
                scanner_state['alerts'].append(alert)
                socketio.emit('alert', alert)
                
                # Save to database
                conn = get_db_connection()
                conn.execute(
                    'INSERT INTO alerts (ts, sid, symbol, message) VALUES (?, ?, ?, ?)',
                    (alert['timestamp'], '0', alert['symbol'], alert['message'])
                )
                conn.commit()
                conn.close()
            
            time.sleep(5)  # Update every 5 seconds
    
    # Start scanner in background thread
    scanner_thread = threading.Thread(target=scanner_loop)
    scanner_thread.daemon = True
    scanner_thread.start()

# Scanner integration
def integrate_with_scanner():
    """
    Integration point with the actual scanner.py
    This function would be called by scanner.py to send real data
    """
    def send_scanner_update(data):
        """Send scanner data update to all connected clients"""
        scanner_state['scanner_data'] = data
        socketio.emit('scanner_data', data)
    
    def send_alert(alert):
        """Send alert to all connected clients"""
        scanner_state['alerts'].append(alert)
        socketio.emit('alert', alert)
    
    return send_scanner_update, send_alert

# API for scanner.py to send data
@app.route('/api/scanner/update', methods=['POST'])
def scanner_update():
    """Receive updates from scanner.py"""
    data = request.json
    
    if data.get('type') == 'data':
        scanner_state['scanner_data'] = data.get('data', [])
        socketio.emit('scanner_data', scanner_state['scanner_data'])
    
    elif data.get('type') == 'alert':
        alert = data.get('alert')
        scanner_state['alerts'].append(alert)
        socketio.emit('alert', alert)
    
    elif data.get('type') == 'stats':
        stats = data.get('stats')
        scanner_state.update(stats)
        socketio.emit('stats', stats)
    
    return jsonify({'success': True})

@app.route('/api/historical/fetch', methods=['POST'])
def fetch_historical():
    """Fetch historical data for analysis"""
    if scanner_state['historical_analysis_running']:
        return jsonify({'error': 'Historical analysis already running'}), 400
    
    def run_historical_fetch():
        scanner_state['historical_analysis_running'] = True
        try:
            # Run async function in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            historical_data = loop.run_until_complete(fetch_and_analyze_historical_data())
            
            # Update scanner state
            scanner_state['historical_data'] = historical_data
            scanner_state['active_symbols'] = len(historical_data)
            
            # Convert to scanner data format
            scanner_data = []
            for sec_id, data in historical_data.items():
                analysis = data['current_analysis']
                scanner_data.append({
                    'symbol': analysis['symbol'],
                    'ltp': analysis['close'],
                    'change': analysis['change_pct'],
                    'volume': analysis['volume'],
                    'resistance': analysis['resistance'],
                    'ema8': analysis['ema_short'],
                    'ema13': analysis['ema_long'],
                    'signal': 'BREAKOUT' if analysis['breakout_signal'] else None
                })
            
            scanner_state['scanner_data'] = scanner_data
            scanner_state['last_update'] = datetime.now().isoformat()
            
            # Emit updates to all clients
            socketio.emit('scanner_data', scanner_data)
            socketio.emit('stats', {
                'activeSymbols': len(historical_data),
                'totalAlerts': len(scanner_state['alerts'])
            })
            
            # Generate alerts for breakout signals
            for data in historical_data.values():
                analysis = data['current_analysis']
                if analysis['breakout_signal']:
                    alert = {
                        'symbol': analysis['symbol'],
                        'message': f"Historical Breakout - Close: {analysis['close']:.2f}, Resistance: {analysis['resistance']:.2f}",
                        'timestamp': datetime.now().isoformat(),
                        'type': 'breakout'
                    }
                    scanner_state['alerts'].append(alert)
                    socketio.emit('alert', alert)
            
            loop.close()
            
        except Exception as e:
            print(f"Historical fetch error: {e}")
        finally:
            scanner_state['historical_analysis_running'] = False
    
    # Start in background thread
    thread = threading.Thread(target=run_historical_fetch, daemon=True)
    thread.start()
    
    return jsonify({'message': 'Historical data fetch started'})

@app.route('/api/historical/status')
def historical_status():
    """Get historical data fetch status"""
    return jsonify({
        'running': scanner_state['historical_analysis_running'],
        'symbols_count': len(scanner_state['historical_data']),
        'last_update': scanner_state['last_update']
    })

@app.route('/api/historical/data')
def get_historical_data():
    """Get current historical data"""
    return jsonify({
        'data': scanner_state['scanner_data'],
        'symbols_count': len(scanner_state['historical_data']),
        'last_update': scanner_state['last_update']
    })

@app.route('/api/debug/instruments')
def debug_instruments():
    """Debug endpoint to check instrument master structure"""
    try:
        import requests
        url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = [c.strip() for c in df.columns]
        
        # Get sample data
        sample_rows = df.head(5).to_dict('records') if len(df) > 0 else []
        
        return jsonify({
            'total_instruments': len(df),
            'columns': list(df.columns),
            'sample_data': sample_rows,
            'unique_segments': df[df.columns[0]].unique()[:10].tolist() if len(df.columns) > 0 else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_scanner_background():
    """Run the scanner in a separate thread"""
    try:
        # Check for credentials
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if client_id and access_token:
            print("Starting live scanner with Dhan credentials...")
            # Import and run scanner
            from scanner import main as scanner_main
            asyncio.run(scanner_main('config.json'))
        else:
            print("No Dhan credentials found - running in demo mode only")
            # Keep thread alive but don't run scanner
            while True:
                time.sleep(60)
                
    except Exception as e:
        print(f"Scanner error: {e}", file=sys.stderr)

def init_app():
    """Initialize the application"""
    # Initialize database
    init_db()
    
    # Start scanner in background thread if credentials are available
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if client_id and access_token:
        print("Starting background scanner thread...")
        scanner_thread = threading.Thread(target=run_scanner_background, daemon=True)
        scanner_thread.start()
        time.sleep(2)  # Give scanner time to initialize
    else:
        print("Running in demo mode - add DHAN credentials for live scanning")

# Initialize when module loads
init_app()

if __name__ == '__main__':
    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    host = '0.0.0.0'  # Railway requires binding to 0.0.0.0
    
    print(f"Starting F&O Scanner Dashboard on {host}:{port}")
    
    # Use allow_unsafe_werkzeug for development/Railway
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)