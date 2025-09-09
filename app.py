#!/usr/bin/env python3
"""
app.py - Web server for F&O Scanner Dashboard
Provides real-time web interface for monitoring scanner activity
"""

import os
import json
import sqlite3
import asyncio
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
scanner_state = {
    'running': False,
    'connected_clients': 0,
    'active_symbols': 0,
    'last_update': None,
    'alerts': [],
    'scanner_data': []
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

if __name__ == '__main__':
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
    
    # Get port from Railway environment
    port = int(os.getenv('PORT', 5000))
    host = '0.0.0.0'  # Railway requires binding to 0.0.0.0
    
    print(f"Starting F&O Scanner Dashboard on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)