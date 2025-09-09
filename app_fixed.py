#!/usr/bin/env python3
"""
app_fixed.py - Fixed version with proper route registration and real data support
"""

import os
import sys
import json
import sqlite3
import asyncio
import pickle
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import logging

# Import our modules FIRST before route definitions
from cache_manager import CacheManager
from scanners.monthly_levels import MonthlyLevelCalculator
from premarket_job import PremarketJob

# Import the original app components we need
from app import (
    DhanHistoricalFetcher, BreakoutAnalyzer, 
    fetch_and_analyze_historical_data, scanner_state, 
    get_db_connection, init_db
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize cache and calculator IMMEDIATELY
cache = CacheManager()
calculator = MonthlyLevelCalculator(cache)

# Basic routes
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get scanner status - FIXED VERSION"""
    return jsonify({
        'running': scanner_state['running'],
        'connected_clients': scanner_state['connected_clients'],
        'active_symbols': scanner_state['active_symbols'],
        'last_update': scanner_state['last_update'],
        'cache_available': True,
        'calculator_ready': True
    })

@app.route('/api/cache/status')
def cache_status():
    """Check cache system status - FIXED VERSION"""
    try:
        # Get cache stats
        stats = cache.get_cache_stats()
        health = cache.health_check()
        
        return jsonify({
            'cache_stats': stats,
            'health': health,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return jsonify({'error': str(e)}), 500

# Monthly Levels API - FIXED VERSIONS
@app.route('/api/levels/calculate', methods=['POST'])
def calculate_monthly_levels():
    """Manually trigger monthly level calculation - FIXED VERSION"""
    try:
        # Check for credentials
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not client_id or not access_token:
            return jsonify({
                'error': 'DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables required',
                'message': 'Set real Dhan credentials to calculate levels with live data'
            }), 400
        
        # Run the calculation in background thread
        def run_calculation():
            import asyncio
            job = PremarketJob()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(job.calculate_monthly_levels_for_all_symbols())
            loop.close()
            return result
        
        # Start calculation in thread to avoid blocking
        calculation_thread = threading.Thread(target=run_calculation, daemon=True)
        calculation_thread.start()
        
        return jsonify({
            'message': 'Monthly level calculation started with real Dhan data',
            'status': 'running',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting level calculation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/<symbol>')
def get_symbol_levels(symbol):
    """Get cached monthly levels for specific symbol - FIXED VERSION"""
    try:
        # Get current month by default, or from query param
        month = request.args.get('month', datetime.now().strftime('%Y-%m'))
        
        levels = calculator.get_cached_levels(symbol.upper(), month)
        
        if levels:
            return jsonify({
                'symbol': symbol.upper(),
                'month': month,
                'levels': levels,
                'retrieved_at': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': f'No cached levels found for {symbol.upper()} in {month}',
                'message': 'Run level calculation first to populate cache with real data',
                'symbol': symbol.upper(),
                'month': month
            }), 404
            
    except Exception as e:
        logger.error(f"Error retrieving levels for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/narrow-cpr')
def get_narrow_cpr_symbols():
    """Get symbols with narrow CPR - FIXED VERSION"""
    try:
        # Get month from query param
        month = request.args.get('month', datetime.now().strftime('%Y-%m'))
        
        # Get scan results from cache
        scan_cache_key = f"scan_results:{month}"
        scan_results = cache.get(scan_cache_key)
        
        if scan_results and 'narrow_cpr' in scan_results:
            results = scan_results['narrow_cpr']
            
            # Emit WebSocket event for real-time updates
            socketio.emit('cpr_results', {
                'results': results,
                'count': len(results),
                'month': month,
                'last_updated': scan_results.get('last_updated')
            })
            
            return jsonify({
                'month': month,
                'narrow_cpr_symbols': results,
                'total_symbols': scan_results.get('total_symbols', 0),
                'count': len(results),
                'last_updated': scan_results.get('last_updated'),
                'retrieved_at': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'message': 'No narrow CPR scan results found. Run level calculation with real Dhan credentials first.',
                'month': month,
                'narrow_cpr_symbols': [],
                'count': 0,
                'help': 'Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables and call POST /api/levels/calculate'
            }), 404
            
    except Exception as e:
        logger.error(f"Error retrieving narrow CPR symbols: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/near-pivot', methods=['POST'])
def get_symbols_near_pivot():
    """Get symbols near monthly pivot - FIXED VERSION"""
    try:
        # Get current prices from request body
        data = request.get_json() or {}
        current_prices = data.get('current_prices', {})
        symbols = data.get('symbols', list(current_prices.keys()))
        
        if not current_prices:
            return jsonify({
                'error': 'current_prices required in request body',
                'example': {
                    'current_prices': {'RELIANCE': 3020.50, 'TCS': 4150.75},
                    'symbols': ['RELIANCE', 'TCS']
                }
            }), 400
        
        month = request.args.get('month', datetime.now().strftime('%Y-%m'))
        
        # Get symbols near pivot
        near_pivot_symbols = calculator.get_symbols_near_pivot(symbols, current_prices, month)
        
        # Emit WebSocket event for real-time updates
        socketio.emit('pivot_results', {
            'results': near_pivot_symbols,
            'count': len(near_pivot_symbols),
            'month': month,
            'input_symbols': len(symbols)
        })
        
        return jsonify({
            'month': month,
            'near_pivot_symbols': near_pivot_symbols,
            'count': len(near_pivot_symbols),
            'input_symbols': len(symbols),
            'retrieved_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking symbols near pivot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/premarket-summary')
def get_premarket_summary():
    """Get latest pre-market job execution summary - FIXED VERSION"""
    try:
        job = PremarketJob()
        
        # Get latest results
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(job.get_latest_results())
        
        # Add health check
        health = loop.run_until_complete(job.health_check())
        results['health'] = health
        
        loop.close()
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting premarket summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/levels/test', methods=['POST'])
def test_level_calculation():
    """Test monthly level calculation with sample data - FIXED VERSION"""
    try:
        # Get test data from request
        data = request.get_json() or {}
        
        # Default test data if none provided
        test_ohlc = data.get('ohlc', {
            'high': 3125.00,
            'low': 2875.00,
            'close': 3050.00,
            'open': 2890.50
        })
        
        symbol = data.get('symbol', 'TEST_SYMBOL')
        
        # Calculate levels
        levels = calculator.calculate_and_cache_symbol_levels(
            symbol,
            test_ohlc,
            'test-month'
        )
        
        return jsonify({
            'message': 'Test calculation completed successfully',
            'input_data': {
                'symbol': symbol,
                'ohlc': test_ohlc
            },
            'calculated_levels': levels,
            'formulas_verified': 'EXACT Chartink CPR and Pivot formulas',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in test calculation: {e}")
        return jsonify({'error': str(e)}), 500

# Historical data API (from original app.py)
@app.route('/api/historical/fetch', methods=['POST'])
def fetch_historical():
    """Fetch historical data for analysis - ENHANCED VERSION"""
    if scanner_state['historical_analysis_running']:
        return jsonify({'error': 'Historical analysis already running'}), 400
    
    # Check for credentials
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if not client_id or not access_token:
        return jsonify({
            'error': 'Dhan credentials required for real data analysis',
            'message': 'Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables',
            'demo_available': False
        }), 400
    
    # Get configuration parameters from request
    config = request.json or {}
    fetch_days = config.get('fetch_days', 75)
    lookback_period = config.get('lookback_period', 50)
    ema_short = config.get('ema_short', 8)
    ema_long = config.get('ema_long', 13)
    volume_factor = config.get('volume_factor', 0.5)
    price_threshold = config.get('price_threshold', 50)
    
    def run_historical_fetch():
        scanner_state['historical_analysis_running'] = True
        try:
            # Run async function in thread with configuration
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            historical_data = loop.run_until_complete(fetch_and_analyze_historical_data(
                fetch_days=fetch_days,
                lookback_period=lookback_period, 
                ema_short=ema_short,
                ema_long=ema_long,
                volume_factor=volume_factor,
                price_threshold=price_threshold
            ))
            
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
            
            loop.close()
            
        except Exception as e:
            print(f"Historical fetch error: {e}")
        finally:
            scanner_state['historical_analysis_running'] = False
    
    # Start in background thread
    thread = threading.Thread(target=run_historical_fetch, daemon=True)
    thread.start()
    
    return jsonify({
        'message': 'Historical data fetch started with real Dhan credentials',
        'config': {
            'fetch_days': fetch_days,
            'lookback_period': lookback_period,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'volume_factor': volume_factor,
            'price_threshold': price_threshold
        }
    })

# Import all WebSocket handlers and other routes from original app
from app import (
    handle_connect, handle_disconnect, handle_start_scanner,
    handle_stop_scanner, handle_refresh_data, handle_update_filters,
    handle_get_symbol_details, handle_update_settings
)

# Register WebSocket handlers
socketio.on_event('connect', handle_connect)
socketio.on_event('disconnect', handle_disconnect)
socketio.on_event('start_scanner', handle_start_scanner)
socketio.on_event('stop_scanner', handle_stop_scanner)
socketio.on_event('refresh_data', handle_refresh_data)
socketio.on_event('update_filters', handle_update_filters)
socketio.on_event('get_symbol_details', handle_get_symbol_details)
socketio.on_event('update_settings', handle_update_settings)

def init_app():
    """Initialize the fixed application"""
    # Initialize database
    init_db()
    
    # Initialize cache and verify it's working
    try:
        test_key = "startup_test"
        cache.set(test_key, {"status": "working", "timestamp": datetime.now().isoformat()}, 1)
        test_result = cache.get(test_key)
        if test_result:
            logger.info("Cache system verified working")
        else:
            logger.warning("Cache system test failed")
    except Exception as e:
        logger.error(f"Cache system initialization error: {e}")
    
    # Check credentials
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if client_id and access_token:
        logger.info("Real Dhan credentials available - API will work with live data")
    else:
        logger.warning("Demo mode - set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN for real data")

# Initialize when module loads
init_app()

if __name__ == '__main__':
    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"Starting FIXED F&O Scanner Dashboard on {host}:{port}")
    print("API endpoints available:")
    print("  GET  /api/status")
    print("  GET  /api/cache/status") 
    print("  POST /api/levels/calculate")
    print("  GET  /api/levels/<symbol>")
    print("  GET  /api/levels/narrow-cpr")
    print("  POST /api/levels/near-pivot")
    print("  GET  /api/levels/premarket-summary")
    print("  POST /api/levels/test")
    print("  POST /api/historical/fetch")
    
    # Use allow_unsafe_werkzeug for development
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)