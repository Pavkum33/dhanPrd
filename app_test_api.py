#!/usr/bin/env python3
"""
app_test_api.py - Simple test app with working API routes and real data
"""

import os
from flask import Flask, jsonify
from cache_manager import CacheManager
from scanners.monthly_levels import MonthlyLevelCalculator
from datetime import datetime

# Set credentials
os.environ['DHAN_CLIENT_ID'] = '1106283829'
os.environ['DHAN_ACCESS_TOKEN'] = 'test'

app = Flask(__name__)

# Initialize components
cache = CacheManager()
calculator = MonthlyLevelCalculator(cache)

@app.route('/')
def index():
    return "Multi-Scan API Test Server Running"

@app.route('/api/test')
def test():
    return jsonify({"status": "API working", "timestamp": datetime.now().isoformat()})

@app.route('/api/levels/RELIANCE')
def get_reliance_levels():
    """Get cached RELIANCE levels from our real data test"""
    levels = calculator.get_cached_levels('RELIANCE')
    if levels:
        return jsonify({
            "success": True,
            "symbol": "RELIANCE",
            "data": levels,
            "source": "Real Dhan API data cached in SQLite"
        })
    else:
        return jsonify({
            "success": False,
            "message": "No cached data for RELIANCE",
            "hint": "Run test_real_data.py first"
        }), 404

@app.route('/api/levels/narrow-cpr')
def get_narrow_cpr():
    """Get narrow CPR results from cache"""
    # Get from cache
    scan_key = "scan_results:2025-09"
    scan_results = cache.get(scan_key)
    
    if scan_results and 'narrow_cpr' in scan_results:
        return jsonify({
            "success": True,
            "narrow_cpr_symbols": scan_results['narrow_cpr'],
            "total_symbols": scan_results.get('total_symbols', 0),
            "last_updated": scan_results.get('last_updated'),
            "source": "Real Dhan API data"
        })
    else:
        # Try to find any cached levels with narrow CPR
        test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
        narrow_results = []
        
        for symbol in test_symbols:
            levels = calculator.get_cached_levels(symbol)
            if levels and levels.get('cpr', {}).get('is_narrow'):
                narrow_results.append({
                    'symbol': symbol,
                    'cpr_width_percent': levels['cpr']['width_percent'],
                    'breakout_level': levels['cpr']['breakout_level'],
                    'trend': levels['cpr']['trend']
                })
        
        if narrow_results:
            return jsonify({
                "success": True,
                "narrow_cpr_symbols": narrow_results,
                "source": "Direct cache lookup"
            })
        else:
            return jsonify({
                "success": False,
                "message": "No narrow CPR data in cache",
                "hint": "Run test_real_data.py to populate"
            }), 404

@app.route('/api/cache/stats')
def cache_stats():
    """Get cache statistics"""
    stats = cache.get_cache_stats()
    health = cache.health_check()
    
    # List all cached symbols
    cached_symbols = []
    for symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ADANIPORTS']:
        if calculator.get_cached_levels(symbol):
            cached_symbols.append(symbol)
    
    return jsonify({
        "cache_stats": stats,
        "health": health,
        "cached_symbols": cached_symbols,
        "count": len(cached_symbols),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = 6000
    print(f"\nSTARTING TEST API SERVER on port {port}")
    print("=" * 60)
    print("Available endpoints:")
    print("  GET http://localhost:6000/api/test")
    print("  GET http://localhost:6000/api/levels/RELIANCE")
    print("  GET http://localhost:6000/api/levels/narrow-cpr")
    print("  GET http://localhost:6000/api/cache/stats")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)