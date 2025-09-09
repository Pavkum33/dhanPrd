#!/usr/bin/env python3
"""
start.py - Railway entry point for running both scanner and web UI
"""

import os
import sys
import threading
import asyncio
import time
from app import app, socketio, init_db

def run_scanner():
    """Run the scanner in a separate thread"""
    try:
        # Import scanner after environment is ready
        from scanner import main as scanner_main
        
        # Run scanner with asyncio
        asyncio.run(scanner_main('config.json'))
    except Exception as e:
        print(f"Scanner error: {e}", file=sys.stderr)
        # Don't exit - let web UI continue running

def run_web_server():
    """Run the Flask web server"""
    # Initialize database
    init_db()
    
    # Get port from Railway environment
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Railway requires binding to 0.0.0.0
    
    print(f"Starting F&O Scanner Dashboard on {host}:{port}")
    
    # Set environment flag for Docker detection
    os.environ['DOCKER_CONTAINER'] = 'true'
    
    # Run Flask with SocketIO
    socketio.run(app, host=host, port=port, debug=False)

def main():
    """Main entry point for Railway deployment"""
    print("Starting Dhan F&O Scanner System...")
    
    # Check for required environment variables
    required_vars = ['DHAN_CLIENT_ID', 'DHAN_ACCESS_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"WARNING: Missing required environment variables: {', '.join(missing_vars)}")
        print("Scanner will run in demo mode with mock data.")
        print("Please set environment variables in Railway dashboard for live trading.")
    else:
        print("Credentials found. Starting live scanner...")
        # Start scanner in background thread
        scanner_thread = threading.Thread(target=run_scanner, daemon=True)
        scanner_thread.start()
        
        # Give scanner time to initialize
        time.sleep(2)
    
    # Start web server (this blocks)
    run_web_server()

if __name__ == '__main__':
    main()