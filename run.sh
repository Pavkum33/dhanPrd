#!/bin/bash
set -e

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container..."
    # Run the integrated Flask app with historical analysis
    exec python app.py
else
    echo "Running locally..."
    # Check for credentials
    if [ -z "$DHAN_CLIENT_ID" ] || [ -z "$DHAN_ACCESS_TOKEN" ]; then
        echo "Warning: DHAN credentials not set. Running in demo mode."
        echo "Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN for live scanning."
    fi
    
    # Start the integrated Flask app (includes both web dashboard and scanner)
    python app.py
fi