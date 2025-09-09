#!/usr/bin/env python3
"""
test_routes.py - Test which routes are actually registered in the Flask app
"""

import os
import sys

# Mock environment to avoid startup issues
os.environ['DHAN_CLIENT_ID'] = '1106283829'
os.environ['DHAN_ACCESS_TOKEN'] = 'test_token'

# Import the app
from app import app

print("REGISTERED FLASK ROUTES:")
print("=" * 60)

for rule in app.url_map.iter_rules():
    endpoint = rule.endpoint
    methods = ', '.join(rule.methods - {'HEAD', 'OPTIONS'})
    print(f"{rule.rule:40} {methods:10} -> {endpoint}")

print("=" * 60)

# Check specifically for API routes
api_routes = [r for r in app.url_map.iter_rules() if '/api/' in r.rule]
print(f"\nTotal API routes found: {len(api_routes)}")

if len(api_routes) == 0:
    print("WARNING: No API routes registered! This is why you're getting 404 errors.")
    print("\nPossible reasons:")
    print("1. Route decorators have syntax errors")
    print("2. Routes are inside conditional blocks that aren't executing")
    print("3. Import errors preventing route registration")
else:
    print("\nAPI routes registered successfully:")
    for route in api_routes:
        print(f"  - {route.rule}")