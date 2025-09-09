#!/usr/bin/env python3
"""
debug_imports.py - Test imports to find the issue causing 404 errors
"""

import sys
import os

print("Testing imports...")

# Test basic imports
try:
    from flask import Flask
    print("OK: Flask import")
except Exception as e:
    print(f"FAIL: Flask import - {e}")

# Test cache manager
try:
    from cache_manager import CacheManager
    print("OK: CacheManager import")
except Exception as e:
    print(f"FAIL: CacheManager import - {e}")

# Test monthly levels
try:
    from scanners.monthly_levels import MonthlyLevelCalculator
    print("OK: MonthlyLevelCalculator import")
except Exception as e:
    print(f"FAIL: MonthlyLevelCalculator import - {e}")

# Test premarket job
try:
    from premarket_job import PremarketJob
    print("OK: PremarketJob import")
except Exception as e:
    print(f"FAIL: PremarketJob import - {e}")

print("\nTesting scanners module initialization...")
try:
    import scanners
    print("OK: scanners module import")
except Exception as e:
    print(f"FAIL: scanners module import - {e}")

# Check if scanners directory has __init__.py
scanners_init = "scanners/__init__.py"
if os.path.exists(scanners_init):
    print("OK: scanners/__init__.py exists")
else:
    print("FAIL: scanners/__init__.py missing")

print("\nAll import tests completed.")