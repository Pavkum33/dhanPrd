#!/usr/bin/env python3
"""Test script to debug import issues"""

import sys
import traceback

print("Testing imports...")

# Test cache_manager
try:
    from cache_manager import CacheManager
    print("✓ CacheManager imported successfully")
except Exception as e:
    print("✗ CacheManager import failed:")
    traceback.print_exc()

# Test monthly_levels
try:
    from scanners.monthly_levels import MonthlyLevelCalculator
    print("✓ MonthlyLevelCalculator imported successfully")
except Exception as e:
    print("✗ MonthlyLevelCalculator import failed:")
    traceback.print_exc()

# Test initialization
try:
    cache = CacheManager()
    print("✓ CacheManager initialized successfully")
    
    calc = MonthlyLevelCalculator(cache)
    print("✓ MonthlyLevelCalculator initialized successfully")
    
except Exception as e:
    print("✗ Initialization failed:")
    traceback.print_exc()

print("\nImport test complete!")