#!/usr/bin/env python3
"""
test_cache.py - Test cache manager functionality
"""

from cache_manager import CacheManager
import json
from datetime import datetime

def test_cache_functionality():
    """Test basic cache operations"""
    print("[TEST] Testing Cache Manager...")
    
    # Initialize cache
    cache = CacheManager()
    
    # Test data
    test_data = {
        "symbol": "RELIANCE",
        "pivot": 3016.67,
        "cpr": {
            "tc": 3020.84,
            "pivot": 3016.67,
            "bc": 3000.00,
            "is_narrow": False
        },
        "calculated_at": datetime.now().isoformat()
    }
    
    print(f"[STATS] Cache Stats: {cache.get_cache_stats()}")
    print(f"[HEALTH] Health Check: {cache.health_check()}")
    
    # Test set operation
    print("\n[TEST] Testing SET operation...")
    success = cache.set("test:RELIANCE:2024-12", test_data, expiry_hours=1)
    print(f"[RESULT] Set operation: {'SUCCESS' if success else 'FAILED'}")
    
    # Test get operation
    print("\n[TEST] Testing GET operation...")
    retrieved = cache.get("test:RELIANCE:2024-12")
    
    if retrieved:
        print("[RESULT] Retrieved data successfully")
        print(f"Symbol: {retrieved['symbol']}")
        print(f"Pivot: {retrieved['pivot']}")
        print(f"CPR: {retrieved['cpr']}")
        
        # Verify data integrity
        if retrieved == test_data:
            print("[PASS] Data integrity verified - EXACT match")
        else:
            print("[FAIL] Data mismatch!")
            print(f"Original: {test_data}")
            print(f"Retrieved: {retrieved}")
    else:
        print("[FAIL] Failed to retrieve data")
    
    # Test non-existent key
    print("\n[TEST] Testing non-existent key...")
    missing = cache.get("non_existent_key")
    print(f"[RESULT] Non-existent key handling: {'CORRECT' if missing is None else 'FAILED'}")
    
    # Test delete operation
    print("\n[TEST] Testing DELETE operation...")
    deleted = cache.delete("test:RELIANCE:2024-12")
    print(f"[RESULT] Delete operation: {'SUCCESS' if deleted else 'FAILED'}")
    
    # Verify deletion
    after_delete = cache.get("test:RELIANCE:2024-12")
    print(f"[RESULT] Verify deletion: {'SUCCESS' if after_delete is None else 'FAILED'}")
    
    # Final stats
    print(f"\n[STATS] Final Cache Stats: {cache.get_cache_stats()}")
    
    print("\n[COMPLETE] Cache Manager test completed!")

if __name__ == "__main__":
    test_cache_functionality()