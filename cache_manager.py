#!/usr/bin/env python3
"""
cache_manager.py - Dual cache system with Redis primary and SQLite fallback
Provides high-performance caching for monthly levels and scanner data
"""

import redis
import json
import sqlite3
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Dual cache system with Redis as primary and SQLite as fallback
    Automatically handles Redis connection failures gracefully
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, sqlite_path: str = 'cache.db'):
        """
        Initialize cache manager with Redis and SQLite
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port  
            redis_db: Redis database number
            sqlite_path: Path to SQLite database file
        """
        self.redis_client = None
        self.use_redis = False
        self.sqlite_path = sqlite_path
        
        # Try to connect to Redis
        self._init_redis(redis_host, redis_port, redis_db)
        
        # Always initialize SQLite as fallback
        self._init_sqlite()
        
        logger.info(f"CacheManager initialized - Redis: {self.use_redis}, SQLite: True")
    
    def _init_redis(self, host: str, port: int, db: int) -> None:
        """Initialize Redis connection with error handling"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError]
            )
            
            # Test connection
            self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ Redis cache connected successfully")
            
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            logger.warning(f"⚠️  Redis not available: {e}, falling back to SQLite")
            self.redis_client = None
            self.use_redis = False
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database for fallback cache"""
        try:
            self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            
            # Create cache table with expiry support
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    data_type TEXT DEFAULT 'pickle'
                )
            ''')
            
            # Create index on expiry for faster cleanup
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
            ''')
            
            self.conn.commit()
            logger.info("✅ SQLite cache initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQLite cache: {e}")
            raise
    
    def set(self, key: str, value: Any, expiry_hours: int = 24) -> bool:
        """
        Set cache value with expiry
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized for Redis, pickled for SQLite)
            expiry_hours: Hours until expiry
            
        Returns:
            bool: True if successful
        """
        try:
            # Try Redis first if available
            if self.use_redis and self.redis_client:
                try:
                    serialized_value = json.dumps(value, default=str)
                    expiry_seconds = expiry_hours * 3600
                    self.redis_client.setex(key, expiry_seconds, serialized_value)
                    logger.debug(f"✅ Set in Redis: {key}")
                    return True
                except (redis.ConnectionError, redis.TimeoutError, json.JSONEncodeError) as e:
                    logger.warning(f"Redis set failed: {e}, trying SQLite")
                    # Fall through to SQLite
            
            # Use SQLite fallback
            expires_at = datetime.now() + timedelta(hours=expiry_hours)
            pickled_value = pickle.dumps(value)
            
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache (key, value, expires_at, data_type) 
                    VALUES (?, ?, ?, 'pickle')
                ''', (key, pickled_value, expires_at.isoformat()))
                conn.commit()
            
            logger.debug(f"✅ Set in SQLite: {key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set error for key '{key}': {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value if not expired
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            # Try Redis first if available
            if self.use_redis and self.redis_client:
                try:
                    value = self.redis_client.get(key)
                    if value is not None:
                        parsed_value = json.loads(value)
                        logger.debug(f"✅ Retrieved from Redis: {key}")
                        return parsed_value
                except (redis.ConnectionError, redis.TimeoutError, json.JSONDecodeError) as e:
                    logger.warning(f"Redis get failed: {e}, trying SQLite")
                    # Fall through to SQLite
            
            # Use SQLite fallback
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute('''
                    SELECT value, expires_at, data_type FROM cache 
                    WHERE key = ? AND expires_at > ?
                ''', (key, datetime.now().isoformat()))
                
                row = cursor.fetchone()
                if row:
                    value_blob, expires_at, data_type = row
                    
                    if data_type == 'pickle':
                        value = pickle.loads(value_blob)
                    else:
                        # Handle legacy data
                        value = pickle.loads(value_blob)
                    
                    logger.debug(f"✅ Retrieved from SQLite: {key}")
                    return value
            
            logger.debug(f"❌ Key not found or expired: {key}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache get error for key '{key}': {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a cache entry
        
        Args:
            key: Cache key to delete
            
        Returns:
            bool: True if deleted
        """
        try:
            deleted = False
            
            # Delete from Redis if available
            if self.use_redis and self.redis_client:
                try:
                    result = self.redis_client.delete(key)
                    deleted = bool(result)
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    logger.warning(f"Redis delete failed: {e}")
            
            # Delete from SQLite
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute('DELETE FROM cache WHERE key = ?', (key,))
                deleted = deleted or cursor.rowcount > 0
                conn.commit()
            
            logger.debug(f"✅ Deleted: {key}")
            return deleted
            
        except Exception as e:
            logger.error(f"❌ Cache delete error for key '{key}': {e}")
            return False
    
    def clear_expired(self) -> int:
        """
        Clear expired entries from SQLite cache
        Redis handles expiry automatically
        
        Returns:
            int: Number of entries cleared
        """
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute('''
                    DELETE FROM cache WHERE expires_at < ?
                ''', (datetime.now().isoformat(),))
                cleared_count = cursor.rowcount
                conn.commit()
            
            logger.info(f"✅ Cleared {cleared_count} expired entries from SQLite")
            return cleared_count
            
        except Exception as e:
            logger.error(f"❌ Error clearing expired entries: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict with cache statistics
        """
        stats = {
            'redis_available': self.use_redis,
            'sqlite_available': True,
            'current_backend': 'Redis' if self.use_redis else 'SQLite'
        }
        
        try:
            # SQLite stats
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache WHERE expires_at > ?', 
                                    (datetime.now().isoformat(),))
                stats['sqlite_active_entries'] = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM cache WHERE expires_at <= ?', 
                                    (datetime.now().isoformat(),))
                stats['sqlite_expired_entries'] = cursor.fetchone()[0]
        
        except Exception as e:
            logger.error(f"Error getting SQLite stats: {e}")
            stats['sqlite_active_entries'] = 0
            stats['sqlite_expired_entries'] = 0
        
        # Redis stats
        if self.use_redis and self.redis_client:
            try:
                info = self.redis_client.info()
                stats['redis_used_memory'] = info.get('used_memory_human', 'Unknown')
                stats['redis_connected_clients'] = info.get('connected_clients', 0)
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        
        return stats
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of both cache systems
        
        Returns:
            Dict with health status
        """
        health = {'redis': False, 'sqlite': False}
        
        # Test Redis
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.ping()
                health['redis'] = True
            except Exception:
                health['redis'] = False
        
        # Test SQLite
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute('SELECT 1').fetchone()
                health['sqlite'] = True
        except Exception:
            health['sqlite'] = False
        
        return health

# Global cache instance
cache = CacheManager()