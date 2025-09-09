#!/usr/bin/env python3
"""
monthly_levels.py - Monthly CPR and Pivot level calculator with EXACT Chartink formulas
Calculates Central Pivot Range (CPR) and Pivot Points using previous month's OHLC data
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MonthlyLevelCalculator:
    """
    Calculates monthly CPR and Pivot levels using exact Chartink formulas
    Based on previous month's High, Low, Close data
    """
    
    def __init__(self, cache_manager=None):
        """
        Initialize calculator with optional cache manager
        
        Args:
            cache_manager: Cache instance for storing calculated levels
        """
        self.cache = cache_manager
        
    def calculate_monthly_cpr(self, high: float, low: float, close: float) -> Dict:
        """
        Calculate Central Pivot Range (CPR) using EXACT Chartink formula
        
        Formula from Chartink:
        - Pivot = (High + Low + Close) / 3
        - Bottom Central (BC) = (High + Low) / 2  
        - Top Central (TC) = (Pivot - BC) + Pivot
        - Width = |TC - BC|
        - Width % = (Width / Pivot) * 100
        - Narrow CPR: Width % < 0.5%
        
        Args:
            high: Previous month's highest price
            low: Previous month's lowest price
            close: Previous month's closing price
            
        Returns:
            Dict with CPR levels and analysis
        """
        
        # Validate inputs
        if not all(isinstance(x, (int, float)) and x > 0 for x in [high, low, close]):
            raise ValueError("High, Low, Close must be positive numbers")
            
        if low > high:
            raise ValueError("Low cannot be greater than High")
            
        # Core CPR calculations (exact Chartink formulas)
        pivot = (high + low + close) / 3.0
        bc = (high + low) / 2.0  # Bottom Central
        tc = (pivot - bc) + pivot  # Top Central
        
        # Width calculations
        width = abs(tc - bc)
        width_percent = (width / pivot) * 100 if pivot > 0 else 0
        
        # Narrow CPR detection (Chartink threshold: 0.5%)
        is_narrow = width_percent < 0.5
        
        # Trend determination
        trend = 'bullish' if tc > bc else 'bearish'
        
        # Range determination
        if width_percent < 0.3:
            range_type = 'extremely_narrow'
        elif width_percent < 0.5:
            range_type = 'narrow'
        elif width_percent < 1.0:
            range_type = 'normal'
        else:
            range_type = 'wide'
        
        return {
            'tc': round(tc, 2),
            'pivot': round(pivot, 2),
            'bc': round(bc, 2),
            'width': round(width, 2),
            'width_percent': round(width_percent, 4),
            'is_narrow': is_narrow,
            'trend': trend,
            'range_type': range_type,
            'breakout_level': tc if trend == 'bullish' else bc  # Key level to watch
        }
    
    def calculate_monthly_pivots(self, high: float, low: float, close: float) -> Dict:
        """
        Calculate Pivot Points with Support/Resistance levels
        Using standard pivot point formulas (same as Chartink)
        
        Formulas:
        - Pivot = (H + L + C) / 3
        - R1 = 2 * Pivot - Low
        - R2 = Pivot + (High - Low)
        - R3 = High + 2 * (Pivot - Low)
        - S1 = 2 * Pivot - High
        - S2 = Pivot - (High - Low)
        - S3 = Low - 2 * (High - Pivot)
        
        Args:
            high: Previous month's highest price
            low: Previous month's lowest price  
            close: Previous month's closing price
            
        Returns:
            Dict with pivot levels and proximity zones
        """
        
        # Validate inputs
        if not all(isinstance(x, (int, float)) and x > 0 for x in [high, low, close]):
            raise ValueError("High, Low, Close must be positive numbers")
            
        if low > high:
            raise ValueError("Low cannot be greater than High")
        
        # Core pivot calculation
        pivot = (high + low + close) / 3.0
        
        # Resistance levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)  
        r3 = high + 2 * (pivot - low)
        
        # Support levels
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # Chartink "near pivot" thresholds for scanning
        # Near pivot: within 1% below to 0.1% above
        near_upper = pivot * 1.001  # 0.1% above (tight range)
        near_lower = pivot * 0.99   # 1% below (gives some room)
        
        return {
            'pivot': round(pivot, 2),
            'r1': round(r1, 2),
            'r2': round(r2, 2),
            'r3': round(r3, 2),
            's1': round(s1, 2),
            's2': round(s2, 2),
            's3': round(s3, 2),
            'near_upper': round(near_upper, 2),
            'near_lower': round(near_lower, 2),
            'range': round(high - low, 2),
            'pivot_strength': self._calculate_pivot_strength(pivot, high, low, close)
        }
    
    def _calculate_pivot_strength(self, pivot: float, high: float, low: float, close: float) -> str:
        """
        Determine pivot strength based on close position relative to range
        
        Args:
            pivot: Calculated pivot level
            high: Month high
            low: Month low  
            close: Month close
            
        Returns:
            str: Strength indicator
        """
        range_size = high - low
        close_from_low = close - low
        close_position = close_from_low / range_size if range_size > 0 else 0.5
        
        if close_position > 0.8:
            return 'strong_bullish'
        elif close_position > 0.6:
            return 'moderate_bullish'
        elif close_position > 0.4:
            return 'neutral'
        elif close_position > 0.2:
            return 'moderate_bearish'
        else:
            return 'strong_bearish'
    
    def calculate_and_cache_symbol_levels(self, symbol: str, monthly_ohlc: Dict, 
                                        current_month: str = None) -> Dict:
        """
        Calculate all monthly levels for a symbol and cache them
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            monthly_ohlc: Dict with 'high', 'low', 'close', 'open'
            current_month: Month string (YYYY-MM), defaults to current month
            
        Returns:
            Dict with all calculated levels
        """
        
        if current_month is None:
            current_month = datetime.now().strftime('%Y-%m')
            
        # Validate input data
        required_keys = ['high', 'low', 'close']
        if not all(key in monthly_ohlc for key in required_keys):
            raise ValueError(f"monthly_ohlc must contain: {required_keys}")
        
        high = float(monthly_ohlc['high'])
        low = float(monthly_ohlc['low'])
        close = float(monthly_ohlc['close'])
        
        # Calculate CPR and Pivot levels
        cpr_levels = self.calculate_monthly_cpr(high, low, close)
        pivot_levels = self.calculate_monthly_pivots(high, low, close)
        
        # Combine all data
        levels = {
            'symbol': symbol,
            'month': current_month,
            'calculated_at': datetime.now().isoformat(),
            'source_data': {
                'high': high,
                'low': low,
                'close': close,
                'open': monthly_ohlc.get('open', None)
            },
            'cpr': cpr_levels,
            'pivots': pivot_levels,
            'key_levels': {
                # Most important levels for quick scanning
                'narrow_cpr': cpr_levels['is_narrow'],
                'cpr_breakout_level': cpr_levels['breakout_level'],
                'pivot_level': pivot_levels['pivot'],
                'pivot_range_upper': pivot_levels['near_upper'],
                'pivot_range_lower': pivot_levels['near_lower']
            }
        }
        
        # Cache if cache manager available
        if self.cache:
            cache_key = f"levels:{symbol}:{current_month}"
            # Cache for 35 days (expires after month end)
            self.cache.set(cache_key, levels, expiry_hours=35*24)
            logger.info(f"Cached levels for {symbol} (month: {current_month})")
        
        return levels
    
    def get_cached_levels(self, symbol: str, month: str = None) -> Optional[Dict]:
        """
        Get cached levels for a symbol
        
        Args:
            symbol: Stock symbol
            month: Month string (YYYY-MM), defaults to current month
            
        Returns:
            Cached levels or None if not found
        """
        
        if not self.cache:
            return None
            
        if month is None:
            month = datetime.now().strftime('%Y-%m')
            
        cache_key = f"levels:{symbol}:{month}"
        return self.cache.get(cache_key)
    
    def get_symbols_with_narrow_cpr(self, symbols: List[str], month: str = None) -> List[Dict]:
        """
        Get all symbols with narrow CPR from cache
        
        Args:
            symbols: List of symbols to check
            month: Month to check, defaults to current
            
        Returns:
            List of symbols with narrow CPR data
        """
        
        narrow_cpr_symbols = []
        
        for symbol in symbols:
            levels = self.get_cached_levels(symbol, month)
            if levels and levels['cpr']['is_narrow']:
                narrow_cpr_symbols.append({
                    'symbol': symbol,
                    'cpr_width_percent': levels['cpr']['width_percent'],
                    'breakout_level': levels['cpr']['breakout_level'],
                    'trend': levels['cpr']['trend'],
                    'levels': levels
                })
        
        # Sort by CPR width (narrowest first)
        narrow_cpr_symbols.sort(key=lambda x: x['cpr_width_percent'])
        
        return narrow_cpr_symbols
    
    def get_symbols_near_pivot(self, symbols: List[str], current_prices: Dict[str, float],
                             month: str = None) -> List[Dict]:
        """
        Get symbols currently trading near monthly pivot
        
        Args:
            symbols: List of symbols to check
            current_prices: Dict mapping symbol -> current price
            month: Month to check, defaults to current
            
        Returns:
            List of symbols near pivot with proximity data
        """
        
        near_pivot_symbols = []
        
        for symbol in symbols:
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            levels = self.get_cached_levels(symbol, month)
            
            if not levels:
                continue
                
            pivot_data = levels['pivots']
            
            # Check if current price is near pivot
            if (pivot_data['near_lower'] <= current_price <= pivot_data['near_upper']):
                
                # Calculate proximity percentage
                pivot = pivot_data['pivot']
                proximity_percent = abs(current_price - pivot) / pivot * 100
                
                near_pivot_symbols.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'pivot': pivot,
                    'proximity_percent': proximity_percent,
                    'position': 'above' if current_price > pivot else 'below',
                    'levels': levels
                })
        
        # Sort by proximity (closest first)
        near_pivot_symbols.sort(key=lambda x: x['proximity_percent'])
        
        return near_pivot_symbols