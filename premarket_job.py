#!/usr/bin/env python3
"""
premarket_job.py - Automated pre-market job for calculating monthly levels
Runs scheduled tasks to fetch historical data and calculate CPR/Pivot levels
"""

import asyncio
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from cache_manager import CacheManager
from scanners.monthly_levels import MonthlyLevelCalculator
from dhan_fetcher import DhanHistoricalFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PremarketJob:
    """
    Automated pre-market job system for calculating monthly levels
    Handles scheduled execution, error recovery, and progress tracking
    """
    
    def __init__(self):
        """Initialize pre-market job with components"""
        self.cache = CacheManager()
        self.calculator = MonthlyLevelCalculator(self.cache)
        self.scheduler = AsyncIOScheduler()
        
        # Get Dhan credentials from environment
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        # Job tracking
        self.last_run = None
        self.last_success = None
        self.is_running = False
        
        logger.info("PremarketJob initialized")
    
    async def calculate_monthly_levels_for_all_symbols(self) -> dict:
        """
        Calculate monthly levels for all F&O symbols using real Dhan API data
        This is the main job that runs pre-market
        """
        
        if self.is_running:
            logger.warning("Job already running, skipping...")
            return {"error": "Job already in progress"}
        
        if not self.client_id or not self.access_token:
            logger.error("Dhan credentials not found in environment variables")
            return {"error": "DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN required"}
        
        self.is_running = True
        self.last_run = datetime.now()
        
        try:
            logger.info("Starting monthly level calculation for all F&O symbols...")
            
            # Initialize Dhan fetcher
            async with DhanHistoricalFetcher(self.client_id, self.access_token) as fetcher:
                
                # Step 1: Get active F&O instruments
                logger.info("Fetching active F&O instruments...")
                instruments_df = await fetcher.get_instruments()
                
                if instruments_df.empty:
                    raise Exception("No instruments data received from Dhan API")
                
                active_futures = fetcher.get_active_fno_futures(instruments_df)
                logger.info(f"Found {len(active_futures)} active F&O futures")
                
                # Step 2: Extract unique underlying symbols
                underlying_symbols = set()
                for _, row in active_futures.iterrows():
                    # Use the column names that exist in the CSV
                    symbol_col = 'SEM_TRADING_SYMBOL' if 'SEM_TRADING_SYMBOL' in row else 'SYMBOL_NAME'
                    underlying = fetcher.extract_underlying_symbol(row[symbol_col])
                    if underlying:
                        underlying_symbols.add(underlying)

                logger.info(f"Processing {len(underlying_symbols)} underlying symbols")

                # Step 2a: Load equity instruments to get security IDs
                logger.info("Loading equity instrument mappings...")
                equity_mapping = await fetcher.load_equity_instruments()
                if not equity_mapping:
                    raise Exception("Failed to load equity instrument mappings")
                logger.info(f"Loaded {len(equity_mapping)} equity instrument mappings")
                
                # Step 3: Calculate date ranges (previous month)
                today = datetime.now()
                first_day_current = today.replace(day=1)
                last_day_previous = first_day_current - timedelta(days=1)
                first_day_previous = last_day_previous.replace(day=1)
                
                logger.info(f"Fetching data for period: {first_day_previous.date()} to {last_day_previous.date()}")
                
                # Step 4: Process each symbol
                results = []
                failed = []
                processed = 0
                
                for symbol in underlying_symbols:
                    try:
                        # Check if already cached for current month
                        current_month = today.strftime('%Y-%m')
                        cached = self.calculator.get_cached_levels(symbol, current_month)
                        
                        if cached:
                            logger.debug(f"Using cached levels for {symbol}")
                            results.append({
                                'symbol': symbol,
                                'status': 'cached',
                                'data': cached
                            })
                            processed += 1
                            continue
                        
                        # Get security ID for the symbol
                        security_id = equity_mapping.get(symbol)
                        if not security_id:
                            logger.warning(f"No security ID found for {symbol}")
                            failed.append({
                                'symbol': symbol,
                                'error': 'No security ID mapping found'
                            })
                            continue

                        # Fetch historical data for previous month
                        logger.info(f"Fetching historical data for {symbol} (security_id={security_id})...")

                        # Calculate number of days to fetch (previous month + buffer)
                        days_to_fetch = (today - first_day_previous).days + 5  # Add 5 days buffer

                        historical_df = await fetcher.get_historical_data_for_underlying(
                            symbol,
                            int(security_id),  # Ensure it's an integer
                            days=days_to_fetch
                        )

                        if historical_df is None or historical_df.empty:
                            logger.warning(f"No historical data for {symbol}")
                            failed.append({
                                'symbol': symbol,
                                'error': 'No historical data available'
                            })
                            continue

                        # Filter data for the previous month only
                        historical_df['date'] = pd.to_datetime(historical_df['date'])
                        month_data = historical_df[
                            (historical_df['date'] >= first_day_previous) &
                            (historical_df['date'] <= last_day_previous)
                        ]

                        if month_data.empty:
                            logger.warning(f"No data for previous month for {symbol}")
                            failed.append({
                                'symbol': symbol,
                                'error': f'No data for {first_day_previous.strftime("%Y-%m")}'
                            })
                            continue

                        # Calculate monthly OHLC from filtered data
                        monthly_ohlc = {
                            'high': month_data['high'].max(),
                            'low': month_data['low'].min(),
                            'close': month_data.iloc[-1]['close'],  # Last day's close
                            'open': month_data.iloc[0]['open']      # First day's open
                        }
                        
                        # Calculate and cache levels
                        levels = self.calculator.calculate_and_cache_symbol_levels(
                            symbol, 
                            monthly_ohlc, 
                            current_month
                        )
                        
                        results.append({
                            'symbol': symbol,
                            'status': 'calculated',
                            'data': levels
                        })
                        
                        processed += 1
                        logger.info(f"[{processed}/{len(underlying_symbols)}] Processed {symbol}")
                        
                        # Add small delay to avoid overwhelming API
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        failed.append({
                            'symbol': symbol,
                            'error': str(e)
                        })
                    
                # Step 5: Generate summary
                summary = {
                    'job_id': f"premarket_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'completed_at': datetime.now().isoformat(),
                    'execution_time_seconds': (datetime.now() - self.last_run).total_seconds(),
                    'total_symbols': len(underlying_symbols),
                    'successful': len(results),
                    'failed': len(failed),
                    'cached_count': len([r for r in results if r['status'] == 'cached']),
                    'calculated_count': len([r for r in results if r['status'] == 'calculated']),
                    'narrow_cpr_count': len([r for r in results if r['data']['cpr']['is_narrow']]),
                    'failed_symbols': failed[:10]  # Limit failed list to first 10
                }
                
                # Step 6: Cache summary for API access
                summary_cache_key = f"premarket_summary:{today.strftime('%Y-%m-%d')}"
                self.cache.set(summary_cache_key, summary, expiry_hours=24)
                
                # Step 7: Cache aggregated scan results
                narrow_cpr_symbols = [
                    {
                        'symbol': r['symbol'],
                        'cpr_width_percent': r['data']['cpr']['width_percent'],
                        'breakout_level': r['data']['cpr']['breakout_level'],
                        'trend': r['data']['cpr']['trend']
                    }
                    for r in results if r['data']['cpr']['is_narrow']
                ]
                
                scan_results = {
                    'narrow_cpr': narrow_cpr_symbols,
                    'total_symbols': len(underlying_symbols),
                    'last_updated': datetime.now().isoformat()
                }
                
                scan_cache_key = f"scan_results:{today.strftime('%Y-%m')}"
                self.cache.set(scan_cache_key, scan_results, expiry_hours=24*7)  # Cache for a week
                
                logger.info(f"Job completed successfully: {summary}")
                
                self.last_success = datetime.now()
                return summary
                
        except Exception as e:
            logger.error(f"Pre-market job failed: {e}")
            return {
                'error': str(e),
                'failed_at': datetime.now().isoformat(),
                'execution_time_seconds': (datetime.now() - self.last_run).total_seconds()
            }
        
        finally:
            self.is_running = False
    
    async def health_check(self) -> dict:
        """
        Check the health of pre-market job system
        """
        
        health = {
            'timestamp': datetime.now().isoformat(),
            'scheduler_running': self.scheduler.running,
            'cache_health': self.cache.health_check(),
            'credentials_available': bool(self.client_id and self.access_token),
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'last_success': self.last_success.isoformat() if self.last_success else None,
            'currently_running': self.is_running
        }
        
        return health
    
    async def get_latest_results(self) -> dict:
        """
        Get latest pre-market calculation results
        """
        
        today = datetime.now()
        summary_key = f"premarket_summary:{today.strftime('%Y-%m-%d')}"
        scan_key = f"scan_results:{today.strftime('%Y-%m')}"
        
        summary = self.cache.get(summary_key)
        scan_results = self.cache.get(scan_key)
        
        return {
            'summary': summary,
            'scan_results': scan_results,
            'retrieved_at': datetime.now().isoformat()
        }
    
    def schedule_jobs(self):
        """
        Schedule pre-market jobs using APScheduler
        """
        
        # Main job: Run at 8:30 AM every trading day (Mon-Fri)
        self.scheduler.add_job(
            func=self.calculate_monthly_levels_for_all_symbols,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=8,
                minute=30,
                timezone='Asia/Kolkata'  # IST timezone
            ),
            id='daily_premarket_levels',
            name='Daily Pre-market Level Calculation',
            replace_existing=True,
            max_instances=1  # Prevent overlapping runs
        )
        
        # Monthly job: Run on 1st trading day of month at 8:00 AM
        self.scheduler.add_job(
            func=self.calculate_monthly_levels_for_all_symbols,
            trigger=CronTrigger(
                day=1,
                hour=8,
                minute=0,
                timezone='Asia/Kolkata'
            ),
            id='monthly_level_refresh',
            name='Monthly Level Refresh',
            replace_existing=True,
            max_instances=1
        )
        
        # Cleanup job: Clear expired cache entries daily at 2 AM
        self.scheduler.add_job(
            func=self._cleanup_cache,
            trigger=CronTrigger(
                hour=2,
                minute=0,
                timezone='Asia/Kolkata'
            ),
            id='cache_cleanup',
            name='Cache Cleanup',
            replace_existing=True
        )
        
        logger.info("Scheduled pre-market jobs:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job.name}: {job.trigger}")
        
        self.scheduler.start()
        logger.info("Scheduler started successfully")
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            cleared = self.cache.clear_expired()
            logger.info(f"Cleaned up {cleared} expired cache entries")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

# Standalone execution
async def run_premarket_job_now():
    """
    Run the pre-market job immediately (for testing/manual execution)
    """
    job = PremarketJob()
    result = await job.calculate_monthly_levels_for_all_symbols()
    
    print("=" * 60)
    print("PRE-MARKET JOB EXECUTION RESULT")
    print("=" * 60)
    print(f"Result: {result}")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    # Run job immediately for testing
    print("Running pre-market job immediately...")
    result = asyncio.run(run_premarket_job_now())