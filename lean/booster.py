"""
Booster module that enhances performance and optimizes processing.
"""

import asyncio
import time
import logging
import functools
import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Callable, TypeVar, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Define a type variable for generic function signatures
T = TypeVar('T')

class Booster:
    """Enhances performance of document processing with caching and parallel execution."""
    
    def __init__(self, cache_dir: str = ".cache", max_workers: int = 4, enable_caching: bool = True):
        """
        Initialize the performance booster.
        
        Args:
            cache_dir: Directory to store cache files
            max_workers: Maximum number of parallel workers
            enable_caching: Whether to enable result caching
        """
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    async def process_in_parallel(self, 
                                items: List[Any], 
                                process_func: Callable[[Any], T], 
                                max_concurrency: Optional[int] = None) -> List[T]:
        """
        Process items in parallel with controlled concurrency.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item (can be sync or async)
            max_concurrency: Maximum concurrent tasks (defaults to self.max_workers)
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        # Use provided concurrency limit or fall back to max_workers
        concurrency_limit = max_concurrency or self.max_workers
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        # Define worker function that respects the semaphore
        async def worker(item, index):
            # Try to get from cache first if enabled
            if self.enable_caching:
                cached_result = self._get_from_cache(item, process_func)
                if cached_result is not None:
                    logger.debug(f"Cache hit for item {index}")
                    return cached_result
            
            # Not in cache, acquire semaphore and process
            async with semaphore:
                try:
                    # Check if the function is already async
                    if asyncio.iscoroutinefunction(process_func):
                        result = await process_func(item)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, process_func, item)
                    
                    # Cache result if enabled
                    if self.enable_caching:
                        self._save_to_cache(item, process_func, result)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    raise
        
        # Create tasks for all items
        tasks = [worker(item, i) for i, item in enumerate(items)]
        
        # Execute all tasks and wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} raised an exception: {result}")
                raise result  # Re-raise to caller
            processed_results.append(result)
        
        return processed_results
    
    def _get_cache_key(self, item: Any, func: Callable) -> str:
        """Generate a unique cache key for an item and function."""
        # Get function name or qualified name if available
        func_name = getattr(func, "__qualname__", func.__name__)
        
        # Get module name if available
        if hasattr(func, "__module__") and func.__module__ is not None:
            func_name = f"{func.__module__}.{func_name}"
        
        # Handle different item types
        if isinstance(item, dict):
            # For dictionaries, create a deterministic string representation
            # Sort keys to ensure consistency
            item_str = str(sorted(item.items()))
        elif isinstance(item, str):
            # For strings, use a truncated representation with length info
            item_str = f"len={len(item)}_{item[:50]}..{item[-50:]}" if len(item) > 100 else item
        else:
            # For other types, use string representation
            item_str = str(item)
        
        # Create a hash for the cache key
        key = f"{func_name}_{hashlib.md5(item_str.encode()).hexdigest()}"
        return key
    
    def _get_from_cache(self, item: Any, func: Callable) -> Optional[Any]:
        """Get a result from cache if available."""
        if not self.enable_caching:
            return None
            
        try:
            cache_key = self._get_cache_key(item, func)
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                logger.debug(f"Cache hit for {func.__name__}")
                return result
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _save_to_cache(self, item: Any, func: Callable, result: Any) -> None:
        """Save a result to cache."""
        if not self.enable_caching:
            return
            
        try:
            cache_key = self._get_cache_key(item, func)
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Saved result to cache for {func.__name__}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def process_with_retry(self, 
                               func: Callable, 
                               *args, 
                               max_retries: int = 3, 
                               base_delay: float = 2.0, 
                               **kwargs) -> Any:
        """
        Execute a function with exponential backoff for error handling.
        
        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                # Check if function is async
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, 
                        functools.partial(func, *args, **kwargs)
                    )
            except Exception as e:
                last_exception = e
                retries += 1
                
                # Check if we should retry
                if retries > max_retries:
                    logger.error(f"Maximum retries ({max_retries}) exceeded: {e}")
                    break
                
                # Calculate delay with exponential backoff and small jitter
                import random
                jitter = random.uniform(0.0, 0.1) * base_delay
                delay = (base_delay * (2 ** (retries - 1))) + jitter
                
                logger.warning(f"Error in function {func.__name__}: {e}. Retrying in {delay:.2f}s... ({retries}/{max_retries})")
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        raise last_exception if last_exception else RuntimeError("Function failed with no exception")
    
    async def process_hierarchical(self, 
                                 items: List[Any],
                                 hierarchical_funcs: List[Tuple[Callable, int]]) -> Any:
        """
        Process items in a hierarchical manner with different grouping levels.
        
        Args:
            items: Initial list of items to process
            hierarchical_funcs: List of (function, group_size) tuples defining each level
            
        Returns:
            Final result after all hierarchical processing levels
        """
        current_items = items
        
        # Process each level
        for level, (level_func, group_size) in enumerate(hierarchical_funcs):
            logger.info(f"Processing hierarchical level {level+1} with {len(current_items)} items")
            
            # If only one item remains, just process it directly
            if len(current_items) == 1:
                if asyncio.iscoroutinefunction(level_func):
                    current_items = [await level_func(current_items[0])]
                else:
                    current_items = [level_func(current_items[0])]
                continue
            
            # Group items
            groups = self._create_groups(current_items, group_size)
            logger.info(f"Created {len(groups)} groups for level {level+1}")
            
            # Process each group
            group_results = []
            for group_idx, group in enumerate(groups):
                try:
                    if asyncio.iscoroutinefunction(level_func):
                        result = await level_func(group)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, level_func, group)
                    
                    group_results.append(result)
                    logger.debug(f"Processed group {group_idx+1}/{len(groups)} in level {level+1}")
                except Exception as e:
                    logger.error(f"Error processing group {group_idx} in level {level+1}: {e}")
                    raise
            
            # Update current items for next level
            current_items = group_results
        
        # Return final result (should be a single item after all levels)
        if len(current_items) == 1:
            return current_items[0]
        else:
            return current_items
    
    def _create_groups(self, items: List[Any], group_size: int) -> List[List[Any]]:
        """Create groups of items with the specified size."""
        return [items[i:i+group_size] for i in range(0, len(items), group_size)]
    
    @staticmethod
    def measure_time(func: Callable) -> Callable:
        """
        Decorator to measure and log function execution time.
        
        Args:
            func: Function to measure
            
        Returns:
            Wrapped function that logs execution time
        """
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper