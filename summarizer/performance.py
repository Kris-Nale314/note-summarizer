"""
Performance optimizations for the document processing pipeline.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Callable, Coroutine, TypeVar, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import pickle
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')

class PerformanceOptimizer:
    """Utilities to optimize document processing performance."""
    
    def __init__(self, cache_dir: str = ".cache", max_workers: int = 4, enable_caching: bool = True):
        """
        Initialize the performance optimizer.
        
        Args:
            cache_dir: Directory to store cache files
            max_workers: Maximum number of parallel workers
            enable_caching: Whether to enable result caching
        """
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    async def process_in_parallel(self, items: List[Any], process_func: Callable[[Any], T], 
                                 chunk_size: int = 1) -> List[T]:
        """
        Process items in parallel using asyncio.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            chunk_size: Number of items to process in each worker
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Create chunks
        chunked_items = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Create a thread pool executor
        executor = ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunked_items)))
        loop = asyncio.get_event_loop()
        
        # Define the processing function for each chunk
        async def process_chunk(chunk: List[Any]) -> List[T]:
            tasks = []
            for item in chunk:
                # Check cache first if enabled
                if self.enable_caching:
                    cached_result = self._get_from_cache(item, process_func)
                    if cached_result is not None:
                        tasks.append(cached_result)
                        continue
                
                # Not in cache, process the item
                task = loop.run_in_executor(executor, process_func, item)
                tasks.append(task)
            
            # Wait for all tasks in this chunk to complete
            results = await asyncio.gather(*tasks)
            
            # Cache results
            if self.enable_caching:
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        self._save_to_cache(chunk[i], process_func, result)
            
            return results
        
        # Process all chunks in parallel
        tasks = [process_chunk(chunk) for chunk in chunked_items]
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    def _get_cache_key(self, item: Any, func: Callable) -> str:
        """Generate a unique cache key for an item and function."""
        # Generate a hash based on the item and function name
        func_name = func.__name__
        
        # Special handling for different item types
        if isinstance(item, dict):
            # For dictionaries, create a deterministic string representation
            item_str = str(sorted(item.items()))
        elif isinstance(item, str):
            # For strings, use a hash of the content
            item_str = str(len(item)) + "_" + item[:100] + item[-100:] if len(item) > 200 else item
        else:
            # For other types, use str representation
            item_str = str(item)
        
        # Create a hash for the cache key
        key = f"{func_name}_{hashlib.md5(item_str.encode()).hexdigest()}"
        return key
    
    def _get_from_cache(self, item: Any, func: Callable) -> Optional[Any]:
        """Get a result from cache if available."""
        try:
            cache_key = self._get_cache_key(item, func)
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                logger.info(f"Cache hit for {func.__name__}")
                return result
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _save_to_cache(self, item: Any, func: Callable, result: Any) -> None:
        """Save a result to cache."""
        try:
            cache_key = self._get_cache_key(item, func)
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Saved result to cache for {func.__name__}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    @staticmethod
    def measure_execution_time(func: Callable) -> Callable:
        """
        Decorator to measure and log function execution time.
        
        Args:
            func: Function to measure
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
            return result
        return wrapper


async def divide_document_parallel(text: str, min_sections: int, 
                                target_tokens_per_section: int, 
                                api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Use parallel processing to divide document with all strategies.
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        api_key: OpenAI API key for embeddings (complex strategy only)
        
    Returns:
        Dictionary with division results for all strategies
    """
    # Import division functions
    from .division import divide_essential, divide_long, divide_complex
    
    # Create performance optimizer
    optimizer = PerformanceOptimizer(cache_dir=".cache", max_workers=3, enable_caching=True)
    start_time = time.time()
    
    # Define functions for each strategy with timing
    async def process_essential():
        t_start = time.time()
        divisions = divide_essential(text, min_sections, target_tokens_per_section)
        process_time = time.time() - t_start
        return "essential", divisions, process_time
        
    async def process_long():
        t_start = time.time()
        divisions = divide_long(text, min_sections, target_tokens_per_section)
        process_time = time.time() - t_start
        return "long", divisions, process_time
        
    async def process_complex():
        t_start = time.time()
        divisions = divide_complex(text, min_sections, target_tokens_per_section, api_key)
        process_time = time.time() - t_start
        return "complex", divisions, process_time
    
    # Run the tasks in parallel
    tasks = [process_essential(), process_long(), process_complex()]
    results = await asyncio.gather(*tasks)
    
    # Organize results
    comparison_results = {
        "metadata": {
            "total_processing_time": time.time() - start_time,
            "text_length": len(text),
            "estimated_tokens": len(text) // 4
        }
    }
    
    for strategy, divisions, process_time in results:
        comparison_results[strategy] = {
            "divisions": divisions,
            "count": len(divisions),
            "processing_time": process_time
        }
    
    return comparison_results


def tiered_summarization(long_text: str, llm_client: Any, max_tokens: int = 30000) -> str:
    """
    Perform tiered summarization for very long texts.
    
    This uses a multi-pass approach, first summarizing sections with a faster model,
    then combining those summaries with a more powerful model.
    
    Args:
        long_text: The text to summarize
        llm_client: LLM client to use
        max_tokens: Maximum tokens per request
        
    Returns:
        Summarized text
    """
    logger.info(f"Starting tiered summarization for text of length {len(long_text)}")
    
    # Exit early if text isn't that long
    if len(long_text) < max_tokens * 2:  # Rough char to token ratio approximation
        logger.info(f"Text is short enough for direct summarization ({len(long_text)} chars)")
        return llm_client.generate_completion(
            f"Summarize this text thoroughly, capturing all important information:\n\n{long_text}"
        )
    
    # Step 1: Break into sections
    from .division import divide_essential
    sections = divide_essential(long_text, min_sections=3, 
                             target_tokens_per_section=max_tokens // 2,
                             section_overlap=0.05)
    
    logger.info(f"Divided text into {len(sections)} sections for tiered processing")
    
    # Step 2: Summarize each section with a simpler prompt and faster model
    section_summaries = []
    current_model = llm_client.model
    
    # Use a faster model for initial summaries if available
    try:
        if "gpt-4" in current_model:
            llm_client.model = "gpt-3.5-turbo-16k"
            logger.info(f"Using {llm_client.model} for initial section summaries")
        
        for i, section in enumerate(sections):
            logger.info(f"Processing section {i+1}/{len(sections)}")
            prompt = f"""Summarize this section {i+1}/{len(sections)} of a document.
            Focus on extracting all important information, key points, and significant details.
            Be thorough and preserve the context and meaning of the original content.
            
            SECTION TEXT:
            {section['text']}
            """
            
            summary = llm_client.generate_completion(prompt)
            section_summaries.append(summary)
            
        # Step 3: Combine section summaries with the original model
        logger.info(f"Switching back to {current_model} for final synthesis")
        llm_client.model = current_model
    except Exception as e:
        # Ensure model is restored even if something fails
        llm_client.model = current_model
        logger.error(f"Error in tiered summarization: {e}")
        raise
    
    # Combine section summaries
    combined_summaries = "\n\n===== SECTION SEPARATOR =====\n\n".join([
        f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(section_summaries)
    ])
    
    # Final synthesis with original model
    logger.info("Generating final synthesis")
    synthesis_prompt = f"""Create a cohesive, well-structured summary from these section summaries.
    
    The final summary should:
    1. Start with a concise Executive Summary of the main points
    2. Organize content logically by topic instead of by section
    3. Include all significant information and important details
    4. Remove redundancies while preserving completeness
    5. Use clear headings and structure for readability
    
    SECTION SUMMARIES:
    {combined_summaries}
    """
    
    final_summary = llm_client.generate_completion(synthesis_prompt)
    logger.info("Tiered summarization complete")
    return final_summary


async def process_with_backoff(func: Callable, *args, max_retries: int = 3, base_delay: int = 2, **kwargs) -> Any:
    """
    Execute a function with exponential backoff for API rate limits.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result
    """
    retries = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if it's a rate limit error
            error_msg = str(e).lower()
            
            if any(term in error_msg for term in ["rate limit", "too many requests", "429"]):
                retries += 1
                if retries > max_retries:
                    logger.error(f"Maximum retries exceeded: {e}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** (retries - 1))
                logger.warning(f"Rate limit hit. Retrying in {delay}s... (Attempt {retries}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                # Not a rate limit error, just raise it
                raise


def run_with_performance_logging(func: Callable[[str], Dict[str, Any]], text: str, **kwargs) -> Dict[str, Any]:
    """
    Run a document processing function with performance logging.
    
    Args:
        func: Function to execute (usually summarize or process_document)
        text: Document text
        **kwargs: Additional arguments for the function
        
    Returns:
        Function result with performance metrics added
    """
    start_time = time.time()
    
    # Generate cache key for checking if we've processed this before
    cache_key = f"{func.__name__}_{len(text)}_{str(sorted(kwargs.items()))}"
    cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    # Check cache
    optimizer = PerformanceOptimizer(cache_dir=".cache", enable_caching=True)
    cached_result = optimizer._get_from_cache(cache_key_hash, func)
    
    if cached_result:
        cached_result["metadata"]["from_cache"] = True
        cached_result["metadata"]["original_processing_time"] = cached_result["metadata"].get("processing_time_seconds", 0)
        cached_result["metadata"]["processing_time_seconds"] = 0.01  # Nearly instant
        return cached_result
    
    # Execute the function
    result = func(text, **kwargs)
    
    # Add performance metrics
    if "metadata" not in result:
        result["metadata"] = {}
    
    result["metadata"]["processing_time_seconds"] = time.time() - start_time
    result["metadata"]["text_length"] = len(text)
    result["metadata"]["estimated_tokens"] = len(text) // 4
    
    # Cache the result
    optimizer._save_to_cache(cache_key_hash, func, result)
    
    return result