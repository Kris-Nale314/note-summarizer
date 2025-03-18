"""
Base class for LLM adapters.
"""

from abc import ABC, abstractmethod
import asyncio
import time
import logging
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)

class LLMAdapter(ABC):
    """Base class for LLM adapters."""
    
    @abstractmethod
    def generate_completion(self, prompt: str) -> str:
        """
        Generate text completion from a prompt.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Generated text
        """
        pass
    
    async def generate_completion_with_backoff(self, prompt: str, 
                                            max_retries: int = 3, 
                                            base_delay: int = 2) -> str:
        """
        Generate completion with exponential backoff for API rate limits.
        
        Args:
            prompt: The prompt text
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            
        Returns:
            Generated text
        """
        retries = 0
        while True:
            try:
                return self.generate_completion(prompt)
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