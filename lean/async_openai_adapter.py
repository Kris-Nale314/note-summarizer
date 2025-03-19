"""
Async OpenAI adapter for the summarization engine.
Supports both synchronous and asynchronous API calls.
"""

import os
import logging
import time
import asyncio
from typing import Optional, Dict, Any, List

from openai import OpenAI
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncOpenAIAdapter:
    """OpenAI adapter with async support."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, 
                temperature: float = 0.2):
        """
        Initialize the OpenAI adapter.
        
        Args:
            model: Model name to use
            api_key: OpenAI API key (defaults to environment variable)
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        
        # Set API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        # Initialize both sync and async clients
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
    
    def generate_completion(self, prompt: str, max_retries: int = 2) -> str:
        """
        Generate a completion from the OpenAI API (synchronous).
        
        Args:
            prompt: Prompt to complete
            max_retries: Maximum number of retries on error
            
        Returns:
            Generated completion text
        """
        retries = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max retries exceeded: {e}")
                    raise
                
                # Simple exponential backoff
                wait_time = 2 ** retries
                logger.warning(f"API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    async def generate_completion_async(self, prompt: str, max_retries: int = 2) -> str:
        """
        Generate a completion from the OpenAI API (asynchronous).
        
        Args:
            prompt: Prompt to complete
            max_retries: Maximum number of retries on error
            
        Returns:
            Generated completion text
        """
        retries = 0
        while True:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max retries exceeded in async call: {e}")
                    raise
                
                # Simple exponential backoff
                wait_time = 2 ** retries
                logger.warning(f"API error in async call: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def generate_completions_concurrently(self, prompts: List[str], 
                                               max_concurrent: int = 3) -> List[str]:
        """
        Generate multiple completions concurrently.
        
        Args:
            prompts: List of prompts to complete
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of generated completion texts
        """
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_prompt(prompt):
            async with semaphore:
                return await self.generate_completion_async(prompt)
        
        # Create tasks for all prompts
        tasks = [process_prompt(prompt) for prompt in prompts]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing prompt {i}: {result}")
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results