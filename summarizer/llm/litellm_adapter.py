"""
LiteLLM adapter for multi-provider support.
"""

import os
import logging
from typing import Optional
from .base import LLMAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiteLLMAdapter(LLMAdapter):
    """LiteLLM adapter for multi-provider support."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, temperature: float = 0.2):
        """
        Initialize the LiteLLM adapter.
        
        Args:
            model: Model identifier to use
            api_key: API key (uses env var if None)
            temperature: Temperature for text generation
        """
        try:
            import litellm
            self.litellm = litellm
            
            if api_key:
                litellm.api_key = api_key
                
            self.model = model
            self.temperature = temperature
            
            logger.info(f"Initialized LiteLLM adapter with model {model}")
        except ImportError as e:
            logger.error(f"LiteLLM package not installed: {e}")
            raise ImportError("LiteLLM package not installed. Install with: pip install litellm") from e
    
    def generate_completion(self, prompt: str) -> str:
        """
        Generate text completion using LiteLLM.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Generated text
        """
        try:
            # Check if prompt may be too long and truncate if needed
            if len(prompt) > 12000 and "gpt-3.5-turbo" in self.model and "16k" not in self.model:
                logger.warning(f"Prompt may be too long ({len(prompt)} chars), truncating for {self.model}")
                prompt_length = len(prompt)
                keep_chars = 10000
                half = keep_chars // 2
                prompt = prompt[:half] + "\n\n[...content truncated for length...]\n\n" + prompt[-half:]
                logger.info(f"Truncated prompt from {prompt_length} to {len(prompt)} chars")
            
            response = self.litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates detailed, well-organized document summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LiteLLM completion: {e}")
            
            # Check for specific error types
            if "maximum context length" in str(e) or "context window" in str(e) or "ContextWindowExceededError" in str(e):
                # Return a more helpful error for context length issues
                logger.error(f"Context window exceeded with {self.model}")
                return f"Error: This document is too large for the {self.model} model. Please try using a model with a larger context window (like gpt-3.5-turbo-16k or gpt-4) or adjust the division settings to create smaller sections."
            
            return f"Error generating completion: {str(e)}"