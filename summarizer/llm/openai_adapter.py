"""
OpenAI API adapter for LLM interactions.
"""

import os
import logging
from typing import Optional
from .base import LLMAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIAdapter(LLMAdapter):
    """Direct OpenAI API adapter."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, temperature: float = 0.2):
        """
        Initialize the OpenAI adapter.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (uses env var if None)
            temperature: Temperature for text generation
        """
        try:
            import openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            
            if not self.api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
                
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = model
            self.temperature = temperature
            
            logger.info(f"Initialized OpenAI adapter with model {model}")
        except ImportError as e:
            logger.error(f"OpenAI package not installed: {e}")
            raise ImportError("OpenAI package not installed. Install with: pip install openai>=1.0.0") from e
    
    def generate_completion(self, prompt: str) -> str:
        """
        Generate text completion using OpenAI API.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Generated text
        """
        try:
            # Check if prompt may be too long and truncate if needed
            if len(prompt) > 12000 and "3.5" in self.model and "16k" not in self.model:
                logger.warning(f"Prompt may be too long ({len(prompt)} chars), truncating for {self.model}")
                prompt_length = len(prompt)
                keep_chars = 10000
                half = keep_chars // 2
                prompt = prompt[:half] + "\n\n[...content truncated for length...]\n\n" + prompt[-half:]
                logger.info(f"Truncated prompt from {prompt_length} to {len(prompt)} chars")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates detailed, well-organized document summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI completion: {e}")
            
            # Check for specific error types
            if "maximum context length" in str(e) or "context window" in str(e):
                # Return a more helpful error for context length issues
                logger.error(f"Context window exceeded with {self.model}")
                return f"Error: This document is too large for the {self.model} model. Please try using a model with a larger context window (like gpt-3.5-turbo-16k or gpt-4) or adjust the division settings to create smaller sections."
            
            return f"Error generating completion: {str(e)}"