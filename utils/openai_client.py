"""
Simple OpenAI API client for transcript processing.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIClient:
    """Simple wrapper for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (uses env var if None)
            model: Default model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        
        logger.info(f"Initialized OpenAI client with model {model}")
    
    def generate_completion(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Generate text completion using OpenAI API.
        
        Args:
            prompt: The prompt text
            model: Model to use (defaults to instance model)
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI completion: {e}")
            return f"Error: {str(e)}"