import os
import logging
from .base import LLMAdapter

logger = logging.getLogger(__name__)

class LiteLLMAdapter(LLMAdapter):
    """LiteLLM adapter for multi-provider support."""
    
    def __init__(self, model="gpt-3.5-turbo", api_key=None, temperature=0.2):
        try:
            import litellm
            self.litellm = litellm
            if api_key:
                litellm.api_key = api_key
            self.model = model
            self.temperature = temperature
            logger.info(f"Initialized LiteLLM adapter with model {model}")
        except ImportError:
            logger.error("LiteLLM package not installed. Install with: pip install litellm")
            raise
    
    def generate_completion(self, prompt: str) -> str:
        """Generate text completion using LiteLLM."""
        try:
            response = self.litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates detailed, well-organized meeting summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LiteLLM completion: {e}")
            return f"Error generating completion: {str(e)}"