import os
import logging
from .base import LLMAdapter

logger = logging.getLogger(__name__)

class OpenAIAdapter(LLMAdapter):
    """Direct OpenAI API adapter."""
    
    def __init__(self, model="gpt-3.5-turbo", api_key=None, temperature=0.2):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
            self.temperature = temperature
            logger.info(f"Initialized OpenAI adapter with model {model}")
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def generate_completion(self, prompt: str) -> str:
        """Generate text completion using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates detailed, well-organized meeting summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI completion: {e}")
            return f"Error generating completion: {str(e)}"