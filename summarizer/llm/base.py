"""
Base class for LLM adapters.
"""

from abc import ABC, abstractmethod

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