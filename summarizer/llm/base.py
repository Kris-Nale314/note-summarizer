from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAdapter(ABC):
    """Base class for LLM adapters."""
    
    @abstractmethod
    def generate_completion(self, prompt: str) -> str:
        """Generate text completion from a prompt."""
        pass