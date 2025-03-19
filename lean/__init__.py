"""
Note-Summarizer with lean architecture.
"""

__version__ = "1.0.0"

# Import main components for easy access
from .options import ProcessingOptions
from .factory import SummarizerFactory
from summarizer.llm.async_openai_adapter import AsyncOpenAIAdapter
from .orchestrator import Orchestrator
from .booster import Booster

# Create a convenience function for getting a complete pipeline
def create_pipeline(api_key=None, options=None):
    """Create a complete summarization pipeline with default or custom options."""
    return factory.create_pipeline(api_key, options)

