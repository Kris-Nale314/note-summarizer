"""
Configuration options for the document summarization engine.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SummaryOptions:
    """Configuration options for document summarization."""
    # Model configuration
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.2
    
    # Max tokens in each chunk sent to the model
    max_chunk_size: int = 12000  # Conservative size for most OpenAI models
    
    # Processing options
    include_action_items: bool = True  # Whether to extract action items