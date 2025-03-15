"""
Configuration options for the document summarization engine.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SummaryOptions:
    """Configuration options for document summarization."""
    # Division strategy to use
    division_strategy: str = "basic"  # "basic", "speaker", "boundary", or "context_aware"
    
    # Model configuration
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.2
    
    # Division parameters
    min_sections: int = 3  # Minimum number of sections to divide into
    target_tokens_per_section: int = 25000  # Target tokens per section
    section_overlap: float = 0.1  # Overlap between sections as a fraction
    
    # Processing options
    include_action_items: bool = True  # Whether to extract action items
    verbose: bool = False  # Whether to log verbose details