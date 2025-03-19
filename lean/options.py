"""
Configuration options for the lean summarization engine.
"""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ProcessingOptions:
    """Configuration options for document processing."""
    # Model configuration
    model_name: str = "default"  # Default model name
    temperature: float = 0.2  # Controls randomness (0.0 to 1.0)
    
    # Chunking parameters
    min_chunks: int = 3  # Minimum number of chunks to create
    max_chunk_size: Optional[int] = None  # Maximum size per chunk, None = auto-calculate
    
    # Analysis options
    preview_length: int = 2000  # Characters to analyze in preview
    
    # Summary options
    detail_level: str = "detailed"  # 'essential', 'detailed', or 'detailed-complex'
    include_action_items: bool = True  # Extract and highlight action items
    
    # Performance options
    max_concurrent_chunks: int = 5  # Maximum concurrent chunk processing
    
    # Output options
    include_metadata: bool = True  # Include metadata in output