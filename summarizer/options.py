from dataclasses import dataclass
from typing import Optional

@dataclass
class SummaryOptions:
    """Configuration options for transcript summarization."""
    chunk_strategy: str = "simple"  # "simple", "speaker", or "boundary"
    model_name: str = "gpt-3.5-turbo"
    max_chunk_size: int = 2000
    chunk_overlap: int = 200
    include_action_items: bool = True
    temperature: float = 0.2
    verbose: bool = False
    
    # Advanced options - can be expanded later
    max_tokens: Optional[int] = None