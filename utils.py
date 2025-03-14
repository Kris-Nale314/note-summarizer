"""
Utility functions for transcript analysis.
"""

import nltk
from typing import List, Optional
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

def chunk_transcript_by_character(transcript: str, chunk_size: int = 2000, overlap: int = 100) -> List[str]:
    """
    Splits a transcript into chunks of approximately `chunk_size` characters,
    with an overlap of `overlap` characters between chunks.

    Args:
        transcript (str): The text of the transcript to chunk.
        chunk_size (int): The desired size of each chunk (in characters).
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of string chunks.
    """
    try:
        # Try to use the enhanced chunking module if available
        from utils.chunking import chunk_by_character
        chunk_objects = chunk_by_character(transcript, chunk_size, overlap)
        return [chunk.text for chunk in chunk_objects]
    except ImportError:
        # Fall back to basic chunking
        chunks = []
        start = 0
        while start < len(transcript):
            end = min(start + chunk_size, len(transcript))
            chunks.append(transcript[start:end])
            start += chunk_size - overlap  # Overlap to maintain context
        return chunks


def chunk_transcript_advanced(
    transcript: str, 
    num_chunks: int = 5, 
    overlap: int = 200, 
    strategy: str = "speaker_aware"
) -> List[str]:
    """
    Advanced transcript chunking that uses speaker awareness and natural boundaries.
    
    Args:
        transcript: The transcript text
        num_chunks: Target number of chunks (for fixed-size chunking)
        overlap: Overlap between chunks in characters
        strategy: Chunking strategy ('fixed_size', 'speaker_aware', 'boundary_aware', 'semantic')
        
    Returns:
        List of text chunks
    """
    try:
        # Try to use the advanced chunking module if available
        from utils.chunking import chunk_transcript_advanced as advanced_chunker
        
        # Calculate appropriate chunk size based on num_chunks if using fixed_size
        chunk_size = len(transcript) // max(1, num_chunks - 1) if strategy == "fixed_size" else 2000
        
        return advanced_chunker(
            transcript=transcript,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    except ImportError:
        # Fall back to basic character chunking if advanced module not available
        print("Advanced chunking not available, falling back to character-based chunking")
        chunk_size = len(transcript) // max(1, num_chunks - 1)
        return chunk_transcript_by_character(transcript, chunk_size, overlap)


# For backward compatibility
def extract_speakers(text: str) -> List[str]:
    """
    Extract speaker names from transcript text.
    
    Args:
        text: Transcript text to analyze
        
    Returns:
        List of identified speaker names
    """
    try:
        from utils.chunking import extract_speakers as extract_func
        return extract_func(text)
    except ImportError:
        # Very basic fallback speaker extraction
        import re
        speakers = re.findall(r'([A-Z][a-z]+):', text)
        return list(set(speakers))  # Remove duplicates