import re
from typing import List, Set, Dict, Any, Optional
import logging
import nltk

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

def chunk_simple(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Simple chunking with intelligent breaks at sentences or paragraphs.
    
    Args:
        text: Text to chunk
        chunk_size: Target size for chunks
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Calculate end position
        end_pos = min(current_pos + chunk_size, len(text))
        
        # If we're not at the end, try to find a better break point
        if end_pos < len(text):
            # Look for a good break point in the last ~20% of the chunk
            search_start = end_pos - min(int(chunk_size * 0.2), 500)
            search_text = text[search_start:end_pos]
            
            # Try to find a paragraph break
            paragraph_pos = search_text.rfind('\n\n')
            if paragraph_pos > 0:
                end_pos = search_start + paragraph_pos + 2  # Include the newlines
            else:
                # Try to find a newline
                newline_pos = search_text.rfind('\n')
                if newline_pos > 0:
                    end_pos = search_start + newline_pos + 1  # Include the newline
                else:
                    # Try to find a sentence break
                    sentence_pos = max(
                        search_text.rfind('. '),
                        search_text.rfind('? '),
                        search_text.rfind('! ')
                    )
                    if sentence_pos > 0:
                        end_pos = search_start + sentence_pos + 2  # Include the punctuation and space
        
        # Add the chunk
        chunks.append(text[current_pos:end_pos])
        
        # Move to next position with overlap
        current_pos = end_pos - overlap
        
        # Ensure we make progress
        if current_pos >= end_pos:
            current_pos = end_pos
    
    logger.info(f"Created {len(chunks)} chunks with simple strategy")
    return chunks

def extract_speakers(text: str) -> List[str]:
    """Extract speakers from transcript text."""
    speakers = []
    
    # Common speaker patterns
    patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+):',  # Full name: pattern
        r'([A-Z][a-z]+):',  # First name: pattern
        r'(Dr\. [A-Z][a-z]+):',  # Dr. Name: pattern
        r'(Mr\. [A-Z][a-z]+):',  # Mr. Name: pattern
        r'(Mrs\. [A-Z][a-z]+):',  # Mrs. Name: pattern
        r'(Ms\. [A-Z][a-z]+):'  # Ms. Name: pattern
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        speakers.extend(matches)
    
    # Remove duplicates while preserving order
    seen: Set[str] = set()
    unique_speakers = []
    for speaker in speakers:
        if speaker not in seen:
            seen.add(speaker)
            unique_speakers.append(speaker)
    
    return unique_speakers

def chunk_by_speaker(text: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Chunk transcript by speaker changes.
    
    Args:
        text: Transcript text
        max_chunk_size: Maximum chunk size
        overlap: Desired overlap between chunks
        
    Returns:
        List of chunks
    """
    # Extract speakers first
    speakers = extract_speakers(text)
    
    if not speakers:
        # Fall back to simple chunking if no speakers found
        logger.info("No speakers detected, falling back to simple chunking")
        return chunk_simple(text, max_chunk_size, overlap)
    
    # Create a pattern to match any speaker
    speaker_pattern = '|'.join([re.escape(s) for s in speakers])
    pattern = f"((?:^|\n)(?:{speaker_pattern}):)"
    
    # Split by speaker markers but keep the markers
    segments = re.split(pattern, text)
    
    chunks = []
    current_chunk = ""
    
    # Reassemble ensuring speakers stay with their content
    i = 1
    while i < len(segments):
        speaker_marker = segments[i]
        content = segments[i+1] if i+1 < len(segments) else ""
        
        segment = speaker_marker + content
        
        # If adding this would exceed max size and we have content, finish chunk
        if len(current_chunk) + len(segment) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            
            # Create overlap by including the last speaker segment in the next chunk
            if len(current_chunk) > overlap:
                # Find the last complete speaker segment to use as overlap
                last_speaker_match = re.search(pattern, current_chunk[::-1])
                if last_speaker_match:
                    last_speaker_pos = len(current_chunk) - last_speaker_match.start()
                    current_chunk = current_chunk[last_speaker_pos-1:]
                else:
                    current_chunk = ""
            else:
                current_chunk = ""
                
            current_chunk += segment
        else:
            current_chunk += segment
        
        i += 2
    
    # Add the final chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Created {len(chunks)} chunks with speaker-aware strategy")
    return chunks

def chunk_by_boundary(text: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Chunk text based on natural boundaries like paragraphs and headings.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Define boundary patterns in order of preference
    boundary_patterns = [
        r'\n\s*#{1,3}\s+[A-Za-z]',  # Markdown headers (highest priority)
        r'\n\s*\n',                  # Paragraph breaks
        r'\n\s*\d+\.\s+',            # Numbered lists
        r'\n\s*[-*•]\s+',            # Bullet points
        r'(?<=[.!?])\s+(?=[A-Z])'    # Sentence boundaries (lowest priority)
    ]
    
    # Find all boundary positions
    boundaries = []
    for pattern in boundary_patterns:
        for match in re.finditer(pattern, text):
            boundaries.append((match.start(), 1.0 if '##' in pattern else 0.8))
    
    # If no boundaries found, fall back to simple chunking
    if not boundaries:
        logger.info("No boundaries detected, falling back to simple chunking")
        return chunk_simple(text, max_chunk_size, overlap)
    
    # Sort boundaries by position
    boundaries.sort(key=lambda x: x[0])
    
    chunks = []
    start_pos = 0
    
    while start_pos < len(text):
        # Find the furthest boundary within max_chunk_size
        end_pos = min(start_pos + max_chunk_size, len(text))
        best_boundary = None
        
        for pos, confidence in boundaries:
            if start_pos < pos < end_pos:
                best_boundary = pos
        
        # If we found a boundary, use it; otherwise use the calculated end
        if best_boundary:
            chunk_end = best_boundary
        else:
            chunk_end = end_pos
        
        # Add the chunk
        chunks.append(text[start_pos:chunk_end])
        
        # Move to next position with overlap handling
        start_pos = max(chunk_end - overlap, start_pos + 1)
    
    logger.info(f"Created {len(chunks)} chunks with boundary-aware strategy")
    return chunks


def chunk_context_aware(text: str, max_chunk_size: int = 2500, min_chunk_size: int = 1000) -> List[str]:
    """
    Context-aware chunking that preserves semantic units and natural boundaries.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum chunk size
        min_chunk_size: Minimum chunk size before creating a new chunk
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    logger.info(f"Using context-aware chunking with max_size={max_chunk_size}, min_size={min_chunk_size}")
    
    # Define priority boundaries (in order of preference)
    boundary_patterns = [
        # 1. Section headers (highest priority)
        (r'\n#{1,3}\s+[A-Za-z]', 0.9),  # Markdown headers
        (r'\n[A-Z][A-Z\s]+\n', 0.85),   # ALL CAPS HEADERS
        
        # 2. Paragraph breaks
        (r'\n\s*\n', 0.8),              # Double line breaks
        
        # 3. List item boundaries
        (r'\n\s*\d+\.\s+', 0.75),       # Numbered lists
        (r'\n\s*[-*•]\s+', 0.7),        # Bullet points
        
        # 4. Topic transitions
        (r'(?i)(?:Next|Now|Moving on to|Let\'s discuss|Let\'s talk about|Turning to|Regarding|About)', 0.65),
        
        # 5. Speaker transitions (for transcripts)
        (r'\n[A-Z][a-z]+\s*[A-Z][a-z]*\s*:', 0.9),  # Full Name:
        (r'\n[A-Z][a-z]+:', 0.85),                  # Name:
        
        # 6. Sentence boundaries (lowest priority)
        (r'(?<=[.!?])\s+(?=[A-Z])', 0.5)
    ]
    
    # Find all potential breakpoints with their positions and strengths
    breakpoints = []
    
    for pattern, strength in boundary_patterns:
        for match in re.finditer(pattern, text):
            position = match.start()
            # Ensure we break at the beginning of the pattern
            breakpoints.append((position, strength, pattern))
    
    # Sort breakpoints by position
    breakpoints.sort(key=lambda x: x[0])
    
    # Create chunks based on boundaries and size constraints
    chunks = []
    current_start = 0
    
    while current_start < len(text):
        # Find the furthest breakpoint within max_chunk_size
        next_breakpoint = None
        best_strength = 0
        
        for position, strength, pattern in breakpoints:
            if (current_start < position < current_start + max_chunk_size and 
                position - current_start >= min_chunk_size and
                strength > best_strength):
                next_breakpoint = position
                best_strength = strength
        
        # If no suitable breakpoint found, use max size or end of text
        if next_breakpoint is None:
            next_breakpoint = min(current_start + max_chunk_size, len(text))
            
            # Try to find a sentence boundary near the breakpoint if possible
            if next_breakpoint < len(text):
                # Look for a sentence ending within the last 20% of the chunk
                last_portion = text[next_breakpoint - int(max_chunk_size * 0.2):next_breakpoint]
                last_sentence_end = max(
                    last_portion.rfind('. '),
                    last_portion.rfind('! '),
                    last_portion.rfind('? ')
                )
                
                if last_sentence_end != -1:
                    # Adjust the breakpoint to this sentence ending
                    next_breakpoint = next_breakpoint - int(max_chunk_size * 0.2) + last_sentence_end + 2
        
        # Add the chunk
        chunk_text = text[current_start:next_breakpoint].strip()
        if chunk_text:  # Only add non-empty chunks
            chunks.append(chunk_text)
        
        # Move to next position
        current_start = next_breakpoint
    
    logger.info(f"Created {len(chunks)} context-aware chunks")
    return chunks

def chunk_semantic(text: str, approx_chunks: int = 5) -> List[str]:
    """
    Simple semantic chunking based on topic segmentation.
    
    Args:
        text: Text to chunk
        approx_chunks: Target number of chunks to create
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # If text is small, just return it as a single chunk
    if len(text) < 3000:
        return [text]
    
    # Calculate approximate chunk size
    target_size = len(text) // approx_chunks
    
    # Tokenize into sentences
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        # Fall back to simple sentence splitting if NLTK fails
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Simple TextTiling-inspired algorithm
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Window parameters
    window_size = min(10, len(sentences) // (approx_chunks * 2))
    
    # Keep track of lexical similarity between adjacent blocks
    for i in range(len(sentences)):
        current_chunk.append(sentences[i])
        current_length += len(sentences[i])
        
        # Check if we should consider a boundary
        if current_length >= target_size and i > window_size:
            # Compare lexical similarity of blocks before and after current position
            before_text = ' '.join(sentences[i-window_size:i])
            after_text = ' '.join(sentences[i:min(i+window_size, len(sentences))])
            
            # Calculate lexical differences (simple bag of words comparison)
            before_words = set(re.findall(r'\b\w+\b', before_text.lower()))
            after_words = set(re.findall(r'\b\w+\b', after_text.lower()))
            
            # Jaccard similarity
            intersection = len(before_words.intersection(after_words))
            union = len(before_words.union(after_words))
            similarity = intersection / union if union > 0 else 0
            
            # If similarity is low, this might be a good boundary
            if similarity < 0.5:  # Threshold for topic change
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
    
    # Add the last chunk if there's content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # If we didn't find good semantic boundaries, fall back to context-aware chunking
    if len(chunks) < 2:
        logger.warning("Semantic chunking couldn't find good boundaries, falling back to context-aware")
        return chunk_context_aware(text)
    
    logger.info(f"Created {len(chunks)} semantic chunks")
    return chunks