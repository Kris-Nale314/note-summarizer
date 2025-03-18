"""
Document division strategies for the summarization engine.

This module provides utility functions for dividing documents into
manageable sections using three main strategies:
- Essential: Basic division for simple, shorter documents
- Long: Optimized division for lengthy but straightforward content
- Complex: Advanced division for documents with sophisticated structure
"""

import re
import math
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from .performance import PerformanceOptimizer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assess_document(text: str) -> Dict[str, Any]:
    """
    Analyze document to determine optimal processing strategy.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary with document assessment information
    """
    # Sample the beginning of the document
    sample = text[:2000]
    
    # Estimate total token count
    estimated_tokens = len(text) / 4  # Rough approximation: 4 chars per token
    
    # Detect if document has speakers (transcript-like)
    speaker_patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+):',  # Full name: pattern
        r'([A-Z][a-z]+):',  # First name: pattern
        r'(Dr\. [A-Z][a-z]+):',  # Dr. Name: pattern
        r'(Mr\. [A-Z][a-z]+):',  # Mr. Name: pattern
        r'(Mrs\. [A-Z][a-z]+):',  # Mrs. Name: pattern
        r'(Ms\. [A-Z][a-z]+):'  # Ms. Name: pattern
    ]
    
    has_speakers = False
    for pattern in speaker_patterns:
        if re.search(pattern, sample):
            has_speakers = True
            break
    
    # Detect if document has clear structure markers
    has_structure = bool(
        re.search(r'\n#{1,3}\s+', sample) or  # Markdown headers
        re.search(r'\n[A-Z][A-Z\s]+[A-Z]:', sample) or  # ALL CAPS HEADERS with colon
        re.search(r'\n[A-Z][A-Z\s]+\n', sample)  # ALL CAPS HEADERS with newline
    )
    
    # Determine document type
    doc_type = "general"
    if has_speakers:
        doc_type = "transcript"
    elif re.search(r'(quarterly|earnings|revenue|fiscal|dividend|EPS)', sample, re.I):
        doc_type = "earnings"
    elif re.search(r'(abstract|introduction|methodology|conclusion)', sample, re.I):
        doc_type = "academic"
    
    # Determine optimal strategy based on document characteristics
    if has_speakers:
        recommended_strategy = "long"  # For transcripts, Long strategy works well
    elif has_structure:
        recommended_strategy = "complex"  # For structured documents, Complex strategy is best
    elif estimated_tokens > 50000:
        recommended_strategy = "long"  # For very long documents without clear structure
    else:
        recommended_strategy = "essential"  # Default for simpler documents
    
    # Determine optimal division count based on size
    if estimated_tokens < 25000:
        recommended_divisions = 1
    elif estimated_tokens < 50000:
        recommended_divisions = 2
    elif estimated_tokens < 75000:
        recommended_divisions = 3
    else:
        recommended_divisions = max(3, int(estimated_tokens / 25000))
    
    return {
        "doc_type": doc_type,
        "estimated_tokens": estimated_tokens,
        "has_speakers": has_speakers,
        "has_structure": has_structure,
        "recommended_strategy": recommended_strategy,
        "recommended_divisions": recommended_divisions
    }

def extract_speakers(text: str) -> List[str]:
    """
    Extract speaker names from transcript text.
    
    Args:
        text: Transcript text
        
    Returns:
        List of speaker names
    """
    speakers = []
    
    # Common speaker patterns in transcripts
    patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+):', # Full name: pattern
        r'([A-Z][a-z]+):', # First name: pattern
        r'(Dr\. [A-Z][a-z]+):', # Dr. Name: pattern
        r'(Mr\. [A-Z][a-z]+):', # Mr. Name: pattern
        r'(Mrs\. [A-Z][a-z]+):', # Mrs. Name: pattern
        r'(Ms\. [A-Z][a-z]+):' # Ms. Name: pattern
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        speakers.extend(matches)
    
    # Remove duplicates
    return list(set(speakers))

def divide_essential(text: str, min_sections: int = 1, 
                    target_tokens_per_section: int = 25000) -> List[Dict[str, Any]]:
    """
    Essential division strategy for simple documents.
    Optimized for smaller documents with straightforward content.
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        
    Returns:
        List of division dictionaries
    """
    divisions = []
    
    # For very small documents, return a single division
    estimated_tokens = len(text) / 4
    if estimated_tokens <= target_tokens_per_section and min_sections <= 1:
        return [{
            'start': 0,
            'end': len(text),
            'text': text,
            'strategy': 'essential'
        }]
    
    # Calculate number of sections based on max section size
    num_sections = max(min_sections, math.ceil(estimated_tokens / target_tokens_per_section))
    
    # Calculate section size in characters
    chars_per_section = len(text) / num_sections
    
    logger.info(f"Essential division: {num_sections} sections, ~{int(chars_per_section)} chars each")
    
    for i in range(num_sections):
        # Calculate section boundaries
        start = max(0, int(i * chars_per_section))
        end = min(len(text), int((i + 1) * chars_per_section))
        
        # Try to find clean break points
        if i > 0 and start > 0:
            # Look for paragraph break near the start
            paragraph_break = text.rfind('\n\n', start - 500, start + 500)
            if paragraph_break != -1 and abs(paragraph_break - start) < 500:
                start = paragraph_break + 2  # +2 to include the newline chars
            else:
                # Look for sentence break
                sentence_breaks = [
                    text.rfind('. ', start - 200, start + 200),
                    text.rfind('! ', start - 200, start + 200),
                    text.rfind('? ', start - 200, start + 200)
                ]
                best_break = max(filter(lambda x: x != -1, sentence_breaks + [-1]))
                if best_break != -1 and abs(best_break - start) < 200:
                    start = best_break + 2  # +2 to include the punctuation and space
        
        if i < num_sections - 1 and end < len(text):
            # Look for paragraph break near the end
            paragraph_break = text.find('\n\n', end - 500, end + 500)
            if paragraph_break != -1 and abs(paragraph_break - end) < 500:
                end = paragraph_break
            else:
                # Look for sentence break
                sentence_breaks = [
                    text.find('. ', end - 200, end + 200),
                    text.find('! ', end - 200, end + 200),
                    text.find('? ', end - 200, end + 200)
                ]
                best_break = max(filter(lambda x: x != -1, sentence_breaks + [-1]))
                if best_break != -1 and abs(best_break - end) < 200:
                    end = best_break + 2  # +2 to include the punctuation and space
        
        # Create division
        division_text = text[start:end]
        if division_text.strip():  # Only add non-empty divisions
            divisions.append({
                'start': start,
                'end': end,
                'text': division_text,
                'strategy': 'essential'
            })
    
    return divisions

def divide_long(text: str, min_sections: int = 2, 
               target_tokens_per_section: int = 25000) -> List[Dict[str, Any]]:
    """
    Long division strategy optimized for lengthy documents.
    Handles transcripts and conversations particularly well.
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        
    Returns:
        List of division dictionaries
    """
    # Check if document has speakers (transcript-like)
    speakers = extract_speakers(text)
    
    if speakers:
        # Use speaker-based division for transcripts
        return _divide_by_speakers(text, speakers, min_sections, target_tokens_per_section)
    else:
        # Use topic transition division for regular text
        return _divide_by_topics(text, min_sections, target_tokens_per_section)

def _divide_by_speakers(text: str, speakers: List[str], min_sections: int, 
                      target_tokens_per_section: int) -> List[Dict[str, Any]]:
    """Speaker-aware division for transcripts and conversations."""
    # Calculate target section size (in characters)
    estimated_tokens = len(text) / 4
    target_sections = max(min_sections, math.ceil(estimated_tokens / target_tokens_per_section))
    target_chars_per_section = len(text) / target_sections
    
    # Build speaker pattern
    speaker_pattern = '|'.join([re.escape(s) for s in speakers])
    pattern = f"((?:^|\n)(?:{speaker_pattern}):)"
    
    # Split by speaker transitions
    segments = re.split(pattern, text)
    
    # Process segments
    divisions = []
    current_division = ""
    current_start = 0
    
    # Skip first segment if it's not a speaker marker
    start_idx = 1 if segments and not any(s+":" in segments[0] for s in speakers) else 0
    
    for i in range(start_idx, len(segments), 2):
        if i+1 >= len(segments):
            break
            
        speaker_marker = segments[i]
        content = segments[i+1] if i+1 < len(segments) else ""
        segment = speaker_marker + content
        
        # Check if adding this would exceed target size
        if len(current_division) + len(segment) > target_chars_per_section * 1.2 and current_division:
            # Finish current division
            divisions.append({
                'start': current_start,
                'end': current_start + len(current_division),
                'text': current_division,
                'strategy': 'long'
            })
            
            # Start new division
            current_division = segment
            current_start = current_start + len(current_division)
        else:
            # Add to current division
            current_division += segment
    
    # Add final division if there's content
    if current_division:
        divisions.append({
            'start': current_start,
            'end': current_start + len(current_division),
            'text': current_division,
            'strategy': 'long'
        })
    
    logger.info(f"Long division (speaker-based): {len(divisions)} sections")
    return divisions

def _divide_by_topics(text: str, min_sections: int, target_tokens_per_section: int) -> List[Dict[str, Any]]:
    """Division based on topic transitions for non-transcript content."""
    # Define topic transition markers
    topic_markers = [
        # Explicit transitions
        r'(?i)\n(?:Next|Now|Moving on to|Let\'s discuss|Let\'s talk about|Turning to|Regarding|About)',
        # Paragraph breaks
        r'\n\s*\n',
        # Lists
        r'\n\s*\d+\.\s+',
        r'\n\s*[-*•]\s+',
        # Sentence boundaries (lowest priority)
        r'(?<=[.!?])\s+(?=[A-Z])'
    ]
    
    # Calculate target section size
    estimated_tokens = len(text) / 4
    target_sections = max(min_sections, math.ceil(estimated_tokens / target_tokens_per_section))
    target_chars_per_section = len(text) / target_sections
    
    # Find potential break points
    potential_breaks = []
    
    for marker in topic_markers:
        for match in re.finditer(marker, text):
            potential_breaks.append(match.start())
    
    # Sort breaks by position
    potential_breaks.sort()
    
    # If too few potential breaks, fall back to essential division
    if len(potential_breaks) < min_sections - 1:
        logger.info("Too few topic transitions, falling back to essential division")
        return divide_essential(text, min_sections, target_tokens_per_section)
    
    # Select optimal breaks for creating target_sections
    selected_breaks = _select_optimal_breaks(potential_breaks, len(text), target_sections)
    
    # Create divisions based on selected breaks
    divisions = []
    current_start = 0
    
    for break_point in selected_breaks:
        divisions.append({
            'start': current_start,
            'end': break_point,
            'text': text[current_start:break_point],
            'strategy': 'long'
        })
        current_start = break_point
    
    # Add final section
    divisions.append({
        'start': current_start,
        'end': len(text),
        'text': text[current_start:],
        'strategy': 'long'
    })
    
    logger.info(f"Long division (topic-based): {len(divisions)} sections")
    return divisions

# In division.py

def divide_complex(text: str, min_sections: int = 2, 
                 target_tokens_per_section: int = 25000, 
                 api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Complex division strategy with performance optimization.
    """
    from concurrent.futures import ThreadPoolExecutor
    import functools
    
    # Define the approaches to try
    approaches = [
        functools.partial(_divide_by_structure, text, min_sections, target_tokens_per_section),
        functools.partial(_divide_context_aware, text, min_sections, target_tokens_per_section)
    ]
    
    if api_key:
        approaches.append(
            functools.partial(_divide_semantic, text, min_sections, target_tokens_per_section, api_key)
        )
    
    # Try approaches in parallel
    with ThreadPoolExecutor(max_workers=min(3, len(approaches))) as executor:
        results = list(executor.map(lambda func: func(), approaches))
    
    # Find the best result (non-empty with most divisions close to target)
    valid_results = [r for r in results if r and len(r) >= min_sections]
    
    if not valid_results:
        # Fall back to essential if nothing worked
        return divide_essential(text, min_sections, target_tokens_per_section)
    
    # Choose the result closest to the desired number of sections
    target_sections = max(min_sections, len(text) // (target_tokens_per_section * 4))
    best_result = min(valid_results, key=lambda r: abs(len(r) - target_sections))
    
    return best_result

def _divide_by_structure(text: str, min_sections: int, target_tokens_per_section: int) -> List[Dict[str, Any]]:
    """Division based on document structure (headers, lists, etc.)."""
    # Define boundary patterns in priority order
    boundary_patterns = [
        r'\n#{1,3}\s+',          # Markdown headers
        r'\n[A-Z][A-Z\s]+[A-Z]:', # ALL CAPS HEADERS with colon
        r'\n[A-Z][A-Z\s]+\n',    # ALL CAPS HEADERS with newline
        r'\n\s*\n',               # Paragraph breaks
        r'\n\s*\d+\.\s+',         # Numbered lists
        r'\n\s*[-*•]\s+'          # Bullet points
    ]
    
    # Find all potential boundaries
    boundaries = []
    for pattern in boundary_patterns:
        for match in re.finditer(pattern, text):
            boundaries.append(match.start())
    
    # Sort boundaries by position
    boundaries.sort()
    
    # If no boundaries found, return empty list to trigger fallback
    if not boundaries:
        logger.info("No document structure boundaries found")
        return []
    
    # Calculate target section size (in characters)
    estimated_tokens = len(text) / 4
    target_sections = max(min_sections, math.ceil(estimated_tokens / target_tokens_per_section))
    
    # Add document start and end to boundaries
    all_boundaries = [0] + boundaries + [len(text)]
    
    # Create initial sections based on natural boundaries
    sections = []
    for i in range(len(all_boundaries) - 1):
        start = all_boundaries[i]
        end = all_boundaries[i + 1]
        
        # Skip very small sections
        if end - start < 200:
            continue
            
        sections.append({
            'start': start, 
            'end': end, 
            'text': text[start:end]
        })
    
    # Merge sections if there are too many
    if len(sections) > 2 * target_sections:
        target_size = len(text) / target_sections
        merged_sections = []
        current_section = sections[0]
        
        for i in range(1, len(sections)):
            current_size = current_section['end'] - current_section['start']
            next_size = sections[i]['end'] - sections[i]['start']
            
            # If adding the next section doesn't exceed target size too much, merge
            if current_size + next_size < target_size * 1.5:
                current_section['end'] = sections[i]['end']
                current_section['text'] = text[current_section['start']:current_section['end']]
            else:
                merged_sections.append(current_section)
                current_section = sections[i]
        
        # Add the last section
        merged_sections.append(current_section)
        sections = merged_sections
    
    # Create final divisions with proper metadata
    divisions = []
    for section in sections:
        divisions.append({
            'start': section['start'],
            'end': section['end'],
            'text': section['text'],
            'strategy': 'complex'
        })
    
    logger.info(f"Complex division (structure-based): {len(divisions)} sections")
    return divisions

def _divide_context_aware(text: str, min_sections: int, target_tokens_per_section: int) -> List[Dict[str, Any]]:
    """Context-aware division that preserves semantic coherence."""
    # Define priority boundaries (in order of preference)
    boundary_patterns = [
        (r'\n#{1,3}\s+[A-Za-z]', 0.9),   # Markdown headers
        (r'\n[A-Z][A-Z\s]+\n', 0.85),    # ALL CAPS HEADERS
        (r'\n\s*\n', 0.8),               # Double line breaks
        (r'\n\s*\d+\.\s+', 0.75),        # Numbered lists
        (r'\n\s*[-*•]\s+', 0.7),         # Bullet points
        
        # Topic transitions
        (r'(?i)(?:Next|Now|Moving on to|Let\'s discuss|Let\'s talk about|Turning to|Regarding|About)', 0.65),
        
        # Speaker transitions (for transcripts)
        (r'\n[A-Z][a-z]+\s*[A-Z][a-z]*\s*:', 0.9),  # Full Name:
        (r'\n[A-Z][a-z]+:', 0.85),                  # Name:
        
        # Sentence boundaries (lowest priority)
        (r'(?<=[.!?])\s+(?=[A-Z])', 0.5)
    ]
    
    # Find all potential breakpoints with their positions and strengths
    breakpoints = []
    for pattern, strength in boundary_patterns:
        for match in re.finditer(pattern, text):
            position = match.start()
            breakpoints.append((position, strength))
    
    # Sort breakpoints by position
    breakpoints.sort(key=lambda x: x[0])
    
    # If no breakpoints found, fall back to essential division
    if not breakpoints:
        return divide_essential(text, min_sections, target_tokens_per_section)
    
    # Calculate target section size
    estimated_tokens = len(text) / 4
    target_sections = max(min_sections, math.ceil(estimated_tokens / target_tokens_per_section))
    target_chars_per_section = len(text) / target_sections
    min_section_size = max(1000, target_chars_per_section // 3)  # Minimum size to avoid tiny sections
    
    # Create divisions based on breakpoints and size constraints
    divisions = []
    current_start = 0
    
    while current_start < len(text):
        # Find the best breakpoint within target size range
        next_breakpoint = None
        best_strength = 0
        
        target_end = min(current_start + target_chars_per_section, len(text))
        acceptable_range = (0.8 * target_chars_per_section, 1.2 * target_chars_per_section)
        
        for position, strength in breakpoints:
            if position <= current_start:
                continue
                
            section_size = position - current_start
            
            # Prioritize breakpoints that give us sections close to target size
            if section_size >= min_section_size:
                # Adjust strength based on how close this breakpoint gets us to target size
                size_factor = 1.0
                if acceptable_range[0] <= section_size <= acceptable_range[1]:
                    size_factor = 1.2  # Bonus for sections in the ideal range
                elif section_size < 0.5 * target_chars_per_section:
                    size_factor = 0.8  # Penalty for sections much smaller than target
                elif section_size > 1.5 * target_chars_per_section:
                    size_factor = 0.7  # Larger penalty for sections much larger than target
                
                adjusted_strength = strength * size_factor
                
                if adjusted_strength > best_strength:
                    next_breakpoint = position
                    best_strength = adjusted_strength
            
            # If we're getting too far past our target size, stop looking
            if position > current_start + 1.5 * target_chars_per_section:
                break
        
        # If no suitable breakpoint found, use a calculated end point
        if next_breakpoint is None:
            next_breakpoint = min(current_start + int(target_chars_per_section), len(text))
            
            # Try to find a sentence boundary near the breakpoint if possible
            if next_breakpoint < len(text):
                # Look for a sentence ending within the last 20% of the section
                last_portion = text[next_breakpoint - int(target_chars_per_section * 0.2):next_breakpoint]
                last_sentence_end = max(
                    last_portion.rfind('. '),
                    last_portion.rfind('! '),
                    last_portion.rfind('? ')
                )
                
                if last_sentence_end != -1:
                    # Adjust the breakpoint to this sentence ending
                    next_breakpoint = next_breakpoint - int(target_chars_per_section * 0.2) + last_sentence_end + 2
        
        # Add the section
        section_text = text[current_start:next_breakpoint].strip()
        if section_text:  # Only add non-empty sections
            divisions.append({
                'start': current_start,
                'end': next_breakpoint,
                'text': section_text,
                'strategy': 'complex'
            })
        
        # Move to next position
        current_start = next_breakpoint
    
    logger.info(f"Complex division (context-aware): {len(divisions)} sections")
    return divisions

def _divide_semantic(text: str, min_sections: int, target_tokens_per_section: int,
                   api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Semantic division using embeddings to detect topic transitions."""
    # First, split text into paragraphs
    paragraphs = _split_into_paragraphs(text)
    
    if len(paragraphs) <= min_sections:
        # Too few paragraphs to do semantic division effectively
        logger.info(f"Document has only {len(paragraphs)} paragraphs, falling back to context-aware division")
        return _divide_context_aware(text, min_sections, target_tokens_per_section)
    
    # Generate embeddings for each paragraph
    try:
        paragraph_embeddings = _generate_embeddings(paragraphs, api_key)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return _divide_context_aware(text, min_sections, target_tokens_per_section)
    
    # Calculate similarity between adjacent paragraphs
    similarities = _calculate_similarities(paragraph_embeddings)
    
    # Identify potential section boundaries based on similarity drops
    boundaries = _identify_boundaries(similarities, min_sections, len(paragraphs))
    
    # Create sections based on boundaries, ensuring they don't exceed size limits
    sections = _create_sections_from_boundaries(
        paragraphs, boundaries, target_tokens_per_section)
    
    # Convert sections to division format
    divisions = []
    current_pos = 0
    
    for section in sections:
        section_text = "\n\n".join(section)
        section_start = text.find(section[0], current_pos)
        
        if section_start == -1:
            # Fallback if exact match fails
            section_start = current_pos
        
        section_end = section_start + len(section_text)
        current_pos = section_end
        
        divisions.append({
            'start': section_start,
            'end': section_end,
            'text': section_text,
            'strategy': 'complex'
        })
    
    logger.info(f"Complex division (semantic): {len(divisions)} sections")
    return divisions

def _select_optimal_breaks(breaks: List[int], text_length: int, target_sections: int) -> List[int]:
    """Select optimal break points to create the desired number of sections."""
    if len(breaks) <= target_sections - 1:
        return breaks
    
    ideal_section_size = text_length / target_sections
    ideal_positions = [int(i * ideal_section_size) for i in range(1, target_sections)]
    
    selected_breaks = []
    for position in ideal_positions:
        # Find closest break to the ideal position
        closest = min(breaks, key=lambda x: abs(x - position))
        selected_breaks.append(closest)
        
        # Remove nearby breaks to prevent tiny sections
        min_section_gap = ideal_section_size * 0.3
        breaks = [b for b in breaks if abs(b - closest) > min_section_gap]
        
        if not breaks:
            break
    
    return sorted(set(selected_breaks))

def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, handling large paragraphs appropriately."""
    # Split by double newline
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Handle case where paragraphs are too large
    max_paragraph_size = 1000  # Characters
    result = []
    
    for p in paragraphs:
        if len(p) > max_paragraph_size:
            # Split large paragraphs into sentences
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                if current_size + len(sentence) > max_paragraph_size and current_chunk:
                    # Add current chunk and start a new one
                    result.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_size += len(sentence)
            
            if current_chunk:
                result.append(' '.join(current_chunk))
        else:
            result.append(p)
    
    return result

def _generate_embeddings(paragraphs: List[str], api_key: Optional[str] = None) -> np.ndarray:
    """Generate embeddings for paragraphs using OpenAI API."""
    try:
        import openai
        import os
        
        # Set API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        client = openai.OpenAI(api_key=api_key)
        
        # Process in batches to avoid hitting API limits
        batch_size = 20
        batches = [paragraphs[i:i+batch_size] for i in range(0, len(paragraphs), batch_size)]
        
        all_embeddings = []
        for batch in batches:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array for easier manipulation
        return np.array(all_embeddings)
        
    except ImportError:
        logger.error("OpenAI package not installed. Install with: pip install openai>=1.0.0")
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def _calculate_similarities(embeddings: np.ndarray) -> List[float]:
    """Calculate cosine similarity between adjacent paragraph embeddings."""
    similarities = []
    
    for i in range(len(embeddings) - 1):
        # Cosine similarity between current and next paragraph
        similarity = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(float(similarity))
    
    return similarities

def _identify_boundaries(similarities: List[float], min_sections: int, paragraph_count: int) -> List[int]:
   """
   Identify section boundaries based on similarity drops.
   
   This function finds the places where there's a significant drop in similarity
   between adjacent paragraphs, suggesting a topic change.
   """
   # Calculate target number of sections
   target_sections = max(min_sections, min(paragraph_count // 5, 10))
   
   # Identify significant drops in similarity
   similarity_drops = []
   for i, sim in enumerate(similarities):
       if i > 0 and i < len(similarities) - 1:
           # Calculate rolling average similarity
           local_avg = (similarities[i-1] + similarities[i] + similarities[i+1]) / 3
           
           # How much this similarity drops compared to local average
           drop = max(0, local_avg - sim)
           similarity_drops.append((i+1, drop))  # i+1 is the paragraph index after the drop
   
   # Sort by drop size (largest first)
   similarity_drops.sort(key=lambda x: x[1], reverse=True)
   
   # Take the top N-1 drops as boundaries (N = target sections)
   boundaries = [sd[0] for sd in similarity_drops[:target_sections-1]]
   
   # Add the end boundary
   boundaries.append(paragraph_count)
   
   # Sort boundaries in ascending order
   boundaries.sort()
   
   return boundaries

def _create_sections_from_boundaries(
   paragraphs: List[str], 
   boundaries: List[int], 
   target_tokens_per_section: int
) -> List[List[str]]:
   """
   Create sections from paragraphs using boundaries, ensuring size constraints.
   """
   # Estimate tokens per paragraph (4 chars per token is a rough approximation)
   estimated_tokens = [len(p) // 4 for p in paragraphs]
   
   # Create initial sections based on boundaries
   sections = []
   start = 0
   
   for boundary in boundaries:
       sections.append(paragraphs[start:boundary])
       start = boundary
   
   # Check if any section exceeds the target size and subdivide if needed
   final_sections = []
   
   for section in sections:
       section_tokens = sum(estimated_tokens[i] for i in range(len(paragraphs)) 
                           if i < len(paragraphs) and paragraphs[i] in section)
       
       if section_tokens > target_tokens_per_section * 1.2 and len(section) > 3:
           # Section is too large, subdivide it
           subsections = []
           current_subsection = []
           current_tokens = 0
           
           for paragraph in section:
               para_tokens = len(paragraph) // 4
               
               if current_tokens + para_tokens > target_tokens_per_section and current_subsection:
                   subsections.append(current_subsection)
                   current_subsection = [paragraph]
                   current_tokens = para_tokens
               else:
                   current_subsection.append(paragraph)
                   current_tokens += para_tokens
           
           # Add the last subsection
           if current_subsection:
               subsections.append(current_subsection)
           
           final_sections.extend(subsections)
       else:
           # Section is fine as is
           final_sections.append(section)
   
   return final_sections

def divide_document(text: str, strategy: str = "essential", min_sections: int = 1, 
                  target_tokens_per_section: int = 25000, api_key: Optional[str] = None,
                  compare_all: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main division function that orchestrates the document division process.
    
    Args:
        text: Document text
        strategy: Division strategy ("essential", "long", "complex", or "auto")
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        api_key: OpenAI API key for embeddings (complex strategy only)
        compare_all: Whether to compare all strategies (for development)
        
    Returns:
        If compare_all is False: List of division dictionaries
        If compare_all is True: Dictionary with results from all strategies
    """
    # If we're comparing all strategies
    if compare_all:
        # Run division for all strategies
        start_time = time.time()
        
        # Use performance module for parallel processing if available
        try:
            from .performance import PerformanceOptimizer
            optimizer = PerformanceOptimizer(cache_dir=".cache", max_workers=3, enable_caching=True)
            
            # Define tasks for parallel processing
            tasks = []
            
            # Essential strategy task
            def run_essential():
                start = time.time()
                result = divide_essential(text, min_sections, target_tokens_per_section)
                return "essential", result, time.time() - start
                
            # Long strategy task
            def run_long():
                start = time.time()
                result = divide_long(text, min_sections, target_tokens_per_section)
                return "long", result, time.time() - start
                
            # Complex strategy task
            def run_complex():
                start = time.time()
                result = divide_complex(text, min_sections, target_tokens_per_section, api_key)
                return "complex", result, time.time() - start
            
            # Try to run in parallel
            try:
                import asyncio
                
                async def parallel_process():
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        tasks = [
                            loop.run_in_executor(executor, run_essential),
                            loop.run_in_executor(executor, run_long),
                            loop.run_in_executor(executor, run_complex)
                        ]
                        return await asyncio.gather(*tasks)
                
                # Get or create an event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the tasks in parallel
                results = loop.run_until_complete(parallel_process())
                
            except (ImportError, Exception) as e:
                # Fall back to sequential processing if parallel fails
                logger.warning(f"Parallel processing failed, using sequential: {e}")
                results = [run_essential(), run_long(), run_complex()]
                
        except ImportError:
            # Performance module not available, run sequentially
            logger.warning("PerformanceOptimizer not available, running sequential comparison")
            
            # Run the division strategies sequentially
            results = []
            
            # Essential strategy
            essential_time = time.time()
            essential_divisions = divide_essential(text, min_sections, target_tokens_per_section)
            essential_elapsed = time.time() - essential_time
            results.append(("essential", essential_divisions, essential_elapsed))
            
            # Long strategy
            long_time = time.time()
            long_divisions = divide_long(text, min_sections, target_tokens_per_section)
            long_elapsed = time.time() - long_time
            results.append(("long", long_divisions, long_elapsed))
            
            # Complex strategy
            complex_time = time.time()
            complex_divisions = divide_complex(text, min_sections, target_tokens_per_section, api_key)
            complex_elapsed = time.time() - complex_time
            results.append(("complex", complex_divisions, complex_elapsed))
        
        # Total time
        total_elapsed = time.time() - start_time
        
        # Compile results
        comparison_results = {
            "metadata": {
                "total_processing_time": total_elapsed,
                "text_length": len(text),
                "estimated_tokens": len(text) // 4
            }
        }
        
        # Add each strategy's results
        for strategy_name, divisions, process_time in results:
            comparison_results[strategy_name] = {
                "divisions": divisions,
                "count": len(divisions),
                "processing_time": process_time
            }
        
        return comparison_results
    
    # Auto detect strategy if "auto"
    if strategy == "auto":
        assessment = assess_document(text)
        strategy = assessment["recommended_strategy"]
        logger.info(f"Auto-selected strategy: {strategy}")
    
    # Apply the selected strategy
    if strategy == "essential":
        return divide_essential(text, min_sections, target_tokens_per_section)
    elif strategy == "long":
        return divide_long(text, min_sections, target_tokens_per_section)
    elif strategy == "complex":
        return divide_complex(text, min_sections, target_tokens_per_section, api_key)
    else:
        logger.warning(f"Unknown strategy '{strategy}', falling back to essential")
        return divide_essential(text, min_sections, target_tokens_per_section)


def generate_division_prompt(division: Dict[str, Any], index: int, total: int, doc_type: str = "general") -> str:
   """
   Generate an intelligent prompt for the division based on document type and position.
   
   Args:
       division: Division dictionary
       index: Division index
       total: Total number of divisions
       doc_type: Document type (general, transcript, earnings, academic)
       
   Returns:
       Prompt for the LLM
   """
   text = division["text"]
   strategy = division.get("strategy", "essential")
   
   # Create common instructions
   instructions = ""
   
   # Adjust instructions based on document type
   if doc_type == "transcript":
       instructions = """
       Create a detailed summary of this transcript section that:
       1. Preserves who said what (maintain speaker attribution for key points)
       2. Captures the main discussion topics, decisions, and action items
       3. Highlights important questions, answers, and exchanges
       4. Notes any agreements, disagreements, or unresolved points
       5. Extracts direct quotes for particularly important statements
       
       Be thorough but also consolidate redundant discussions.
       """
   elif doc_type == "earnings":
       instructions = """
       Create a detailed summary of this earnings call section that:
       1. Captures key financial metrics and performance highlights
       2. Notes management's forward guidance and strategic priorities
       3. Highlights market conditions and competitive landscape insights
       4. Extracts important analyst questions and management responses
       5. Identifies specific challenges, opportunities, and planned initiatives
       
       Focus on facts, figures, and specific commitments rather than general statements.
       """
   elif doc_type == "academic":
       instructions = """
       Create a detailed summary of this academic content that:
       1. Identifies the key arguments, findings, and conclusions
       2. Preserves important methodological details and limitations
       3. Captures statistical results and their significance
       4. Notes relationships to prior research and theoretical frameworks
       5. Maintains the logical flow of the argument structure
       
       Be precise with technical terminology and maintain scholarly tone.
       """
   else:  # general content
       instructions = """
       Create a detailed summary of this content that:
       1. Captures all key points, arguments, and information
       2. Preserves important details, examples, and evidence
       3. Maintains the logical structure and flow of ideas
       4. Highlights connections between different concepts
       5. Notes any conclusions, recommendations, or implications
       
       Aim for comprehensiveness while eliminating redundancy.
       """
   
   # Tailor prompt based on division position
   division_context = ""
   if total > 1:
       if index == 0:
           division_context = "This is the BEGINNING section of the document. Focus on establishing context and introducing key topics."
       elif index == total - 1:
           division_context = "This is the FINAL section of the document. Focus on conclusions, next steps, and wrapping up discussions."
       else:
           division_context = f"This is section {index+1} of {total}. Focus on continuing the narrative from previous sections."
   
   # Build the complete prompt
   prompt = f"""
   {instructions}
   
   {division_context}
   
   SECTION TEXT:
   {text}
   """
   
   return prompt.strip()

def synthesize_summaries(division_summaries: List[str], doc_type: str = "general") -> str:
   """
   Intelligently synthesize division summaries into a cohesive document.
   
   Args:
       division_summaries: List of division summaries
       doc_type: Document type (general, transcript, earnings, academic)
       
   Returns:
       Synthesis prompt
   """
   # Combine division summaries with clear separation
   combined = "\n\n===== SECTION SEPARATOR =====\n\n".join([
       f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(division_summaries)
   ])
   
   # Build synthesis prompt based on document type
   if doc_type == "transcript":
       synthesis_prompt = f"""
       Create a comprehensive yet concise summary of this conversation/meeting.
       
       Your summary should:
       1. Start with a brief overview of the discussion's purpose and participants
       2. Organize content by topic rather than chronologically 
       3. Preserve key speaker attributions for important points
       4. Highlight decisions, action items, and follow-ups
       5. Note areas of agreement and any unresolved questions
       
       Format with clear markdown headings and bullet points for readability.
       
       SECTION SUMMARIES:
       {combined}
       """
   elif doc_type == "earnings":
       synthesis_prompt = f"""
       Create a comprehensive analysis of this earnings call.
       
       Your analysis should:
       1. Start with an executive summary of financial performance
       2. Organize insights by business segment or strategic priority
       3. Include a section specifically on forward guidance and outlook
       4. Highlight key analyst questions and management responses
       5. Note important metrics, trends, and management commentary
       
       Format with clear markdown headings and include a "Key Metrics" section.
       
       SECTION SUMMARIES:
       {combined}
       """
   elif doc_type == "academic":
       synthesis_prompt = f"""
       Create a comprehensive summary of this academic document.
       
       Your summary should:
       1. Begin with an abstract-like overview of the main findings/arguments
       2. Organize by research components (methodology, results, discussion)
       3. Preserve statistical significance and key technical details
       4. Highlight limitations and implications of the research
       5. Maintain precision in technical terminology
       
       Format with proper academic structure and clear section headings.
       
       SECTION SUMMARIES:
       {combined}
       """
   else:
       synthesis_prompt = f"""
       Create a comprehensive, well-structured summary of this document.
       
       Your summary should:
       1. Begin with a concise executive summary capturing the main points
       2. Organize content logically by topic rather than by section
       3. Preserve important details and supporting evidence
       4. Eliminate redundancies while maintaining completeness
       5. Connect related ideas across different sections
       
       Use markdown formatting with clear headings and structure.
       
       SECTION SUMMARIES:
       {combined}
       """
   
   return synthesis_prompt.strip()