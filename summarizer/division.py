"""
Document division strategies for the summarization engine.

This module provides utility functions for dividing documents into
manageable sections using different strategies:
- Basic: Simple division with smart paragraph/sentence breaks
- Boundary: Division based on natural document boundaries (headers, paragraphs)
- Speaker: Speaker-aware division for transcripts and conversations
- Context-aware: Intelligent division that preserves semantic coherence
"""

import re
import math
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def divide_basic(text: str, min_sections: int = 3, target_tokens_per_section: int = 25000, 
                 section_overlap: float = 0.1) -> List[Dict[str, Any]]:
    """
    Basic division strategy with fixed section sizes and overlap.
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        section_overlap: Overlap between sections as a fraction of section size
        
    Returns:
        List of division dictionaries
    """
    divisions = []
    
    # Calculate the number of tokens (rough estimate)
    estimated_tokens = len(text) / 4  # Rough approximation: 4 chars per token
    
    # Calculate number of sections based on max section size
    num_sections = max(min_sections, math.ceil(estimated_tokens / target_tokens_per_section))
    
    # Calculate section size in characters
    chars_per_section = len(text) / num_sections
    overlap_chars = chars_per_section * section_overlap
    
    logger.info(f"Basic division: {num_sections} sections, ~{int(chars_per_section)} chars each, {int(overlap_chars)} char overlap")
    
    for i in range(num_sections):
        # Calculate section boundaries
        start = max(0, int(i * chars_per_section - (i > 0) * overlap_chars))
        end = min(len(text), int((i + 1) * chars_per_section + (i < num_sections - 1) * overlap_chars))
        
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
                best_break = max(sentence_breaks)
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
                'strategy': 'basic'
            })
    
    return divisions

def divide_boundary(text: str, min_sections: int = 3, target_tokens_per_section: int = 25000) -> List[Dict[str, Any]]:
    """
    Boundary-aware division that respects document structure.
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        
    Returns:
        List of division dictionaries
    """
    divisions = []
    
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
    
    # If no boundaries found, fall back to basic division
    if not boundaries:
        logger.info("No document boundaries found, falling back to basic division")
        return divide_basic(text, min_sections, target_tokens_per_section)
    
    # Calculate target section size (in characters)
    estimated_tokens = len(text) / 4
    target_sections = max(min_sections, math.ceil(estimated_tokens / target_tokens_per_section))
    target_chars_per_section = len(text) / target_sections
    
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
    
    # Merge small sections if needed to approach target section count
    if len(sections) > 2 * target_sections:
        merged_sections = []
        current_section = sections[0]
        
        for i in range(1, len(sections)):
            current_size = current_section['end'] - current_section['start']
            next_size = sections[i]['end'] - sections[i]['start']
            
            # If adding the next section doesn't exceed target size too much, merge
            if current_size + next_size < target_chars_per_section * 1.5:
                current_section['end'] = sections[i]['end']
                current_section['text'] = text[current_section['start']:current_section['end']]
            else:
                merged_sections.append(current_section)
                current_section = sections[i]
        
        # Add the last section
        merged_sections.append(current_section)
        sections = merged_sections
    
    # Create final divisions with proper metadata
    for i, section in enumerate(sections):
        divisions.append({
            'start': section['start'],
            'end': section['end'],
            'text': section['text'],
            'strategy': 'boundary'
        })
    
    logger.info(f"Boundary division: {len(divisions)} sections based on document structure")
    return divisions

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

def divide_speaker(text: str, min_sections: int = 3, target_tokens_per_section: int = 25000) -> List[Dict[str, Any]]:
    """
    Speaker-aware division optimized for transcripts.
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        
    Returns:
        List of division dictionaries
    """
    # Extract speakers
    speakers = extract_speakers(text)
    
    # If no speakers found, fall back to basic division
    if not speakers:
        logger.info("No speakers detected, falling back to basic division")
        return divide_basic(text, min_sections, target_tokens_per_section)
    
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
                'strategy': 'speaker'
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
            'strategy': 'speaker'
        })
    
    logger.info(f"Speaker division: {len(divisions)} sections based on speaker transitions")
    return divisions

def divide_context_aware(text: str, min_sections: int = 3, target_tokens_per_section: int = 25000) -> List[Dict[str, Any]]:
    """
    Context-aware division that preserves semantic coherence.
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        
    Returns:
        List of division dictionaries
    """
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
    
    # If no breakpoints found, fall back to basic division
    if not breakpoints:
        return divide_basic(text, min_sections, target_tokens_per_section)
    
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
                'strategy': 'context_aware'
            })
        
        # Move to next position
        current_start = next_breakpoint
    
    logger.info(f"Context-aware division: {len(divisions)} semantically coherent sections")
    return divisions

def divide_document(text: str, strategy: str = "basic", min_sections: int = 3, 
                   target_tokens_per_section: int = 25000, section_overlap: float = 0.1) -> List[Dict[str, Any]]:
    """
    Divide a document using the specified strategy.
    
    Args:
        text: Document text
        strategy: Division strategy ("basic", "boundary", "speaker", or "context_aware")
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        section_overlap: Overlap between sections as a fraction of section size
        
    Returns:
        List of division dictionaries
    """
    if strategy == "speaker":
        return divide_speaker(text, min_sections, target_tokens_per_section)
    elif strategy == "boundary":
        return divide_boundary(text, min_sections, target_tokens_per_section)
    elif strategy == "context_aware":
        return divide_context_aware(text, min_sections, target_tokens_per_section)
    else:  # "basic" or fallback
        return divide_basic(text, min_sections, target_tokens_per_section, section_overlap)


def divide_semantic(text: str, min_sections: int = 3, target_tokens_per_section: int = 25000,
                   api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Semantic division that preserves topic coherence using embedding-based similarity.
    
    This approach:
    1. Breaks text into smaller segments (paragraphs)
    2. Generates embeddings for each segment
    3. Uses similarity to group segments into coherent sections
    4. Ensures sections don't exceed token limits
    
    Args:
        text: Document text
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        api_key: OpenAI API key for embeddings (uses env var if None)
        
    Returns:
        List of division dictionaries
    """
    logger.info("Starting semantic division")
    
    # First, split text into paragraphs
    paragraphs = _split_into_paragraphs(text)
    
    if len(paragraphs) <= min_sections:
        # Too few paragraphs to do semantic division effectively
        logger.info(f"Document has only {len(paragraphs)} paragraphs, falling back to context_aware division")
        from .division import divide_context_aware
        return divide_context_aware(text, min_sections, target_tokens_per_section)
    
    # Generate embeddings for each paragraph
    try:
        paragraph_embeddings = _generate_embeddings(paragraphs, api_key)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        logger.info("Falling back to context_aware division")
        from .division import divide_context_aware
        return divide_context_aware(text, min_sections, target_tokens_per_section)
    
    # Calculate similarity between adjacent paragraphs
    similarities = _calculate_similarities(paragraph_embeddings)
    
    # Identify potential section boundaries based on similarity drops
    boundaries = _identify_boundaries(similarities, min_sections, len(paragraphs))
    
    # Create sections based on boundaries, ensuring they don't exceed size limits
    sections = _create_sections_with_size_constraints(
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
            'strategy': 'semantic'
        })
    
    logger.info(f"Semantic division: {len(divisions)} semantically coherent sections")
    return divisions

def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on double newlines."""
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
        import numpy as np
        
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
    # Calculate similarities between adjacent paragraphs
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

def _create_sections_with_size_constraints(
    paragraphs: List[str], 
    boundaries: List[int], 
    target_tokens_per_section: int
) -> List[List[str]]:
    """
    Create sections from paragraphs using boundaries, ensuring size constraints.
    
    This function takes the identified boundaries and adjusts them to ensure
    no section exceeds the target token size.
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
                            if paragraphs[i] in section)
        
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

# Update the main divide_document function to include the semantic strategy
def divide_document(text: str, strategy: str = "basic", min_sections: int = 3, 
                   target_tokens_per_section: int = 25000, section_overlap: float = 0.1,
                   api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Divide a document using the specified strategy.
    
    Args:
        text: Document text
        strategy: Division strategy ("basic", "boundary", "speaker", "context_aware", or "semantic")
        min_sections: Minimum number of sections to create
        target_tokens_per_section: Target tokens per section
        section_overlap: Overlap between sections as a fraction of section size
        api_key: OpenAI API key for embeddings (semantic strategy only)
        
    Returns:
        List of division dictionaries
    """
    from .division import divide_speaker, divide_boundary, divide_context_aware, divide_basic
    
    if strategy == "speaker":
        return divide_speaker(text, min_sections, target_tokens_per_section)
    elif strategy == "boundary":
        return divide_boundary(text, min_sections, target_tokens_per_section)
    elif strategy == "context_aware":
        return divide_context_aware(text, min_sections, target_tokens_per_section)
    elif strategy == "semantic":
        return divide_semantic(text, min_sections, target_tokens_per_section, api_key)
    else:  # "basic" or fallback
        return divide_basic(text, min_sections, target_tokens_per_section, section_overlap)