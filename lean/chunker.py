"""
Smart document chunking module that divides text into balanced, meaningful sections.
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Smart text chunking that adapts to document structure."""
    
    def __init__(self):
        """Initialize the document chunker."""
        pass
    
    def chunk_document(self, 
                      text: str, 
                      min_chunks: int = 3, 
                      max_chunk_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Divide a document into smart chunks based on content and structure.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk in characters
                           (if None, will be calculated based on min_chunks)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # If max_chunk_size is not provided, calculate it based on min_chunks
        if max_chunk_size is None:
            # Simple estimate: total length divided by min_chunks, with some buffer
            max_chunk_size = (len(text) // min_chunks) * 1.2
            # Ensure a reasonable maximum (about 4000 tokens)
            max_chunk_size = min(max_chunk_size, 16000)
        
        # Check if the document already fits in a single chunk
        if len(text) <= max_chunk_size and min_chunks <= 1:
            return [{
                'index': 0,
                'text': text,
                'start_pos': 0,
                'end_pos': len(text),
                'chunk_type': 'full_document'
            }]
        
        # Detect if this is a transcript-like document
        is_transcript = self._is_transcript_like(text)
        
        # Choose chunking strategy based on document type
        if is_transcript:
            chunks = self._chunk_transcript(text, min_chunks, max_chunk_size)
        else:
            chunks = self._chunk_by_structure(text, min_chunks, max_chunk_size)
        
        # If we didn't get enough chunks, fall back to simpler chunking
        if len(chunks) < min_chunks:
            chunks = self._chunk_evenly(text, min_chunks, max_chunk_size)
        
        # Ensure each chunk has an index
        for i, chunk in enumerate(chunks):
            chunk['index'] = i
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _is_transcript_like(self, text: str) -> bool:
        """
        Detect if text appears to be a transcript or conversation.
        
        Args:
            text: Text to analyze
            
        Returns:
            Boolean indicating if the text is transcript-like
        """
        # Check a sample of the text for patterns
        sample = text[:min(len(text), 3000)]
        
        # Check for speaker patterns (e.g., "John:", "Jane Doe:")
        speaker_patterns = [
            r'^\s*[A-Z][a-z]+:',
            r'^\s*[A-Z][a-z]+ [A-Z][a-z]+:',
            r'\n\s*[A-Z][a-z]+:',
            r'\n\s*[A-Z][a-z]+ [A-Z][a-z]+:'
        ]
        
        # Count speaker pattern matches
        speaker_matches = 0
        for pattern in speaker_patterns:
            speaker_matches += len(re.findall(pattern, sample))
        
        # If we have multiple speaker patterns, likely a transcript
        if speaker_matches >= 5:
            return True
            
        # Check for time stamps often found in transcripts
        time_patterns = [
            r'\d{1,2}:\d{2}(:\d{2})?\s*[AP]M',
            r'\[\d{1,2}:\d{2}(:\d{2})?\]'
        ]
        
        # Count time pattern matches
        time_matches = 0
        for pattern in time_patterns:
            time_matches += len(re.findall(pattern, sample))
        
        # If we have multiple time stamps, likely a transcript
        if time_matches >= 3:
            return True
        
        return False
    
    def _chunk_transcript(self, text: str, min_chunks: int, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk a transcript-like document based on speaker turns.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Common speaker patterns
        speaker_patterns = [
            # "Name:"
            r'(^|\n)\s*([A-Z][a-z]+ ?[A-Z]?[a-z]*):',
            # "Name Lastname:"
            r'(^|\n)\s*([A-Z][a-z]+ [A-Z][a-z]+):',
            # Timestamps with speakers
            r'(^|\n)\s*\d{1,2}:\d{2}(:\d{2})?\s*[AP]M\s*([A-Z][a-z]+ ?[A-Z]?[a-z]*):',
            r'(^|\n)\s*\d{1,2}:\d{2}(:\d{2})?\s*[AP]M\s*([A-Z][a-z]+ [A-Z][a-z]+):'
        ]
        
        # Find all potential speaker transitions
        transitions = []
        for pattern in speaker_patterns:
            for match in re.finditer(pattern, text):
                transitions.append((match.start(), match.group()))
        
        # Sort transitions by position
        transitions.sort(key=lambda x: x[0])
        
        # If no transitions found, fall back to structure-based chunking
        if not transitions:
            return self._chunk_by_structure(text, min_chunks, max_chunk_size)
        
        # Create chunks based on speaker transitions
        chunks = []
        current_start = 0
        current_text = ""
        
        for pos, marker in transitions:
            # Skip if this is the start of the document
            if pos == 0:
                continue
                
            # Check if adding this transition would exceed max size
            if len(current_text) + (pos - current_start) > max_chunk_size:
                # Finish current chunk
                chunks.append({
                    'text': text[current_start:pos],
                    'start_pos': current_start,
                    'end_pos': pos,
                    'chunk_type': 'transcript_segment'
                })
                current_start = pos
            
            # Update for last transition
            if pos == transitions[-1][0]:
                chunks.append({
                    'text': text[current_start:],
                    'start_pos': current_start,
                    'end_pos': len(text),
                    'chunk_type': 'transcript_segment'
                })
        
        # If we created no chunks (rare edge case), fall back
        if not chunks:
            return self._chunk_evenly(text, min_chunks, max_chunk_size)
        
        return chunks
    
    def _chunk_by_structure(self, text: str, min_chunks: int, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk document based on structural elements like paragraphs, headers, etc.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Find structural breaks in priority order
        structure_patterns = [
            # Headers (markdown-style)
            (r'(^|\n)#{1,3}\s+[^\n]+', 0.9),
            # Headers (uppercase followed by newline)
            (r'(^|\n)[A-Z][A-Z\s]+[A-Z]:\s*\n', 0.8),
            (r'(^|\n)[A-Z][A-Z\s]+[A-Z]\n', 0.8),
            # Double line breaks (paragraph)
            (r'\n\s*\n', 0.7),
            # Numbered lists
            (r'(^|\n)\s*\d+\.\s+', 0.6),
            # Bullet points
            (r'(^|\n)\s*[-*•]\s+', 0.6),
            # Single line breaks (weaker boundary)
            (r'\n', 0.4)
        ]
        
        # Find potential break points with their positions and strengths
        break_points = []
        for pattern, strength in structure_patterns:
            for match in re.finditer(pattern, text):
                # For newlines and other patterns that might create empty chunks,
                # make sure we're not at the very beginning
                if match.start() > 10 or pattern not in [r'\n\s*\n', r'\n']:
                    break_points.append((match.start(), strength))
        
        # Sort break points by position
        break_points.sort(key=lambda x: x[0])
        
        # If no break points found, fall back to even chunking
        if not break_points:
            return self._chunk_evenly(text, min_chunks, max_chunk_size)
        
        # Calculate desired chunk size
        target_size = len(text) / min_chunks
        
        # Create chunks based on break points and size constraints
        chunks = []
        current_start = 0
        current_length = 0
        best_break = None
        best_break_score = 0
        
        for pos, strength in break_points:
            # Skip if we're already past this position
            if pos <= current_start:
                continue
                
            current_length = pos - current_start
            
            # If adding this section would exceed max_chunk_size
            if current_length > max_chunk_size:
                # If we have a candidate break point, use it
                if best_break is not None:
                    chunks.append({
                        'text': text[current_start:best_break],
                        'start_pos': current_start,
                        'end_pos': best_break,
                        'chunk_type': 'structural_segment'
                    })
                    current_start = best_break
                    current_length = pos - current_start
                    best_break = None
                    best_break_score = 0
                else:
                    # No good break found, just cut at max_chunk_size
                    end = current_start + max_chunk_size
                    chunks.append({
                        'text': text[current_start:end],
                        'start_pos': current_start,
                        'end_pos': end,
                        'chunk_type': 'size_limited_segment'
                    })
                    current_start = end
                    current_length = pos - current_start
            
            # Update best break if this break is better
            # We prefer breaks that are close to the target size
            size_factor = 1.0 - abs(current_length - target_size) / target_size
            break_score = strength * max(0.5, size_factor)
            
            if break_score > best_break_score:
                best_break = pos
                best_break_score = break_score
            
            # If we're near the target size and have a decent break, take it
            if current_length >= 0.8 * target_size and best_break_score > 0.6:
                chunks.append({
                    'text': text[current_start:best_break],
                    'start_pos': current_start,
                    'end_pos': best_break,
                    'chunk_type': 'structural_segment'
                })
                current_start = best_break
                best_break = None
                best_break_score = 0
        
        # Add the last chunk
        if current_start < len(text):
            chunks.append({
                'text': text[current_start:],
                'start_pos': current_start,
                'end_pos': len(text),
                'chunk_type': 'structural_segment'
            })
        
        return chunks
    
    def _chunk_evenly(self, text: str, min_chunks: int, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk document into roughly even pieces, trying to break at sentence boundaries.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Determine target chunk size
        total_length = len(text)
        target_size = min(max_chunk_size, total_length / min_chunks)
        
        # Find all sentence boundaries
        sentence_ends = []
        for match in re.finditer(r'[.!?]\s+', text):
            sentence_ends.append(match.end())
        
        # If no sentence boundaries found, just divide evenly
        if not sentence_ends:
            chunks = []
            for i in range(min_chunks):
                start = i * (total_length // min_chunks)
                end = (i + 1) * (total_length // min_chunks) if i < min_chunks - 1 else total_length
                chunks.append({
                    'text': text[start:end],
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_type': 'even_segment'
                })
            return chunks
        
        # Create chunks at sentence boundaries closest to target positions
        chunks = []
        current_pos = 0
        
        while current_pos < total_length:
            # Find target end position
            target_end = min(current_pos + target_size, total_length)
            
            # Find closest sentence boundary
            closest_boundary = None
            min_distance = float('inf')
            
            for boundary in sentence_ends:
                if boundary <= current_pos:
                    continue
                    
                distance = abs(boundary - target_end)
                if distance < min_distance:
                    closest_boundary = boundary
                    min_distance = distance
                
                # If we're getting too far past target, stop searching
                if boundary > target_end + (target_size * 0.3):
                    break
            
            # If no suitable boundary found, just use target end
            if closest_boundary is None or min_distance > target_size * 0.3:
                end_pos = min(current_pos + int(target_size), total_length)
            else:
                end_pos = closest_boundary
            
            # Create chunk
            chunks.append({
                'text': text[current_pos:end_pos],
                'start_pos': current_pos,
                'end_pos': end_pos,
                'chunk_type': 'sentence_boundary_segment'
            })
            
            # Move to next position
            current_pos = end_pos
        
        return chunks