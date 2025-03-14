"""
Advanced chunking processor for transcript analysis.

This module provides different document chunking strategies:
1. Traditional fixed-size chunking
2. Speaker-aware intelligent chunking
3. Boundary-aware intelligent chunking
4. Topic-based semantic chunking

It also includes comparison metrics for evaluating chunking quality.
"""

import re
import nltk
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class TranscriptChunk:
    """Class representing a transcript chunk with enhanced metadata."""
    text: str
    start_idx: int
    end_idx: int
    speakers: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    time_markers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        """Return the length of the chunk in characters."""
        return len(self.text)
    
    @property
    def primary_speaker(self) -> Optional[str]:
        """Return the primary speaker in this chunk, if any."""
        if not self.speakers:
            return None
        
        # Count speaker occurrences and return the most frequent
        speaker_counts = {}
        for speaker in self.speakers:
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        return max(speaker_counts.items(), key=lambda x: x[1])[0]
    
    def __str__(self) -> str:
        speaker_info = f", speakers: {len(self.speakers)}" if self.speakers else ""
        topic_info = f", topics: {len(self.topics)}" if self.topics else ""
        return f"TranscriptChunk({self.start_idx}:{self.end_idx}, {len(self.text)} chars{speaker_info}{topic_info})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "speakers": self.speakers,
            "topics": self.topics,
            "time_markers": self.time_markers,
            "metadata": self.metadata or {}
        }


@dataclass
class DocumentBoundary:
    """Class representing a detected document boundary."""
    position: int
    boundary_type: str  # e.g., 'paragraph', 'section', 'speaker_change'
    confidence: float  # 0.0 to 1.0
    context: str = None  # surrounding text for context
    
    def __str__(self) -> str:
        return f"Boundary({self.position}, {self.boundary_type}, conf={self.confidence:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert boundary to dictionary representation."""
        return {
            "position": self.position,
            "type": self.boundary_type,
            "confidence": self.confidence,
            "context": self.context
        }


class ChunkingProcessor:
    """
    Advanced processor for chunking transcripts with speaker awareness,
    topic detection, and intelligent boundary recognition.
    """
    
    def __init__(self, 
                default_chunk_size: int = 2000, 
                default_chunk_overlap: int = 200,
                boundary_detection_threshold: float = 0.7,
                use_llm: bool = False):
        """
        Initialize the chunking processor.
        
        Args:
            default_chunk_size: Default size for fixed-size chunks (characters)
            default_chunk_overlap: Default overlap between chunks (characters)
            boundary_detection_threshold: Confidence threshold for boundary detection
            use_llm: Whether to use LLM for enhanced chunking (requires OpenAI client)
        """
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.boundary_detection_threshold = boundary_detection_threshold
        self.use_llm = use_llm
        
        # Initialize OpenAI client if LLM is enabled
        if self.use_llm:
            try:
                from .openai_client import OpenAIClient
                self.openai_client = OpenAIClient()
                logger.info("OpenAI client initialized for enhanced chunking")
            except ImportError:
                logger.warning("OpenAI client not available, disabling LLM-enhanced chunking")
                self.use_llm = False
        
        logger.info(f"Initialized ChunkingProcessor with default chunk size {default_chunk_size}, " 
                  f"overlap {default_chunk_overlap}, use_llm={use_llm}")
    
    def chunk_text_fixed_size(self, 
                             text: str, 
                             chunk_size: Optional[int] = None, 
                             chunk_overlap: Optional[int] = None) -> List[TranscriptChunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: The text to chunk
            chunk_size: Size of each chunk in characters (uses default if None)
            chunk_overlap: Overlap between chunks in characters (uses default if None)
            
        Returns:
            List of TranscriptChunk objects
        """
        if not text:
            return []
        
        # Use defaults if not specified
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        
        logger.info(f"Chunking text ({len(text)} chars) with fixed-size strategy: "
                  f"size={chunk_size}, overlap={chunk_overlap}")
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position, ensuring we don't go beyond text length
            end = min(start + chunk_size, len(text))
            
            # If we're not at the start and not at the end, try to find a better break point
            if start > 0 and end < len(text):
                # Find the last sentence-ending punctuation
                last_period = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('\n', start, end)
                )
                
                # If found and reasonably positioned, use it as the end point
                if last_period != -1 and last_period > start + (chunk_size / 2):
                    end = last_period + 2  # Include the punctuation and space
            
            # Create the chunk
            chunk_text = text[start:end]
            
            # Extract metadata
            speakers = self._extract_speakers(chunk_text)
            time_markers = self._extract_time_markers(chunk_text)
            
            chunks.append(TranscriptChunk(
                text=chunk_text,
                start_idx=start,
                end_idx=end,
                speakers=speakers,
                time_markers=time_markers,
                metadata={"strategy": "fixed_size"}
            ))
            
            # Move start position for next chunk, accounting for overlap
            start = end - chunk_overlap
            
            # Ensure we make progress even with large overlaps
            if start <= chunks[-1].start_idx:
                start = chunks[-1].start_idx + 1
                
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def detect_document_boundaries(self, text: str) -> List[DocumentBoundary]:
        """
        Detect natural document boundaries in text, with special focus on 
        transcript-specific boundaries like speaker changes.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of DocumentBoundary objects
        """
        logger.info(f"Detecting document boundaries in text ({len(text)} chars)")
        
        boundaries = []
        
        # 1. Find speaker transitions
        speakers = self._extract_speakers(text)
        if speakers:
            speaker_pattern = '|'.join([re.escape(s) for s in speakers])
            for match in re.finditer(f"(?:^|\n)({speaker_pattern}):", text):
                position = match.start()
                speaker = match.group(1)
                
                # Get context (20 chars before and after)
                context_start = max(0, position - 20)
                context_end = min(len(text), position + 20)
                context = text[context_start:context_end]
                
                boundaries.append(DocumentBoundary(
                    position=position,
                    boundary_type="speaker_change",
                    confidence=0.9,  # High confidence for speaker transitions
                    context=context
                ))
        
        # 2. Find paragraph breaks (double newlines)
        paragraph_pattern = r'\n\s*\n'
        for match in re.finditer(paragraph_pattern, text):
            position = match.start()
            
            # Get context
            context_start = max(0, position - 20)
            context_end = min(len(text), position + 20)
            context = text[context_start:context_end]
            
            boundaries.append(DocumentBoundary(
                position=position,
                boundary_type="paragraph",
                confidence=0.8,  # High confidence for clear paragraph breaks
                context=context
            ))
        
        # 3. Find section headers or topic transitions
        # Pattern for common section headers (e.g., "## Section Title")
        header_patterns = [
            (r'#+\s+[A-Z]', 0.9),  # Markdown headers
            (r'\n[A-Z][A-Z\s]+\n', 0.8),  # ALL CAPS HEADERS
            (r'\n\d+\.\s+[A-Z]', 0.7),  # Numbered sections
            (r'(?i)(?:^|\n)(?:Next|Now|Moving on to|Let\'s discuss|Let\'s talk about|Turning to|Regarding|About)', 0.7)  # Topic transitions
        ]
        
        for pattern, confidence in header_patterns:
            for match in re.finditer(pattern, text):
                position = match.start()
                
                # Get context
                context_start = max(0, position - 20)
                context_end = min(len(text), position + 20)
                context = text[context_start:context_end]
                
                boundaries.append(DocumentBoundary(
                    position=position,
                    boundary_type="section",
                    confidence=confidence,
                    context=context
                ))
        
        # 4. Find format shifts (e.g., from prose to a list)
        format_shifts = [
            (r'\n\s*[-*•]\s', 0.7),  # Start of a list
            (r'\n\s*\d+\.\s', 0.7),  # Start of a numbered list
            (r'\n\s*```', 0.9),      # Code block
            (r'\n\s*\|', 0.8)        # Table
        ]
        
        for pattern, confidence in format_shifts:
            for match in re.finditer(pattern, text):
                position = match.start()
                
                # Get context
                context_start = max(0, position - 20)
                context_end = min(len(text), position + 20)
                context = text[context_start:context_end]
                
                boundaries.append(DocumentBoundary(
                    position=position,
                    boundary_type="format_shift",
                    confidence=confidence,
                    context=context
                ))
        
        # 5. Find time markers as potential boundaries
        time_markers = self._extract_time_markers(text)
        for marker in time_markers:
            marker_pattern = re.escape(marker)
            for match in re.finditer(marker_pattern, text):
                position = match.start()
                
                # Get context
                context_start = max(0, position - 20)
                context_end = min(len(text), position + 20)
                context = text[context_start:context_end]
                
                boundaries.append(DocumentBoundary(
                    position=position,
                    boundary_type="time_marker",
                    confidence=0.75,
                    context=context
                ))
        
        # Use LLM for advanced boundary detection if available
        if self.use_llm:
            try:
                llm_boundaries = self._detect_boundaries_with_llm(text)
                if llm_boundaries:
                    # Add LLM-detected boundaries with higher confidence
                    for b in llm_boundaries:
                        boundaries.append(DocumentBoundary(
                            position=b["position"],
                            boundary_type=b["type"],
                            confidence=b.get("confidence", 0.85),  # LLM boundaries get high confidence
                            context=b.get("context", "")
                        ))
            except Exception as e:
                logger.warning(f"Error detecting boundaries with LLM: {e}")
            
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x.position)
        
        logger.info(f"Detected {len(boundaries)} potential document boundaries")
        return boundaries
    
    def _detect_boundaries_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to detect document boundaries, especially useful for detecting
        semantic boundaries in transcript texts.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of boundary dictionaries with position, type, and context
        """
        # Only analyze if text is long enough to potentially have boundaries
        if len(text) < 1000:
            return []
            
        # Sample the text if it's too long
        if len(text) > 8000:
            # Take samples from beginning, middle, and end
            sample_size = 2500
            beginning = text[:sample_size]
            middle_start = max(0, len(text)//2 - sample_size//2)
            middle = text[middle_start:middle_start+sample_size]
            end = text[-sample_size:]
            
            # Clear markers for the samples
            sample_text = f"[BEGINNING]\n{beginning}\n\n[MIDDLE]\n{middle}\n\n[END]\n{end}"
        else:
            sample_text = text
            
        prompt = """
        Analyze the following transcript and identify potential boundaries (places where the discussion shifts).
        
        For a meeting transcript, look for:
        - Speaker changes (new person speaking)
        - Topic shifts (discussion moves to a new subject)
        - Question-answer transitions
        - Format changes (e.g., presentation to open discussion)
        - Time transitions (moving to a new agenda item)
        
        Return a JSON list of boundary objects with these properties:
        - position: approximate character position in the original text
        - type: the boundary type (speaker_change, topic_shift, question_answer, format_change, time_transition)
        - confidence: your confidence level (0.0-1.0)
        - context: a short snippet showing the boundary context
        
        TRANSCRIPT TO ANALYZE:
        """
        
        try:
            result = self.openai_client.generate_completion(prompt + sample_text)
            
            # Extract JSON part of the response
            import json
            import re
            
            # Find JSON in the response (handling potential explanatory text)
            json_pattern = r'\[\s*\{.*\}\s*\]'
            json_match = re.search(json_pattern, result, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                boundaries = json.loads(json_str)
                
                # Adjust positions if we used samples
                if len(text) > 8000:
                    for b in boundaries:
                        if "[BEGINNING]" in b.get("context", ""):
                            # Position is already correct for beginning
                            pass
                        elif "[MIDDLE]" in b.get("context", ""):
                            b["position"] += len(text)//2 - 2500//2
                        elif "[END]" in b.get("context", ""):
                            b["position"] += len(text) - 2500
                
                return boundaries
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error processing LLM boundary detection: {e}")
            return []
    
    def chunk_text_boundary_aware(self, 
                                 text: str, 
                                 min_chunk_size: int = 200,
                                 max_chunk_size: int = 1500) -> List[TranscriptChunk]:
        """
        Split text into chunks based on detected natural boundaries,
        with special awareness of transcript-specific boundaries like speaker changes.
        
        Args:
            text: The text to chunk
            min_chunk_size: Minimum size for chunks in characters
            max_chunk_size: Maximum size for chunks in characters
            
        Returns:
            List of TranscriptChunk objects
        """
        if not text:
            return []
        
        logger.info(f"Chunking text ({len(text)} chars) with boundary-aware strategy")
        
        # Detect boundaries
        boundaries = self.detect_document_boundaries(text)
        
        # In utils/chunking.py, find this section within chunk_text_boundary_aware:

        # Filter boundaries by confidence threshold
        high_confidence_boundaries = [
            b for b in boundaries 
            if b.confidence >= self.boundary_detection_threshold
        ]

        logger.info(f"Using {len(high_confidence_boundaries)} high-confidence boundaries " 
                f"out of {len(boundaries)} detected")

        # If no high-confidence boundaries, fall back to fixed-size chunking
        if not high_confidence_boundaries:
            logger.warning("No high-confidence boundaries found, falling back to fixed-size chunking")
            return self.chunk_text_fixed_size(text)

        # Replace it with this improved version:

        # Filter boundaries by confidence threshold
        high_confidence_boundaries = [
            b for b in boundaries 
            if b.confidence >= self.boundary_detection_threshold
        ]

        logger.info(f"Using {len(high_confidence_boundaries)} high-confidence boundaries " 
                f"out of {len(boundaries)} detected")

        # If no high-confidence boundaries or text is short, use speaker-aware chunking
        if not high_confidence_boundaries or len(text) < max_chunk_size:
            if len(text) < max_chunk_size:
                logger.info(f"Text is shorter than max_chunk_size ({len(text)} < {max_chunk_size}), returning single chunk")
                speakers = self._extract_speakers(text)
                time_markers = self._extract_time_markers(text)
                return [TranscriptChunk(
                    text=text,
                    start_idx=0,
                    end_idx=len(text),
                    speakers=speakers,
                    time_markers=time_markers,
                    metadata={"strategy": "boundary_aware_single"}
                )]
            
            logger.warning("No high-confidence boundaries found, falling back to speaker-aware chunking")
            return self.chunk_text_speaker_aware(text, max_chunk_size, min_chunk_size)
        
        # Create chunks based on boundaries
        chunks = []
        
        # Add text start and end as boundary positions
        positions = [0] + [b.position for b in high_confidence_boundaries] + [len(text)]
        boundary_types = ["document_start"] + [b.boundary_type for b in high_confidence_boundaries] + ["document_end"]
        
        # Process each segment between boundaries
        for i in range(len(positions) - 1):
            segment_start = positions[i]
            segment_end = positions[i+1]
            segment_length = segment_end - segment_start
            
            # If segment is larger than max_chunk_size, subdivide it
            if segment_length > max_chunk_size:
                logger.info(f"Segment {i} exceeds max size ({segment_length} > {max_chunk_size}), subdividing")
                
                # Use sentence-aware subdivision
                segment_text = text[segment_start:segment_end]
                
                # Try to split at sentence boundaries
                try:
                    sentences = nltk.sent_tokenize(segment_text)
                    current_chunk_text = ""
                    current_chunk_start = segment_start
                    
                    for sentence in sentences:
                        # If adding this sentence would exceed max_chunk_size, create a chunk and start a new one
                        if len(current_chunk_text) + len(sentence) > max_chunk_size and len(current_chunk_text) > min_chunk_size:
                            # Add current chunk
                            current_chunk_end = current_chunk_start + len(current_chunk_text)
                            
                            # Extract metadata
                            speakers = self._extract_speakers(current_chunk_text)
                            time_markers = self._extract_time_markers(current_chunk_text)
                            
                            chunks.append(TranscriptChunk(
                                text=current_chunk_text,
                                start_idx=current_chunk_start,
                                end_idx=current_chunk_end,
                                speakers=speakers,
                                time_markers=time_markers,
                                metadata={
                                    "strategy": "boundary_aware_subdivided",
                                    "boundary_type": boundary_types[i],
                                    "subdivision": "sentence"
                                }
                            ))
                            
                            # Start new chunk
                            current_chunk_text = sentence
                            current_chunk_start = current_chunk_end
                        else:
                            # Add sentence to current chunk
                            current_chunk_text += sentence
                    
                    # Add the last chunk if it has content
                    if current_chunk_text:
                        # Extract metadata
                        speakers = self._extract_speakers(current_chunk_text)
                        time_markers = self._extract_time_markers(current_chunk_text)
                        
                        chunks.append(TranscriptChunk(
                            text=current_chunk_text,
                            start_idx=current_chunk_start,
                            end_idx=current_chunk_start + len(current_chunk_text),
                            speakers=speakers,
                            time_markers=time_markers,
                            metadata={
                                "strategy": "boundary_aware_subdivided",
                                "boundary_type": boundary_types[i],
                                "subdivision": "sentence"
                            }
                        ))
                        
                except Exception as e:
                    logger.warning(f"Error in sentence-aware subdivision: {e}. Falling back to fixed-size.")
                    # Fall back to fixed-size subdivision
                    segment_chunks = self.chunk_text_fixed_size(
                        text[segment_start:segment_end],
                        chunk_size=max_chunk_size,
                        chunk_overlap=min(200, max_chunk_size // 5)
                    )
                    
                    # Adjust indices to be relative to the original text
                    for chunk in segment_chunks:
                        chunk.start_idx += segment_start
                        chunk.end_idx += segment_start
                        chunk.metadata = {
                            "strategy": "boundary_aware_subdivided",
                            "boundary_type": boundary_types[i],
                            "subdivision": "fixed_size"
                        }
                    
                    chunks.extend(segment_chunks)
            
            # If segment is smaller than min_chunk_size and not the last segment,
            # consider merging with the next segment
            elif segment_length < min_chunk_size and i < len(positions) - 2:
                # Look ahead to see if combining with next segment is reasonable
                next_segment_length = positions[i+2] - positions[i+1]
                
                if segment_length + next_segment_length <= max_chunk_size:
                    logger.info(f"Segment {i} below min size ({segment_length} < {min_chunk_size}), "
                               f"will merge with next segment")
                    continue
                else:
                    # Small segment but merging not feasible, keep as is
                    chunk_text = text[segment_start:segment_end]
                    
                    # Extract metadata
                    speakers = self._extract_speakers(chunk_text)
                    time_markers = self._extract_time_markers(chunk_text)
                    
                    chunks.append(TranscriptChunk(
                        text=chunk_text,
                        start_idx=segment_start,
                        end_idx=segment_end,
                        speakers=speakers,
                        time_markers=time_markers,
                        metadata={
                            "strategy": "boundary_aware",
                            "boundary_type": boundary_types[i]
                        }
                    ))
            
            # Otherwise, create a chunk for this segment
            else:
                chunk_text = text[segment_start:segment_end]
                
                # Extract metadata
                speakers = self._extract_speakers(chunk_text)
                time_markers = self._extract_time_markers(chunk_text)
                
                chunks.append(TranscriptChunk(
                    text=chunk_text,
                    start_idx=segment_start,
                    end_idx=segment_end,
                    speakers=speakers,
                    time_markers=time_markers,
                    metadata={
                        "strategy": "boundary_aware",
                        "boundary_type": boundary_types[i]
                    }
                ))
        
        logger.info(f"Created {len(chunks)} boundary-aware chunks")
        return chunks
    
    def chunk_text_speaker_aware(self, 
                                text: str, 
                                max_chunk_size: int = 2000,
                                min_chunk_size: int = 500) -> List[TranscriptChunk]:
        """
        Split text into chunks based on speaker transitions, optimized for meeting transcripts.
        
        Args:
            text: The text to chunk
            max_chunk_size: Maximum size for chunks in characters
            min_chunk_size: Minimum size for chunks in characters
            
        Returns:
            List of TranscriptChunk objects
        """
        if not text:
            return []
        
        logger.info(f"Chunking text ({len(text)} chars) with speaker-aware strategy")
        
        # Extract speaker segments
        speaker_segments = self._segment_by_speakers(text)
        
        # If no speaker segments detected, fall back to boundary-aware chunking
        if not speaker_segments:
            logger.warning("No speaker segments detected, falling back to boundary-aware chunking")
            return self.chunk_text_boundary_aware(text)
        
        # Create chunks based on speaker segments
        chunks = []
        current_chunk_text = ""
        current_chunk_start = 0
        current_speakers = set()
        current_time_markers = []
        
        for segment in speaker_segments:
            speaker = segment["speaker"]
            segment_text = segment["text"]
            position = segment["position"]
            
            # Extract time markers for this segment
            time_markers = self._extract_time_markers(segment_text)
            
            # If adding this segment would exceed max_chunk_size, finalize the current chunk
            if len(current_chunk_text) + len(segment_text) > max_chunk_size and len(current_chunk_text) >= min_chunk_size:
                chunks.append(TranscriptChunk(
                    text=current_chunk_text,
                    start_idx=current_chunk_start,
                    end_idx=current_chunk_start + len(current_chunk_text),
                    speakers=list(current_speakers),
                    time_markers=current_time_markers,
                    metadata={"strategy": "speaker_aware"}
                ))
                
                # Start new chunk with this segment
                current_chunk_text = segment_text
                current_chunk_start = position
                current_speakers = {speaker}
                current_time_markers = time_markers
            else:
                # Add segment to current chunk
                current_chunk_text += segment_text
                current_speakers.add(speaker)
                current_time_markers.extend(time_markers)
        
        # Add the final chunk if it has content
        if current_chunk_text:
            chunks.append(TranscriptChunk(
                text=current_chunk_text,
                start_idx=current_chunk_start,
                end_idx=current_chunk_start + len(current_chunk_text),
                speakers=list(current_speakers),
                time_markers=current_time_markers,
                metadata={"strategy": "speaker_aware"}
            ))
        
        logger.info(f"Created {len(chunks)} speaker-aware chunks")
        return chunks
    
    def chunk_text_semantic(self, text: str, target_chunk_count: int = None) -> List[TranscriptChunk]:
        """
        Split text into semantically coherent chunks using LLM assistance.
        Requires use_llm=True when initializing the processor.
        
        Args:
            text: The text to chunk
            target_chunk_count: Approximately how many chunks to create
            
        Returns:
            List of TranscriptChunk objects
        """
        if not text:
            return []
            
        logger.info(f"Chunking text ({len(text)} chars) with semantic strategy")
        
        # Check if LLM is available
        if not self.use_llm:
            logger.warning("LLM not available for semantic chunking, falling back to speaker-aware chunking")
            return self.chunk_text_speaker_aware(text)
        
        # For very long texts, fall back to boundary-aware chunking first
        if len(text) > 8000:
            logger.info("Text too long for direct semantic chunking, using boundary-aware chunking first")
            initial_chunks = self.chunk_text_boundary_aware(text)
            
            # Now apply semantic chunking to each large chunk if needed
            result_chunks = []
            for chunk in initial_chunks:
                if chunk.length > 4000:
                    # Further chunk this large piece semantically
                    semantic_subchunks = self._apply_semantic_chunking(chunk.text, chunk.start_idx)
                    result_chunks.extend(semantic_subchunks)
                else:
                    result_chunks.append(chunk)
                    
            return result_chunks
        
        # For manageable text, apply semantic chunking directly
        return self._apply_semantic_chunking(text, 0)
    
    def _apply_semantic_chunking(self, text: str, base_offset: int = 0) -> List[TranscriptChunk]:
        """
        Apply semantic chunking to text using LLM.
        
        Args:
            text: Text to chunk semantically
            base_offset: Starting character offset for the text in the original document
            
        Returns:
            List of semantically chunked pieces
        """
        prompt = """
        Analyze the following meeting transcript and divide it into meaningful semantic chunks.
        Each chunk should:
        1. Contain a complete discussion on a specific topic or subtopic
        2. Preserve the natural flow of conversation
        3. Keep related questions and answers together
        4. Not break apart important context
        
        Return a JSON array with each chunk's information:
        [
            {
                "start_char": <approximate starting character position>,
                "end_char": <approximate ending character position>,
                "topic": "<brief topic description>",
                "speakers": ["<speaker1>", "<speaker2>", ...],
                "reason": "<why this is a good semantic boundary>"
            }
        ]
        
        TRANSCRIPT TO CHUNK:
        """
        
        try:
            result = self.openai_client.generate_completion(prompt + text)
            
            # Extract JSON part of the response
            import json
            import re
            
            # Find JSON in the response
            json_pattern = r'\[\s*\{.*\}\s*\]'
            json_match = re.search(json_pattern, result, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                chunk_boundaries = json.loads(json_str)
                
                # Create chunks based on identified boundaries
                chunks = []
                prev_end = 0
                
                for i, boundary in enumerate(chunk_boundaries):
                    # Get approximate positions
                    start_char = boundary.get("start_char", prev_end)
                    end_char = boundary.get("end_char", len(text))
                    
                    # Adjust to ensure we don't miss text or overlap
                    start_char = max(prev_end, min(start_char, len(text)))
                    end_char = max(start_char, min(end_char, len(text)))
                    
                    # Create chunk
                    chunk_text = text[start_char:end_char]
                    
                    # Get speakers and topics
                    speakers_from_llm = boundary.get("speakers", [])
                    detected_speakers = self._extract_speakers(chunk_text)
                    all_speakers = speakers_from_llm + [s for s in detected_speakers if s not in speakers_from_llm]
                    
                    topic = boundary.get("topic", "")
                    
                    # Extract time markers
                    time_markers = self._extract_time_markers(chunk_text)
                    
                    chunks.append(TranscriptChunk(
                        text=chunk_text,
                        start_idx=base_offset + start_char,
                        end_idx=base_offset + end_char,
                        speakers=all_speakers,
                        topics=[topic] if topic else [],
                        time_markers=time_markers,
                        metadata={
                            "strategy": "semantic",
                            "reason": boundary.get("reason", "")
                        }
                    ))
                    
                    prev_end = end_char
                
                # If there's remaining text, add it as a final chunk
                if prev_end < len(text):
                    final_text = text[prev_end:]
                    final_speakers = self._extract_speakers(final_text)
                    final_time_markers = self._extract_time_markers(final_text)
                    
                    chunks.append(TranscriptChunk(
                        text=final_text,
                        start_idx=base_offset + prev_end,
                        end_idx=base_offset + len(text),
                        speakers=final_speakers,
                        topics=["Concluding discussion"],
                        time_markers=final_time_markers,
                        metadata={
                            "strategy": "semantic",
                            "reason": "Final content"
                        }
                    ))
                
                logger.info(f"Created {len(chunks)} semantic chunks")
                return chunks
            else:
                logger.warning("Could not extract semantic chunks from LLM response")
                # Fall back to speaker-aware chunking
                return self.chunk_text_speaker_aware(text)
                
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            # Fall back to speaker-aware chunking
            return self.chunk_text_speaker_aware(text)
    
    def _extract_speakers(self, text: str) -> List[str]:
        """
        Extract speaker names from transcript text.
        
        Args:
            text: Transcript text to analyze
            
        Returns:
            List of identified speaker names
        """
        speakers = []
        
        # Common speaker patterns in transcripts
        patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+):', # Full name: pattern
            r'([A-Z][a-z]+):', # First name: pattern
            r'(Dr\. [A-Z][a-z]+):', # Dr. Name: pattern
            r'(Mr\. [A-Z][a-z]+):', # Mr. Name: pattern
            r'(Mrs\. [A-Z][a-z]+):', # Mrs. Name: pattern
            r'(Ms\. [A-Z][a-z]+):', # Ms. Name: pattern
            r'(Speaker \d+):', # Speaker #: pattern
            r'\[(.*?)\]' # [Speaker] pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            speakers.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_speakers = []
        for speaker in speakers:
            if speaker not in seen:
                seen.add(speaker)
                unique_speakers.append(speaker)
        
        return unique_speakers
    
    def _extract_time_markers(self, text: str) -> List[str]:
        """
        Extract time markers from transcript text.
        
        Args:
            text: Transcript text to analyze
            
        Returns:
            List of identified time markers
        """
        time_markers = []
        
        # Time marker patterns
        patterns = [
            r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]',  # [HH:MM:SS] or [HH:MM]
            r'\[(\d{1,2}:\d{2}(?::\d{2})? (?:AM|PM))\]',  # [HH:MM:SS AM/PM] or [HH:MM AM/PM]
            r'\((\d{1,2}:\d{2}(?::\d{2})?)\)',  # (HH:MM:SS) or (HH:MM)
            r'\((\d{1,2}:\d{2}(?::\d{2})? (?:AM|PM))\)'  # (HH:MM:SS AM/PM) or (HH:MM AM/PM)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            time_markers.extend(matches)
        
        return time_markers
    
    def _segment_by_speakers(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment transcript by speaker transitions.
        
        Args:
            text: The transcript text to segment
            
        Returns:
            List of speaker segments with speaker, text, and position
        """
        # Extract all speakers first
        all_speakers = self._extract_speakers(text)
        
        if not all_speakers:
            return []
        
        # Create a pattern to match any speaker
        speaker_pattern = '|'.join([re.escape(speaker) for speaker in all_speakers])
        pattern = f"(?:^|\n)({speaker_pattern}):(.*?)(?=(?:^|\n)(?:{speaker_pattern}):|$)"
        
        segments = []
        
        # Find all speaker segments
        for match in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
            speaker = match.group(1)
            content = match.group(2).strip()
            position = match.start()
            
            segments.append({
                "speaker": speaker,
                "text": f"{speaker}: {content}\n\n",
                "position": position
            })
        
        # If we couldn't find any segments with the regex, try a simpler approach
        if not segments:
            lines = text.split('\n')
            current_position = 0
            current_speaker = None
            current_content = ""
            
            for line in lines:
                # Check if this line starts with a speaker
                speaker_match = False
                
                for speaker in all_speakers:
                    if line.startswith(f"{speaker}:"):
                        # If we have content from a previous speaker, add it as a segment
                        if current_speaker and current_content:
                            segments.append({
                                "speaker": current_speaker,
                                "text": current_content,
                                "position": current_position - len(current_content)
                            })
                        
                        # Start a new segment with this speaker
                        current_speaker = speaker
                        current_content = line + "\n"
                        speaker_match = True
                        break
                
                # If not a speaker line, add to current content if we have a speaker
                if not speaker_match and current_speaker:
                    current_content += line + "\n"
                
                current_position += len(line) + 1  # +1 for newline
            
            # Add the final segment if we have one
            if current_speaker and current_content:
                segments.append({
                    "speaker": current_speaker,
                    "text": current_content,
                    "position": current_position - len(current_content)
                })
        
        logger.info(f"Segmented transcript into {len(segments)} speaker segments")
        return segments
    
    def chunk_document(self, 
                      text: str, 
                      strategy: str = "speaker_aware", 
                      compute_metrics: bool = True,
                      **kwargs) -> Dict[str, Any]:
        """
        Chunk a document using the specified strategy and compute metrics.
        
        Args:
            text: The text to chunk
            strategy: Chunking strategy ('fixed_size', 'speaker_aware', 'boundary_aware', 'semantic')
            compute_metrics: Whether to compute and return chunking metrics
            **kwargs: Additional parameters for the chunking strategy
            
        Returns:
            Dictionary with chunks, metrics, and boundaries (if applicable)
        """
        logger.info(f"Chunking document ({len(text)} chars) with strategy: {strategy}")
        
        results = {
            "strategy": strategy,
            "document_length": len(text)
        }
        
        # Apply the selected chunking strategy
        if strategy == "fixed_size":
            chunks = self.chunk_text_fixed_size(
                text=text,
                chunk_size=kwargs.get("chunk_size", self.default_chunk_size),
                chunk_overlap=kwargs.get("chunk_overlap", self.default_chunk_overlap)
            )
        elif strategy == "speaker_aware":
            chunks = self.chunk_text_speaker_aware(
                text=text,
                max_chunk_size=kwargs.get("max_chunk_size", 2000),
                min_chunk_size=kwargs.get("min_chunk_size", 500)
            )
        elif strategy == "boundary_aware":
            chunks = self.chunk_text_boundary_aware(
                text=text,
                min_chunk_size=kwargs.get("min_chunk_size", 200),
                max_chunk_size=kwargs.get("max_chunk_size", 1500)
            )
        elif strategy == "semantic":
            chunks = self.chunk_text_semantic(
                text=text,
                target_chunk_count=kwargs.get("target_chunk_count", None)
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        results["chunks"] = chunks
        
        # Compute metrics if requested
        if compute_metrics:
            metrics = self._compute_chunking_metrics(chunks, text)
            results["metrics"] = metrics
        
        return results
    
    def _compute_chunking_metrics(self, 
                                chunks: List[TranscriptChunk], 
                                original_text: str) -> Dict[str, Any]:
        """
        Compute metrics to evaluate chunking quality.
        
        Args:
            chunks: List of chunks to evaluate
            original_text: Original text that was chunked
            
        Returns:
            Dictionary of metric names and values
        """
        if not chunks:
            return {"chunk_count": 0, "error": "No chunks provided"}
        
        logger.info(f"Computing metrics for {len(chunks)} chunks")
        
        metrics = {
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(c.length for c in chunks) / len(chunks),
            "min_chunk_size": min(c.length for c in chunks),
            "max_chunk_size": max(c.length for c in chunks),
            "size_std_dev": np.std([c.length for c in chunks]),
            "total_text_coverage": sum(c.length for c in chunks) / len(original_text) if original_text else 0,
        }
        
        # Calculate speaker preservation metrics if speakers are available
        all_speakers = set()
        for chunk in chunks:
            all_speakers.update(chunk.speakers)
        
        if all_speakers:
            metrics["unique_speakers"] = len(all_speakers)
            
            # Count speakers per chunk
            speaker_counts = [len(c.speakers) for c in chunks]
            metrics["avg_speakers_per_chunk"] = sum(speaker_counts) / len(chunks)
            metrics["max_speakers_per_chunk"] = max(speaker_counts) if speaker_counts else 0
            
            # Calculate how many chunks each speaker appears in
            speaker_chunk_counts = {}
            for speaker in all_speakers:
                speaker_chunk_counts[speaker] = sum(1 for c in chunks if speaker in c.speakers)
            
            metrics["avg_chunks_per_speaker"] = sum(speaker_chunk_counts.values()) / len(speaker_chunk_counts)
            metrics["max_chunks_per_speaker"] = max(speaker_chunk_counts.values()) if speaker_chunk_counts else 0
        
        # Calculate sentence integrity metrics
        try:
            sentences = nltk.sent_tokenize(original_text)
            sentence_breaks = 0
            
            # Find sentence starts
            sentence_starts = []
            start_pos = 0
            for sentence in sentences:
                idx = original_text.find(sentence, start_pos)
                if idx != -1:
                    sentence_starts.append(idx)
                    start_pos = idx + len(sentence)
            
            # Count broken sentences
            for start_pos in sentence_starts:
                for chunk in chunks:
                    if chunk.start_idx < start_pos < chunk.end_idx - 10:  # Sentence starts mid-chunk
                        sentence_breaks += 1
                        break
            
            metrics["sentence_count"] = len(sentences)
            metrics["broken_sentence_count"] = sentence_breaks
            metrics["sentence_integrity_score"] = 1 - (sentence_breaks / len(sentences)) if sentences else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating sentence metrics: {e}")
            metrics["sentence_integrity_score"] = None
        
        logger.info(f"Computed chunking metrics: avg_size={metrics['avg_chunk_size']:.1f}, "
                  f"speakers_per_chunk={metrics.get('avg_speakers_per_chunk', 'N/A')}")
                  
        return metrics


# Public functions for easy usage
def chunk_by_character(transcript: str, chunk_size: int = 2000, overlap: int = 200) -> List[TranscriptChunk]:
    """
    Split transcript into fixed-size chunks with overlap.
    
    Args:
        transcript: The transcript text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of TranscriptChunk objects
    """
    processor = ChunkingProcessor()
    return processor.chunk_text_fixed_size(transcript, chunk_size, overlap)


def chunk_by_speaker(transcript: str, max_chunk_size: int = 2000) -> List[TranscriptChunk]:
    """
    Split transcript into chunks based on speaker transitions.
    
    Args:
        transcript: The transcript text to chunk
        max_chunk_size: Maximum size for any chunk in characters
        
    Returns:
        List of TranscriptChunk objects
    """
    processor = ChunkingProcessor()
    return processor.chunk_text_speaker_aware(transcript, max_chunk_size)


def chunk_by_boundary(transcript: str, max_chunk_size: int = 1500) -> List[TranscriptChunk]:
    """
    Split transcript into chunks based on natural boundaries.
    
    Args:
        transcript: The transcript text to chunk
        max_chunk_size: Maximum size for any chunk in characters
        
    Returns:
        List of TranscriptChunk objects
    """
    processor = ChunkingProcessor()
    return processor.chunk_text_boundary_aware(transcript, max_chunk_size=max_chunk_size)


def extract_speakers(text: str) -> List[str]:
    """
    Extract speaker names from transcript text.
    
    Args:
        text: Transcript text to analyze
        
    Returns:
        List of identified speaker names
    """
    processor = ChunkingProcessor()
    return processor._extract_speakers(text)


def segment_by_speakers(text: str) -> List[Dict[str, Any]]:
    """
    Segment transcript by speaker transitions.
    
    Args:
        text: The transcript text to segment
        
    Returns:
        List of speaker segments with speaker, text, and position
    """
    processor = ChunkingProcessor()
    return processor._segment_by_speakers(text)


def chunk_transcript_advanced(
    transcript: str,
    strategy: str = "speaker_aware",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    use_llm: bool = False
) -> List[str]:
    """
    Enhanced transcript chunking with multiple strategies.
    
    Args:
        transcript: The transcript text to chunk
        strategy: Chunking strategy ('fixed_size', 'speaker_aware', 'boundary_aware', 'semantic')
        chunk_size: Size for fixed-size chunks
        chunk_overlap: Overlap for fixed-size chunks
        use_llm: Whether to use LLM for advanced chunking
        
    Returns:
        List of text chunks
    """
    processor = ChunkingProcessor(
        default_chunk_size=chunk_size, 
        default_chunk_overlap=chunk_overlap,
        use_llm=use_llm
    )
    
    # Apply the selected chunking strategy
    if strategy == "fixed_size":
        chunks = processor.chunk_text_fixed_size(transcript)
    elif strategy == "speaker_aware":
        chunks = processor.chunk_text_speaker_aware(transcript)
    elif strategy == "boundary_aware":
        chunks = processor.chunk_text_boundary_aware(transcript)
    elif strategy == "semantic":
        chunks = processor.chunk_text_semantic(transcript)
    else:
        logger.warning(f"Unknown chunking strategy: {strategy}, falling back to speaker_aware")
        chunks = processor.chunk_text_speaker_aware(transcript)
    
    # Convert chunk objects to plain text for compatibility with existing code
    return [chunk.text for chunk in chunks]