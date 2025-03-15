"""
Core transcript summarization engine with flexible document division strategies.
"""

import time
import logging
import re
from typing import List, Dict, Any, Optional
import os

from .options import SummaryOptions
from .llm.base import LLMAdapter
from .llm.openai_adapter import OpenAIAdapter

try:
    from .llm.litellm_adapter import LiteLLMAdapter
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptSummarizer:
    """
    Core transcript summarization engine using document division strategies.
    """
    
    def __init__(self, options: Optional[SummaryOptions] = None):
        """
        Initialize the summarizer with options.
        
        Args:
            options: Configuration options
        """
        self.options = options or SummaryOptions()
        self.llm_client = self._initialize_llm_client()
    
    def _initialize_llm_client(self) -> LLMAdapter:
        """Initialize the appropriate LLM client based on config."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if LITELLM_AVAILABLE:
            try:
                logger.info("Initializing LiteLLM adapter")
                return LiteLLMAdapter(
                    model=self.options.model_name,
                    api_key=api_key,
                    temperature=self.options.temperature
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LiteLLM adapter: {e}. Falling back to direct OpenAI.")
        
        logger.info("Initializing direct OpenAI adapter")
        return OpenAIAdapter(
            model=self.options.model_name,
            api_key=api_key,
            temperature=self.options.temperature
        )
    
    def summarize(self, transcript: str) -> Dict[str, Any]:
        """
        Summarize a transcript using the configured strategy.
        
        Args:
            transcript: The transcript text
            
        Returns:
            Dictionary with summary and metadata
        """
        start_time = time.time()
        
        # Divide the transcript into manageable pieces
        divisions = self._divide_transcript(transcript)
        
        if self.options.verbose:
            logger.info(f"Divided transcript into {len(divisions)} pieces using {self.options.chunk_strategy} strategy")
        
        # Process the divisions to create a summary
        summary = self._process_divisions(transcript, divisions)
        
        # Extract action items if requested
        action_items = None
        if self.options.include_action_items:
            action_items = self._extract_action_items(transcript, divisions)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "summary": summary,
            "action_items": action_items,
            "divisions": divisions,
            "metadata": {
                "division_count": len(divisions),
                "division_strategy": self.options.chunk_strategy,
                "model": self.options.model_name,
                "processing_time_seconds": processing_time
            }
        }
        
        return result
    
    def _divide_transcript(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Divide transcript into manageable pieces based on selected strategy.
        
        Args:
            transcript: The transcript text
            
        Returns:
            List of division dictionaries with position and text
        """
        max_size = self.options.max_chunk_size
        
        if self.options.chunk_strategy == "speaker":
            return self._divide_by_speaker(transcript, max_size)
        elif self.options.chunk_strategy == "boundary":
            return self._divide_by_boundary(transcript, max_size)
        elif self.options.chunk_strategy == "context_aware":
            return self._divide_context_aware(transcript, max_size)
        else:  # "basic" or fallback
            return self._divide_basic(transcript, max_size)
    
    def _divide_basic(self, text: str, max_size: int) -> List[Dict[str, Any]]:
        """
        Basic division into manageable pieces with smart breaks.
        
        Args:
            text: Text to divide
            max_size: Maximum size for each division
            
        Returns:
            List of division dictionaries
        """
        divisions = []
        start_pos = 0
        
        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + max_size, len(text))
            
            # If not at the end, try to find a better break point
            if end_pos < len(text):
                # Look for sentence endings in the last ~20% of the piece
                search_start = end_pos - min(int(max_size * 0.2), 500)
                search_text = text[search_start:end_pos]
                
                # Try paragraph break, then sentence break
                paragraph_pos = search_text.rfind('\n\n')
                if paragraph_pos > 0:
                    end_pos = search_start + paragraph_pos + 2
                else:
                    sentence_end = max(
                        search_text.rfind('. '),
                        search_text.rfind('! '),
                        search_text.rfind('? ')
                    )
                    if sentence_end > 0:
                        end_pos = search_start + sentence_end + 2
            
            # Add the division
            divisions.append({
                'start': start_pos,
                'end': end_pos,
                'text': text[start_pos:end_pos]
            })
            
            # Move to next position
            start_pos = end_pos
        
        return divisions
    
    def _divide_by_speaker(self, text: str, max_size: int) -> List[Dict[str, Any]]:
        """
        Divide text based on speaker transitions.
        
        Args:
            text: Text to divide
            max_size: Maximum size for each division
            
        Returns:
            List of division dictionaries
        """
        divisions = []
        
        # Extract speakers
        speakers = self._extract_speakers(text)
        
        # If no speakers found, fall back to basic division
        if not speakers:
            logger.info("No speakers detected, falling back to basic division")
            return self._divide_basic(text, max_size)
        
        # Build speaker pattern
        speaker_pattern = '|'.join([re.escape(s) for s in speakers])
        pattern = f"((?:^|\n)(?:{speaker_pattern}):)"
        
        # Split by speaker transitions
        segments = re.split(pattern, text)
        
        # Process segments
        current_division = ""
        current_start = 0
        
        # Skip first segment if it's not a speaker marker
        start_idx = 1 if segments and not any(s+":" in segments[0] for s in speakers) else 0
        
        for i in range(start_idx, len(segments), 2):
            if i+1 >= len(segments):
                break
                
            speaker_marker = segments[i]
            content = segments[i+1]
            segment = speaker_marker + content
            
            # Check if adding this would exceed max size
            if len(current_division) + len(segment) > max_size and current_division:
                # Finish current division
                divisions.append({
                    'start': current_start,
                    'end': current_start + len(current_division),
                    'text': current_division
                })
                
                # Start new division
                current_division = segment
                current_start += len(current_division)
            else:
                # Add to current division
                current_division += segment
        
        # Add final division if there's content
        if current_division:
            divisions.append({
                'start': current_start,
                'end': current_start + len(current_division),
                'text': current_division
            })
        
        return divisions
    
    def _divide_by_boundary(self, text: str, max_size: int) -> List[Dict[str, Any]]:
        """
        Divide text based on natural document boundaries.
        
        Args:
            text: Text to divide
            max_size: Maximum size for each division
            
        Returns:
            List of division dictionaries
        """
        divisions = []
        
        # Define boundary patterns in priority order
        boundary_patterns = [
            r'\n#{1,3}\s+',          # Markdown headers
            r'\n\s*\n',               # Paragraph breaks
            r'\n\s*\d+\.\s+',         # Numbered lists
            r'\n\s*[-*•]\s+',         # Bullet points
            r'(?<=[.!?])\s+(?=[A-Z])' # Sentence boundaries (lowest priority)
        ]
        
        # Find all boundary positions
        boundaries = []
        for pattern in boundary_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
        
        # Sort boundaries by position
        boundaries.sort()
        
        # If no boundaries found, fall back to basic division
        if not boundaries:
            return self._divide_basic(text, max_size)
        
        # Create divisions based on boundaries and max size
        start_pos = 0
        
        while start_pos < len(text):
            # Find the furthest boundary within max_size
            end_pos = min(start_pos + max_size, len(text))
            
            # Look for a boundary before the max position
            boundary_pos = None
            for pos in boundaries:
                if start_pos < pos < end_pos:
                    boundary_pos = pos
            
            # If boundary found, use it; otherwise use calculated end
            if boundary_pos:
                end_pos = boundary_pos
            
            # Add the division
            divisions.append({
                'start': start_pos,
                'end': end_pos,
                'text': text[start_pos:end_pos]
            })
            
            # Move to next position
            start_pos = end_pos
        
        return divisions
    
    def _divide_context_aware(self, text: str, max_size: int) -> List[Dict[str, Any]]:
        """
        Context-aware division that preserves semantic units and natural boundaries.
        
        Args:
            text: Text to divide
            max_size: Maximum size for each division
            
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
            return self._divide_basic(text, max_size)
        
        # Create divisions based on boundaries and size constraints
        divisions = []
        current_start = 0
        min_size = max(500, max_size // 3)  # Minimum size to avoid tiny divisions
        
        while current_start < len(text):
            # Find the best breakpoint within max_size
            next_breakpoint = None
            best_strength = 0
            
            for position, strength in breakpoints:
                if (current_start < position < current_start + max_size and 
                    position - current_start >= min_size and
                    strength > best_strength):
                    next_breakpoint = position
                    best_strength = strength
            
            # If no suitable breakpoint found, use max size or end of text
            if next_breakpoint is None:
                next_breakpoint = min(current_start + max_size, len(text))
                
                # Try to find a sentence boundary near the breakpoint if possible
                if next_breakpoint < len(text):
                    # Look for a sentence ending within the last 20% of the division
                    last_portion = text[next_breakpoint - int(max_size * 0.2):next_breakpoint]
                    last_sentence_end = max(
                        last_portion.rfind('. '),
                        last_portion.rfind('! '),
                        last_portion.rfind('? ')
                    )
                    
                    if last_sentence_end != -1:
                        # Adjust the breakpoint to this sentence ending
                        next_breakpoint = next_breakpoint - int(max_size * 0.2) + last_sentence_end + 2
            
            # Add the division
            division_text = text[current_start:next_breakpoint].strip()
            if division_text:  # Only add non-empty divisions
                divisions.append({
                    'start': current_start,
                    'end': next_breakpoint,
                    'text': division_text
                })
            
            # Move to next position
            current_start = next_breakpoint
        
        return divisions
    
    def _process_divisions(self, full_transcript: str, divisions: List[Dict[str, Any]]) -> str:
        """
        Process the transcript divisions to create a summary.
        
        Args:
            full_transcript: The complete transcript
            divisions: List of division dictionaries
            
        Returns:
            Summarized text
        """
        # For very short transcripts, just summarize directly
        if len(divisions) == 1:
            return self._summarize_single(full_transcript)
        
        # For texts with few divisions, include context
        if len(divisions) <= 3:
            return self._summarize_with_context(full_transcript, divisions)
        
        # For longer texts, use division-based approach
        return self._summarize_with_divisions(full_transcript, divisions)
    
    def _summarize_single(self, transcript: str) -> str:
        """
        Summarize a short transcript directly.
        
        Args:
            transcript: Transcript text
            
        Returns:
            Summary text
        """
        prompt = """Create a comprehensive, well-structured summary of this transcript.
        
        The summary should:
        1. Start with a concise Executive Summary
        2. Organize content logically by topic
        3. Include all significant details, decisions, and discussion points
        4. Preserve who said what when important
        5. Use clear headings and structure for readability
        
        FORMAT WITH MARKDOWN:
        # Meeting Summary
        
        ## Executive Summary
        
        ## Key Discussion Topics
        
        ### [Topic 1]
        
        ### [Topic 2]
        
        ## Decisions and Next Steps
        
        TRANSCRIPT:
        """
        
        try:
            return self.llm_client.generate_completion(prompt + transcript)
        except Exception as e:
            logger.error(f"Error summarizing transcript: {e}")
            return f"Error: {str(e)}"
    
    def _summarize_with_context(self, transcript: str, divisions: List[Dict[str, Any]]) -> str:
        """
        Summarize with full context for smaller transcripts.
        
        Args:
            transcript: Complete transcript text
            divisions: List of division dictionaries
            
        Returns:
            Summary text
        """
        # Calculate approx token count to avoid overflow
        approx_tokens = len(transcript) / 4
        
        # If it's likely to fit in context window, summarize directly
        if approx_tokens < 12000:  # Conservative estimate for gpt-3.5-turbo
            return self._summarize_single(transcript)
        
        # Otherwise, provide context about the division structure
        divisions_text = "\n\n---\n\n".join([
            f"SECTION {i+1}:\n{div['text']}" for i, div in enumerate(divisions)
        ])
        
        prompt = f"""Create a comprehensive, well-structured summary of this transcript.
        
        The transcript has been divided into {len(divisions)} sections.
        
        Your summary should:
        1. Start with a concise Executive Summary
        2. Organize content logically by topic
        3. Include all significant details, decisions, and discussion points
        4. Preserve who said what when important
        5. Use clear headings and structure for readability
        
        FORMAT WITH MARKDOWN:
        # Meeting Summary
        
        ## Executive Summary
        
        ## Key Discussion Topics
        
        ### [Topic 1]
        
        ### [Topic 2]
        
        ## Decisions and Next Steps
        
        TRANSCRIPT BY SECTION:
        {divisions_text}
        """
        
        try:
            return self.llm_client.generate_completion(prompt)
        except Exception as e:
            logger.error(f"Error summarizing with context: {e}")
            return self._fallback_summary(divisions)
    
    def _summarize_with_divisions(self, transcript: str, divisions: List[Dict[str, Any]]) -> str:
        """
        Summarize using divisions for longer transcripts.
        
        Args:
            transcript: Complete transcript text
            divisions: List of division dictionaries
            
        Returns:
            Summary text
        """
        # Extract key details from transcript to provide context
        speakers = self._extract_speakers(transcript)
        word_count = len(transcript.split())
        
        # Create context info
        context = f"""This is a transcript with approximately {word_count} words"""
        if speakers:
            context += f" featuring {len(speakers)} speakers: {', '.join(speakers)}"
        
        # Create division descriptions
        division_info = []
        for i, div in enumerate(divisions):
            # Extract a brief preview of each division
            preview = div['text'][:100].replace('\n', ' ').strip() + "..."
            division_info.append(f"Section {i+1}: {preview}")
        
        # Build the prompt
        prompt = f"""Create a comprehensive, well-structured summary of a transcript.

        CONTEXT:
        {context}
        
        TRANSCRIPT SECTIONS:
        {chr(10).join(division_info)}
        
        Your task is to create a summary that:
        1. Starts with a concise Executive Summary
        2. Organizes content logically by topic
        3. Includes all significant details, decisions, and discussion points
        4. Preserves who said what when important
        5. Uses clear headings and structure for readability
        
        I'll provide each section one by one, and you should build your summary progressively.
        
        FORMAT WITH MARKDOWN:
        # Meeting Summary
        
        ## Executive Summary
        
        ## Key Discussion Topics
        
        ### [Topic 1]
        
        ### [Topic 2]
        
        ## Decisions and Next Steps
        """
        
        # Process the first division
        first_prompt = f"{prompt}\n\nSECTION 1:\n{divisions[0]['text']}"
        
        try:
            # Get initial summary
            summary = self.llm_client.generate_completion(first_prompt)
            
            # Process remaining divisions
            for i in range(1, min(len(divisions), 5)):  # Process up to 5 divisions to keep reasonable
                # Create continuation prompt
                continuation_prompt = f"""
                Below is a partial summary followed by the next section of the transcript.
                Please update and expand the summary to incorporate information from this new section.
                Maintain the structure but add/modify content as needed based on the new information.
                
                CURRENT SUMMARY:
                {summary}
                
                SECTION {i+1}:
                {divisions[i]['text']}
                """
                
                try:
                    # Update summary
                    summary = self.llm_client.generate_completion(continuation_prompt)
                except Exception as e:
                    logger.error(f"Error processing division {i+1}: {e}")
                    # Continue with current summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return self._fallback_summary(divisions)
    
    def _extract_action_items(self, transcript: str, divisions: List[Dict[str, Any]]) -> str:
        """
        Extract action items from the transcript.
        
        Args:
            transcript: Complete transcript text
            divisions: List of division dictionaries
            
        Returns:
            Action items text
        """
        # For short transcripts, extract directly
        if len(transcript) < 6000:
            prompt = """Extract all action items, tasks, and commitments from this transcript.
            For each action item:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a prioritized, well-organized list. Don't miss any action items, even if implied.
            
            TRANSCRIPT:
            """
            
            try:
                return self.llm_client.generate_completion(prompt + transcript)
            except Exception as e:
                logger.error(f"Error extracting action items: {e}")
                return "Could not extract action items due to an error."
        
        # For longer transcripts, extract from key divisions
        divisions_to_check = min(4, len(divisions))
        
        # Focus on the first and last divisions, plus a sample from the middle
        indices = [0]  # Always check first division
        
        # Add last division if there are more than 2
        if len(divisions) > 2:
            indices.append(len(divisions) - 1)
        
        # Add 1-2 divisions from the middle
        if len(divisions) > 3:
            middle_idx = len(divisions) // 2
            indices.append(middle_idx)
            
            if len(divisions) > 6:
                indices.append(middle_idx + 2)  # Add another from middle-end
        
        # Ensure we don't have duplicates
        indices = sorted(set(indices))
        
        # Extract from selected divisions
        action_items = []
        
        for idx in indices:
            division = divisions[idx]
            prompt = f"""Extract all action items, tasks, and commitments from this transcript section.
            For each action item:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a bulleted list. Don't miss any action items, even if implied.
            
            TRANSCRIPT SECTION {idx+1}/{len(divisions)}:
            {division['text']}
            """
            
            try:
                result = self.llm_client.generate_completion(prompt)
                action_items.append(result)
            except Exception as e:
                logger.error(f"Error extracting action items from division {idx+1}: {e}")
        
        # Consolidate action items
        if not action_items:
            return "No action items identified."
        
        if len(action_items) == 1:
            return action_items[0]
        
        # Deduplicate multiple sets of action items
        combined = "\n\n".join(action_items)
        
        dedup_prompt = f"""Here are action items extracted from different parts of a transcript.
        Create a final, consolidated list that:
        1. Removes duplicates
        2. Organizes by responsible person or priority
        3. Merges related items
        4. Ensures all unique items are preserved
        
        EXTRACTED ITEMS:
        {combined}
        """
        
        try:
            return self.llm_client.generate_completion(dedup_prompt)
        except Exception as e:
            logger.error(f"Error deduplicating action items: {e}")
            return "# Action Items\n\n" + combined
    
    def _extract_speakers(self, text: str) -> List[str]:
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
    
    def _fallback_summary(self, divisions: List[Dict[str, Any]]) -> str:
        """
        Create a basic summary when other methods fail.
        
        Args:
            divisions: List of division dictionaries
            
        Returns:
            Basic summary text
        """
        summary = "# Meeting Summary\n\n"
        summary += "## Overview\n\n"
        summary += "This transcript contains multiple sections of content.\n\n"
        
        for i, div in enumerate(divisions):
            # Add brief info about each division
            preview = div['text'][:200].replace('\n', ' ').strip()
            if len(div['text']) > 200:
                preview += "..."
            
            summary += f"## Section {i+1}\n\n{preview}\n\n"
        
        return summary