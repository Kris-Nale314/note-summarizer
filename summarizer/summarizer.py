"""
Core document summarization engine with async support.
"""

import re
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable

from summarizer.options import SummaryOptions
from summarizer.llm.async_openai_adapter import AsyncOpenAIAdapter
from summarizer.synthesis import SummaryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncTranscriptSummarizer:
    """Transcript summarization engine with async support."""
    
    def __init__(self, options: Optional[SummaryOptions] = None):
        """Initialize the summarizer with options."""
        self.options = options or SummaryOptions()
        self.llm_client = AsyncOpenAIAdapter(
            model=self.options.model_name,
            temperature=self.options.temperature
        )
        self.synthesizer = SummaryProcessor(
            model=self.options.model_name,
            temperature=self.options.temperature
        )
    
    async def summarize_async(self, text: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Summarize a transcript using async processing.
        
        Args:
            text: The transcript text
            progress_callback: Optional callback function for progress updates
                Function signature: callback(status_message, progress_fraction)
            
        Returns:
            Dictionary with summary and metadata
        """
        start_time = time.time()
        
        # Update progress
        if progress_callback:
            progress_callback("Analyzing transcript format...", 0.1)
        
        # Detect if this is a Teams transcript
        is_teams_transcript = self._is_teams_transcript(text)
        
        # Divide the document into manageable chunks
        if progress_callback:
            progress_callback("Dividing transcript into manageable sections...", 0.2)
        
        if is_teams_transcript:
            divisions = self._divide_teams_transcript(text)
        else:
            divisions = self._divide_generic_transcript(text)
        
        # Process divisions to generate summaries
        if progress_callback:
            progress_callback(f"Processing {len(divisions)} sections concurrently...", 0.3)
        
        # Generate prompts for each division
        prompts = []
        for i, division in enumerate(divisions):
            prompt = self._create_division_prompt(division, i, len(divisions))
            prompts.append(prompt)
        
        # Process prompts concurrently
        # Limit concurrent requests to avoid rate limits
        max_concurrent = min(5, len(divisions))
        division_summaries = await self.llm_client.generate_completions_concurrently(
            prompts, max_concurrent=max_concurrent
        )
        
        # Update progress
        if progress_callback:
            progress_callback("Creating final synthesis...", 0.8)
        
        # Create metadata for synthesis
        metadata = {
            "is_teams_transcript": is_teams_transcript
        }
        
        # Extract speakers if it's a Teams transcript
        if is_teams_transcript:
            speakers = self._extract_speakers(text)
            if speakers:
                metadata["speakers"] = speakers
        
        # Generate final summary using the synthesizer
        final_summary = await self.synthesizer.synthesize_summaries(
            division_summaries,
            detail_level="standard",
            doc_type="transcript",
            metadata=metadata
        )
        
        # Extract action items if requested
        action_items = None
        if self.options.include_action_items:
            if progress_callback:
                progress_callback("Extracting action items...", 0.9)
            action_items = await self._extract_action_items_async(text, divisions)
        
        # Prepare result
        processing_time = time.time() - start_time
        
        if progress_callback:
            progress_callback("Summary complete!", 1.0)
        
        return {
            "summary": final_summary,
            "action_items": action_items,
            "divisions": divisions,
            "division_summaries": division_summaries,
            "metadata": {
                "division_count": len(divisions),
                "is_teams_transcript": is_teams_transcript,
                "model": self.options.model_name,
                "processing_time_seconds": processing_time
            }
        }
    
    def summarize(self, text: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for summarize_async.
        
        Args:
            text: The transcript text
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with summary and metadata
        """
        # Create an event loop and run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.summarize_async(text, progress_callback))
        finally:
            loop.close()
    
    def _is_teams_transcript(self, text: str) -> bool:
        """
        Detect if this is a Microsoft Teams transcript.
        
        Args:
            text: Transcript text
            
        Returns:
            True if this appears to be a Teams transcript
        """
        # Check for Teams-specific patterns
        teams_patterns = [
            r"\d{1,2}:\d{2}:\d{2}\s+[AP]M\s+From\s+.+:",  # Timestamp pattern
            r"Meeting\s+started:",                        # Meeting start marker
            r"Meeting\s+ended:",                         # Meeting end marker
            r"[\w\s]+\s+\(\d{1,2}:\d{2}:\d{2}\s+[AP]M\)"  # Speaker with timestamp
        ]
        
        # Sample the beginning of the document
        sample = text[:5000]
        
        # Check for Teams patterns
        for pattern in teams_patterns:
            if re.search(pattern, sample):
                logger.info("Detected Microsoft Teams transcript format")
                return True
        
        return False
    
    def _divide_teams_transcript(self, text: str) -> List[Dict[str, Any]]:
        """
        Divide a Teams transcript into speaker-based sections.
        
        Args:
            text: Teams transcript text
            
        Returns:
            List of division dictionaries
        """
        # Find speaker patterns in Teams transcripts
        speaker_pattern = r"(\d{1,2}:\d{2}:\d{2}\s+[AP]M\s+From\s+[^:]+:)"
        
        # Split by speaker transitions
        divisions = []
        
        # First check if we need to divide at all (small transcript)
        if len(text) < self.options.max_chunk_size * 0.8:
            logger.info("Transcript is small enough to process in one piece")
            return [{
                'start': 0,
                'end': len(text),
                'text': text
            }]
        
        # Find all speaker transitions
        transitions = list(re.finditer(speaker_pattern, text))
        
        if not transitions:
            # Fallback if no transitions found
            return self._divide_by_size(text)
        
        # Process transitions
        current_start = 0
        current_text = ""
        
        for i, match in enumerate(transitions):
            # If this isn't the first transition, add the previous section
            if i > 0:
                current_text = text[current_start:match.start()]
                
                # If adding this would exceed the limit, start a new chunk
                if len(current_text) > self.options.max_chunk_size:
                    divisions.append({
                        'start': current_start,
                        'end': match.start(),
                        'text': current_text
                    })
                    current_start = match.start()
                
            # For the last transition, add the remaining text
            if i == len(transitions) - 1:
                current_text = text[match.start():]
                divisions.append({
                    'start': match.start(),
                    'end': len(text),
                    'text': current_text
                })
        
        # If no divisions were created, use the whole text
        if not divisions:
            divisions.append({
                'start': 0,
                'end': len(text),
                'text': text
            })
        
        logger.info(f"Created {len(divisions)} divisions from Teams transcript")
        return divisions
    
    def _divide_generic_transcript(self, text: str) -> List[Dict[str, Any]]:
        """
        Divide a generic transcript into manageable sections.
        
        Args:
            text: Transcript text
            
        Returns:
            List of division dictionaries
        """
        # For generic transcripts, fall back to size-based division
        return self._divide_by_size(text)
    
    def _divide_by_size(self, text: str) -> List[Dict[str, Any]]:
        """
        Divide text into sections based on size constraints.
        
        Args:
            text: Document text
            
        Returns:
            List of division dictionaries
        """
        # Simple size-based division
        divisions = []
        max_size = self.options.max_chunk_size
        
        # If text is small enough, return it as is
        if len(text) <= max_size:
            return [{
                'start': 0,
                'end': len(text),
                'text': text
            }]
        
        # Calculate rough number of divisions needed
        num_divisions = (len(text) + max_size - 1) // max_size  # Ceiling division
        target_size = len(text) // num_divisions
        
        current_pos = 0
        while current_pos < len(text):
            # Calculate the target end position
            target_end = min(current_pos + target_size, len(text))
            
            # Try to find a good break point (paragraph)
            if target_end < len(text):
                # Look for paragraph breaks
                paragraph_break = text.rfind('\n\n', current_pos, target_end + 500)
                
                if paragraph_break != -1 and paragraph_break <= target_end + 500:
                    end_pos = paragraph_break + 2  # Include the newlines
                else:
                    # Look for sentence breaks
                    sentence_break = max(
                        text.rfind('. ', current_pos, target_end + 200),
                        text.rfind('! ', current_pos, target_end + 200),
                        text.rfind('? ', current_pos, target_end + 200)
                    )
                    
                    if sentence_break != -1 and sentence_break <= target_end + 200:
                        end_pos = sentence_break + 2  # Include the punctuation and space
                    else:
                        # Fall back to target size
                        end_pos = target_end
            else:
                end_pos = target_end
            
            # Create the division
            division_text = text[current_pos:end_pos]
            if division_text.strip():  # Only add non-empty divisions
                divisions.append({
                    'start': current_pos,
                    'end': end_pos,
                    'text': division_text
                })
            
            # Move to next position
            current_pos = end_pos
        
        logger.info(f"Created {len(divisions)} size-based divisions")
        return divisions
    
    def _create_division_prompt(self, division: Dict[str, Any], index: int, total: int) -> str:
        """
        Generate a prompt for a division.
        
        Args:
            division: Division dictionary
            index: Division index
            total: Total number of divisions
            
        Returns:
            Prompt text
        """
        text = division['text']
        
        # Create division context based on position
        position_context = ""
        if total > 1:
            if index == 0:
                position_context = "This is the BEGINNING section of the transcript. Focus on establishing context and introducing participants."
            elif index == total - 1:
                position_context = "This is the FINAL section of the transcript. Focus on conclusions, next steps, and final points."
            else:
                position_context = f"This is section {index+1} of {total}."
        
        # Create prompt
        prompt = f"""
        Create a detailed summary of this transcript section that:
        1. Preserves who said what (maintain speaker attribution for key points)
        2. Captures the main discussion topics, decisions, and action items
        3. Highlights important questions, answers, and exchanges
        4. Notes any agreements, disagreements, or unresolved points
        5. Extracts direct quotes for particularly important statements
        
        {position_context}
        
        Be thorough but also consolidate redundant discussions.
        
        TRANSCRIPT SECTION:
        {text}
        """
        
        return prompt.strip()
    
    async def _extract_action_items_async(self, text: str, divisions: List[Dict[str, Any]]) -> str:
        """
        Extract action items from the transcript asynchronously.
        
        Args:
            text: Full transcript text
            divisions: List of division dictionaries
            
        Returns:
            Action items text
        """
        # For short transcripts, extract directly
        if len(text) < self.options.max_chunk_size:
            prompt = """
            Extract all action items, tasks, and commitments from this transcript.
            For each action item, include:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a prioritized, well-organized list with markdown.
            
            TRANSCRIPT:
            """
            
            return await self.llm_client.generate_completion_async(prompt + text)
        
        # For longer transcripts, extract from key sections
        # Focus on beginning and end sections, plus sample from middle
        indices = [0]  # Always include first division
        
        if len(divisions) > 1:
            indices.append(len(divisions) - 1)  # Include last division
        
        if len(divisions) > 3:
            indices.append(len(divisions) // 2)  # Include middle division
        
        # Extract from selected divisions concurrently
        prompts = []
        for idx in indices:
            division = divisions[idx]
            prompt = f"""
            Extract all action items, tasks, and commitments from this transcript section.
            For each action item, include:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a markdown list. Don't include anything that's not an action item.
            
            TRANSCRIPT SECTION {idx+1}/{len(divisions)}:
            {division['text']}
            """
            prompts.append(prompt)
        
        # Process prompts concurrently
        action_items_results = await self.llm_client.generate_completions_concurrently(prompts)
        
        # If we have multiple sections with action items, deduplicate
        if len(action_items_results) > 1:
            combined = "\n\n".join(action_items_results)
            
            dedup_prompt = f"""
            Here are action items extracted from different sections of a transcript.
            Create a final, consolidated list that:
            1. Removes duplicates
            2. Organizes by responsible person or priority
            3. Merges related items
            
            Format with markdown headings and bullet points.
            
            EXTRACTED ITEMS:
            {combined}
            """
            
            return await self.llm_client.generate_completion_async(dedup_prompt)
        elif action_items_results:
            return action_items_results[0]
        else:
            return "No action items identified."
    
    def _extract_speakers(self, text: str) -> List[str]:
        """
        Extract speaker names from a Teams transcript.
        
        Args:
            text: Transcript text
            
        Returns:
            List of speaker names
        """
        speakers = set()
        
        # Look for Teams speaker patterns
        pattern = r"\d{1,2}:\d{2}:\d{2}\s+[AP]M\s+From\s+([^:]+):"
        matches = re.findall(pattern, text)
        
        for match in matches:
            speakers.add(match.strip())
        
        return list(speakers)