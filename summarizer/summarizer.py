"""
Enhanced document summarization engine with metadata extraction.
"""

import re
import time
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Callable

from summarizer.options import SummaryOptions
from summarizer.llm.async_openai_adapter import AsyncOpenAIAdapter
from summarizer.synthesis import SummaryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncTranscriptSummarizer:
    """Transcript summarization engine with metadata extraction."""
    
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
        
        # Process divisions to generate summaries with metadata
        if progress_callback:
            progress_callback(f"Processing {len(divisions)} sections concurrently...", 0.3)
        
        # Generate prompts for each division
        division_data = []
        for i, division in enumerate(divisions):
            position = self._get_section_position(i, len(divisions))
            
            # For each division, create a data structure with section info
            division_data.append({
                'text': division['text'],
                'position': position,
                'index': i,
                'total': len(divisions)
            })
        
        # Process divisions concurrently
        max_concurrent = min(5, len(division_data))
        division_results = await self._process_divisions_with_metadata(division_data, max_concurrent)
        
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
            division_results,
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
        
        # Format result with original summaries and enhanced metadata
        division_summaries = [result["summary"] for result in division_results]
        
        return {
            "summary": final_summary,
            "action_items": action_items,
            "divisions": divisions,
            "division_summaries": division_summaries,
            "division_metadata": division_results,  # Include the full metadata
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
    
    async def _process_divisions_with_metadata(self, division_data: List[Dict[str, Any]], max_concurrent: int) -> List[Dict[str, Any]]:
        """
        Process divisions with enhanced metadata extraction.
        
        Args:
            division_data: List of division data dictionaries
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of processed division data with summaries and metadata
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_division(data):
            async with semaphore:
                return await self._summarize_division_with_metadata(data)
        
        # Create tasks for all divisions
        tasks = [process_division(data) for data in division_data]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing division {i}: {result}")
                # Include a basic fallback result for failed divisions
                processed_results.append({
                    "summary": f"Error processing section {i+1}: {str(result)}",
                    "index": i,
                    "keywords": [],
                    "importance": 1,
                    "speakers": [],
                    "position": division_data[i].get('position', 'middle')
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _summarize_division_with_metadata(self, division_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced summary with metadata for a division.
        
        Args:
            division_data: Division data
            
        Returns:
            Dictionary with summary and extracted metadata
        """
        text = division_data['text']
        index = division_data['index']
        total = division_data['total']
        position = division_data['position']
        
        # Create prompt for metadata extraction and summary
        prompt = f"""
        Summarize this transcript section and extract key metadata.

        Return the result in JSON format with these fields:
        - summary: A detailed summary of the section preserving key points and speaker attributions
        - keywords: A list of key topics/terms (5-10) that best represent this section
        - importance: A score from 1-5 indicating how important this section is (5 being highest)
        - speakers: List of speakers in this section (if identifiable)
        - next_steps: Any action items or next steps mentioned in this section
        
        Position context: This is the {position} section of the transcript ({index+1}/{total}).
        
        TRANSCRIPT SECTION:
        {text}
        """
        
        try:
            # Get the enhanced summary with metadata
            result = await self.llm_client.generate_completion_async(prompt)
            
            # Parse the JSON response
            try:
                parsed_result = json.loads(result)
                # Add original index for ordering
                parsed_result["index"] = index
                parsed_result["position"] = position
                return parsed_result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON from section {index+1}. Using fallback extraction.")
                # Extract what we can from the text result
                return self._fallback_metadata_extraction(result, index, position)
                
        except Exception as e:
            logger.error(f"Error in division {index+1} processing: {e}")
            raise
    
    def _fallback_metadata_extraction(self, text_result: str, index: int, position: str) -> Dict[str, Any]:
        """
        Extract metadata from text when JSON parsing fails.
        
        Args:
            text_result: The text result from the LLM
            index: Section index
            position: Section position
            
        Returns:
            Dictionary with summary and extracted metadata
        """
        # Extract summary - everything up to keywords or the whole text
        summary = text_result
        if "keywords:" in text_result.lower():
            summary = text_result.split("keywords:", 1)[0].strip()
        
        # Try to extract keywords
        keywords = []
        keywords_match = re.search(r"keywords:(.+?)(?:importance:|speakers:|next_steps:|$)", 
                                  text_result.lower(), re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # Try to parse as list
            keywords = [k.strip() for k in re.split(r'[,\n•\-]+', keywords_text) if k.strip()]
        
        # Extract importance
        importance = 3  # Default mid-importance
        importance_match = re.search(r"importance:\s*(\d)", text_result.lower())
        if importance_match:
            try:
                importance = int(importance_match.group(1))
            except ValueError:
                pass
        
        # Extract speakers
        speakers = []
        speakers_match = re.search(r"speakers:(.+?)(?:next_steps:|$)", 
                                  text_result.lower(), re.DOTALL)
        if speakers_match:
            speakers_text = speakers_match.group(1).strip()
            speakers = [s.strip() for s in re.split(r'[,\n•\-]+', speakers_text) if s.strip()]
        
        # Extract next steps
        next_steps = ""
        next_steps_match = re.search(r"next_steps:(.+?)$", text_result.lower(), re.DOTALL)
        if next_steps_match:
            next_steps = next_steps_match.group(1).strip()
        
        return {
            "summary": summary,
            "keywords": keywords,
            "importance": importance,
            "speakers": speakers,
            "next_steps": next_steps,
            "index": index,
            "position": position
        }
    
    def _get_section_position(self, index: int, total: int) -> str:
        """
        Determine the position of a section in the document.
        
        Args:
            index: Section index
            total: Total number of sections
            
        Returns:
            Position label ('beginning', 'middle', or 'end')
        """
        if index == 0:
            return "beginning"
        elif index == total - 1:
            return "end"
        elif index < total * 0.25:
            return "early"
        elif index > total * 0.75:
            return "late"
        else:
            return "middle"
    
    def _is_teams_transcript(self, text: str) -> bool:
        """Detect if this is a Microsoft Teams transcript."""
        teams_patterns = [
            r"\d{1,2}:\d{2}:\d{2}\s+[AP]M\s+From\s+.+:",  # Timestamp pattern
            r"Meeting\s+started:",                        # Meeting start marker
            r"Meeting\s+ended:",                         # Meeting end marker
            r"[\w\s]+\s+\(\d{1,2}:\d{2}:\d{2}\s+[AP]M\)"  # Speaker with timestamp
        ]
        
        sample = text[:5000]
        
        for pattern in teams_patterns:
            if re.search(pattern, sample):
                logger.info("Detected Microsoft Teams transcript format")
                return True
        
        return False
    
    def _divide_teams_transcript(self, text: str) -> List[Dict[str, Any]]:
        """Divide a Teams transcript into speaker-based sections."""
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
        """Divide a generic transcript into manageable sections."""
        return self._divide_by_size(text)
    
    def _divide_by_size(self, text: str) -> List[Dict[str, Any]]:
        """Divide text into sections based on size constraints."""
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
            target_end = min(current_pos + target_size, len(text))
            
            # Try to find a good break point
            if target_end < len(text):
                paragraph_break = text.rfind('\n\n', current_pos, target_end + 500)
                
                if paragraph_break != -1 and paragraph_break <= target_end + 500:
                    end_pos = paragraph_break + 2
                else:
                    # Look for sentence breaks
                    sentence_break = max(
                        text.rfind('. ', current_pos, target_end + 200),
                        text.rfind('! ', current_pos, target_end + 200),
                        text.rfind('? ', current_pos, target_end + 200)
                    )
                    
                    if sentence_break != -1 and sentence_break <= target_end + 200:
                        end_pos = sentence_break + 2
                    else:
                        end_pos = target_end
            else:
                end_pos = target_end
            
            # Create the division
            division_text = text[current_pos:end_pos]
            if division_text.strip():
                divisions.append({
                    'start': current_pos,
                    'end': end_pos,
                    'text': division_text
                })
            
            current_pos = end_pos
        
        logger.info(f"Created {len(divisions)} size-based divisions")
        return divisions
    
    async def _extract_action_items_async(self, text: str, divisions: List[Dict[str, Any]]) -> str:
        """Extract action items from the transcript asynchronously."""
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
        
        # For longer transcripts, focus on key sections
        indices = [0]  # Always include first division
        
        if len(divisions) > 1:
            indices.append(len(divisions) - 1)  # Include last division
        
        if len(divisions) > 3:
            indices.append(len(divisions) // 2)  # Include middle division
        
        # Get action items concurrently
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
        
        # Deduplicate results if needed
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
        """Extract speaker names from a Teams transcript."""
        speakers = set()
        
        # Look for Teams speaker patterns
        pattern = r"\d{1,2}:\d{2}:\d{2}\s+[AP]M\s+From\s+([^:]+):"
        matches = re.findall(pattern, text)
        
        for match in matches:
            speakers.add(match.strip())
        
        return list(speakers)