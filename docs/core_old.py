"""
Enhanced core document summarization engine with performance optimizations.
"""

import time
import logging
import os
from typing import List, Dict, Any, Optional, Union, Callable

from ..summarizer.options import SummaryOptions
from ..summarizer.division import assess_document, divide_document, generate_division_prompt, synthesize_summaries
from ..summarizer.llm.base import LLMAdapter
from ..summarizer.llm.async_openai_adapter import OpenAIAdapter

try:
    from ..summarizer.llm.litellm_adapter import LiteLLMAdapter
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory LRU cache (simple implementation)
_CACHE = {}
_CACHE_MAX_SIZE = 10  # Adjust based on memory constraints

class TranscriptSummarizer:
    """Enhanced document summarization engine with performance optimizations."""
    
    def __init__(self, options: Optional[SummaryOptions] = None):
        """Initialize the summarizer with options."""
        self.options = options or SummaryOptions()
        self.llm_client = self._initialize_llm_client()
    
    def _initialize_llm_client(self) -> LLMAdapter:
        """Initialize the appropriate LLM client based on config."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Check if we should try to use LiteLLM adapter (removed use_litellm check)
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

    def summarize(self, text: str) -> Dict[str, Any]:
        """
        Summarize a document using intelligent processing.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary with summary and metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in _CACHE:
            logger.info("Using cached summary result")
            cached_result = _CACHE[cache_key]
            cached_result["metadata"]["from_cache"] = True
            return cached_result
            
        start_time = time.time()
        
        # Check if this is a very large document that would benefit from tiered processing
        is_large_document = len(text) > 100000  # ~25k tokens
        use_tiered_processing = is_large_document and "gpt-4" in self.options.model_name
        
        if use_tiered_processing:
            return self._process_with_tiered_approach(text, start_time)
        
        # Standard approach: Analyze document to determine optimal strategy
        doc_assessment = assess_document(text)
        doc_type = doc_assessment["doc_type"]
        
        # Use the strategy from options or the recommended one if auto-detect
        strategy = self.options.division_strategy
        if not strategy or strategy == "auto":
            strategy = doc_assessment["recommended_strategy"]
            logger.info(f"Using auto-detected strategy: {strategy}")
        
        # Divide document
        divisions = divide_document(
            text=text,
            strategy=strategy,
            min_sections=self.options.min_sections,
            target_tokens_per_section=self.options.target_tokens_per_section
        )
        
        # Process divisions to generate summaries
        division_summaries = self._process_divisions(divisions, doc_type, text)
        
        # Synthesize final summary
        synthesis_prompt = synthesize_summaries(division_summaries, doc_type)
        final_summary = self.llm_client.generate_completion(synthesis_prompt)
        
        # Extract action items if requested
        action_items = None
        if self.options.include_action_items:
            action_items = self._extract_action_items(text, divisions)
        
        # Prepare result
        processing_time = time.time() - start_time
        result = {
            "summary": final_summary,
            "action_items": action_items,
            "divisions": divisions,
            "division_summaries": division_summaries,
            "metadata": {
                "division_count": len(divisions),
                "division_strategy": strategy,
                "document_type": doc_type,
                "model": self.options.model_name,
                "processing_time_seconds": processing_time
            }
        }
        
        # Save to cache (with LRU eviction if needed)
        if len(_CACHE) >= _CACHE_MAX_SIZE:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(_CACHE))
            del _CACHE[oldest_key]
        _CACHE[cache_key] = result
        
        return result
    
    def _process_with_tiered_approach(self, text: str, start_time: float) -> Dict[str, Any]:
        """Process a very large document with tiered approach."""
        logger.info("Using tiered processing for large document")
        
        # Step 1: Use essential division strategy with fewer, larger chunks
        divisions = divide_document(
            text=text,
            strategy="essential",
            min_sections=3,
            target_tokens_per_section=30000
        )
        
        # Step 2: Use a faster model for initial summaries
        original_model = self.llm_client.model
        try:
            # Switch to faster model if using GPT-4
            if "gpt-4" in original_model:
                self.llm_client.model = "gpt-3.5-turbo-16k"
                logger.info(f"Switched to {self.llm_client.model} for initial summaries")
            
            # Process each division with simpler prompts
            division_summaries = []
            for i, division in enumerate(divisions):
                prompt = f"""Summarize this section {i+1}/{len(divisions)} of a large document.
                Focus on extracting all key information, important details, and main points.
                Be thorough but concise.
                
                SECTION:
                {division['text']}
                """
                
                summary = self.llm_client.generate_completion(prompt)
                division_summaries.append(summary)
            
            # Switch back to original model for synthesis
            self.llm_client.model = original_model
            logger.info(f"Switched back to {original_model} for synthesis")
            
            # Generate synthesis
            combined_summaries = "\n\n===== SECTION SEPARATOR =====\n\n".join([
                f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(division_summaries)
            ])
            
            synthesis_prompt = f"""Create a comprehensive, well-structured summary from these section summaries.
            
            The final summary should:
            1. Start with a concise Executive Summary of the main points
            2. Organize content logically by topic instead of by section
            3. Include all significant information and important details
            4. Remove redundancies while preserving completeness
            5. Use clear headings and structure for readability
            
            SECTION SUMMARIES:
            {combined_summaries}
            """
            
            final_summary = self.llm_client.generate_completion(synthesis_prompt)
            
        except Exception as e:
            # Ensure model is restored even if there's an error
            self.llm_client.model = original_model
            logger.error(f"Error in tiered processing: {e}")
            raise
        
        # Calculate processing time and prepare result
        processing_time = time.time() - start_time
        result = {
            "summary": final_summary,
            "divisions": divisions,
            "division_summaries": division_summaries,
            "metadata": {
                "division_count": len(divisions),
                "division_strategy": "tiered",
                "model": self.options.model_name,
                "processing_time_seconds": processing_time,
                "tiered_processing": True
            }
        }
        
        return result
    
    def _process_divisions(self, divisions: List[Dict[str, Any]], doc_type: str, original_text: str) -> List[str]:
        """Process divisions efficiently, possibly in batches for similar-sized divisions."""
        # For a small number of divisions, process them individually
        if len(divisions) <= 3:
            return self._process_divisions_individually(divisions, doc_type)
        
        # For more divisions, use a batch approach if the model supports it
        batch_size = 2  # Default batch size
        
        # GPT-4 models may benefit from even larger batches
        if "gpt-4" in self.llm_client.model:
            batch_size = 3
        
        try:
            return self._process_divisions_batched(divisions, doc_type, batch_size)
        except Exception as e:
            # If batched processing fails, fall back to individual processing
            logger.warning(f"Batched processing failed: {e}. Falling back to individual processing")
            return self._process_divisions_individually(divisions, doc_type)
    
    def _process_divisions_individually(self, divisions: List[Dict[str, Any]], doc_type: str) -> List[str]:
        """Process each division individually."""
        division_summaries = []
        
        for i, division in enumerate(divisions):
            try:
                logger.info(f"Processing division {i+1}/{len(divisions)}")
                
                # Generate a tailored prompt for this division
                prompt = generate_division_prompt(
                    division=division,
                    index=i,
                    total=len(divisions),
                    doc_type=doc_type
                )
                
                summary = self.llm_client.generate_completion(prompt)
                division_summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing division {i+1}: {e}")
                division_summaries.append(f"Error summarizing division {i+1}: {str(e)}")
        
        return division_summaries
    
    def _process_divisions_batched(self, divisions: List[Dict[str, Any]], doc_type: str, batch_size: int) -> List[str]:
        """Process divisions in batches to reduce API calls."""
        all_summaries = []
        
        # Process divisions in batches
        for i in range(0, len(divisions), batch_size):
            batch = divisions[i:i+batch_size]
            batch_indices = list(range(i, min(i+batch_size, len(divisions))))
            
            logger.info(f"Processing batch of {len(batch)} divisions")
            
            # Create a combined prompt
            combined_prompt = "Summarize each of these document sections separately:\n\n"
            
            for j, division in enumerate(batch):
                section_index = batch_indices[j]
                
                # Add section-specific instructions based on position and document type
                if section_index == 0:
                    combined_prompt += f"SECTION {section_index+1}/{len(divisions)} (BEGINNING SECTION):\n"
                elif section_index == len(divisions) - 1:
                    combined_prompt += f"SECTION {section_index+1}/{len(divisions)} (FINAL SECTION):\n"
                else:
                    combined_prompt += f"SECTION {section_index+1}/{len(divisions)}:\n"
                
                combined_prompt += f"{division['text']}\n\n"
                combined_prompt += "===== SECTION SEPARATOR =====\n\n"
            
            # Add instructions for formatting response
            combined_prompt += f"""For each section, provide a summary that captures all important information appropriate to a {doc_type} document.
            
            Format your response with clear section headings like:
            
            ## SUMMARY FOR SECTION 1
            [Summary content]
            
            ## SUMMARY FOR SECTION 2
            [Summary content]
            
            And so on for each section."""
            
            # Get combined summary
            combined_result = self.llm_client.generate_completion(combined_prompt)
            
            # Parse out individual summaries
            section_markers = [f"## SUMMARY FOR SECTION {idx+1}" for idx in batch_indices]
            if not any(marker in combined_result for marker in section_markers):
                # Try alternative markers
                section_markers = [f"SECTION {idx+1}" for idx in batch_indices]
            
            section_texts = []
            
            for j, marker in enumerate(section_markers):
                start_idx = combined_result.find(marker)
                
                if start_idx != -1:
                    if j < len(section_markers) - 1:
                        next_marker = section_markers[j + 1]
                        end_idx = combined_result.find(next_marker, start_idx)
                        if end_idx != -1:
                            section_texts.append(combined_result[start_idx:end_idx].strip())
                        else:
                            section_texts.append(combined_result[start_idx:].strip())
                    else:
                        section_texts.append(combined_result[start_idx:].strip())
            
            # Fall back if parsing fails
            if len(section_texts) != len(batch):
                logger.warning(f"Failed to parse combined result, falling back to individual processing for this batch")
                
                for division in batch:
                    batch_index = divisions.index(division)
                    prompt = generate_division_prompt(
                        division=division,
                        index=batch_index,
                        total=len(divisions),
                        doc_type=doc_type
                    )
                    summary = self.llm_client.generate_completion(prompt)
                    section_texts.append(summary)
            
            all_summaries.extend(section_texts)
        
        return all_summaries
    
    def _extract_action_items(self, text: str, divisions: List[Dict[str, Any]]) -> str:
        """Extract action items efficiently based on document size."""
        logger.info("Extracting action items")
        
        # For short documents, extract directly
        if len(text) < 8000:
            prompt = """Extract all action items, tasks, and commitments from this document.
            For each action item, include:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a prioritized, well-organized list with markdown.
            
            DOCUMENT:
            """
            
            return self.llm_client.generate_completion(prompt + text)
        
        # For longer documents, extract from key sections
        # Determine which sections to check (beginning, middle, and end)
        if not divisions:
            return "Could not extract action items: no document divisions available."
        
        # Initialize indices - always include first and last division 
        indices = [0]
        if len(divisions) > 1:
            indices.append(len(divisions) - 1)
        
        # Add a middle division if we have at least 3
        if len(divisions) >= 3:
            indices.append(len(divisions) // 2)
        
        # Ensure indices are valid and sort
        indices = sorted(set([i for i in indices if 0 <= i < len(divisions)]))
        
        # Extract from selected divisions (in parallel if possible)
        action_items = []
        for idx in indices:
            division = divisions[idx]
            prompt = f"""Extract all action items, tasks, and commitments from this document section.
            For each action item, include:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a markdown list. Don't include anything that's not an action item.
            
            DOCUMENT SECTION {idx+1}/{len(divisions)}:
            {division['text']}
            """
            
            result = self.llm_client.generate_completion(prompt)
            action_items.append(result)
        
        # If we have multiple sections with action items, deduplicate
        if len(action_items) > 1:
            combined = "\n\n".join(action_items)
            
            dedup_prompt = f"""Here are action items extracted from different sections of a document.
            Create a final, consolidated list that:
            1. Removes duplicates
            2. Organizes by responsible person or priority
            3. Merges related items
            
            Format with markdown headings and bullet points.
            
            EXTRACTED ITEMS:
            {combined}
            """
            
            return self.llm_client.generate_completion(dedup_prompt)
        elif action_items:
            return action_items[0]
        else:
            return "No action items identified."
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a document."""
        import hashlib
        
        # Create a hash from text sample, options, and model
        hash_input = (
            f"{self.options.division_strategy}_{self.options.model_name}_"
            f"{self.options.min_sections}_{len(text)}_"
            f"{text[:500]}_{text[-500:]}"  # Use beginning and end of text
        )
        
        return hashlib.md5(hash_input.encode()).hexdigest()

    def process_divisions(self, divisions: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """
        Process pre-divided sections (for comparison mode).
        
        Args:
            divisions: List of division dictionaries
            text: Original document text
            
        Returns:
            Dictionary with summary and metadata
        """
        start_time = time.time()
        
        # Assess document to determine type
        doc_assessment = assess_document(text)
        doc_type = doc_assessment["doc_type"]
        
        # Process each division individually
        division_summaries = self._process_divisions_individually(divisions, doc_type)
        
        # Generate synthesis prompt
        synthesis_prompt = synthesize_summaries(division_summaries, doc_type)
        
        # Generate the final summary
        final_summary = self.llm_client.generate_completion(synthesis_prompt)
        
        # Extract action items if requested
        action_items = None
        if self.options.include_action_items:
            action_items = self._extract_action_items(text, divisions)
        
        # Prepare result
        processing_time = time.time() - start_time
        result = {
            "summary": final_summary,
            "action_items": action_items,
            "divisions": divisions,
            "division_summaries": division_summaries,
            "metadata": {
                "division_count": len(divisions),
                "division_strategy": self.options.division_strategy,
                "document_type": doc_type,
                "model": self.options.model_name,
                "processing_time_seconds": processing_time
            }
        }
        
        return result