"""
Enhanced core document summarization engine with tiered processing for long documents.
"""

import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
import os

from .options import SummaryOptions
from .llm.base import LLMAdapter
from .llm.openai_adapter import OpenAIAdapter
from .division import divide_document, extract_speakers

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
    Enhanced document summarization engine with tiered processing for long documents.
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
    
    def summarize(self, text: str) -> Dict[str, Any]:
        """
        Summarize a document using the selected division strategy with tiered processing.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary with summary and metadata
        """
        start_time = time.time()
        
        # For very large documents, use tiered processing
        is_large_document = len(text) > 50000  # Approximately 12.5k tokens
        use_tiered_processing = is_large_document and "gpt-4" in self.options.model_name
        
        # Divide the document using the selected strategy
        divisions = divide_document(
            text=text,
            strategy=self.options.division_strategy,
            min_sections=self.options.min_sections,
            target_tokens_per_section=self.options.target_tokens_per_section,
            section_overlap=self.options.section_overlap
        )
        
        if self.options.verbose:
            logger.info(f"Divided document into {len(divisions)} sections using {self.options.division_strategy} strategy")
        
        # Process each division
        division_summaries = []
        
        if use_tiered_processing:
            logger.info("Using tiered processing for large document")
            division_summaries = self._process_with_tiered_approach(divisions)
        else:
            # Standard approach for smaller documents
            for i, division in enumerate(divisions):
                try:
                    logger.info(f"Processing division {i+1}/{len(divisions)}")
                    summary = self._summarize_division(division, i, len(divisions))
                    division_summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error summarizing division {i+1}: {e}")
                    # Add placeholder for failed division
                    division_summaries.append(f"Error summarizing division {i+1}: {str(e)}")
        
        # Synthesize the final summary
        final_summary = self._synthesize_summary(division_summaries)
        
        # Extract action items if requested
        action_items = None
        if self.options.include_action_items:
            action_items = self._extract_action_items(text, divisions)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "summary": final_summary,
            "action_items": action_items,
            "divisions": divisions,
            "division_summaries": division_summaries,
            "metadata": {
                "division_count": len(divisions),
                "division_strategy": self.options.division_strategy,
                "model": self.options.model_name,
                "processing_time_seconds": processing_time,
                "tiered_processing": use_tiered_processing
            }
        }
        
        return result
    
    def _process_with_tiered_approach(self, divisions: List[Dict[str, Any]]) -> List[str]:
        """
        Process divisions using a tiered approach with different models for efficiency.
        
        Args:
            divisions: List of division dictionaries
            
        Returns:
            List of division summaries
        """
        division_summaries = []
        
        # Store original model
        original_model = self.llm_client.model
        
        try:
            # Use faster model for initial summaries if using a powerful model
            if "gpt-4" in original_model:
                # Switch to faster model for initial summarization
                logger.info(f"Switching to gpt-3.5-turbo-16k for initial summaries (from {original_model})")
                self.llm_client.model = "gpt-3.5-turbo-16k"
            
            # First pass: Process each division with simpler prompts and faster model
            for i, division in enumerate(divisions):
                try:
                    logger.info(f"Tier 1: Processing division {i+1}/{len(divisions)}")
                    # Use a simpler prompt focused on factual extraction
                    strategy = division.get('strategy', 'basic')
                    
                    prompt = f"""Extract the key information from this section ({i+1}/{len(divisions)}) of a document.
                    
                    Focus ONLY on:
                    1. Main facts, points, and arguments
                    2. Important details and data
                    3. Key decisions or conclusions
                    
                    Be thorough but concise. Prioritize accuracy and completeness.
                    
                    SECTION TEXT:
                    {division['text']}
                    """
                    
                    # Generate the first-tier summary
                    summary = self.llm_client.generate_completion(prompt)
                    division_summaries.append(summary)
                    
                except Exception as e:
                    logger.error(f"Error in tier 1 summary for division {i+1}: {e}")
                    division_summaries.append(f"Error summarizing division {i+1}: {str(e)}")
            
            # Return to original model for final synthesis (handled by caller)
            self.llm_client.model = original_model
            logger.info(f"Switched back to {original_model} for synthesis")
            
        except Exception as e:
            # Make sure we restore the original model even if there's an error
            self.llm_client.model = original_model
            logger.error(f"Error in tiered processing: {e}")
            raise
            
        return division_summaries
        
    def _summarize_division(self, division: Dict[str, Any], index: int, total: int) -> str:
        """
        Summarize a single division using a strategy-appropriate prompt.
        
        Args:
            division: Division dictionary
            index: Division index
            total: Total number of divisions
            
        Returns:
            Summary text
        """
        strategy = division.get('strategy', 'basic')
        
        # Create a tailored prompt based on division strategy
        if strategy == "speaker":
            prompt = f"""Summarize this section ({index+1}/{total}) of a transcript or conversation.
            
            Focus on:
            1. Preserving who said what (maintain speaker attribution)
            2. Key discussion points, decisions, and action items
            3. Important details and arguments made by each participant
            4. Any agreements, disagreements, or questions raised
            
            Create a comprehensive, well-structured summary that captures the conversation flow
            and all significant content.
            
            SECTION TEXT:
            {division['text']}
            """
        elif strategy == "boundary" or strategy == "context_aware":
            prompt = f"""Summarize this section ({index+1}/{total}) of the document.
            
            Focus on:
            1. Maintaining the document structure and headings
            2. Key points, arguments, and significant details
            3. Preserving the logical flow and narrative
            4. Technical details that are important for understanding
            
            Create a comprehensive, well-structured summary that captures all important content
            while maintaining the document's organization.
            
            SECTION TEXT:
            {division['text']}
            """
        else:  # basic or fallback
            prompt = f"""Summarize this section ({index+1}/{total}) of the document.
            
            Create a detailed, comprehensive summary that:
            1. Captures all significant content
            2. Preserves important details
            3. Maintains the logical flow
            4. Is clear and concise
            
            SECTION TEXT:
            {division['text']}
            """
        
        try:
            return self.llm_client.generate_completion(prompt)
        except Exception as e:
            logger.error(f"Error summarizing division {index+1}: {e}")
            # Return error message as summary
            return f"Error summarizing this section: {str(e)}"
    
    def _synthesize_summary(self, division_summaries: List[str]) -> str:
        """
        Synthesize division summaries into a final summary.
        
        Args:
            division_summaries: List of division summaries
            
        Returns:
            Final synthesized summary
        """
        logger.info("Synthesizing final summary from division summaries")
        
        # Combine the division summaries
        combined_summaries = "\n\n===== SECTION SEPARATOR =====\n\n".join([
            f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(division_summaries)
        ])
        
        # Build synthesis prompt
        synthesis_prompt = f"""Create a cohesive, well-structured summary from these section summaries.
        
        The final summary should:
        1. Start with a concise Executive Summary that captures the main points
        2. Organize content logically by topic instead of by section
        3. Include all significant information and important details
        4. Remove redundancies while preserving completeness
        5. Use clear headings and structure for readability
        
        FORMAT WITH MARKDOWN:
        # Document Summary
        
        ## Executive Summary
        
        ## Key Topics
        
        ### [Topic 1]
        
        ### [Topic 2]
        
        ## Conclusions and Next Steps
        
        SECTION SUMMARIES:
        {combined_summaries}
        """
        
        try:
            result = self.llm_client.generate_completion(synthesis_prompt)
            return result
        except Exception as e:
            logger.error(f"Error synthesizing final summary: {e}")
            
            # Fallback to simple concatenation with a header
            fallback = "# Document Summary\n\n"
            fallback += "## Note\n\nThe following sections could not be synthesized due to an error.\n\n"
            
            for i, summary in enumerate(division_summaries):
                fallback += f"## Section {i+1}\n\n{summary}\n\n"
                
            return fallback
    
    def _extract_action_items(self, text: str, divisions: List[Dict[str, Any]]) -> str:
        """
        Extract action items from the document.
        
        Args:
            text: Complete document text
            divisions: List of division dictionaries
            
        Returns:
            Action items text
        """
        logger.info("Extracting action items")
        
        # For short documents, extract directly
        if len(text) < 6000:
            prompt = """Extract all action items, tasks, and commitments from this document.
            For each action item:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a prioritized, well-organized list with markdown. 
            Don't miss any action items, even if implied.
            
            DOCUMENT:
            """
            
            try:
                return self.llm_client.generate_completion(prompt + text)
            except Exception as e:
                logger.error(f"Error extracting action items: {e}")
                return "Could not extract action items due to an error."
        
        # For longer documents, extract from key divisions
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
        
        # Ensure we don't have duplicates and sort
        indices = sorted(set(indices))
        
        # Extract from selected divisions
        action_items = []
        
        for idx in indices:
            division = divisions[idx]
            prompt = f"""Extract all action items, tasks, and commitments from this document section.
            For each action item:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a bulleted list with markdown. Don't miss any action items, even if implied.
            
            DOCUMENT SECTION {idx+1}/{len(divisions)}:
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
        
        dedup_prompt = f"""Here are action items extracted from different sections of a document.
        Create a final, consolidated list that:
        1. Removes duplicates
        2. Organizes by responsible person or priority
        3. Merges related items
        4. Ensures all unique items are preserved
        
        Format with markdown.
        
        EXTRACTED ITEMS:
        {combined}
        """
        
        try:
            return self.llm_client.generate_completion(dedup_prompt)
        except Exception as e:
            logger.error(f"Error deduplicating action items: {e}")
            return "# Action Items\n\n" + combined