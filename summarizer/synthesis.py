"""
Synthesis module for combining section summaries into a cohesive final summary.
"""

import logging
from typing import List, Dict, Any, Optional

from summarizer.llm.async_openai_adapter import AsyncOpenAIAdapter

logger = logging.getLogger(__name__)

class SummaryProcessor:
    """Processor for creating final summaries from individual section summaries."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.2):
        """
        Initialize the summary processor.
        
        Args:
            model: Model name to use
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        self.llm_client = AsyncOpenAIAdapter(model=model, temperature=temperature)
    
    async def synthesize_summaries(self, 
                                  summaries: List[str], 
                                  detail_level: str = "standard",
                                  doc_type: str = "transcript",
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a cohesive summary from multiple section summaries.
        
        Args:
            summaries: List of individual section summaries
            detail_level: Level of detail for the synthesis ('brief', 'standard', 'detailed')
            doc_type: Type of document ('transcript', 'article', 'report', etc.)
            metadata: Optional metadata about the document
            
        Returns:
            Synthesized summary
        """
        # Combine section summaries with clear separation
        combined = "\n\n===== SECTION SEPARATOR =====\n\n".join([
            f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(summaries)
        ])
        
        # Build a prompt based on detail level and document type
        prompt = self._create_synthesis_prompt(combined, detail_level, doc_type, metadata)
        
        # Generate the synthesis
        synthesis = await self.llm_client.generate_completion_async(prompt)
        return synthesis
    
    def _create_synthesis_prompt(self, 
                                combined_summaries: str, 
                                detail_level: str,
                                doc_type: str,
                                metadata: Optional[Dict[str, Any]]) -> str:
        """
        Create a prompt for synthesis based on detail level and document type.
        
        Args:
            combined_summaries: Combined text of all section summaries
            detail_level: Level of detail ('brief', 'standard', 'detailed')
            doc_type: Type of document
            metadata: Optional metadata
            
        Returns:
            Prompt for the language model
        """
        # Basic instructions for all summaries
        basic_instructions = f"""
        Create a coherent, well-organized summary from these section summaries.
        
        The final summary should:
        - Be logically organized by topic rather than by section
        - Remove redundancies while preserving important details
        - Use clear headings and structure for readability
        """
        
        # Detail level specific instructions
        detail_instructions = {
            "brief": """
            Focus on creating a concise overview with only the most important points.
            Keep the summary short, highlighting only key decisions, main topics, and critical information.
            Aim for brevity while still capturing the essential content.
            Use bullet points where appropriate for clarity and brevity.
            """,
            
            "standard": """
            Create a balanced summary with important details and supporting information.
            Include key points, decisions, and noteworthy discussions.
            Maintain a moderate level of detail that gives good understanding without excessive length.
            Use a mix of paragraphs and bullet points for readability.
            """,
            
            "detailed": """
            Create a comprehensive summary that captures all significant content.
            Include important details, context, supporting information, and nuances of discussions.
            Preserve specific examples, figures, and evidence that support main points.
            Organize with clear headings and subheadings to maintain readability despite the detail level.
            """
        }
        
        # Document type specific instructions
        doc_type_instructions = {
            "transcript": """
            For this meeting transcript:
            - Preserve key speaker attributions for important points
            - Highlight decisions, action items, and follow-ups
            - Note areas of agreement and any unresolved questions
            - Start with a brief overview of the meeting's purpose and participants
            """,
            
            "article": """
            For this article:
            - Maintain the author's key arguments and conclusions
            - Preserve important evidence, examples, and data points
            - Note any counterarguments or limitations discussed
            - Start with a brief overview of the article's main thesis
            """,
            
            "report": """
            For this report:
            - Present key findings, conclusions, and recommendations
            - Include important data points, metrics, and trends
            - Preserve methodological details when relevant
            - Start with an executive summary of the main findings
            """
        }
        
        # Set default document type if not recognized
        if doc_type not in doc_type_instructions:
            doc_type = "transcript"  # default
        
        # Set default detail level if not recognized
        if detail_level not in detail_instructions:
            detail_level = "standard"  # default
        
        # Add metadata context if provided
        metadata_context = ""
        if metadata:
            metadata_context = "Additional context:\n"
            
            if "is_teams_transcript" in metadata:
                metadata_context += "- This is a Microsoft Teams meeting transcript\n"
            
            if "speakers" in metadata:
                speakers = metadata["speakers"]
                metadata_context += f"- Participants: {', '.join(speakers)}\n"
        
        # Combine all instructions
        prompt = f"""
        {basic_instructions}
        
        {detail_instructions[detail_level]}
        
        {doc_type_instructions[doc_type]}
        
        {metadata_context}
        
        FORMAT WITH MARKDOWN:
        # Meeting Summary

        ## Key Points

        ## Discussion Topics
        
        ## Conclusions & Next Steps
        
        SECTION SUMMARIES:
        {combined_summaries}
        """
        
        return prompt
    
    def synthesize_summaries_sync(self, 
                                summaries: List[str], 
                                detail_level: str = "standard",
                                doc_type: str = "transcript",
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Synchronous version of synthesize_summaries.
        
        Args:
            summaries: List of individual section summaries
            detail_level: Level of detail for the synthesis ('brief', 'standard', 'detailed')
            doc_type: Type of document ('transcript', 'article', 'report', etc.)
            metadata: Optional metadata about the document
            
        Returns:
            Synthesized summary
        """
        # Combine section summaries with clear separation
        combined = "\n\n===== SECTION SEPARATOR =====\n\n".join([
            f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(summaries)
        ])
        
        # Build a prompt based on detail level and document type
        prompt = self._create_synthesis_prompt(combined, detail_level, doc_type, metadata)
        
        # Generate the synthesis
        synthesis = self.llm_client.generate_completion(prompt)
        return synthesis