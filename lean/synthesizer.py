"""
Lean synthesizer module that creates organized, categorized notes from document chunks.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class Synthesizer:
    """Converts chunk summaries into organized, bulleted notes with logical categorization."""
    
    def __init__(self, llm_client):
        """
        Initialize the synthesizer.
        
        Args:
            llm_client: LLM client for generating organized notes
        """
        self.llm_client = llm_client
    
    async def synthesize_summaries(self, 
                                 chunk_results: List[Dict[str, Any]],
                                 detail_level: str = "detailed",
                                 document_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create organized, categorized notes from chunk summaries.
        
        Args:
            chunk_results: List of chunk summary results
            detail_level: Level of detail ('essential', 'detailed', or 'detailed-complex')
            document_context: Optional document context
            
        Returns:
            Organized notes in markdown format
        """
        # Sort chunk results by index
        sorted_results = sorted(chunk_results, key=lambda x: x.get('chunk_index', 0))
        
        # Extract key topics and metadata
        topics, speakers, action_items = self._extract_key_elements(sorted_results)
        
        # Prepare the input for summarization
        combined_text = self._prepare_synthesis_input(sorted_results)
        
        logger.info(f"Synthesizing with detail level: {detail_level}, topics: {', '.join(topics[:5])}")
        
        # Generate organized notes based on detail level
        if detail_level == "essential":
            return await self._generate_essential_notes(
                combined_text, topics, speakers, action_items, document_context
            )
        elif detail_level == "detailed-complex":
            return await self._generate_detailed_notes(
                combined_text, topics, speakers, action_items, document_context,
                is_complex=True
            )
        else:  # "detailed" (default)
            return await self._generate_detailed_notes(
                combined_text, topics, speakers, action_items, document_context,
                is_complex=False
            )
    
    def _extract_key_elements(self, chunk_results: List[Dict[str, Any]]) -> tuple:
        """
        Extract key topics, speakers, and action items from chunk results.
        
        Args:
            chunk_results: List of chunk summary results
            
        Returns:
            Tuple of (topics, speakers, action_items)
        """
        # Extract and count all keywords
        all_keywords = []
        for chunk in chunk_results:
            keywords = chunk.get('keywords', [])
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
        
        # Get top keywords by frequency
        keyword_counter = Counter(all_keywords)
        topics = [keyword for keyword, _ in keyword_counter.most_common(15)]
        
        # Extract all speakers
        all_speakers = []
        for chunk in chunk_results:
            speakers = chunk.get('speakers', [])
            if isinstance(speakers, list):
                all_speakers.extend(speakers)
        
        # Get unique speakers
        speakers = list(set(all_speakers))
        
        # Extract all action items
        all_action_items = []
        for chunk in chunk_results:
            action_items = chunk.get('action_items', [])
            if isinstance(action_items, list):
                all_action_items.extend(action_items)
            elif isinstance(action_items, str) and action_items.strip():
                # Try to split by lines or bullets
                import re
                items = re.findall(r'[-*•]?\s*([^•\n-][^\n]+)', action_items)
                if items:
                    all_action_items.extend(items)
                else:
                    all_action_items.append(action_items)
        
        # Simple deduplication for action items
        unique_action_items = []
        for item in all_action_items:
            if item and item not in unique_action_items:
                unique_action_items.append(item)
        
        return topics, speakers, unique_action_items
    
    def _prepare_synthesis_input(self, chunk_results: List[Dict[str, Any]]) -> str:
        """
        Prepare the input text for synthesis from chunk results.
        
        Args:
            chunk_results: List of chunk summary results
            
        Returns:
            Combined text ready for synthesis
        """
        combined_text = ""
        
        for i, chunk in enumerate(chunk_results):
            summary = chunk.get('summary', '')
            position = chunk.get('position', 'unknown')
            
            # Add formatted chunk text
            combined_text += f"SECTION {i+1} ({position.upper()}):\n{summary}\n\n"
        
        return combined_text
    
    async def _generate_essential_notes(self, 
                                      combined_text: str,
                                      topics: List[str],
                                      speakers: List[str],
                                      action_items: List[str],
                                      document_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate concise, organized notes for essential detail level.
        
        Args:
            combined_text: Combined input text
            topics: List of key topics
            speakers: List of speakers
            action_items: List of action items
            document_context: Optional document context
            
        Returns:
            Organized notes in markdown format
        """
        # Get document type
        is_transcript = document_context.get('is_meeting_transcript', False) if document_context else False
        meeting_purpose = document_context.get('meeting_purpose', '') if document_context else ''
        
        # Get user instructions if available
        user_instructions = ""
        if document_context and document_context.get('user_instructions'):
            user_instructions = f"""
            USER INSTRUCTIONS:
            {document_context['user_instructions']}
            
            Please incorporate these instructions when creating the notes.
            """
        
        # Create format guidance based on document type
        format_guidance = """
        # Document Notes
        
        ## Main Points
        - [Key points organized as bulleted lists]
        
        ## Additional Information
        - [Any important context or background information]
        
        ## Next Steps
        - [Action items and next steps]
        """
        
        if is_transcript:
            format_guidance = """
            # Meeting Notes
            
            ## Discussion Topics
            - [Key discussion topics organized as bulleted lists]
            
            ## Decisions Made
            - [Major decisions reached in the meeting]
            
            ## Action Items
            - [Tasks, assignments, and follow-ups]
            
            ## Participants' Contributions
            - [Notable points made by specific participants]
            """
        
        # Create the synthesis prompt
        prompt = f"""
        Create organized, bulleted notes from the following document summaries.
        
        IMPORTANT INSTRUCTIONS:
        1. Organize the notes into logical categories based on the content
        2. Use bullet points for all information (NOT numbered lists or paragraphs)
        3. Group related points together under appropriate headings
        4. Keep the notes concise and focused on key information
        5. Do not include an executive summary
        6. Use the document's own terminology and language
        7. Include specific details, numbers, and examples where important
        
        {user_instructions}
        
        DOCUMENT TYPE: {'Meeting Transcript' if is_transcript else 'Document'}
        {f'MEETING PURPOSE: {meeting_purpose}' if meeting_purpose else ''}
        
        KEY TOPICS: {', '.join(topics[:10])}
        
        SPEAKERS: {', '.join(speakers) if speakers else 'None specified'}
        
        SUGGESTED FORMAT:
        {format_guidance}
        
        DOCUMENT CONTENT:
        {combined_text}
        """
        
        # Generate the organized notes
        return await self.llm_client.generate_completion_async(prompt)
    
    async def _generate_detailed_notes(self, 
                                     combined_text: str,
                                     topics: List[str],
                                     speakers: List[str],
                                     action_items: List[str],
                                     document_context: Optional[Dict[str, Any]] = None,
                                     is_complex: bool = False) -> str:
        """
        Generate detailed, organized notes.
        
        Args:
            combined_text: Combined input text
            topics: List of key topics
            speakers: List of speakers
            action_items: List of action items
            document_context: Optional document context
            is_complex: Whether to generate more complex notes
            
        Returns:
            Organized notes in markdown format
        """
        # Get document type
        is_transcript = document_context.get('is_meeting_transcript', False) if document_context else False
        meeting_purpose = document_context.get('meeting_purpose', '') if document_context else ''
        client_name = document_context.get('client_name', '') if document_context else ''
        
        # Get user instructions if available
        user_instructions = ""
        if document_context and document_context.get('user_instructions'):
            user_instructions = f"""
            USER INSTRUCTIONS:
            {document_context['user_instructions']}
            
            Please incorporate these instructions when creating the notes.
            """
        
        # Adjust detail level based on is_complex flag
        detail_instruction = """
        Create moderately detailed, well-organized notes with:
        - Key points and important details
        - 3-5 categories of information
        - Specific examples and context where relevant
        """
        
        if is_complex:
            detail_instruction = """
        Create comprehensive, well-structured notes with:
        - Detailed points preserving important nuances
        - 5-8 categories of information organized by theme
        - Specific examples, data points, and context
        - Greater detail for high-priority topics
        """
        
        # Create format guidance based on document type
        format_guidance = """
        # Document Notes
        
        ## [Category 1: Derived from content]
        - [Related points as bullets]
        - [More points...]
        
        ## [Category 2: Derived from content]
        - [Related points as bullets]
        - [More points...]
        
        ## [Additional categories as needed]
        - [Related points...]
        
        ## Key Takeaways
        - [Important conclusions or insights]
        
        ## Next Steps
        - [Action items and follow-up tasks]
        """
        
        if is_transcript:
            format_guidance = """
            # Meeting Notes
            
            ## [Topic 1: Derived from discussion]
            - [Points discussed under this topic]
            - [Specific details and context]
            
            ## [Topic 2: Derived from discussion]
            - [Points discussed under this topic]
            - [Specific details and context]
            
            ## [Additional topics as needed]
            - [Related points...]
            
            ## Decisions
            - [Decisions made during the meeting]
            
            ## Action Items
            - [Person responsible]: [Task description] [Deadline if specified]
            
            ## Discussion Highlights
            - [Speaker name]: [Notable quote or contribution]
            - [Another speaker]: [Notable quote or contribution]
            """
        
        # Create the synthesis prompt
        prompt = f"""
        {detail_instruction}
        
        IMPORTANT INSTRUCTIONS:
        1. Create a structured set of notes organized by logical categories derived from the content
        2. Use bullet points for all information (NOT numbered lists or paragraphs)
        3. Derive appropriate section headings from the content itself
        4. Do not include an executive summary
        5. Include specific details, numbers, quotes, and examples
        6. For action items, include owner/responsible party when mentioned
        7. Maintain the specific terminology and language used in the document
        8. Each bullet point should be a complete, meaningful piece of information
        
        {user_instructions}
        
        DOCUMENT TYPE: {'Meeting Transcript' if is_transcript else 'Document'}
        {f'MEETING PURPOSE: {meeting_purpose}' if meeting_purpose else ''}
        {f'CLIENT: {client_name}' if client_name else ''}
        
        KEY TOPICS: {', '.join(topics)}
        
        SPEAKERS: {', '.join(speakers) if speakers else 'None specified'}
        
        SUGGESTED FORMAT STRUCTURE:
        {format_guidance}
        
        DOCUMENT CONTENT:
        {combined_text}
        """
        
        # Generate the organized notes
        return await self.llm_client.generate_completion_async(prompt)
    
    async def extract_topics(self, chunk_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key topics from chunk results.
        
        Args:
            chunk_results: List of chunk summary results
            
        Returns:
            List of key topics
        """
        # Extract all keywords
        all_keywords = []
        for chunk in chunk_results:
            keywords = chunk.get('keywords', [])
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
        
        # Count and return top keywords
        keyword_counter = Counter(all_keywords)
        return [keyword for keyword, _ in keyword_counter.most_common(10)]