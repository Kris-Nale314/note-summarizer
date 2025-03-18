"""
Enhanced synthesis module for combining section summaries using contextual awareness.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from collections import Counter

from summarizer.llm.async_openai_adapter import AsyncOpenAIAdapter

logger = logging.getLogger(__name__)

class SummaryProcessor:
    """Enhanced processor for creating final summaries from individual section summaries."""
    
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
                                  division_results: List[Dict[str, Any]], 
                                  detail_level: str = "standard",
                                  doc_type: str = "transcript",
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a cohesive summary from multiple section summaries with contextual awareness.
        
        Args:
            division_results: List of division results with summaries and metadata
            detail_level: Level of detail for the synthesis ('brief', 'standard', 'detailed')
            doc_type: Type of document ('transcript', 'article', 'report', etc.)
            metadata: Optional metadata about the document
            
        Returns:
            Synthesized summary
        """
        # Sort the division results by index to ensure proper order
        sorted_results = sorted(division_results, key=lambda x: x.get('index', 0))
        
        # Extract just the summary texts for compatibility with older code
        summaries = [result.get('summary', '') for result in sorted_results]
        
        # First pass: Analyze the first section for context
        if sorted_results and len(sorted_results) > 0:
            first_section = sorted_results[0]
            context_analysis = await self._analyze_first_section(first_section)
            
            # Extract key topics across all sections
            key_topics = self._extract_key_topics(sorted_results)
            
            # Generate the synthesis with context
            return await self._generate_contextual_synthesis(
                sorted_results, 
                context_analysis, 
                key_topics, 
                detail_level, 
                doc_type, 
                metadata
            )
        else:
            # Fallback to standard synthesis if no results
            return await self._standard_synthesis(summaries, detail_level, doc_type, metadata)
    
    async def _analyze_first_section(self, first_section: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the first section to extract meeting context.
        
        Args:
            first_section: First section data with summary and metadata
            
        Returns:
            Dictionary with contextual information
        """
        # We already have keywords, let's extract more specific context
        first_summary = first_section.get('summary', '')
        
        if not first_summary:
            return {'purpose': 'Unknown', 'key_topics': []}
        
        context_prompt = f"""
        Analyze this first section of a meeting transcript summary to identify:
        1. The main purpose of the meeting
        2. Any specific goals or decisions that need to be made
        3. Expected outcomes from the meeting
        
        Return your analysis in JSON format with these fields:
        - purpose: A concise statement of the meeting's purpose
        - goals: A list of specific goals mentioned
        - expected_outcomes: Expected results or deliverables
        - priority_topics: Topics that should be emphasized in the summary
        
        FIRST SECTION SUMMARY:
        {first_summary}
        """
        
        try:
            # Get context analysis
            response = await self.llm_client.generate_completion_async(context_prompt)
            
            # Parse JSON response
            try:
                context = json.loads(response)
                return context
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse context analysis JSON. Using basic extraction.")
                return {
                    'purpose': 'Discussion meeting',
                    'goals': ['Share information', 'Make decisions'],
                    'expected_outcomes': ['Action items', 'Shared understanding'],
                    'priority_topics': first_section.get('keywords', [])
                }
        except Exception as e:
            logger.error(f"Error in first section analysis: {e}")
            # Provide a basic fallback
            return {
                'purpose': 'Meeting',
                'priority_topics': first_section.get('keywords', [])
            }
    
    def _extract_key_topics(self, division_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract key topics across all sections with frequency and importance.
        
        Args:
            division_results: List of division results with summaries and metadata
            
        Returns:
            List of topic dictionaries with frequency and importance
        """
        # Extract all keywords and their sections
        all_keywords = []
        for result in division_results:
            section_keywords = result.get('keywords', [])
            section_importance = result.get('importance', 3)
            section_index = result.get('index', -1)
            section_position = result.get('position', 'middle')
            
            # Add each keyword with its metadata
            for keyword in section_keywords:
                all_keywords.append({
                    'keyword': keyword,
                    'importance': section_importance,
                    'section': section_index,
                    'position': section_position
                })
        
        # Count keyword frequency
        keyword_counter = Counter([k['keyword'].lower() for k in all_keywords])
        
        # Group by keyword
        keyword_data = {}
        for item in all_keywords:
            keyword = item['keyword'].lower()
            if keyword not in keyword_data:
                keyword_data[keyword] = {
                    'keyword': item['keyword'],  # Keep original casing
                    'frequency': keyword_counter[keyword],
                    'total_importance': 0,
                    'sections': [],
                    'positions': set()
                }
            
            keyword_data[keyword]['total_importance'] += item['importance']
            if item['section'] not in keyword_data[keyword]['sections']:
                keyword_data[keyword]['sections'].append(item['section'])
            keyword_data[keyword]['positions'].add(item['position'])
        
        # Calculate a relevance score for each keyword
        # Formula: frequency * avg_importance * position_factor
        for keyword, data in keyword_data.items():
            avg_importance = data['total_importance'] / data['frequency']
            
            # Position factor: topics that appear in both beginning and end are more important
            position_factor = 1.0
            if 'beginning' in data['positions'] and 'end' in data['positions']:
                position_factor = 1.5
            elif 'beginning' in data['positions']:
                position_factor = 1.2
            elif 'end' in data['positions']:
                position_factor = 1.1
            
            # Spread factor: topics that appear across many sections are more important
            spread_factor = min(1.0 + (len(data['sections']) / len(division_results) * 0.5), 1.5)
            
            data['relevance'] = data['frequency'] * avg_importance * position_factor * spread_factor
        
        # Convert to list and sort by relevance
        topics_list = list(keyword_data.values())
        topics_list.sort(key=lambda x: x['relevance'], reverse=True)
        
        return topics_list[:10]  # Return top 10 topics
    
    async def _generate_contextual_synthesis(self, 
                                           division_results: List[Dict[str, Any]],
                                           context: Dict[str, Any],
                                           key_topics: List[Dict[str, Any]],
                                           detail_level: str,
                                           doc_type: str,
                                           metadata: Optional[Dict[str, Any]]) -> str:
        """
        Generate a contextually-aware synthesis of all sections.
        
        Args:
            division_results: List of division results with summaries and metadata
            context: Context analysis from first section
            key_topics: Key topics across all sections
            detail_level: Detail level for synthesis
            doc_type: Document type
            metadata: Additional metadata
            
        Returns:
            Synthesized text
        """
        # Extract summaries and format with section positions
        formatted_summaries = []
        for i, result in enumerate(division_results):
            summary = result.get('summary', '')
            importance = result.get('importance', 3)
            position = result.get('position', 'middle')
            
            formatted_summaries.append(f"SECTION {i+1} ({position.upper()}, Importance: {importance}/5):\n{summary}")
        
        # Combine formatted summaries
        combined = "\n\n===== SECTION SEPARATOR =====\n\n".join(formatted_summaries)
        
        # Format key topics for the prompt
        formatted_topics = ""
        for topic in key_topics:
            formatted_topics += f"- {topic['keyword']}: relevance {topic['relevance']:.1f}, mentioned in {len(topic['sections'])} sections\n"
        
        # Create a structured prompt
        prompt = f"""
        You are creating a comprehensive meeting summary with special attention to the meeting's purpose and key topics.
        
        MEETING CONTEXT:
        - Purpose: {context.get('purpose', 'Discussion')}
        - Goals: {', '.join(context.get('goals', ['Information sharing']))}
        - Expected Outcomes: {', '.join(context.get('expected_outcomes', ['Decisions and next steps']))}
        
        KEY TOPICS (in order of relevance):
        {formatted_topics}
        
        INSTRUCTIONS:
        1. Structure the summary around the key meeting topics rather than chronologically
        2. Give more attention to topics with higher relevance scores
        3. Focus primarily on supporting the meeting's stated purpose and goals
        4. Include specific decisions, action items, and commitments
        5. Use {detail_level} level of detail
        
        FORMAT WITH MARKDOWN:
        # Meeting Summary: {context.get('purpose', 'Discussion')}

        ## Key Points

        ## Discussed Topics

        ## Conclusions & Next Steps
        
        SECTION SUMMARIES:
        {combined}
        """
        
        # Generate the synthesis
        synthesis = await self.llm_client.generate_completion_async(prompt)
        return synthesis
    
    async def _standard_synthesis(self, 
                                 summaries: List[str], 
                                 detail_level: str,
                                 doc_type: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a standard synthesis as fallback when rich metadata isn't available.
        
        Args:
            summaries: List of section summaries
            detail_level: Level of detail for the synthesis
            doc_type: Type of document
            metadata: Optional metadata about the document
            
        Returns:
            Synthesized text
        """
        # Combine section summaries with clear separation
        combined = "\n\n===== SECTION SEPARATOR =====\n\n".join([
            f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(summaries)
        ])
        
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
            
            if metadata.get("is_teams_transcript", False):
                metadata_context += "- This is a Microsoft Teams meeting transcript\n"
            
            if "speakers" in metadata:
                speakers = metadata["speakers"]
                metadata_context += f"- Participants: {', '.join(speakers)}\n"
        
        # Create the prompt
        prompt = f"""
        Create a coherent, well-organized summary from these section summaries.
        
        {detail_instructions[detail_level]}
        
        {doc_type_instructions[doc_type]}
        
        {metadata_context}
        
        FORMAT WITH MARKDOWN:
        # Meeting Summary

        ## Key Points

        ## Discussion Topics
        
        ## Conclusions & Next Steps
        
        SECTION SUMMARIES:
        {combined}
        """
        
        # Generate the synthesis
        synthesis = await self.llm_client.generate_completion_async(prompt)
        return synthesis
    
    def synthesize_summaries_sync(self, 
                                division_results: List[Dict[str, Any]], 
                                detail_level: str = "standard",
                                doc_type: str = "transcript",
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Synchronous version of synthesize_summaries.
        
        Args:
            division_results: List of division results with summaries and metadata
            detail_level: Level of detail for the synthesis
            doc_type: Type of document
            metadata: Optional metadata about the document
            
        Returns:
            Synthesized summary
        """
        import asyncio
        
        # Create an event loop and run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.synthesize_summaries(
                division_results, detail_level, doc_type, metadata
            ))
        finally:
            loop.close()