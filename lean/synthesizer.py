"""
Synthesizer module for combining chunk summaries into coherent documents.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class Synthesizer:
    """Combines chunk summaries into coherent, context-aware documents."""
    
    def __init__(self, llm_client):
        """
        Initialize the synthesizer.
        
        Args:
            llm_client: LLM client for generating syntheses
        """
        self.llm_client = llm_client
    
    async def synthesize_summaries(self, 
                                 chunk_results: List[Dict[str, Any]],
                                 detail_level: str = "standard",
                                 document_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a cohesive synthesis from chunk summaries.
        
        Args:
            chunk_results: List of chunk summary results
            detail_level: Level of detail ('essential', 'detailed', 'detailed-complex')
            document_context: Optional document context
            
        Returns:
            Synthesized text
        """
        # Sort chunk results by index
        sorted_results = sorted(chunk_results, key=lambda x: x.get('chunk_index', 0))
        
        # Extract key insights from all chunks
        insights = self._extract_document_insights(sorted_results)
        
        # Generate synthesis based on detail level
        if detail_level == "essential":
            return await self._generate_essential_synthesis(sorted_results, insights, document_context)
        elif detail_level == "detailed-complex":
            return await self._generate_complex_synthesis(sorted_results, insights, document_context)
        else:  # "detailed" (default)
            return await self._generate_detailed_synthesis(sorted_results, insights, document_context)
    
    def _extract_document_insights(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract key insights across all chunks.
        
        Args:
            chunk_results: List of chunk summary results
            
        Returns:
            Dictionary of document-wide insights
        """
        # Extract all keywords with importance scores
        keyword_scores = {}
        for result in chunk_results:
            # Get keywords and importance
            keywords = result.get('keywords', [])
            importance = result.get('importance', 3)
            
            # Add each keyword with its importance score
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in keyword_scores:
                    # Update existing keyword
                    keyword_scores[keyword_lower]['occurrences'] += 1
                    keyword_scores[keyword_lower]['total_importance'] += importance
                    # Keep the original casing with highest importance
                    if importance > keyword_scores[keyword_lower]['highest_importance']:
                        keyword_scores[keyword_lower]['keyword'] = keyword
                        keyword_scores[keyword_lower]['highest_importance'] = importance
                else:
                    # Add new keyword
                    keyword_scores[keyword_lower] = {
                        'keyword': keyword,  # Original casing
                        'occurrences': 1,
                        'total_importance': importance,
                        'highest_importance': importance
                    }
        
        # Calculate relevance scores
        top_keywords = []
        for k, v in keyword_scores.items():
            relevance = (v['occurrences'] * v['total_importance']) / len(chunk_results)
            top_keywords.append({
                'keyword': v['keyword'],
                'relevance': relevance,
                'occurrences': v['occurrences']
            })
        
        # Sort by relevance
        top_keywords.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Extract all speakers
        all_speakers = []
        for result in chunk_results:
            speakers = result.get('speakers', [])
            all_speakers.extend(speakers)
        
        # Count speaker occurrences
        speaker_counter = Counter(all_speakers)
        top_speakers = [{"name": name, "count": count} 
                       for name, count in speaker_counter.most_common(10)]
        
        # Extract all action items
        all_action_items = []
        for result in chunk_results:
            action_items = result.get('action_items', [])
            all_action_items.extend(action_items)
        
        # Extract key quotes
        all_quotes = []
        for result in chunk_results:
            quotes = result.get('key_quotes', [])
            all_quotes.extend(quotes)
        
        return {
            'top_keywords': top_keywords[:15],  # Top 15 keywords
            'top_speakers': top_speakers,
            'action_items': all_action_items,
            'key_quotes': all_quotes
        }
    
    async def _generate_essential_synthesis(self, 
                                          chunk_results: List[Dict[str, Any]],
                                          insights: Dict[str, Any],
                                          document_context: Optional[Dict[str, Any]]) -> str:
        """
        Generate a concise, essential synthesis.
        
        Args:
            chunk_results: List of chunk summary results
            insights: Document-wide insights
            document_context: Optional document context
            
        Returns:
            Synthesized text
        """
        # Format summaries for prompt
        formatted_summaries = []
        for i, result in enumerate(chunk_results):
            summary = result.get('summary', '')
            position = result.get('position', 'unknown')
            importance = result.get('importance', 3)
            
            # Only include high-importance chunks for essential summary
            if importance >= 3:
                formatted_summaries.append(f"SECTION {i+1} ({position.upper()}, Importance: {importance}/5):\n{summary}")
        
        # Combine formatted summaries
        combined_summaries = "\n\n===== SECTION SEPARATOR =====\n\n".join(formatted_summaries)
        
        # Format keywords
        formatted_keywords = "\n".join([
            f"- {kw['keyword']} (relevance: {kw['relevance']:.1f})"
            for kw in insights['top_keywords'][:10]
        ])
        
        # Create essential synthesis prompt
        prompt = f"""
        Create a concise, focused summary of this document. Focus only on the most essential information.
        
        KEY INSTRUCTIONS:
        1. Create a brief, to-the-point summary (around 3-5 paragraphs)
        2. Focus only on the most important points and decisions
        3. Eliminate all non-essential details
        4. Use clear, direct language
        5. Highlight only the critical action items or next steps
        
        The document appears to focus on these key topics:
        {formatted_keywords}
        
        FORMAT WITH MARKDOWN:
        # Document Summary
        
        ## Key Points
        
        ## Essential Action Items (if any)
        
        SECTION SUMMARIES:
        {combined_summaries}
        """
        
        # Generate the synthesis
        return await self.llm_client.generate_completion_async(prompt)
    
    async def _generate_detailed_synthesis(self, 
                                         chunk_results: List[Dict[str, Any]],
                                         insights: Dict[str, Any],
                                         document_context: Optional[Dict[str, Any]]) -> str:
        """
        Generate a detailed synthesis with good context and structure.
        
        Args:
            chunk_results: List of chunk summary results
            insights: Document-wide insights
            document_context: Optional document context
            
        Returns:
            Synthesized text
        """
        # Format summaries for prompt with all details
        formatted_summaries = []
        for i, result in enumerate(chunk_results):
            summary = result.get('summary', '')
            position = result.get('position', 'unknown')
            importance = result.get('importance', 3)
            formatted_summaries.append(f"SECTION {i+1} ({position.upper()}, Importance: {importance}/5):\n{summary}")
        
        # Combine formatted summaries
        combined_summaries = "\n\n===== SECTION SEPARATOR =====\n\n".join(formatted_summaries)
        
        # Format keywords with relevance
        formatted_keywords = "\n".join([
            f"- {kw['keyword']} (relevance: {kw['relevance']:.1f})"
            for kw in insights['top_keywords']
        ])
        
        # Get document type
        is_transcript = document_context.get('is_meeting_transcript', False) if document_context else False
        
        # Create type-specific instructions
        type_instructions = ""
        if is_transcript:
            # Format speakers if available
            top_speakers = ""
            if insights['top_speakers']:
                top_speakers = "\nKey participants:\n" + "\n".join([
                    f"- {s['name']} (mentioned {s['count']} times)" 
                    for s in insights['top_speakers']
                ])
                
            type_instructions = f"""
            This appears to be a meeting transcript or conversation.
            {top_speakers}
            
            For your summary:
            - Organize by topics rather than chronologically
            - Include key points from all main speakers
            - Highlight areas of agreement and disagreement
            - Clearly identify decisions and commitments
            """
        
        # Format document context
        context_info = ""
        if document_context:
            client = document_context.get('client_name')
            purpose = document_context.get('meeting_purpose')
            
            if client:
                context_info += f"\nClient: {client}"
            if purpose:
                context_info += f"\nPurpose: {purpose}"
        
        # Create detailed synthesis prompt
        prompt = f"""
        Create a comprehensive, well-structured summary of this document.
        
        {type_instructions}
        {context_info}
        
        KEY INSTRUCTIONS:
        1. Create a detailed summary that captures all important information
        2. Organize content by topics rather than by section
        3. Include important details while eliminating redundancies
        4. Use clear headings and structure
        5. Highlight decisions, action items, and key insights

        The document focuses on these key topics:
        {formatted_keywords}
        
        FORMAT WITH MARKDOWN:
        # Document Summary
        
        ## Executive Summary
        
        ## Key Topics and Discussion Points
        
        ## Decisions and Outcomes
        
        ## Action Items and Next Steps
        
        SECTION SUMMARIES:
        {combined_summaries}
        """
        
        # Generate the synthesis
        return await self.llm_client.generate_completion_async(prompt)
    
    async def _generate_complex_synthesis(self, 
                                        chunk_results: List[Dict[str, Any]],
                                        insights: Dict[str, Any],
                                        document_context: Optional[Dict[str, Any]]) -> str:
        """
        Generate an advanced synthesis using a hierarchical approach.
        
        Args:
            chunk_results: List of chunk summary results
            insights: Document-wide insights
            document_context: Optional document context
            
        Returns:
            Synthesized text
        """
        # For detailed-complex, we use a two-stage approach:
        # 1. Create intermediate summaries for groups of chunks
        # 2. Synthesize those intermediate summaries
        
        # Group chunks into clusters (3-4 chunks per group)
        group_size = 3
        chunk_groups = []
        current_group = []
        
        for i, chunk in enumerate(chunk_results):
            current_group.append(chunk)
            
            # When group is full or it's the last chunk, add group to list
            if len(current_group) >= group_size or i == len(chunk_results) - 1:
                if current_group:
                    chunk_groups.append(current_group)
                    current_group = []
        
        # Process each group to get intermediate summaries
        intermediate_summaries = []
        for i, group in enumerate(chunk_groups):
            # Format group summaries
            group_text = []
            for j, chunk in enumerate(group):
                summary = chunk.get('summary', '')
                position = chunk.get('position', 'unknown')
                importance = chunk.get('importance', 3)
                group_text.append(f"SECTION {j+1} ({position.upper()}, Importance: {importance}/5):\n{summary}")
            
            combined_group = "\n\n---\n\n".join(group_text)
            
            # Create group context description
            if i == 0:
                group_context = "This is the BEGINNING section of the document"
            elif i == len(chunk_groups) - 1:
                group_context = "This is the ENDING section of the document"
            else:
                group_context = f"This is section {i+1} of {len(chunk_groups)} of the document"
            
            # Create intermediate synthesis prompt
            intermediate_prompt = f"""
            Create a detailed synthesis of these related document sections.
            
            {group_context}
            
            INSTRUCTIONS:
            1. Synthesize these sections into one coherent, detailed summary
            2. Organize by topics rather than by section
            3. Preserve all important details and context
            4. Note any key decisions or action items
            5. Identify the most important themes across these sections
            
            SECTIONS:
            {combined_group}
            """
            
            # Generate intermediate summary
            intermediate_summary = await self.llm_client.generate_completion_async(intermediate_prompt)
            intermediate_summaries.append({
                'summary': intermediate_summary,
                'group_index': i,
                'is_beginning': i == 0,
                'is_ending': i == len(chunk_groups) - 1
            })
        
        # Now create the final synthesis from intermediate summaries
        formatted_intermediates = []
        for i, intermediate in enumerate(intermediate_summaries):
            section_type = "BEGINNING" if intermediate['is_beginning'] else "ENDING" if intermediate['is_ending'] else "MIDDLE"
            formatted_intermediates.append(f"SUMMARY SECTION {i+1} ({section_type}):\n{intermediate['summary']}")
        
        combined_intermediates = "\n\n===== MAJOR SECTION SEPARATOR =====\n\n".join(formatted_intermediates)
        
        # Format keywords with relevance
        formatted_keywords = "\n".join([
            f"- {kw['keyword']} (relevance: {kw['relevance']:.1f})"
            for kw in insights['top_keywords']
        ])
        
        # Format document context
        context_info = ""
        if document_context:
            client = document_context.get('client_name')
            purpose = document_context.get('meeting_purpose')
            
            if client:
                context_info += f"\nClient: {client}"
            if purpose:
                context_info += f"\nPurpose: {purpose}"
        
        # Create final synthesis prompt
        prompt = f"""
        Create a comprehensive, expertly structured summary of this document.
        
        {context_info}
        
        KEY INSTRUCTIONS:
        1. Create a professional-quality executive summary
        2. Develop thorough analysis organized by key themes and topics
        3. Include important context, nuances, and different perspectives
        4. Use a clear, logical structure with detailed sub-sections
        5. Provide in-depth coverage of decisions, action items, and implications

        The document centers on these key topics:
        {formatted_keywords}
        
        FORMAT WITH MARKDOWN:
        # Document Analysis
        
        ## Executive Summary
        
        ## Key Themes and Analysis
        
        ## Detailed Discussion Points
        
        ## Decisions and Implications
        
        ## Action Items and Next Steps
        
        ## Key Quotes and Insights
        
        SECTION SUMMARIES:
        {combined_intermediates}
        """
        
        # Generate the final synthesis
        return await self.llm_client.generate_completion_async(prompt)
    
    def synthesize_summaries_sync(self, 
                                chunk_results: List[Dict[str, Any]],
                                detail_level: str = "standard",
                                document_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Synchronous version of synthesize_summaries.
        
        Args:
            chunk_results: List of chunk summary results
            detail_level: Level of detail ('essential', 'detailed', 'detailed-complex')
            document_context: Optional document context
            
        Returns:
            Synthesized text
        """
        import asyncio
        
        # Create an event loop and run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.synthesize_summaries(chunk_results, detail_level, document_context)
            )
        finally:
            loop.close()