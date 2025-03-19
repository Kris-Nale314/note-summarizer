"""
Action item extraction module with specialized logic for identifying and validating action items.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ActionItemExtractor:
    """
    Specialized class for extracting valid action items from document summaries.
    Uses a multi-stage approach to identify, validate, deduplicate, and prioritize
    action items.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the action item extractor.
        
        Args:
            llm_client: LLM client for action item analysis
        """
        self.llm_client = llm_client
    
    async def extract_action_items(self, chunk_summaries: List[Dict[str, Any]], 
                                document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract action items from chunk summaries using a multi-stage approach.
        
        Args:
            chunk_summaries: List of chunk summary dictionaries
            document_context: Document context from initial analysis
            
        Returns:
            List of validated and prioritized action items
        """
        # Stage 1: Identify candidate action items from each chunk
        candidate_items = await self._identify_candidates(chunk_summaries)
        
        # No candidates found
        if not candidate_items:
            logger.info("No action item candidates identified")
            return []
        
        # Stage 2: Validate candidates for specificity and actionability
        validated_items = await self._validate_candidates(candidate_items, document_context)
        
        # Stage 3: Deduplicate and prioritize the validated items
        final_items = self._deduplicate_and_prioritize(validated_items)
        
        logger.info(f"Extracted {len(final_items)} action items from document")
        return final_items
    
    async def _identify_candidates(self, chunk_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify candidate action items from each chunk summary.
        
        Args:
            chunk_summaries: List of chunk summary dictionaries
            
        Returns:
            List of candidate action items with metadata
        """
        # Collect existing action items from chunk summaries
        candidates = []
        
        # First pass: collect any action items already identified in chunks
        for i, chunk in enumerate(chunk_summaries):
            chunk_items = chunk.get('action_items', [])
            
            # Skip if no action items in this chunk
            if not chunk_items:
                continue
            
            # Normalize format to list if it's a string
            if isinstance(chunk_items, str):
                # Try to extract individual items from the text
                items = [item.strip() for item in chunk_items.split('\n') 
                        if item.strip() and (item.strip().startswith('-') or item.strip().startswith('*'))]
                
                # Fall back to treating it as a single item if we couldn't parse it
                if not items:
                    items = [chunk_items]
            else:
                items = chunk_items
            
            # Add each item to candidates
            for item in items:
                # Clean up the item text
                if isinstance(item, str):
                    item_text = item.strip()
                    # Remove leading dash/bullet if present
                    if item_text.startswith('-') or item_text.startswith('*'):
                        item_text = item_text[1:].strip()
                    
                    candidates.append({
                        'text': item_text,
                        'source_chunk': i,
                        'position': chunk.get('position', 'unknown'),
                        'importance': chunk.get('importance', 3),
                        'speakers': chunk.get('speakers', [])
                    })
        
        # If no action items found through direct extraction, use LLM to identify candidates
        if not candidates:
            candidates = await self._extract_with_llm(chunk_summaries)
        
        return candidates
    
    async def _extract_with_llm(self, chunk_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to identify action items from chunk summaries when direct extraction fails.
        
        Args:
            chunk_summaries: List of chunk summary dictionaries
            
        Returns:
            List of candidate action items
        """
        # Process chunks with high importance first to prioritize key sections
        sorted_chunks = sorted(chunk_summaries, key=lambda x: x.get('importance', 3), reverse=True)
        
        # Combine summaries and other data for context
        combined_text = ""
        
        for i, chunk in enumerate(sorted_chunks):
            summary = chunk.get('summary', '')
            speakers = ", ".join(chunk.get('speakers', []))
            position = chunk.get('position', 'unknown')
            
            # Add formatted chunk content
            combined_text += f"SECTION {i+1} ({position}):\n"
            if speakers:
                combined_text += f"Speakers: {speakers}\n"
            combined_text += f"{summary}\n\n"
        
        # Create prompt for action item extraction
        prompt = f"""
        Analyze the following text and identify specific action items, tasks, or follow-up items.
        
        Focus only on concrete, specific actions that were clearly agreed upon or assigned.
        Do NOT include vague suggestions, ideas for future consideration, or general discussion points.
        
        Good examples:
        - "John will update the project timeline by Friday"
        - "Marketing team to prepare campaign materials for Q3 launch"
        - "Sarah and Mike to investigate the performance issue and report back next week"
        
        Bad examples (too vague):
        - "The team should think about the strategy going forward"
        - "It might be good to look into that sometime"
        - "We could potentially consider a new approach"
        
        TEXT TO ANALYZE:
        {combined_text}
        
        Return your response as a JSON array of objects, where each object has:
        - "text": The text of the action item
        - "assignee": Who is assigned this item (if specified), or null if unspecified
        - "due": Due date or timeframe (if specified), or null if unspecified
        - "source_section": Section number where this was found
        - "confidence": A value from 1-5 indicating how confident you are this is a true action item (5 being highest)
        """
        
        try:
            # Get response from LLM
            response = await self.llm_client.generate_completion_async(prompt)
            
            # Parse response as JSON
            try:
                items = json.loads(response)
                
                # Validate format and transform
                candidates = []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and 'text' in item:
                            # Map to our candidate format
                            source_section = item.get('source_section')
                            if isinstance(source_section, str) and source_section.isdigit():
                                source_section = int(source_section) - 1  # Convert to 0-based index
                            
                            # Limit to valid sections
                            if not isinstance(source_section, int) or source_section < 0 or source_section >= len(sorted_chunks):
                                source_section = 0
                            
                            candidates.append({
                                'text': item['text'],
                                'assignee': item.get('assignee'),
                                'due': item.get('due'),
                                'source_chunk': source_section,
                                'position': sorted_chunks[source_section].get('position', 'unknown'),
                                'importance': sorted_chunks[source_section].get('importance', 3),
                                'confidence': item.get('confidence', 3),
                                'speakers': sorted_chunks[source_section].get('speakers', [])
                            })
                return candidates
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON. Attempting to extract manually.")
                return self._extract_from_text_response(response, sorted_chunks)
        except Exception as e:
            logger.error(f"Error in LLM action item extraction: {e}")
            return []
    
    def _extract_from_text_response(self, text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract action items from a text response when JSON parsing fails.
        
        Args:
            text: Text response from LLM
            chunks: Sorted chunk summaries
            
        Returns:
            List of candidate action items
        """
        # This is a fallback method for when the LLM doesn't return valid JSON
        candidates = []
        
        # Look for items as bullet points or numbered lists
        import re
        
        # Patterns to look for
        patterns = [
            r'- ([^\n]+)',  # Bullet points
            r'\* ([^\n]+)',  # Asterisk bullet points
            r'\d+\.\s+([^\n]+)'  # Numbered items
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                candidates.append({
                    'text': match.strip(),
                    'source_chunk': 0,  # Default to first chunk as we can't determine source
                    'position': chunks[0].get('position', 'unknown') if chunks else 'unknown',
                    'importance': 3,  # Default importance
                    'confidence': 3,  # Default confidence
                    'speakers': []
                })
        
        return candidates
    
    async def _validate_candidates(self, candidates: List[Dict[str, Any]], 
                                document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate candidate action items for specificity and actionability.
        
        Args:
            candidates: List of candidate action item dictionaries
            document_context: Document context from initial analysis
            
        Returns:
            List of validated action items with scores
        """
        # Skip validation if no candidates
        if not candidates:
            return []
        
        # Extract texts only for validation
        texts = [item['text'] for item in candidates]
        
        # Create validation prompt
        prompt = f"""
        For each of the following potential action items, evaluate whether it represents a REAL, ACTIONABLE task.
        
        Action items should be:
        1. Specific - Clear what needs to be done
        2. Actionable - Something that can actually be completed
        3. Assigned - Preferably has a person or team responsible
        4. Scoped - Has a timeframe or clear completion criteria
        
        Validation criteria:
        - Rate each on a scale of 1-5 (5 being highest)
        - Score 4-5: Valid action item with clear responsibility
        - Score 3: Probable action item but missing some specificity
        - Score 1-2: Too vague, not really actionable, or just a suggestion
        
        ITEMS TO EVALUATE:
        {chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}
        
        Return your response as a JSON array with the format:
        [
          {{"index": 0, "score": 4, "reason": "Clear action with owner and timeframe"}},
          {{"index": 1, "score": 2, "reason": "Too vague, no clear owner or deliverable"}}
        ]
        
        Use 0-based indices matching the original list order.
        """
        
        try:
            # Get response from LLM
            response = await self.llm_client.generate_completion_async(prompt)
            
            # Parse response as JSON
            try:
                validation_results = json.loads(response)
                
                # Apply validation results to candidates
                validated_items = []
                if isinstance(validation_results, list):
                    for result in validation_results:
                        if isinstance(result, dict) and 'index' in result and 'score' in result:
                            idx = result['index']
                            score = result['score']
                            reason = result.get('reason', '')
                            
                            # Only keep items with score of 3 or higher
                            if idx < len(candidates) and score >= 3:
                                item = candidates[idx].copy()
                                item['validation_score'] = score
                                item['validation_reason'] = reason
                                validated_items.append(item)
                
                return validated_items
            except json.JSONDecodeError:
                logger.warning("Failed to parse validation response as JSON. Using all candidates with default scores.")
                # Return all candidates with default validation score
                for item in candidates:
                    item['validation_score'] = 3
                    item['validation_reason'] = "Default score (validation failed)"
                return candidates
        except Exception as e:
            logger.error(f"Error in action item validation: {e}")
            # Return all candidates with default validation score
            for item in candidates:
                item['validation_score'] = 3
                item['validation_reason'] = "Default score (validation failed)"
            return candidates
    
    def _deduplicate_and_prioritize(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate and prioritize validated action items.
        
        Args:
            items: List of validated action item dictionaries
            
        Returns:
            List of final action items, sorted by priority
        """
        if not items:
            return []
        
        # Group similar items
        grouped_items = self._group_similar_items(items)
        
        # For each group, select the best representative
        final_items = []
        for group in grouped_items:
            # Sort by validation score, then by source importance
            sorted_group = sorted(
                group, 
                key=lambda x: (
                    x.get('validation_score', 3),
                    x.get('importance', 3),
                    # Prefer items with assignees
                    1 if x.get('assignee') else 0,
                    # Prefer items with due dates
                    1 if x.get('due') else 0
                ), 
                reverse=True
            )
            
            # Select the best item from the group
            best_item = sorted_group[0]
            
            # Collect assignee and due date from other items if best doesn't have them
            if not best_item.get('assignee') and any(item.get('assignee') for item in sorted_group):
                for item in sorted_group:
                    if item.get('assignee'):
                        best_item['assignee'] = item['assignee']
                        break
            
            if not best_item.get('due') and any(item.get('due') for item in sorted_group):
                for item in sorted_group:
                    if item.get('due'):
                        best_item['due'] = item['due']
                        break
            
            final_items.append(best_item)
        
        # Sort final items by priority
        final_items.sort(key=lambda x: (
            x.get('validation_score', 3), 
            x.get('importance', 3)
        ), reverse=True)
        
        # Transform to final format
        formatted_items = []
        for item in final_items:
            formatted_text = item['text']
            
            # Add assignee if available
            if item.get('assignee'):
                if item['assignee'].lower() not in formatted_text.lower():
                    formatted_text = f"{item['assignee']}: {formatted_text}"
            
            # Add due date if available
            if item.get('due'):
                if item['due'].lower() not in formatted_text.lower():
                    formatted_text = f"{formatted_text} (Due: {item['due']})"
            
            formatted_items.append(formatted_text)
        
        return formatted_items
    
    def _group_similar_items(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group similar action items based on text similarity.
        
        Args:
            items: List of action item dictionaries
            
        Returns:
            List of grouped similar items
        """
        if not items:
            return []
        
        import re
        from difflib import SequenceMatcher
        
        # Normalize text for comparison
        def normalize_text(text):
            # Convert to lowercase and remove punctuation
            return re.sub(r'[^\w\s]', '', text.lower())
        
        # Calculate similarity between two texts
        def similarity(text1, text2):
            return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()
        
        # Group similar items
        groups = []
        remaining_items = items.copy()
        
        while remaining_items:
            current_item = remaining_items.pop(0)
            current_group = [current_item]
            
            i = 0
            while i < len(remaining_items):
                if similarity(current_item['text'], remaining_items[i]['text']) > 0.7:
                    current_group.append(remaining_items.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups