"""
Clean orchestrator module that seamlessly integrates enhanced features with lean architecture.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrates the document processing pipeline with a focus on balanced detail 
    preservation and hierarchical processing.
    """
    
    def __init__(self, 
                 llm_client, 
                 document_analyzer, 
                 document_chunker, 
                 chunk_summarizer, 
                 synthesizer,
                 options=None):
        """
        Initialize the orchestrator with component instances and options.
        
        Args:
            llm_client: LLM client for generating completions
            document_analyzer: DocumentAnalyzer for initial document analysis
            document_chunker: DocumentChunker for splitting text into chunks
            chunk_summarizer: ChunkSummarizer for processing individual chunks
            synthesizer: Synthesizer for creating final summaries
            options: ProcessingOptions instance with configuration
        """
        self.llm_client = llm_client
        self.document_analyzer = document_analyzer
        self.document_chunker = document_chunker
        self.chunk_summarizer = chunk_summarizer
        self.synthesizer = synthesizer
        self.options = options
        
        # Import here to avoid circular imports
        try:
            from lean.itemizer import ActionItemExtractor
            self.action_item_extractor = ActionItemExtractor(llm_client)
            self.has_action_extractor = True
        except ImportError:
            self.has_action_extractor = False
            logger.warning("ActionItemExtractor not available. Action item extraction will be limited.")
    
    async def process_document(self, text: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            text: Document text to process
            progress_callback: Optional callback function for progress updates
                               Function signature: callback(progress_float, status_message)
            
        Returns:
            Dictionary with complete summary and metadata
        """
        # Start timing
        start_time = time.time()
        total_steps = 5  # analyze, chunk, summarize, synthesize, extract actions
        current_step = 0
        
        # Set LLM temperature if available
        original_temperature = None
        if hasattr(self.llm_client, 'temperature'):
            original_temperature = self.llm_client.temperature
            self.llm_client.temperature = self.options.temperature
        
        try:
            # Step 1: Analyze document to get context
            logger.info("Starting document analysis")
            self._update_progress(progress_callback, current_step / total_steps, "Analyzing document...")
            
            analysis_result = await self.document_analyzer.analyze_preview(
                text, 
                preview_length=getattr(self.options, 'preview_length', 2000)
            )
            current_step += 1
            
            # Step 2: Chunk the document
            self._update_progress(progress_callback, current_step / total_steps, "Chunking document...")
            
            # Scale min_chunks based on detail level
            min_chunks = getattr(self.options, 'min_chunks', 3)
            detail_level = getattr(self.options, 'detail_level', 'detailed')
            
            scaled_min_chunks = min_chunks
            if detail_level == "detailed":
                scaled_min_chunks = min_chunks * 2
            elif detail_level == "detailed-complex":
                scaled_min_chunks = min_chunks * 3
                
            logger.info(f"Using scaled min_chunks: {scaled_min_chunks} for detail level: {detail_level}")
            
            chunks = self.document_chunker.chunk_document(
                text, 
                min_chunks=scaled_min_chunks, 
                max_chunk_size=getattr(self.options, 'max_chunk_size', None)
            )
            
            logger.info(f"Document divided into {len(chunks)} chunks")
            
            # Store total chunks in document context
            document_context = analysis_result.copy()
            document_context['total_chunks'] = len(chunks)
            document_context['original_text_length'] = len(text)
            
            # Apply user instructions if provided
            if hasattr(self.options, 'user_instructions') and self.options.user_instructions:
                document_context['user_instructions'] = self.options.user_instructions
            
            current_step += 1
            
            # Step 3: Process chunks with concurrency control
            self._update_progress(progress_callback, current_step / total_steps, "Summarizing chunks...")
            
            # Create a semaphore to limit concurrency
            max_concurrent = getattr(self.options, 'max_concurrent_chunks', 5)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Process each chunk with semaphore
            async def process_chunk(chunk):
                async with semaphore:
                    return await self.chunk_summarizer.summarize_chunk(chunk, document_context)
            
            # Create tasks for all chunks
            chunk_tasks = [process_chunk(chunk) for chunk in chunks]
            
            # Process chunks and track progress
            chunk_summaries = []
            for i, task in enumerate(asyncio.as_completed(chunk_tasks)):
                try:
                    result = await task
                    chunk_summaries.append(result)
                    
                    # Update progress
                    chunk_progress = (i + 1) / len(chunks)
                    progress_value = current_step / total_steps + (chunk_progress / total_steps)
                    self._update_progress(
                        progress_callback, 
                        progress_value, 
                        f"Processed chunk {i+1}/{len(chunks)}..."
                    )
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    # Continue with chunks we could process
            
            # Sort chunk summaries by index for consistent ordering
            chunk_summaries.sort(key=lambda x: x.get('chunk_index', 0))
            current_step += 1
            
            # Step 4: Synthesize final document
            self._update_progress(progress_callback, current_step / total_steps, "Synthesizing final summary...")
            
            synthesis_result = await self.synthesizer.synthesize_summaries(
                chunk_summaries,
                detail_level=detail_level,
                document_context=document_context
            )
            
            # Create result dictionary
            result = {
                'summary': synthesis_result,
                'document_info': document_context
            }
            
            # Extract key topics from the result if available
            if hasattr(self.synthesizer, 'extract_topics'):
                result['key_topics'] = await self.synthesizer.extract_topics(chunk_summaries)
            
            current_step += 1
            
            # Step 5: Extract action items if requested
            if getattr(self.options, 'include_action_items', True):
                self._update_progress(progress_callback, current_step / total_steps, "Extracting action items...")
                
                if self.has_action_extractor:
                    # Use our dedicated action item extractor
                    action_items = await self.action_item_extractor.extract_action_items(
                        chunk_summaries, 
                        document_context
                    )
                    result['action_items'] = action_items
                else:
                    # Fall back to extracting action items from chunk summaries
                    action_items = self._extract_action_items_fallback(chunk_summaries)
                    result['action_items'] = action_items
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            
            # Add metadata if requested
            if getattr(self.options, 'include_metadata', True):
                result['metadata'] = {
                    'processing_time_seconds': elapsed_time,
                    'chunks_processed': len(chunk_summaries),
                    'detail_level': detail_level,
                    'model': getattr(self.llm_client, 'model', 'unknown'),
                    'temperature': getattr(self.options, 'temperature', 0.2),
                    'timestamp': time.time()
                }
            
            # Add hierarchical metadata
            result['hierarchical_metadata'] = {
                'hierarchical_levels': 3 if detail_level == 'detailed-complex' else 2,
                'level1_groups': len(chunks),
                'level1_summaries': len(chunk_summaries),
                'level2_summaries': len(chunks) // 3 if len(chunks) > 3 else 1,
                'level3_summaries': 1 if detail_level == 'detailed-complex' else 0
            }
            
            # Update progress to complete
            self._update_progress(progress_callback, 1.0, "Processing complete")
            
            return result
            
        finally:
            # Restore original temperature if we changed it
            if original_temperature is not None and hasattr(self.llm_client, 'temperature'):
                self.llm_client.temperature = original_temperature
    
    def _update_progress(self, progress_callback, progress_value, status_message):
        """Update progress if callback is provided."""
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(progress_value, status_message)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")
    
    def _extract_action_items_fallback(self, chunk_summaries: List[Dict[str, Any]]) -> List[str]:
        """
        Extract action items from chunk summaries as a fallback.
        
        Args:
            chunk_summaries: List of chunk summaries
            
        Returns:
            List of action items
        """
        action_items = []
        
        # Collect action items from all chunks
        for chunk in chunk_summaries:
            chunk_items = chunk.get('action_items', [])
            
            # Convert to list if it's a string
            if isinstance(chunk_items, str):
                # Try to split by newlines and bullets
                import re
                items = re.findall(r'[\-\*•]\s*([^\n]+)', chunk_items)
                if items:
                    chunk_items = items
                else:
                    chunk_items = [chunk_items]
            
            # Add each item
            if isinstance(chunk_items, list):
                action_items.extend(chunk_items)
        
        # Basic deduplication
        unique_items = []
        for item in action_items:
            # Skip if empty or too short
            if not item or len(item) < 5:
                continue
                
            # Clean up the text
            item_text = item.strip()
            if item_text.startswith('-') or item_text.startswith('*'):
                item_text = item_text[1:].strip()
            
            # Only add if not already included
            if item_text and item_text not in unique_items:
                unique_items.append(item_text)
        
        return unique_items
    
    def process_document_sync(self, text: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process a document synchronously by running the async method in an event loop.
        
        Args:
            text: Document text to process
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary with complete summary and metadata
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_document(text, progress_callback)
            )
        finally:
            loop.close()