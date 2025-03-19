"""
Enhanced orchestrator module with action item extraction and user instructions support.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class Orchestrator:
    """Coordinates the document processing pipeline with advanced features."""
    
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
        from lean.itemizer import ActionItemExtractor
        self.action_item_extractor = ActionItemExtractor(llm_client)
    
    async def process_document(self, text: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline with enhanced features.
        
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
                preview_length=self.options.preview_length
            )
            current_step += 1
            
            # Step 2: Chunk the document
            self._update_progress(progress_callback, current_step / total_steps, "Chunking document...")
            
            chunks = self.document_chunker.chunk_document(
                text, 
                min_chunks=self.options.min_chunks, 
                max_chunk_size=self.options.max_chunk_size
            )
            
            logger.info(f"Document divided into {len(chunks)} chunks")
            
            # Store total chunks in document context
            document_context = analysis_result.copy()
            document_context['total_chunks'] = len(chunks)
            
            # Apply user instructions if provided
            if hasattr(self.options, 'user_instructions') and self.options.user_instructions:
                document_context['user_instructions'] = self.options.user_instructions
            
            current_step += 1
            
            # Step 3: Process chunks with concurrency control
            self._update_progress(progress_callback, current_step / total_steps, "Summarizing chunks...")
            
            # Create a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.options.max_concurrent_chunks)
            
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
                detail_level=self.options.detail_level,
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
            if self.options.include_action_items:
                self._update_progress(progress_callback, current_step / total_steps, "Extracting action items...")
                
                # Use our dedicated action item extractor
                action_items = await self.action_item_extractor.extract_action_items(
                    chunk_summaries, 
                    document_context
                )
                
                # Add to result
                result['action_items'] = action_items
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            
            # Add metadata if requested
            if self.options.include_metadata:
                result['metadata'] = {
                    'processing_time_seconds': elapsed_time,
                    'chunks_processed': len(chunk_summaries),
                    'detail_level': self.options.detail_level,
                    'model': getattr(self.llm_client, 'model', 'unknown'),
                    'temperature': self.options.temperature,
                    'timestamp': time.time()
                }
            
            # Add hierarchical metadata if available from chunks
            hierarchical_data = {}
            for chunk in chunk_summaries:
                if 'hierarchical_metadata' in chunk:
                    for key, value in chunk['hierarchical_metadata'].items():
                        if key in hierarchical_data:
                            hierarchical_data[key] += value
                        else:
                            hierarchical_data[key] = value
            
            if hierarchical_data:
                result['hierarchical_metadata'] = hierarchical_data
            
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