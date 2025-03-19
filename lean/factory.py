"""
Factory for creating and configuring the summarization pipeline.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SummarizerFactory:
    """Factory for creating a fully configured summarization pipeline."""
    
    @staticmethod
    def create_pipeline(api_key: Optional[str] = None, options=None):
        """
        Create a complete summarization pipeline with all components.
        
        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
            options: ProcessingOptions instance with configuration
            
        Returns:
            Tuple containing (orchestrator, document_analyzer, chunker, summarizer, synthesizer)
        """
        # Import all needed components
        from .async_openai_adapter import AsyncOpenAIAdapter
        from .document import DocumentAnalyzer
        from .chunker import DocumentChunker
        from .summarizer import ChunkSummarizer
        from .synthesizer import Synthesizer
        from .orchestrator import Orchestrator
        from .booster import Booster
        
        # Create options if not provided
        if options is None:
            from .options import ProcessingOptions
            options = ProcessingOptions()
        
        # Create LLM client with appropriate model
        llm_client = AsyncOpenAIAdapter(
            model=options.model_name,
            api_key=api_key,
            temperature=options.temperature
        )
        
        # Create booster for performance optimization
        booster = Booster(
            max_workers=options.max_concurrent_chunks,
            enable_caching=True
        )
        
        # Create all pipeline components
        document_analyzer = DocumentAnalyzer(llm_client)
        document_chunker = DocumentChunker()
        chunk_summarizer = ChunkSummarizer(llm_client)
        synthesizer = Synthesizer(llm_client)
        
        # Create orchestrator with all components
        orchestrator = Orchestrator(
            llm_client=llm_client,
            document_analyzer=document_analyzer,
            document_chunker=document_chunker,
            chunk_summarizer=chunk_summarizer,
            synthesizer=synthesizer,
            options=options
        )
        
        return {
            'orchestrator': orchestrator,
            'llm_client': llm_client,
            'document_analyzer': document_analyzer,
            'document_chunker': document_chunker,
            'chunk_summarizer': chunk_summarizer,
            'synthesizer': synthesizer,
            'booster': booster
        }
    
    