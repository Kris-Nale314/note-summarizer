"""
Updated factory for creating and configuring the enhanced summarization pipeline.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SummarizerFactory:
    """Factory for creating a fully configured summarization pipeline with enhanced features."""
    
    @staticmethod
    def create_pipeline(api_key: Optional[str] = None, options=None):
        """
        Create a complete summarization pipeline with all components.
        
        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
            options: ProcessingOptions instance with configuration
            
        Returns:
            Dictionary containing all pipeline components
        """
        # Import all needed components
        from lean.async_openai_adapter import AsyncOpenAIAdapter
        from lean.document import DocumentAnalyzer
        from lean.chunker import DocumentChunker
        from lean.summarizer import ChunkSummarizer
        from lean.synthesizer import Synthesizer
        from lean.orchestrator import Orchestrator
        from lean.booster import Booster
        from lean.itemizer import ActionItemExtractor
        from lean.refiner import SummaryRefiner
        
        # Create options if not provided
        if options is None:
            from lean.options import ProcessingOptions
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
        
        # Create specialized components
        action_extractor = ActionItemExtractor(llm_client)
        summary_refiner = SummaryRefiner(llm_client)
        
        # Create enhanced orchestrator with all components
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
            'booster': booster,
            'action_extractor': action_extractor,
            'summary_refiner': summary_refiner
        }
    
    @staticmethod
    def create_streamlit_app(api_key: Optional[str] = None):
        """
        Create a Streamlit app with the summarization pipeline.
        
        Args:
            api_key: OpenAI API key (optional)
            
        Returns:
            Streamlit app instance
        """
        try:
            import streamlit as st
            from lean.app import create_app
            
            # Create pipeline
            pipeline = SummarizerFactory.create_pipeline(api_key)
            
            # Create and return app
            app = create_app(pipeline)
            return app
        except ImportError:
            logger.error("Streamlit not installed. Cannot create app.")
            raise