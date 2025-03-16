"""
Simplified output storage functionality for Note-Summarizer.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputStore:
    """Store and manage summarization outputs."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the output store.
        
        Args:
            output_dir: Directory for storing outputs
        """
        self.output_dir = Path(output_dir)
        self._ensure_directory_exists(self.output_dir)
    
    def _ensure_directory_exists(self, directory: Path):
        """Create directory if it doesn't exist."""
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _generate_filename(self, base_name: str, strategy: str) -> str:
        """
        Generate a filename with the format: BaseName_Strategy_YYYY-MM-DD.md
        
        Args:
            base_name: Base name for the file (typically document title)
            strategy: Division strategy used
            
        Returns:
            Filename with date
        """
        # Clean the base name to make it file-system friendly
        safe_name = "".join(c if c.isalnum() or c in ['.', '-', '_'] else '_' for c in base_name)
        
        # Get current date in YYYY-MM-DD format
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        return f"{safe_name}_{strategy}_{date_str}.md"
    
    def store_single_document_summary(self, title: str, content: Dict[str, Any], 
                                    document_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a single document summary.
        
        Args:
            title: Document title
            content: Dictionary containing summary content
            document_info: Optional document metadata
            
        Returns:
            Path to the stored markdown file
        """
        # Get the division strategy used
        strategy = content.get("metadata", {}).get("division_strategy", "unknown")
        
        # Generate filename
        filename = self._generate_filename(title, strategy)
        file_path = self.output_dir / filename
        
        # Create markdown content
        markdown = self._create_single_document_markdown(title, content, document_info)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Summary stored at: {file_path}")
        return str(file_path)
    
    def store_multi_document_summary(self, title: str, content: Dict[str, Any],
                                   document_infos: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Store a multi-document summary.
        
        Args:
            title: Analysis title
            content: Dictionary containing analysis content
            document_infos: Optional list of document metadata
            
        Returns:
            Path to the stored markdown file
        """
        # Get the document type
        doc_type = content.get("metadata", {}).get("document_type", "multi")
        
        # Generate filename
        filename = self._generate_filename(title, doc_type)
        file_path = self.output_dir / filename
        
        # Create markdown content
        markdown = self._create_multi_document_markdown(title, content, document_infos)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Analysis stored at: {file_path}")
        return str(file_path)
    
    def store_comparison(self, title: str, results: Dict[str, Dict[str, Any]],
                      document_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a strategy comparison.
        
        Args:
            title: Comparison title
            results: Dictionary of strategy results
            document_info: Optional document metadata
            
        Returns:
            Path to the stored markdown file
        """
        # Generate filename
        filename = self._generate_filename(title, "comparison")
        file_path = self.output_dir / filename
        
        # Create markdown content
        markdown = self._create_comparison_markdown(title, results, document_info)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Comparison stored at: {file_path}")
        return str(file_path)
    
    def _create_single_document_markdown(self, title: str, content: Dict[str, Any],
                                       document_info: Optional[Dict[str, Any]] = None) -> str:
        """Create markdown content for a single document summary."""
        lines = []
        
        # Add header
        lines.append(f"# Summary: {title}")
        lines.append("")
        
        # Add generation info
        lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*")
        lines.append("")
        
        # Add document stats if available
        if document_info:
            lines.append("## Document Information")
            lines.append("")
            lines.append(f"- Characters: {document_info.get('num_characters', 'N/A'):,}")
            lines.append(f"- Words: {document_info.get('num_words', 'N/A'):,}")
            lines.append(f"- Speakers: {document_info.get('num_speakers', 'N/A')}")
            lines.append("")
        
        # Add processing info if available
        if "metadata" in content:
            lines.append("## Processing Information")
            lines.append("")
            lines.append(f"- Division Strategy: {content['metadata'].get('division_strategy', 'N/A')}")
            lines.append(f"- Model: {content['metadata'].get('model', 'N/A')}")
            lines.append(f"- Processing Time: {content['metadata'].get('processing_time_seconds', 'N/A'):.2f} seconds")
            lines.append(f"- Division Count: {content['metadata'].get('division_count', 'N/A')}")
            lines.append("")
        
        # Add summary content
        if "summary" in content:
            # The summary might already have a header, so we'll add it directly
            lines.append(content["summary"])
            lines.append("")
        
        # Add action items if available
        if "action_items" in content and content["action_items"]:
            if not content["action_items"].startswith("#"):
                lines.append("## Action Items")
                lines.append("")
            lines.append(content["action_items"])
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_multi_document_markdown(self, title: str, content: Dict[str, Any],
                                      document_infos: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create markdown content for a multi-document analysis."""
        lines = []
        
        # Add header
        lines.append(f"# {title}")
        lines.append("")
        
        # Add generation info
        lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*")
        lines.append("")
        
        # Add document collection info if available
        if "metadata" in content:
            lines.append("## Analysis Information")
            lines.append("")
            
            doc_type = content["metadata"].get("document_type", "Generic")
            if doc_type == "earnings_calls_multi_company":
                lines.append("**Analysis Type:** Multi-Company Earnings Call Comparison")
                if "companies" in content["metadata"]:
                    lines.append(f"**Companies:** {', '.join(content['metadata']['companies'])}")
            elif doc_type == "earnings_calls_sequential":
                lines.append("**Analysis Type:** Sequential Earnings Call Analysis")
                if "company" in content["metadata"]:
                    lines.append(f"**Company:** {content['metadata']['company']}")
                if "periods" in content["metadata"]:
                    lines.append(f"**Periods:** {', '.join(content['metadata']['periods'])}")
            else:
                lines.append("**Analysis Type:** Multi-Document Synthesis")
            
            lines.append(f"**Documents Analyzed:** {content['metadata'].get('document_count', 'N/A')}")
            lines.append(f"**Processing Time:** {content['metadata'].get('processing_time_seconds', 'N/A'):.2f} seconds")
            lines.append("")
        
        # Add document list if available
        if document_infos:
            lines.append("## Documents")
            lines.append("")
            for i, doc_info in enumerate(document_infos):
                company = doc_info.get("metadata", {}).get("company", f"Document {i+1}")
                period = doc_info.get("metadata", {}).get("period", "")
                filename = doc_info.get("filename", "")
                
                doc_title = company
                if period:
                    doc_title += f" ({period})"
                
                lines.append(f"### {doc_title}")
                lines.append("")
                if filename:
                    lines.append(f"Filename: `{filename}`")
                lines.append(f"Size: {doc_info.get('size', 'N/A'):,} characters")
                lines.append(f"Words: {doc_info.get('words', 'N/A'):,}")
                lines.append("")
        
        # Add main analysis
        if "summary" in content:
            # The summary might already have a header, so we'll add it directly
            lines.append(content["summary"])
            lines.append("")
        
        # Add action items if available
        if "action_items" in content and content["action_items"]:
            if not content["action_items"].startswith("#"):
                lines.append("## Action Items")
                lines.append("")
            lines.append(content["action_items"])
            lines.append("")
        
        # Add individual summaries if available
        if "individual_summaries" in content and content["individual_summaries"]:
            lines.append("## Individual Document Summaries")
            lines.append("")
            
            # Determine what to call each document
            labels = []
            if "companies" in content.get("metadata", {}):
                labels = content["metadata"]["companies"]
            elif "periods" in content.get("metadata", {}):
                periods = content["metadata"]["periods"]
                company = content["metadata"].get("company", "")
                labels = [f"{company} {period}" for period in periods]
            elif "titles" in content.get("metadata", {}):
                labels = content["metadata"]["titles"]
            else:
                labels = [f"Document {i+1}" for i in range(len(content["individual_summaries"]))]
            
            # Add each summary with its label
            for i, summary in enumerate(content["individual_summaries"]):
                if i < len(labels):
                    lines.append(f"### {labels[i]}")
                else:
                    lines.append(f"### Document {i+1}")
                lines.append("")
                lines.append(summary)
                lines.append("")
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)
    
    def _create_comparison_markdown(self, title: str, results: Dict[str, Dict[str, Any]], 
                                  document_info: Dict[str, Any] = None) -> str:
        """
        Create markdown content for strategy comparison.
        
        Args:
            title: Comparison title
            results: Dictionary of strategy results
            document_info: Document metadata
            
        Returns:
            Markdown content
        """
        lines = []
        
        # Add header
        lines.append(f"# {title}")
        lines.append("")
        
        # Add generation info
        lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*")
        lines.append("")
        
        # Add document info if available
        if document_info:
            lines.append("## Document Information")
            lines.append("")
            lines.append(f"- Characters: {document_info.get('num_characters', 'N/A'):,}")
            lines.append(f"- Words: {document_info.get('num_words', 'N/A'):,}")
            lines.append(f"- Speakers: {document_info.get('num_speakers', 'N/A')}")
            lines.append("")
        
        # Add comparison table
        lines.append("## Strategy Comparison")
        lines.append("")
        lines.append("| Strategy | Processing Time | Divisions | Summary Length |")
        lines.append("|----------|-----------------|-----------|----------------|")
        
        for strategy, result in results.items():
            processing_time = f"{result.get('processing_time', 0):.2f}s"
            divisions = result.get('division_count', 0)
            summary_length = len(result.get('summary', ''))
            
            lines.append(f"| {strategy} | {processing_time} | {divisions} | {summary_length:,} chars |")
        
        lines.append("")
        
        # Add summaries for each strategy
        lines.append("## Strategy Summaries")
        lines.append("")
        
        for strategy, result in results.items():
            lines.append(f"### {strategy.capitalize()} Strategy")
            lines.append("")
            lines.append(result.get('summary', 'No summary available.'))
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    

   