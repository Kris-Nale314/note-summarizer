"""
Output storage functionality for Note-Summarizer.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class OutputStore:
    """Store and manage summarization outputs in markdown format."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the output store."""
        self.output_dir = Path(output_dir)
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, base_name: str, suffix: str) -> str:
        """Generate a filesystem-friendly filename with date."""
        # Clean the base name for filesystem compatibility
        safe_name = "".join(c if c.isalnum() or c in ['.', '-', '_'] else '_' for c in base_name)
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{safe_name}_{suffix}_{date_str}.md"
    
    def store_single_document_summary(self, title: str, content: Dict[str, Any], 
                                     document_info: Optional[Dict[str, Any]] = None) -> str:
        """Store a single document summary as markdown."""
        strategy = content.get("metadata", {}).get("division_strategy", "unknown")
        filename = self._generate_filename(title, strategy)
        file_path = self.output_dir / filename
        
        # Create markdown content
        markdown = []
        markdown.append(f"# Summary: {title}")
        markdown.append("")
        markdown.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*")
        markdown.append("")
        
        # Add document stats if available
        if document_info:
            markdown.append("## Document Information")
            markdown.append("")
            markdown.append(f"- Characters: {document_info.get('num_characters', 'N/A'):,}")
            markdown.append(f"- Words: {document_info.get('num_words', 'N/A'):,}")
            if 'num_speakers' in document_info:
                markdown.append(f"- Speakers: {document_info.get('num_speakers', 'N/A')}")
            markdown.append("")
        
        # Add processing info
        if "metadata" in content:
            markdown.append("## Processing Information")
            markdown.append("")
            markdown.append(f"- Division Strategy: {content['metadata'].get('division_strategy', 'N/A')}")
            markdown.append(f"- Model: {content['metadata'].get('model', 'N/A')}")
            markdown.append(f"- Processing Time: {content['metadata'].get('processing_time_seconds', 'N/A'):.2f} seconds")
            markdown.append(f"- Division Count: {content['metadata'].get('division_count', 'N/A')}")
            markdown.append("")
        
        # Add summary content
        if "summary" in content:
            markdown.append(content["summary"])
            markdown.append("")
        
        # Add action items if available
        if "action_items" in content and content["action_items"]:
            if not content["action_items"].startswith("#"):
                markdown.append("## Action Items")
                markdown.append("")
            markdown.append(content["action_items"])
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown))
        
        logger.info(f"Summary stored at: {file_path}")
        return str(file_path)
    
    def store_multi_document_summary(self, title: str, content: Dict[str, Any],
                                   document_infos: Optional[List[Dict[str, Any]]] = None) -> str:
        """Store a multi-document analysis as markdown."""
        doc_type = content.get("metadata", {}).get("document_type", "multi")
        filename = self._generate_filename(title, doc_type)
        file_path = self.output_dir / filename
        
        # Create markdown content
        markdown = []
        markdown.append(f"# {title}")
        markdown.append("")
        markdown.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*")
        markdown.append("")
        
        # Add metadata
        if "metadata" in content:
            markdown.append("## Analysis Information")
            markdown.append("")
            
            # Format based on document type
            doc_type = content["metadata"].get("document_type", "Generic")
            if doc_type == "earnings_calls_multi_company":
                markdown.append("**Analysis Type:** Multi-Company Earnings Call Comparison")
                if "companies" in content["metadata"]:
                    markdown.append(f"**Companies:** {', '.join(content['metadata']['companies'])}")
            elif doc_type == "earnings_calls_sequential":
                markdown.append("**Analysis Type:** Sequential Earnings Call Analysis")
                if "company" in content["metadata"]:
                    markdown.append(f"**Company:** {content['metadata']['company']}")
                if "periods" in content["metadata"]:
                    markdown.append(f"**Periods:** {', '.join(content['metadata']['periods'])}")
            else:
                markdown.append("**Analysis Type:** Multi-Document Synthesis")
            
            markdown.append(f"**Documents Analyzed:** {content['metadata'].get('document_count', 'N/A')}")
            markdown.append(f"**Processing Time:** {content['metadata'].get('processing_time_seconds', 'N/A'):.2f} seconds")
            markdown.append("")
        
        # Add main analysis
        if "summary" in content:
            markdown.append(content["summary"])
            markdown.append("")
        
        # Add action items if available
        if "action_items" in content and content["action_items"]:
            if not content["action_items"].startswith("#"):
                markdown.append("## Action Items")
                markdown.append("")
            markdown.append(content["action_items"])
            markdown.append("")
        
        # Add individual summaries
        if "individual_summaries" in content and content["individual_summaries"]:
            # Determine document labels based on metadata
            labels = self._get_document_labels(content)
            
            markdown.append("## Individual Document Summaries")
            markdown.append("")
            
            for i, summary in enumerate(content["individual_summaries"]):
                label = labels[i] if i < len(labels) else f"Document {i+1}"
                markdown.append(f"### {label}")
                markdown.append("")
                markdown.append(summary)
                markdown.append("")
                markdown.append("---")
                markdown.append("")
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown))
        
        logger.info(f"Analysis stored at: {file_path}")
        return str(file_path)
    
    def _get_document_labels(self, content: Dict[str, Any]) -> List[str]:
        """Extract appropriate document labels from metadata."""
        if "companies" in content.get("metadata", {}):
            return content["metadata"]["companies"]
        elif "periods" in content.get("metadata", {}):
            periods = content["metadata"]["periods"]
            company = content["metadata"].get("company", "")
            return [f"{company} {period}" for period in periods]
        elif "titles" in content.get("metadata", {}):
            return content["metadata"]["titles"]
        else:
            return [f"Document {i+1}" for i in range(len(content.get("individual_summaries", [])))]
    
    def store_comparison(self, title: str, results: Dict[str, Dict[str, Any]],
                      document_info: Optional[Dict[str, Any]] = None) -> str:
        """Store a strategy comparison as markdown."""
        filename = self._generate_filename(title, "comparison")
        file_path = self.output_dir / filename
        
        # Create markdown content
        markdown = []
        markdown.append(f"# {title}")
        markdown.append("")
        markdown.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d')}*")
        markdown.append("")
        
        # Add document info
        if document_info:
            markdown.append("## Document Information")
            markdown.append("")
            markdown.append(f"- Characters: {document_info.get('num_characters', 'N/A'):,}")
            markdown.append(f"- Words: {document_info.get('num_words', 'N/A'):,}")
            if 'num_speakers' in document_info:
                markdown.append(f"- Speakers: {document_info.get('num_speakers', 'N/A')}")
            markdown.append("")
        
        # Add comparison table
        markdown.append("## Strategy Comparison")
        markdown.append("")
        markdown.append("| Strategy | Processing Time | Divisions | Summary Length |")
        markdown.append("|----------|-----------------|-----------|----------------|")
        
        for strategy, result in results.items():
            processing_time = f"{result.get('metadata', {}).get('processing_time_seconds', 0):.2f}s"
            divisions = result.get('metadata', {}).get('division_count', 0)
            summary_length = len(result.get('summary', ''))
            
            markdown.append(f"| {strategy} | {processing_time} | {divisions} | {summary_length:,} chars |")
        
        markdown.append("")
        
        # Add summaries for each strategy
        markdown.append("## Strategy Summaries")
        markdown.append("")
        
        for strategy, result in results.items():
            markdown.append(f"### {strategy.capitalize()} Strategy")
            markdown.append("")
            markdown.append(result.get('summary', 'No summary available.'))
            markdown.append("")
            markdown.append("---")
            markdown.append("")
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown))
        
        logger.info(f"Comparison stored at: {file_path}")
        return str(file_path)
    
    def list_outputs(self) -> List[Dict[str, Any]]:
        """List all saved outputs with metadata."""
        if not self.output_dir.exists():
            return []
        
        outputs = []
        for file in self.output_dir.glob("*.md"):
            # Extract info from filename
            filename = file.name
            date_match = filename.split("_")[-1].replace(".md", "")
            
            # Basic metadata
            output_info = {
                "path": str(file),
                "filename": filename,
                "date": date_match,
                "size": file.stat().st_size
            }
            
            # Try to extract title from the file
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('# '):
                        output_info["title"] = first_line[2:]
            except Exception:
                output_info["title"] = filename
            
            outputs.append(output_info)
        
        # Sort by date (newest first)
        return sorted(outputs, key=lambda x: x["date"], reverse=True)