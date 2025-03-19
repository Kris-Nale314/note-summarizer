"""
Specialized pass processors for extracting structured information from documents.
Implements a templated approach for easy creation of new pass types.
"""

import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from difflib import SequenceMatcher
import os

logger = logging.getLogger(__name__)

class PassProcessor:
    """
    Base class for document pass processors that extract specific types of information.
    """
    
    def __init__(self, llm_client, document_chunker):
        """
        Initialize the pass processor.
        
        Args:
            llm_client: LLM client for text processing
            document_chunker: DocumentChunker for splitting text
        """
        self.llm_client = llm_client
        self.document_chunker = document_chunker
        
        # Create output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    async def process_document(self, 
                             text: str, 
                             document_info: Optional[Dict[str, Any]] = None,
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a document using this pass.
        
        Args:
            text: Document text
            document_info: Optional document metadata from analysis
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Step 1: Chunk the document
        if progress_callback:
            progress_callback(0.1, "Chunking document...")
        
        chunks = self.document_chunker.chunk_document(
            text,
            min_chunks=3  # Default to 3 chunks minimum
        )
        
        # Step 2: Process each chunk with pass-specific instructions
        if progress_callback:
            progress_callback(0.2, "Processing document chunks...")
        
        chunk_results = await self._process_chunks(chunks, document_info, progress_callback)
        
        # Step 3: Synthesize results
        if progress_callback:
            progress_callback(0.8, "Synthesizing results...")
        
        # Apply pass-specific synthesis
        synthesis_result = await self._synthesize_results(chunk_results, document_info)
        
        # Step 4: Format results
        if progress_callback:
            progress_callback(0.9, "Formatting final output...")
        
        formatted_result = await self._format_results(synthesis_result, document_info)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        result = {
            "result": formatted_result,
            "document_info": document_info or {},
            "processing_metadata": {
                "pass_type": self.pass_type,
                "processing_time_seconds": processing_time,
                "chunks_processed": len(chunks),
                "timestamp": time.time()
            }
        }
        
        # Mark processing as complete
        if progress_callback:
            progress_callback(1.0, "Processing complete")
        
        return result
    
    async def _process_chunks(self, 
                           chunks: List[Dict[str, Any]], 
                           document_info: Optional[Dict[str, Any]],
                           progress_callback: Optional[Callable]) -> List[Dict[str, Any]]:
        """
        Process each chunk with pass-specific instructions.
        
        Args:
            chunks: List of document chunks
            document_info: Optional document metadata
            progress_callback: Optional progress callback
            
        Returns:
            List of chunk processing results
        """
        # Process chunks in parallel with concurrency limit
        tasks = []
        for chunk in chunks:
            tasks.append(self._process_chunk(chunk, document_info))
        
        # Process all chunks and track progress
        chunk_results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            chunk_results.append(result)
            
            # Update progress
            if progress_callback:
                chunk_progress = (i + 1) / len(chunks)
                progress_value = 0.2 + (chunk_progress * 0.6)  # Scale to 20%-80% range
                progress_callback(progress_value, f"Processed chunk {i+1}/{len(chunks)}")
        
        return chunk_results
    
    async def _process_chunk(self, 
                          chunk: Dict[str, Any], 
                          document_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single chunk with pass-specific instructions.
        
        Args:
            chunk: Document chunk with text and metadata
            document_info: Optional document metadata
            
        Returns:
            Dictionary with chunk processing results
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _process_chunk")
    
    async def _synthesize_results(self, 
                               chunk_results: List[Dict[str, Any]],
                               document_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize results from all chunks into a unified output.
        
        Args:
            chunk_results: List of chunk processing results
            document_info: Optional document metadata
            
        Returns:
            Dictionary with synthesized results
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _synthesize_results")
    
    async def _format_results(self, 
                           synthesis_result: Dict[str, Any],
                           document_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format the synthesized results for final output.
        
        Args:
            synthesis_result: Synthesized results
            document_info: Optional document metadata
            
        Returns:
            Dictionary with formatted results
        """
        # Default implementation just returns the synthesis result
        # Subclasses can override for custom formatting
        return synthesis_result
    
    def save_results(self, 
                    result: Dict[str, Any], 
                    filename_base: Optional[str] = None) -> Dict[str, Path]:
        """
        Save results to output directory.
        
        Args:
            result: Processing result to save
            filename_base: Optional base filename
            
        Returns:
            Dictionary with paths to saved files
        """
        from datetime import datetime
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        if filename_base:
            # Clean the filename base
            clean_base = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in filename_base)
            filename = f"{clean_base}_{self.pass_type}_{timestamp}"
        else:
            filename = f"{self.pass_type}_{timestamp}"
        
        # Create JSON version
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Create markdown version if formatter exists
        md_path = None
        if hasattr(self, '_create_markdown'):
            md_content = self._create_markdown(result)
            md_path = self.output_dir / f"{filename}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        return {
            'json': json_path,
            'markdown': md_path
        }
    
    def process_document_sync(self, 
                           text: str, 
                           document_info: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a document synchronously by running the async method in an event loop.
        
        Args:
            text: Document text
            document_info: Optional document metadata
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with processing results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_document(text, document_info, progress_callback)
            )
        finally:
            loop.close()


class TemplatedPass(PassProcessor):
    """
    A templated pass processor that can be easily customized with specific
    categories, instructions, and formatting requirements.
    """
    
    def __init__(self, 
                llm_client, 
                document_chunker,
                pass_config: Dict[str, Any]):
        """
        Initialize the templated pass.
        
        Args:
            llm_client: LLM client for text processing
            document_chunker: DocumentChunker for splitting text
            pass_config: Configuration dictionary with pass definition
        """
        super().__init__(llm_client, document_chunker)
        
        # Set pass type from config
        self.pass_type = pass_config.get("pass_type", "custom_pass")
        
        # Store configuration
        self.pass_config = pass_config
        
        # Extract common configurations
        self.purpose = pass_config.get("purpose", "Analyze document content")
        self.instructions = pass_config.get("instructions", "")
        self.categories = pass_config.get("categories", [])
        self.item_schema = pass_config.get("item_schema", {})
        self.output_format = pass_config.get("output_format", {})
        
        # Validation
        if not self.item_schema:
            logger.warning(f"No item schema defined for pass {self.pass_type}")
            # Set default schema
            self.item_schema = {
                "title": "string - Item title",
                "description": "string - Item description",
                "category": "string - Category from defined list",
                "priority": "string - Priority level"
            }
    
    async def _process_chunk(self, 
                          chunk: Dict[str, Any], 
                          document_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single chunk with pass-specific instructions.
        
        Args:
            chunk: Document chunk with text and metadata
            document_info: Optional document metadata
            
        Returns:
            Dictionary with chunk processing results
        """
        # Extract chunk metadata
        text = chunk.get('text', '')
        position = chunk.get('position', 'unknown')
        
        # Get context information
        is_transcript = document_info.get('is_meeting_transcript', False) if document_info else False
        client_name = document_info.get('client_name', '') if document_info else ''
        
        # Prepare category information
        category_text = ""
        if self.categories:
            category_text = "Use these specific categories for classification:\n"
            for category in self.categories:
                cat_name = category.get("name", "")
                cat_desc = category.get("description", "")
                category_text += f"- {cat_name}: {cat_desc}\n"
        
        # Create prompt with pass-specific instructions
        prompt = f"""
        {self.purpose}
        
        Document type: {"Meeting Transcript" if is_transcript else "Document"}
        {f"Client: {client_name}" if client_name else ""}
        Section position: {position}
        
        INSTRUCTIONS:
        {self.instructions}
        
        {category_text}
        
        For each item identified, provide:
        """
        
        # Add schema requirements to prompt
        for field, description in self.item_schema.items():
            prompt += f"- {field}: {description}\n"
        
        # Get item key
        item_key = self.pass_config.get("item_key", "items")
        
        # Add output format specification
        prompt += f"""
        Return your analysis in JSON format:
        {{
            "{item_key}": [
                {{
                    // Item fields following the schema above
                }},
                // additional items...
            ]
        }}
        
        TEXT TO ANALYZE:
        {text}
        """
        
        # Process with LLM
        try:
            response = await self.llm_client.generate_completion_async(prompt)
            
            # Parse response as JSON
            try:
                parsed_result = json.loads(response)
                # Add chunk position for synthesis
                parsed_result['chunk_position'] = position
                return parsed_result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for chunk in position {position}")
                return {
                    item_key: [],
                    "chunk_position": position,
                    "error": "Failed to parse response as JSON"
                }
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return {item_key: [], "chunk_position": position}
    
    async def _synthesize_results(self, 
                               chunk_results: List[Dict[str, Any]],
                               document_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize results from all chunks.
        
        Args:
            chunk_results: List of chunk processing results
            document_info: Optional document metadata
            
        Returns:
            Dictionary with combined results
        """
        # Collect all items from chunks
        all_items = []
        
        # Main result key, usually "items" but can be customized
        item_key = self.pass_config.get("item_key", "items")
        
        for chunk_result in chunk_results:
            items = chunk_result.get(item_key, [])
            if items:
                # Add chunk position to each item for reference
                chunk_position = chunk_result.get('chunk_position', 'unknown')
                for item in items:
                    item['source_position'] = chunk_position
                
                all_items.extend(items)
        
        # Deduplicate similar items if specified
        if self.pass_config.get("deduplicate", True):
            unique_items = self._deduplicate_items(all_items)
        else:
            unique_items = all_items
        
        # Count items by category
        categories = {}
        category_field = self.pass_config.get("category_field", "category")
        
        for item in unique_items:
            category = item.get(category_field, "Uncategorized")
            categories[category] = categories.get(category, 0) + 1
        
        # Create synthesis result
        synthesis_result = {
            item_key: unique_items,
            "categories": categories,
            f"total_{item_key}": len(unique_items)
        }
        
        return synthesis_result
    
    def _deduplicate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate similar items.
        
        Args:
            items: List of all items
            
        Returns:
            List of unique items
        """
        if not items:
            return []
        
        # Function to calculate text similarity
        def similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()
        
        # Get configuration for deduplication
        priority_field = self.pass_config.get("priority_field", "priority")
        title_field = self.pass_config.get("title_field", "title")
        description_field = self.pass_config.get("description_field", "description")
        similarity_threshold = self.pass_config.get("similarity_threshold", 0.6)
        
        # Sort items by priority (high to low) to prioritize important items
        priority_order = self.pass_config.get("priority_order", {"High": 3, "Medium": 2, "Low": 1})
        
        # Try to sort by priority if the field exists
        try:
            sorted_items = sorted(
                items, 
                key=lambda x: priority_order.get(x.get(priority_field, ""), 0), 
                reverse=True
            )
        except:
            # Fall back to original order if sorting fails
            sorted_items = items
        
        # Group similar items
        unique_items = []
        for item in sorted_items:
            # Check if this item is similar to any already in unique_items
            is_duplicate = False
            for unique in unique_items:
                # Get fields for comparison, defaulting to empty strings
                item_title = item.get(title_field, "")
                unique_title = unique.get(title_field, "")
                
                # Check title similarity
                title_sim = similarity(item_title, unique_title)
                
                # If titles are similar, check description similarity
                if title_sim > similarity_threshold:
                    # Get descriptions, limiting length for efficiency
                    item_desc = item.get(description_field, "")[:200]
                    unique_desc = unique.get(description_field, "")[:200]
                    
                    desc_sim = similarity(item_desc, unique_desc)
                    if desc_sim > similarity_threshold * 0.8:  # Slightly lower threshold for description
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_items.append(item)
        
        return unique_items
    
    async def _format_results(self, 
                           synthesis_result: Dict[str, Any],
                           document_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format the synthesized results for final output.
        
        Args:
            synthesis_result: Synthesized results
            document_info: Optional document metadata
            
        Returns:
            Dictionary with formatted results
        """
        # Get main item key and formatted items
        item_key = self.pass_config.get("item_key", "items")
        items = synthesis_result.get(item_key, [])
        categories = synthesis_result.get("categories", {})
        
        # Generate a summary if requested and items exist
        if self.pass_config.get("generate_summary", True) and items:
            # Use custom summary prompt if provided, otherwise generate a default one
            summary_prompt = self.pass_config.get("summary_prompt", "")
            
            if not summary_prompt:
                # Create a sample of items for the summary prompt
                item_sample = items[:min(10, len(items))]
                sample_text = ""
                
                # Generate sample text based on title field or fallback to the item itself
                title_field = self.pass_config.get("title_field", "title")
                for i, item in enumerate(item_sample):
                    item_title = item.get(title_field, f"Item {i+1}")
                    sample_text += f"{i+1}. {item_title}\n"
                
                if len(items) > 10:
                    sample_text += f"\n... and {len(items) - 10} more items"
                
                # Create default summary prompt
                summary_prompt = f"""
                You've analyzed a document and identified {len(items)} items across {len(categories)} categories:
                
                {sample_text}
                
                Please provide a brief summary (3-5 sentences) of the overall findings.
                Highlight major themes, patterns, and potential areas of focus.
                
                SUMMARY:
                """
            
            try:
                summary = await self.llm_client.generate_completion_async(summary_prompt)
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = f"Analysis identified {len(items)} items across {len(categories)} categories."
        else:
            summary = f"Analysis identified {len(items)} items across {len(categories)} categories."
        
        # Add the summary to the result
        formatted_result = {
            item_key: items,
            "categories": categories,
            f"total_{item_key}": len(items),
            "summary": summary
        }
        
        return formatted_result
    
    def _create_markdown(self, result: Dict[str, Any]) -> str:
        """
        Create a markdown version of the results.
        
        Args:
            result: Processing result
            
        Returns:
            Markdown formatted string
        """
        from datetime import datetime
        
        # Extract data
        formatted_result = result.get('result', {})
        document_info = result.get('document_info', {})
        
        # Get item key and items
        item_key = self.pass_config.get("item_key", "items")
        items = formatted_result.get(item_key, [])
        categories = formatted_result.get("categories", {})
        summary = formatted_result.get("summary", '')
        
        # Get field names for display
        title_field = self.pass_config.get("title_field", "title")
        description_field = self.pass_config.get("description_field", "description")
        category_field = self.pass_config.get("category_field", "category")
        priority_field = self.pass_config.get("priority_field", "priority")
        author_field = self.pass_config.get("author_field", "speaker")
        
        # Get display names
        title_display = self.pass_config.get("title_display", "Title")
        description_display = self.pass_config.get("description_display", "Description")
        category_display = self.pass_config.get("category_display", "Category")
        priority_display = self.pass_config.get("priority_display", "Priority")
        author_display = self.pass_config.get("author_display", "Raised by")
        
        # Create markdown
        report_title = self.pass_config.get("report_title", "Analysis Report")
        md = f"# {report_title}\n\n"
        md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n\n"
        
        # Add document info
        doc_type = "Meeting Transcript" if document_info.get('is_meeting_transcript', False) else "Document"
        md += f"**Document Type**: {doc_type}\n"
        
        if document_info.get('client_name'):
            md += f"**Client**: {document_info['client_name']}\n"
        
        md += "\n"
        
        # Add summary
        if summary:
            md += f"## Summary\n\n{summary}\n\n"
        
        # Add categories
        if categories:
            md += "## Categories\n\n"
            for category, count in categories.items():
                md += f"- **{category}**: {count} items\n"
            md += "\n"
        
        # Add items
        if items:
            md += f"## {self.pass_config.get('items_section_title', 'Identified Items')} ({len(items)})\n\n"
            
            # Group by priority if available
            if priority_field and any(priority_field in item for item in items):
                # Define priority groups (customizable)
                priority_groups = self.pass_config.get("priority_groups", ["High", "Medium", "Low"])
                
                for priority in priority_groups:
                    # Find items with this priority
                    priority_items = [i for i in items if i.get(priority_field, "").lower() == priority.lower()]
                    
                    if priority_items:
                        md += f"### {priority} Priority Items ({len(priority_items)})\n\n"
                        
                        for item in priority_items:
                            item_title = item.get(title_field, "Untitled Item")
                            md += f"#### {item_title}\n\n"
                            
                            # Add category if available
                            if category_field in item:
                                md += f"**{category_display}**: {item.get(category_field)}\n\n"
                            
                            # Add author if available
                            if author_field in item and item.get(author_field):
                                md += f"**{author_display}**: {item.get(author_field)}\n\n"
                            
                            # Add description
                            md += f"{item.get(description_field, 'No description provided')}\n\n"
                            md += "---\n\n"
            else:
                # No priority field, group by category instead
                if category_field and any(category_field in item for item in items):
                    # Get unique categories
                    unique_categories = sorted(set(item.get(category_field, "Uncategorized") for item in items))
                    
                    for category in unique_categories:
                        # Find items in this category
                        category_items = [i for i in items if i.get(category_field, "Uncategorized") == category]
                        
                        if category_items:
                            md += f"### {category} ({len(category_items)})\n\n"
                            
                            for item in category_items:
                                item_title = item.get(title_field, "Untitled Item")
                                md += f"#### {item_title}\n\n"
                                
                                # Add author if available
                                if author_field in item and item.get(author_field):
                                    md += f"**{author_display}**: {item.get(author_field)}\n\n"
                                
                                # Add description
                                md += f"{item.get(description_field, 'No description provided')}\n\n"
                                md += "---\n\n"
                else:
                    # No grouping, just list all items
                    for item in items:
                        item_title = item.get(title_field, "Untitled Item")
                        md += f"### {item_title}\n\n"
                        
                        # Add category if available
                        if category_field in item:
                            md += f"**{category_display}**: {item.get(category_field)}\n\n"
                        
                        # Add author if available
                        if author_field in item and item.get(author_field):
                            md += f"**{author_display}**: {item.get(author_field)}\n\n"
                        
                        # Add description
                        md += f"{item.get(description_field, 'No description provided')}\n\n"
                        md += "---\n\n"
        else:
            md += "## No Items Identified\n\n"
            md += "No items were identified in the document.\n\n"
        
        return md


# Issue identification pass configuration
ISSUE_IDENTIFICATION_CONFIG = {
    "pass_type": "issue_identification",
    "purpose": "Analyze this document to identify issues, challenges, and problems mentioned.",
    "instructions": """
    Look for explicit mentions of problems, challenges, issues, and roadblocks.
    Focus on specific, concrete issues rather than general observations.
    Be thorough in capturing both technical and non-technical issues.
    """,
    "categories": [
        {
            "name": "Strategy",
            "description": "Issues related to business strategy, planning, roadmaps, or vision"
        },
        {
            "name": "People",
            "description": "Issues related to talent, skills, training, organizational structure, or culture"
        },
        {
            "name": "Platforms",
            "description": "Issues related to software platforms, tools, applications, or systems"
        },
        {
            "name": "Infrastructure",
            "description": "Issues related to hardware, networks, cloud resources, or computing environment"
        },
        {
            "name": "Data",
            "description": "Issues related to data quality, availability, governance, or management"
        },
        {
            "name": "Adoption",
            "description": "Issues related to user adoption, change management, or utilization"
        }
    ],
    "item_schema": {
        "title": "string - Clear, concise issue title",
        "description": "string - Detailed explanation of the issue",
        "category": "string - Category from the list above",
        "speaker": "string - Who raised the issue (if known)",
        "severity": "string - High, Medium, or Low based on impact and urgency"
    },
    "item_key": "issues",
    "title_field": "title",
    "description_field": "description",
    "category_field": "category",
    "priority_field": "severity",
    "author_field": "speaker",
    "priority_order": {"High": 3, "Medium": 2, "Low": 1},
    "priority_groups": ["High", "Medium", "Low"],
    "report_title": "Issue Analysis Report",
    "items_section_title": "Identified Issues",
    "deduplicate": True,
    "similarity_threshold": 0.6,
    "generate_summary": True
}


# Data governance pass configuration
DATA_GOVERNANCE_CONFIG = {
    "pass_type": "data_governance",
    "purpose": "Analyze this document to assess data governance maturity across key dimensions.",
    "instructions": """
    Evaluate the organization's data governance practices across the defined dimensions.
    Look for evidence of policies, procedures, tools, and organizational practices.
    Assess maturity level for each dimension based on the evidence.
    """,
    "categories": [
        {
            "name": "Tools & Technology",
            "description": "Systems used for data management, metadata, quality, security"
        },
        {
            "name": "Policies & Standards",
            "description": "Data policies, procedures, standards, documentation"
        },
        {
            "name": "Communications",
            "description": "How data governance is communicated and understood"
        },
        {
            "name": "Roles & Responsibilities",
            "description": "Clarity around data ownership and stewardship"
        },
        {
            "name": "Data Quality",
            "description": "Approaches to ensuring data quality and integrity"
        }
    ],
    "item_schema": {
        "dimension": "string - Dimension name from the categories",
        "level": "string - Maturity level (Initial, Developing, Defined, Managed, Optimized)",
        "evidence": "string - Evidence from document supporting this assessment",
        "recommendations": ["string - Specific improvement recommendations"]
    },
    "item_key": "dimension_assessments",
    "title_field": "dimension",
    "description_field": "evidence",
    "category_field": "dimension",
    "priority_field": "level",
    "priority_order": {"Optimized": 5, "Managed": 4, "Defined": 3, "Developing": 2, "Initial": 1},
    "priority_groups": ["Optimized", "Managed", "Defined", "Developing", "Initial"],
    "report_title": "Data Governance Maturity Assessment",
    "items_section_title": "Dimension Assessments",
    "deduplicate": False,  # No need to deduplicate maturity assessments
    "generate_summary": True
}


# Function to create a pass processor by name
def create_pass_processor(pass_type: str, llm_client, document_chunker) -> PassProcessor:
    """
    Create a pass processor instance by type name.
    
    Args:
        pass_type: Type of pass processor to create
        llm_client: LLM client for text processing
        document_chunker: DocumentChunker for splitting text
        
    Returns:
        PassProcessor instance
    """
    if pass_type == "issue_identification":
        return TemplatedPass(llm_client, document_chunker, ISSUE_IDENTIFICATION_CONFIG)
    elif pass_type == "data_governance":
        return TemplatedPass(llm_client, document_chunker, DATA_GOVERNANCE_CONFIG)
    else:
        # Look for custom pass configuration
        pass_config = load_pass_config(pass_type)
        if pass_config:
            return TemplatedPass(llm_client, document_chunker, pass_config)
        else:
            raise ValueError(f"Unknown pass type: {pass_type}")


def load_pass_config(pass_type: str) -> Optional[Dict[str, Any]]:
    """
    Load a pass configuration from file.
    
    Args:
        pass_type: Type of pass to load
        
    Returns:
        Pass configuration dictionary or None if not found
    """
    # Create passes directory if it doesn't exist
    passes_dir = Path("passes")
    passes_dir.mkdir(exist_ok=True)
    
    # Look for a JSON configuration file
    config_path = passes_dir / f"{pass_type}.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading pass configuration: {e}")
    
    return None


def save_pass_config(pass_config: Dict[str, Any]) -> bool:
    """
    Save a pass configuration to file.
    
    Args:
        pass_config: Pass configuration dictionary
        
    Returns:
        Boolean indicating success
    """
    if "pass_type" not in pass_config:
        logger.error("Cannot save pass configuration without pass_type")
        return False
    
    # Create passes directory if it doesn't exist
    passes_dir = Path("passes")
    passes_dir.mkdir(exist_ok=True)
    
    # Save to JSON file
    try:
        config_path = passes_dir / f"{pass_config['pass_type']}.json"
        with open(config_path, 'w') as f:
            json.dump(pass_config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving pass configuration: {e}")
        return False


# Create a custom pass from configuration
def create_custom_pass(pass_config: Dict[str, Any], llm_client, document_chunker) -> TemplatedPass:
    """
    Create a custom pass from a configuration dictionary.
    
    Args:
        pass_config: Pass configuration dictionary
        llm_client: LLM client for text processing
        document_chunker: DocumentChunker for splitting text
        
    Returns:
        TemplatedPass instance
    """
    return TemplatedPass(llm_client, document_chunker, pass_config)


# List available pass types
def list_available_passes() -> List[Dict[str, str]]:
    """
    List all available pass types with descriptions.
    
    Returns:
        List of dictionaries with pass information
    """
    # Standard passes
    passes = [
        {
            "pass_type": "issue_identification",
            "name": "Issue Identification",
            "description": "Identifies and categorizes issues, challenges, and problems mentioned in the document."
        },
        {
            "pass_type": "data_governance",
            "name": "Data Governance Assessment",
            "description": "Assesses data governance maturity across key dimensions based on the DAMA-DMBOK framework."
        }
    ]
    
    # Look for custom passes
    passes_dir = Path("passes")
    if passes_dir.exists():
        for config_path in passes_dir.glob("*.json"):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                    # Extract info from config
                    pass_type = config.get("pass_type", config_path.stem)
                    name = config.get("report_title", pass_type.replace("_", " ").title())
                    description = config.get("purpose", "Custom document analysis pass")
                    
                    # Add to list if not already included
                    if not any(p["pass_type"] == pass_type for p in passes):
                        passes.append({
                            "pass_type": pass_type,
                            "name": name,
                            "description": description
                        })
            except Exception as e:
                logger.warning(f"Error loading custom pass from {config_path}: {e}")
    
    return passes


# Example function to create a new pass configuration
def create_opportunity_pass_config() -> Dict[str, Any]:
    """
    Create a sample opportunity identification pass configuration.
    
    Returns:
        Pass configuration dictionary
    """
    return {
        "pass_type": "opportunity_identification",
        "purpose": "Analyze this document to identify business opportunities and potential areas for growth.",
        "instructions": """
        Look for mentions of growth opportunities, potential improvements, and areas for innovation.
        Focus on identifying actionable opportunities rather than general discussion.
        Capture both explicitly stated opportunities and implied potential.
        """,
        "categories": [
            {
                "name": "Revenue Growth",
                "description": "Opportunities to increase revenue through new customers or services"
            },
            {
                "name": "Cost Reduction",
                "description": "Opportunities to reduce costs or improve efficiency"
            },
            {
                "name": "Process Improvement",
                "description": "Opportunities to improve internal processes or workflows"
            },
            {
                "name": "Innovation",
                "description": "Opportunities for new products, services, or business models"
            },
            {
                "name": "Market Expansion",
                "description": "Opportunities to enter new markets or segments"
            },
            {
                "name": "Customer Experience",
                "description": "Opportunities to improve customer satisfaction or engagement"
            }
        ],
        "item_schema": {
            "title": "string - Clear, concise opportunity title",
            "description": "string - Detailed explanation of the opportunity",
            "category": "string - Category from the list above",
            "source": "string - Who mentioned or what prompted this opportunity",
            "impact": "string - High, Medium, or Low potential business impact",
            "effort": "string - High, Medium, or Low implementation effort"
        },
        "item_key": "opportunities",
        "title_field": "title",
        "description_field": "description",
        "category_field": "category",
        "priority_field": "impact",
        "author_field": "source",
        "priority_order": {"High": 3, "Medium": 2, "Low": 1},
        "priority_groups": ["High", "Medium", "Low"],
        "report_title": "Opportunity Analysis Report",
        "items_section_title": "Identified Opportunities",
        "deduplicate": True,
        "similarity_threshold": 0.6,
        "generate_summary": True
    }