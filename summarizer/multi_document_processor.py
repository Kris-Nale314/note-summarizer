"""
Multi-document summarization for earnings calls and sequential transcripts.
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from .core import TranscriptSummarizer
from .options import SummaryOptions

logger = logging.getLogger(__name__)

class MultiDocumentProcessor:
    """
    Specialized processor for handling multiple related documents, such as:
    - Earnings calls from different companies
    - Sequential earnings calls from the same company
    - Series of related meeting transcripts
    """
    
    def __init__(self, options: Optional[SummaryOptions] = None):
        """
        Initialize the multi-document processor with options.
        
        Args:
            options: Configuration options
        """
        self.options = options or SummaryOptions()
        self.summarizer = TranscriptSummarizer(options)
    
    def process_multiple_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple documents and generate a comparative analysis.
        
        Args:
            documents: List of document dictionaries, each containing:
                - 'text': The document text
                - 'metadata': Optional metadata dict with keys like 'company', 'date', 'title'
                
        Returns:
            Dictionary with combined summary and analysis
        """
        start_time = time.time()
        
        # Analyze document metadata to detect document type
        doc_type = self._detect_document_type(documents)
        logger.info(f"Detected document type: {doc_type}")
        
        # Process individual documents first
        individual_summaries = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}")
            
            # Use a context-aware approach for individual document summaries
            context_options = SummaryOptions(**vars(self.options))
            context_options.division_strategy = "context_aware"
            
            # Skip action items for individual summaries (we'll extract them later)
            context_options.include_action_items = False
            
            doc_summarizer = TranscriptSummarizer(context_options)
            result = doc_summarizer.summarize(doc['text'])
            
            # Add document metadata to result
            result['metadata'].update(doc.get('metadata', {}))
            individual_summaries.append(result)
        
        # Generate comparative analysis based on document type
        if doc_type == "earnings_calls_multi_company":
            final_result = self._analyze_multi_company_earnings(documents, individual_summaries)
        elif doc_type == "earnings_calls_sequential":
            final_result = self._analyze_sequential_earnings(documents, individual_summaries)
        else:
            # Generic multi-document analysis
            final_result = self._analyze_generic_documents(documents, individual_summaries)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        final_result['metadata']['processing_time_seconds'] = processing_time
        
        return final_result
    
    def _detect_document_type(self, documents: List[Dict[str, Any]]) -> str:
        """
        Detect the type of documents being processed.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Document type identifier string
        """
        # Check if documents have company names in metadata
        companies = set()
        dates = []
        
        # Look for metadata clues
        for doc in documents:
            metadata = doc.get('metadata', {})
            if 'company' in metadata:
                companies.add(metadata['company'])
            if 'date' in metadata:
                dates.append(metadata['date'])
        
        # If no metadata, try to extract from content
        if not companies:
            for doc in documents:
                # Common patterns in earnings call headers
                company_patterns = [
                    r"([A-Z][A-Za-z\s]+)(?:\,? Inc\.|\,? Corp\.|\,? Corporation|\,? Ltd\.)\s+(?:Q\d|Quarter)",
                    r"([A-Z][A-Za-z\s]+)(?:\,? Inc\.|\,? Corp\.|\,? Corporation|\,? Ltd\.)\s+Earnings\s+Call",
                ]
                
                for pattern in company_patterns:
                    matches = re.findall(pattern, doc['text'][:1000])  # Check only first 1000 chars
                    companies.update(matches)
                
                # Extract dates if not in metadata
                if not dates:
                    date_patterns = [
                        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s*,\s*\d{4}",
                        r"\d{1,2}/\d{1,2}/\d{4}",
                        r"\d{4}-\d{2}-\d{2}"
                    ]
                    
                    for pattern in date_patterns:
                        matches = re.findall(pattern, doc['text'][:1000])
                        if matches:
                            dates.extend(matches)
        
        # Check if keywords suggest earnings calls
        earnings_call_indicators = ['quarter', 'earnings', 'revenue', 'guidance', 'dividend', 'fiscal year', 'EPS']
        earnings_call_score = 0
        
        for doc in documents:
            doc_sample = doc['text'][:5000].lower()  # Check first 5000 chars
            for indicator in earnings_call_indicators:
                if indicator in doc_sample:
                    earnings_call_score += 1
        
        # Determine document type based on collected evidence
        if earnings_call_score > len(documents) * 2:  # Strong earnings call indicators
            if len(companies) > 1:
                return "earnings_calls_multi_company"
            elif len(documents) > 1:
                return "earnings_calls_sequential"
        
        # Default to generic document type
        return "generic_documents"
    
    def _analyze_multi_company_earnings(self, documents: List[Dict[str, Any]], 
                                       summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparative analysis for earnings calls from different companies.
        
        Args:
            documents: List of document dictionaries
            summaries: List of individual summary results
            
        Returns:
            Dictionary with combined summary and analysis
        """
        logger.info("Generating multi-company earnings call analysis")
        
        # Extract company names from metadata or summaries
        companies = []
        for summary in summaries:
            company = summary['metadata'].get('company', None)
            if not company:
                # Try to extract from summary
                first_line = summary['summary'].split('\n')[0]
                if 'summary' in first_line.lower():
                    company_match = re.search(r"([A-Z][A-Za-z\s]+)(?:,? Inc\.|\,? Corp\.|\,? Ltd\.)", first_line)
                    if company_match:
                        company = company_match.group(1).strip()
            
            companies.append(company or f"Company {len(companies)+1}")
        
        # Combine individual summaries with company context
        combined_summaries = "\n\n===== COMPANY SEPARATOR =====\n\n".join([
            f"COMPANY: {companies[i]}\n\n{summary['summary']}" 
            for i, summary in enumerate(summaries)
        ])
        
        # Create a comparative analysis prompt
        comparison_prompt = f"""Create a comparative analysis of these different companies' earnings calls.

The analysis should:
1. Start with a high-level executive summary
2. Compare key financial metrics across companies (revenue, earnings, growth, etc.)
3. Identify industry trends that appear across multiple companies
4. Highlight unique challenges or opportunities for each company
5. Compare forward-looking guidance or outlooks
6. Present a sector or market perspective based on all the information

FORMAT WITH MARKDOWN:
# Multi-Company Earnings Analysis

## Executive Summary

## Company Summaries
{', '.join([f'### {company}' for company in companies])}

## Comparative Analysis
### Financial Performance Comparison
### Industry Trends
### Forward Outlook

## Sector Implications

COMPANY SUMMARIES:
{combined_summaries}
"""

        # Generate the analysis
        try:
            comparative_analysis = self.summarizer.llm_client.generate_completion(comparison_prompt)
            
            # Extract action items across all companies
            action_items = None
            if self.options.include_action_items:
                action_items = self._extract_comparative_action_items(documents, companies)
            
            return {
                "summary": comparative_analysis,
                "action_items": action_items,
                "individual_summaries": [s['summary'] for s in summaries],
                "metadata": {
                    "document_count": len(documents),
                    "companies": companies,
                    "document_type": "earnings_calls_multi_company"
                }
            }
        except Exception as e:
            logger.error(f"Error generating comparative analysis: {e}")
            
            # Fallback to a simpler concatenation
            fallback = "# Multi-Company Earnings Analysis\n\n"
            fallback += "## Note\n\nThe comparative analysis could not be generated due to an error.\n\n"
            
            for i, company in enumerate(companies):
                fallback += f"## {company}\n\n{summaries[i]['summary']}\n\n"
                
            return {
                "summary": fallback,
                "individual_summaries": [s['summary'] for s in summaries],
                "metadata": {
                    "document_count": len(documents),
                    "companies": companies,
                    "document_type": "earnings_calls_multi_company",
                    "error": str(e)
                }
            }
    
    def _analyze_sequential_earnings(self, documents: List[Dict[str, Any]], 
                                   summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparative analysis for sequential earnings calls from the same company.
        
        Args:
            documents: List of document dictionaries
            summaries: List of individual summary results
            
        Returns:
            Dictionary with combined summary and analysis
        """
        logger.info("Generating sequential earnings call analysis")
        
        # Extract company name and periods
        company = None
        periods = []
        
        # Try to extract from metadata
        for i, summary in enumerate(summaries):
            if i == 0:
                company = summary['metadata'].get('company', None)
                if not company:
                    # Try to extract from summary
                    first_line = summary['summary'].split('\n')[0]
                    company_match = re.search(r"([A-Z][A-Za-z\s]+)(?:,? Inc\.|\,? Corp\.|\,? Ltd\.)", first_line)
                    if company_match:
                        company = company_match.group(1).strip()
            
            # Extract period (quarter/year)
            period = summary['metadata'].get('period', None)
            if not period:
                # Try to find quarter mention
                quarter_match = re.search(r"(Q\d\s+\d{4}|Quarter\s+\d\s+\d{4})", summary['summary'][:500])
                if quarter_match:
                    period = quarter_match.group(1)
                else:
                    # Use date as fallback
                    date = summary['metadata'].get('date', f"Period {i+1}")
                    period = date
            
            periods.append(period)
        
        company = company or "Company"
        
        # Sort summaries chronologically if dates are available
        if all('date' in s['metadata'] for s in summaries):
            # Convert dates to datetime objects for sorting
            date_summaries = []
            for i, summary in enumerate(summaries):
                try:
                    date_str = summary['metadata']['date']
                    if isinstance(date_str, str):
                        # Try multiple date formats
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y']:
                            try:
                                date = datetime.strptime(date_str, fmt)
                                date_summaries.append((date, summary, periods[i]))
                                break
                            except ValueError:
                                continue
                except (ValueError, KeyError):
                    # If date parsing fails, append with a default date
                    date_summaries.append((datetime(1900, 1, 1), summary, periods[i]))
            
            # Sort by date
            date_summaries.sort(key=lambda x: x[0])
            summaries = [ds[1] for ds in date_summaries]
            periods = [ds[2] for ds in date_summaries]
        
        # Combine individual summaries with period context
        combined_summaries = "\n\n===== PERIOD SEPARATOR =====\n\n".join([
            f"PERIOD: {periods[i]}\n\n{summary['summary']}" 
            for i, summary in enumerate(summaries)
        ])
        
        # Create a temporal analysis prompt
        temporal_prompt = f"""Create a temporal analysis of these earnings calls for {company} across multiple periods ({', '.join(periods)}).

The analysis should:
1. Start with an executive summary of the company's performance trajectory
2. Identify key financial metrics and their trends over time 
3. Highlight significant changes or shifts in business strategy
4. Track evolving challenges and opportunities
5. Analyze management's changing guidance or outlook
6. Provide insights into the company's overall direction

FORMAT WITH MARKDOWN:
# {company}: Temporal Analysis

## Executive Summary

## Performance Trends
### Financial Metrics Over Time
### Business Strategy Evolution
### Key Challenges and Opportunities

## Quarterly Breakdown
{', '.join([f'### {period}' for period in periods])}

## Future Outlook

PERIOD SUMMARIES:
{combined_summaries}
"""

        # Generate the analysis
        try:
            temporal_analysis = self.summarizer.llm_client.generate_completion(temporal_prompt)
            
            # Extract action items across all periods
            action_items = None
            if self.options.include_action_items:
                action_items = self._extract_temporal_action_items(documents, company, periods)
            
            return {
                "summary": temporal_analysis,
                "action_items": action_items,
                "individual_summaries": [s['summary'] for s in summaries],
                "metadata": {
                    "document_count": len(documents),
                    "company": company,
                    "periods": periods,
                    "document_type": "earnings_calls_sequential"
                }
            }
        except Exception as e:
            logger.error(f"Error generating temporal analysis: {e}")
            
            # Fallback to simpler output
            fallback = f"# {company}: Temporal Analysis\n\n"
            fallback += "## Note\n\nThe temporal analysis could not be generated due to an error.\n\n"
            
            for i, period in enumerate(periods):
                fallback += f"## {period}\n\n{summaries[i]['summary']}\n\n"
                
            return {
                "summary": fallback,
                "individual_summaries": [s['summary'] for s in summaries],
                "metadata": {
                    "document_count": len(documents),
                    "company": company,
                    "periods": periods,
                    "document_type": "earnings_calls_sequential",
                    "error": str(e)
                }
            }
    
    def _analyze_generic_documents(self, documents: List[Dict[str, Any]], 
                                  summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate analysis for generic document collections.
        
        Args:
            documents: List of document dictionaries
            summaries: List of individual summary results
            
        Returns:
            Dictionary with combined summary and analysis
        """
        logger.info("Generating generic multi-document analysis")
        
        # Extract document titles/identifiers
        titles = []
        for i, summary in enumerate(summaries):
            title = summary['metadata'].get('title', None)
            if not title:
                # Use first line as title or default
                first_line = summary['summary'].split('\n')[0].strip('#').strip()
                title = first_line if len(first_line) < 100 else f"Document {i+1}"
            
            titles.append(title)
        
        # Combine individual summaries
        combined_summaries = "\n\n===== DOCUMENT SEPARATOR =====\n\n".join([
            f"DOCUMENT: {titles[i]}\n\n{summary['summary']}" 
            for i, summary in enumerate(summaries)
        ])
        
        # Create a synthesis prompt
        synthesis_prompt = f"""Create a cohesive synthesis of these related documents.

The synthesis should:
1. Start with an executive summary of key themes across all documents
2. Organize information by major topics rather than by document
3. Highlight areas of consensus and disagreement
4. Identify connections and relationships between information in different documents
5. Present a unified perspective based on all the information

FORMAT WITH MARKDOWN:
# Multi-Document Synthesis

## Executive Summary

## Key Themes
### Theme 1
### Theme 2
### Theme 3

## Document-Specific Insights
{', '.join([f'### {title}' for title in titles])}

## Conclusions

DOCUMENT SUMMARIES:
{combined_summaries}
"""

        # Generate the synthesis
        try:
            synthesis = self.summarizer.llm_client.generate_completion(synthesis_prompt)
            
            # Extract action items across all documents
            action_items = None
            if self.options.include_action_items:
                action_items = self._extract_generic_action_items(documents, titles)
            
            return {
                "summary": synthesis,
                "action_items": action_items,
                "individual_summaries": [s['summary'] for s in summaries],
                "metadata": {
                    "document_count": len(documents),
                    "titles": titles,
                    "document_type": "generic_documents"
                }
            }
        except Exception as e:
            logger.error(f"Error generating document synthesis: {e}")
            
            # Fallback to simpler output
            fallback = "# Multi-Document Synthesis\n\n"
            fallback += "## Note\n\nThe document synthesis could not be generated due to an error.\n\n"
            
            for i, title in enumerate(titles):
                fallback += f"## {title}\n\n{summaries[i]['summary']}\n\n"
                
            return {
                "summary": fallback,
                "individual_summaries": [s['summary'] for s in summaries],
                "metadata": {
                    "document_count": len(documents),
                    "titles": titles,
                    "document_type": "generic_documents",
                    "error": str(e)
                }
            }
    
    def _extract_comparative_action_items(self, documents: List[Dict[str, Any]], 
                                         companies: List[str]) -> str:
        """
        Extract action items with company attribution.
        
        Args:
            documents: List of document dictionaries
            companies: List of company names
            
        Returns:
            Action items text
        """
        # Process each document for action items
        company_actions = []
        
        for i, (doc, company) in enumerate(zip(documents, companies)):
            prompt = f"""Extract action items, commitments, and key follow-ups from this {company} earnings call.
            
            For each item, identify:
            1. Who is responsible (executives, company)
            2. What specifically was promised or committed to
            3. Any timeline mentioned
            4. The context/importance of this item
            
            Format with markdown and attribute clearly to {company}.
            Focus on concrete commitments, not general statements.
            
            DOCUMENT:
            {doc['text'][:8000]}  # Process first portion for efficiency
            """
            
            try:
                result = self.summarizer.llm_client.generate_completion(prompt)
                company_actions.append(f"## {company} Action Items\n\n{result}")
            except Exception as e:
                logger.error(f"Error extracting action items for {company}: {e}")
                company_actions.append(f"## {company} Action Items\n\nError extracting action items.")
        
        # Combine all company action items
        combined_actions = "\n\n".join(company_actions)
        
        return f"# Multi-Company Action Items and Commitments\n\n{combined_actions}"
    
    def _extract_temporal_action_items(self, documents: List[Dict[str, Any]], 
                                      company: str, periods: List[str]) -> str:
        """
        Extract action items across time periods with status tracking.
        
        Args:
            documents: List of document dictionaries
            company: Company name
            periods: List of time periods
            
        Returns:
            Action items text
        """
        # Process each document for action items
        period_actions = []
        
        for i, (doc, period) in enumerate(zip(documents, periods)):
            prompt = f"""Extract action items, commitments, and key follow-ups from this {company} earnings call for {period}.
            
            For each item, identify:
            1. Who is responsible (specific executives if named)
            2. What specifically was promised or committed to
            3. Any timeline mentioned
            4. The context/importance of this item
            
            Format with markdown and attribute to {period}.
            Focus on concrete commitments, not general statements.
            
            DOCUMENT:
            {doc['text'][:8000]}  # Process first portion for efficiency
            """
            
            try:
                result = self.summarizer.llm_client.generate_completion(prompt)
                period_actions.append(f"## {period} Action Items\n\n{result}")
            except Exception as e:
                logger.error(f"Error extracting action items for {period}: {e}")
                period_actions.append(f"## {period} Action Items\n\nError extracting action items.")
        
        # If we have multiple periods, try to track commitment fulfillment
        commitment_tracking = ""
        if len(periods) > 1:
            # Combine all action items for analysis
            all_actions = "\n\n".join(period_actions)
            
            tracking_prompt = f"""Analyze these action items and commitments from {company} across multiple periods ({', '.join(periods)}).
            
            Create a commitment tracking summary that:
            1. Identifies commitments made in earlier periods and their status in later periods
            2. Highlights recurring commitments or themes
            3. Notes which commitments appear to have been fulfilled
            4. Identifies areas where guidance or commitments changed over time
            
            Format with markdown, focusing on significant commitments and their evolution.
            
            PERIOD ACTION ITEMS:
            {all_actions}
            """
            
            try:
                commitment_tracking = self.summarizer.llm_client.generate_completion(tracking_prompt)
            except Exception as e:
                logger.error(f"Error generating commitment tracking: {e}")
                commitment_tracking = "## Commitment Tracking\n\nError generating commitment tracking analysis."
        
        # Combine all period action items
        combined_actions = "\n\n".join(period_actions)
        
        if commitment_tracking:
            return f"# {company} Action Items and Commitments\n\n{commitment_tracking}\n\n## Detailed Action Items by Period\n\n{combined_actions}"
        else:
            return f"# {company} Action Items and Commitments\n\n{combined_actions}"
    
    def _extract_generic_action_items(self, documents: List[Dict[str, Any]], 
                                     titles: List[str]) -> str:
        """
        Extract action items from generic documents.
        
        Args:
            documents: List of document dictionaries
            titles: List of document titles
            
        Returns:
            Action items text
        """
        # Process each document for action items
        doc_actions = []
        
        for i, (doc, title) in enumerate(zip(documents, titles)):
            prompt = f"""Extract all action items, tasks, and commitments from this document.
            
            For each action item:
            1. Who is responsible
            2. What needs to be done
            3. Any deadline mentioned
            4. Context about why it's important
            
            Format as a prioritized, well-organized list with markdown.
            Don't miss any action items, even if implied.
            
            DOCUMENT TITLE: {title}
            
            DOCUMENT:
            {doc['text'][:8000]}  # Process first portion for efficiency
            """
            
            try:
                result = self.summarizer.llm_client.generate_completion(prompt)
                doc_actions.append(f"## {title} Action Items\n\n{result}")
            except Exception as e:
                logger.error(f"Error extracting action items for {title}: {e}")
                doc_actions.append(f"## {title} Action Items\n\nError extracting action items.")
        
        # Combine all document action items
        combined_actions = "\n\n".join(doc_actions)
        
        return f"# Action Items and Commitments\n\n{combined_actions}"