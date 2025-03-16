"""
Streamlit UI components for multi-document processing.
"""
import streamlit as st
import os
import re
import time
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from wordcloud import WordCloud, STOPWORDS

from summarizer import TranscriptSummarizer, SummaryOptions
from summarizer.multi_document_processor import MultiDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_multi_document_ui() -> Tuple[Optional[List[Dict[str, Any]]], bool, Dict[str, Any]]:
    """
    Display UI for uploading and processing multiple documents.
    
    Returns:
        Tuple containing:
        - List of document dictionaries (or None if no documents)
        - Boolean indicating if process button was clicked
        - Dictionary of processing options
    """
    st.markdown("## Multi-Document Processing")
    
    st.markdown("""
    Process multiple related documents to generate comparative or sequential analysis:
    - Multiple earnings calls from different companies
    - Sequential earnings calls from the same company over time
    - Related meeting transcripts or documents
    """)
    
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload multiple documents", 
        type=["txt"], 
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("Please upload at least two document files to begin multi-document analysis.")
        with st.expander("How to prepare files for multi-document analysis"):
            st.markdown("""
            ### Tips for Multi-Document Analysis
            
            #### For earnings calls from different companies:
            1. Save each company's earnings call as a separate text file
            2. Include the company name in the filename (e.g., "Apple_Q3_2024.txt")
            3. Upload all files at once
            
            #### For sequential earnings calls from the same company:
            1. Save each quarter's earnings call as a separate text file
            2. Include the quarter/period in the filename (e.g., "Apple_Q1_2024.txt", "Apple_Q2_2024.txt")
            3. Upload all files in chronological order
            
            #### For best results:
            - Use consistent file naming
            - Ensure each file contains a single document
            - Include company name and date/period in the transcript if possible
            """)
        
        # Return empty values when no files uploaded
        empty_options = {
            "model_name": "gpt-3.5-turbo-16k",
            "include_action_items": True,
            "temperature": 0.3,
            "section_overlap": 0.1,
            "verbose": False,
            "document_type": "Auto-detect"
        }
        return None, False, empty_options
    
    # We have files, let's set up document processing options
    if len(uploaded_files) < 2:
        st.warning("Please upload at least two documents for multi-document analysis.")
        
        # Return empty values when not enough files
        empty_options = {
            "model_name": "gpt-3.5-turbo-16k",
            "include_action_items": True,
            "temperature": 0.3,
            "section_overlap": 0.1,
            "verbose": False,
            "document_type": "Auto-detect"
        }
        return None, False, empty_options
    
    # Document list for display
    st.markdown("### Documents for Analysis")
    
    # Process the uploaded files
    documents = []
    doc_metadata = []
    
    for i, file in enumerate(uploaded_files):
        try:
            # Try to read with different encodings
            try:
                content = file.read().decode("utf-8")
            except UnicodeDecodeError:
                # Try other encodings
                file.seek(0)
                try:
                    content = file.read().decode("latin-1")
                except UnicodeDecodeError:
                    file.seek(0)
                    content = file.read().decode("cp1252")
            
            # Basic document info
            doc_info = {
                "index": i,
                "filename": file.name,
                "size": len(content),
                "words": len(content.split()),
                "metadata": {}
            }
            
            # Try to extract metadata from filename
            filename = file.name.replace(".txt", "")
            
            # Look for company name and period in filename
            # Patterns: CompanyName_Q1_2024.txt or CompanyName_2024_Q1.txt
            company_period_match = re.search(r"([A-Za-z\s]+)[-_](?:Q(\d)[-_](\d{4})|(\d{4})[-_]Q(\d))", filename)
            
            if company_period_match:
                if company_period_match.group(2) and company_period_match.group(3):
                    # Pattern: CompanyName_Q1_2024
                    doc_info["metadata"]["company"] = company_period_match.group(1).replace("_", " ").strip()
                    doc_info["metadata"]["period"] = f"Q{company_period_match.group(2)} {company_period_match.group(3)}"
                elif company_period_match.group(4) and company_period_match.group(5):
                    # Pattern: CompanyName_2024_Q1
                    doc_info["metadata"]["company"] = company_period_match.group(1).replace("_", " ").strip()
                    doc_info["metadata"]["period"] = f"Q{company_period_match.group(5)} {company_period_match.group(4)}"
            else:
                # Just try to get the company name
                company_match = re.search(r"([A-Za-z\s]+)(?:[-_]|$)", filename)
                if company_match:
                    doc_info["metadata"]["company"] = company_match.group(1).replace("_", " ").strip()
            
            # Add to our lists
            documents.append({"text": content, "metadata": doc_info["metadata"]})
            doc_metadata.append(doc_info)
            
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
            logger.error(f"File processing error: {e}", exc_info=True)
    
    # Display document table
    doc_display_data = []
    for doc in doc_metadata:
        company = doc["metadata"].get("company", "Unknown")
        period = doc["metadata"].get("period", "")
        
        doc_display_data.append({
            "Document": f"{doc['index']+1}. {doc['filename']}",
            "Company": company,
            "Period": period,
            "Size": f"{doc['size']:,} chars",
            "Words": f"{doc['words']:,}"
        })
    
    st.dataframe(doc_display_data)
    
    # Allow user to edit metadata
    with st.expander("Edit Document Metadata (Optional)"):
        st.markdown("Adjust metadata if the automatic extraction wasn't correct:")
        
        for i, doc in enumerate(doc_metadata):
            st.markdown(f"**Document {i+1}: {doc['filename']}**")
            
            # Create columns for metadata fields
            col1, col2 = st.columns(2)
            
            with col1:
                company = st.text_input(
                    f"Company {i+1}",
                    value=doc["metadata"].get("company", ""),
                    key=f"company_{i}"
                )
                doc["metadata"]["company"] = company
                documents[i]["metadata"]["company"] = company
                
            with col2:
                period = st.text_input(
                    f"Period/Quarter {i+1}",
                    value=doc["metadata"].get("period", ""),
                    key=f"period_{i}"
                )
                doc["metadata"]["period"] = period
                documents[i]["metadata"]["period"] = period
            
            # Add date picker for precise dating
            date_value = None
            if "date" in doc["metadata"]:
                try:
                    date_str = doc["metadata"]["date"]
                    date_value = datetime.strptime(date_str, "%Y-%m-%d").date()
                except:
                    date_value = None
                    
            date = st.date_input(
                f"Date {i+1}",
                value=date_value,
                key=f"date_{i}"
            )
            
            doc["metadata"]["date"] = date.strftime("%Y-%m-%d") if date else ""
            documents[i]["metadata"]["date"] = doc["metadata"]["date"]
            
            st.markdown("---")
    
    # Document type hints
    document_type = st.radio(
        "Document Collection Type",
        ["Auto-detect", "Multi-company earnings calls", "Sequential earnings calls", "Generic documents"],
        help="Select the type of documents you're analyzing, or let the system detect it automatically."
    )
    
    # Analysis options
    st.markdown("### Analysis Options")
    
    # Model selection
    model_options = ["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"]
    model_name = st.selectbox(
        "Model", 
        model_options,
        index=0,
        help="Select the model to use for summarization. Multi-document analysis works best with models that have larger context windows."
    )
    
    # Include action items checkbox
    include_action_items = st.checkbox(
        "Extract Action Items",
        value=True,
        help="Extract and track action items, commitments, and follow-ups across documents"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Controls creativity: lower is more factual, higher is more creative"
        )
        
        section_overlap = st.slider(
            "Section Overlap",
            min_value=0.0,
            max_value=0.3,
            value=0.1,
            step=0.05,
            help="Overlap between document sections as a fraction of section size"
        )
        
        verbose_mode = st.checkbox(
            "Show Processing Details",
            value=False,
            help="Display detailed processing information"
        )
    
    # Process button
    process_button = st.button(
        "Process Documents",
        type="primary",
        help="Start the multi-document analysis with the selected settings"
    )
    
    # Return the documents, processing flag, and options
    options = {
        "model_name": model_name,
        "include_action_items": include_action_items,
        "temperature": temperature,
        "section_overlap": section_overlap,
        "verbose": verbose_mode,
        "document_type": document_type
    }
    
    return documents, process_button, options

def display_multi_document_results(result: Dict[str, Any], processing_time: float):
    """
    Display the results of multi-document analysis.
    
    Args:
        result: The analysis result dictionary
        processing_time: Total processing time in seconds
    """
    # Determine the document type
    doc_type = result["metadata"].get("document_type", "generic_documents")
    
    # Create appropriate tabs based on document type
    if doc_type == "earnings_calls_multi_company":
        tabs = st.tabs(["📊 Comparative Analysis", "🏢 Individual Companies", "✅ Action Items", "📈 Visualization"])
        companies = result["metadata"].get("companies", [])
    elif doc_type == "earnings_calls_sequential":
        tabs = st.tabs(["📊 Temporal Analysis", "📅 Individual Periods", "✅ Action Items", "📈 Visualization"])
        company = result["metadata"].get("company", "Company")
        periods = result["metadata"].get("periods", [])
    else:
        tabs = st.tabs(["📊 Synthesis", "📑 Individual Documents", "✅ Action Items", "📈 Visualization"])
        titles = result["metadata"].get("titles", [])
    
    # Main analysis tab
    with tabs[0]:
        # Display metrics
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Processing Time", f"{processing_time:.2f}s")
        metrics_cols[1].metric("Documents Analyzed", result["metadata"]["document_count"])
        
        if doc_type == "earnings_calls_multi_company":
            metrics_cols[2].metric("Companies", len(companies))
        elif doc_type == "earnings_calls_sequential":
            metrics_cols[2].metric("Time Periods", len(periods))
        
        # Display the main summary
        st.markdown("### Analysis")
        st.markdown(result["summary"])
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Copy Analysis"):
                try:
                    import pyperclip
                    pyperclip.copy(result["summary"])
                    st.success("Copied to clipboard!")
                except:
                    st.warning("Could not copy to clipboard. Please select and copy the text manually.")
        
        with col2:
            filename = ""
            if doc_type == "earnings_calls_multi_company":
                filename = "multi_company_analysis.md"
            elif doc_type == "earnings_calls_sequential":
                filename = f"{company}_temporal_analysis.md"
            else:
                filename = "document_synthesis.md"
                
            st.download_button(
                label="Download Analysis",
                data=result["summary"],
                file_name=filename,
                mime="text/markdown"
            )
    
    # Individual documents/companies/periods tab
    with tabs[1]:
        if doc_type == "earnings_calls_multi_company":
            for i, company in enumerate(companies):
                if i < len(result["individual_summaries"]):
                    with st.expander(f"{company}", expanded=False):
                        st.markdown(result["individual_summaries"][i])
                        
                        if st.button(f"Download {company} Summary", key=f"dl_company_{i}"):
                            st.download_button(
                                label=f"Download {company} Summary",
                                data=result["individual_summaries"][i],
                                file_name=f"{company}_summary.md",
                                mime="text/markdown",
                                key=f"dl_button_company_{i}"
                            )
        
        elif doc_type == "earnings_calls_sequential":
            for i, period in enumerate(periods):
                if i < len(result["individual_summaries"]):
                    with st.expander(f"{period}", expanded=False):
                        st.markdown(result["individual_summaries"][i])
                        
                        if st.button(f"Download {period} Summary", key=f"dl_period_{i}"):
                            st.download_button(
                                label=f"Download {period} Summary",
                                data=result["individual_summaries"][i],
                                file_name=f"{company}_{period}_summary.md", 
                                mime="text/markdown",
                                key=f"dl_button_period_{i}"
                            )
        
        else:  # generic_documents
            titles = result["metadata"].get("titles", [f"Document {i+1}" for i in range(len(result["individual_summaries"]))])
            for i, title in enumerate(titles):
                if i < len(result["individual_summaries"]):
                    with st.expander(f"{title}", expanded=False):
                        st.markdown(result["individual_summaries"][i])
                        
                        if st.button(f"Download {title} Summary", key=f"dl_doc_{i}"):
                            st.download_button(
                                label=f"Download Summary",
                                data=result["individual_summaries"][i],
                                file_name=f"{title.replace(' ', '_')}_summary.md",
                                mime="text/markdown",
                                key=f"dl_button_doc_{i}"
                            )
    
    # Action items tab
    with tabs[2]:
        if "action_items" in result and result["action_items"]:
            st.markdown("### Action Items and Commitments")
            st.markdown(result["action_items"])
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Copy Action Items"):
                    try:
                        import pyperclip
                        pyperclip.copy(result["action_items"])
                        st.success("Copied to clipboard!")
                    except:
                        st.warning("Could not copy to clipboard. Please select and copy the text manually.")
            
            with col2:
                filename = ""
                if doc_type == "earnings_calls_multi_company":
                    filename = "multi_company_action_items.md"
                elif doc_type == "earnings_calls_sequential":
                    filename = f"{company}_action_items.md"
                else:
                    filename = "document_action_items.md"
                    
                st.download_button(
                    label="Download Action Items",
                    data=result["action_items"],
                    file_name=filename,
                    mime="text/markdown"
                )
        else:
            st.info("No action items were extracted or action item extraction was disabled.")
    
    # Visualization tab
    with tabs[3]:
        st.markdown("### Document Analysis Visualization")
        
        # Combine all documents for overall word cloud
        all_text = " ".join([doc.get('text', '') for doc in result.get("documents", [])])
        
        if not all_text and "individual_summaries" in result:
            # Fall back to summaries if full documents not available
            all_text = " ".join(result["individual_summaries"])
        
        if all_text:
            try:
                # Set up stopwords
                stopwords = set(STOPWORDS)
                
                # Add common stopwords for business/earnings documents
                business_stopwords = {
                    "think", "know", "going", "yeah", "um", "uh", "like", "just",
                    "quarter", "year", "company", "business", "call", "percent",
                    "thank", "next", "question", "please", "will", "good", "great"
                }
                stopwords.update(business_stopwords)
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=150,
                    stopwords=stopwords,
                    contour_width=3
                ).generate(all_text)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                
                st.pyplot(fig)
                
                # Additional insights based on document type
                if doc_type == "earnings_calls_multi_company":
                    st.markdown("### Company Keyword Comparison")
                    # TODO: Add comparative keyword analysis across companies
                    
                elif doc_type == "earnings_calls_sequential":
                    st.markdown("### Temporal Keyword Evolution")
                    # TODO: Add visualization showing keyword changes over time
            
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
        else:
            st.info("Insufficient text data for visualization.")