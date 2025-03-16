"""
UI components for output storage functionality (simplified version).
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import datetime

def display_save_summary_ui(result: Dict[str, Any], document_info: Dict[str, Any] = None):
    """
    Display UI for saving a single document summary.
    
    Args:
        result: Summary result
        document_info: Document metadata
    """
    from output_store import OutputStore
    
    st.markdown("### Save Summary")
    
    # Create columns for save options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        title = st.text_input(
            "Summary Title",
            value=document_info.get("title", "Document Summary") if document_info else "Document Summary",
            help="Enter a title for this summary"
        )
    
    with col2:
        if st.button("💾 Save Summary", type="primary"):
            if title:
                try:
                    # Initialize the output store
                    output_store = OutputStore()
                    
                    # Store the summary
                    file_path = output_store.store_single_document_summary(title, result, document_info)
                    
                    # Show success message
                    st.success(f"Summary saved to {file_path}")
                except Exception as e:
                    st.error(f"Error saving summary: {str(e)}")
            else:
                st.warning("Please enter a title for the summary")

def display_save_multi_document_ui(result: Dict[str, Any], documents: List[Dict[str, Any]] = None):
    """
    Display UI for saving a multi-document analysis.
    
    Args:
        result: Analysis result
        documents: List of document metadata
    """
    from output_store import OutputStore
    
    st.markdown("### Save Analysis")
    
    # Create columns for save options
    col1, col2 = st.columns([3, 1])
    
    # Determine a good default title
    default_title = "Multi-Document Analysis"
    if result.get("metadata", {}).get("document_type") == "earnings_calls_multi_company":
        if "companies" in result.get("metadata", {}):
            companies = result["metadata"]["companies"]
            if len(companies) <= 3:
                default_title = f"{', '.join(companies)} Comparison"
            else:
                default_title = f"{len(companies)} Companies Comparison"
    elif result.get("metadata", {}).get("document_type") == "earnings_calls_sequential":
        company = result.get("metadata", {}).get("company", "")
        if company:
            default_title = f"{company} Sequential Analysis"
    
    with col1:
        title = st.text_input(
            "Analysis Title",
            value=default_title,
            help="Enter a title for this analysis"
        )
    
    with col2:
        if st.button("💾 Save Analysis", type="primary"):
            if title:
                try:
                    # Initialize the output store
                    output_store = OutputStore()
                    
                    # Store the analysis
                    file_path = output_store.store_multi_document_summary(title, result, documents)
                    
                    # Show success message
                    st.success(f"Analysis saved to {file_path}")
                except Exception as e:
                    st.error(f"Error saving analysis: {str(e)}")
            else:
                st.warning("Please enter a title for the analysis")

def display_save_comparison_ui(results: Dict[str, Dict[str, Any]], document_info: Dict[str, Any] = None):
    """
    Display UI for saving strategy comparison results.
    
    Args:
        results: Dictionary of strategy results
        document_info: Document metadata
    """
    from output_store import OutputStore
    
    st.markdown("### Save Comparison Results")
    
    # Create columns for save options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        title = st.text_input(
            "Comparison Title",
            value=document_info.get("title", "Strategy Comparison") if document_info else "Strategy Comparison",
            help="Enter a title for this comparison"
        )
    
    with col2:
        if st.button("💾 Save Comparison", type="primary"):
            if title:
                try:
                    # Initialize the output store
                    output_store = OutputStore()
                    
                    # Store the comparison
                    file_path = output_store.store_comparison(title, results, document_info)
                    
                    # Show success message
                    st.success(f"Comparison saved to {file_path}")
                except Exception as e:
                    st.error(f"Error saving comparison: {str(e)}")
            else:
                st.warning("Please enter a title for the comparison")

def list_saved_outputs():
    """
    List all saved outputs in the outputs directory.
    
    Returns:
        List of file paths
    """
    output_dir = Path("outputs")
    if not output_dir.exists():
        return []
    
    files = []
    for file in output_dir.glob("*.md"):
        files.append(str(file))
    
    return sorted(files)

def display_outputs_browser():
    """Display a simple browser for saved outputs."""
    st.markdown("## Saved Outputs")
    
    # List all saved outputs
    output_files = list_saved_outputs()
    
    if not output_files:
        st.info("No saved outputs found. Process documents and save the results!")
        return
    
    # Create a list of file names for the select box
    file_names = [os.path.basename(f) for f in output_files]
    
    # Let the user select a file
    selected_file = st.selectbox("Select output to view", file_names)
    
    if selected_file:
        # Get the full path of the selected file
        selected_path = os.path.join("outputs", selected_file)
        
        try:
            # Read the file content
            with open(selected_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Display the content
            st.markdown(content)
            
            # Provide download button
            st.download_button(
                label="Download Markdown",
                data=content,
                file_name=selected_file,
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")