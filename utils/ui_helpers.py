"""
Common UI components and helpers for the Streamlit interface.
"""

import streamlit as st
import time
from typing import Dict, Any, Tuple, List, Optional, Callable
import pandas as pd

def get_logo_path():
    """Get the correct path to the logo."""
    import os
    
    # First, try the docs/images location
    if os.path.exists("docs/images/logo.svg"):
        return "docs/images/logo.svg"
    
    # Then try the root directory
    if os.path.exists("logo.svg"):
        return "logo.svg"
    
    # If running from pages directory, try one level up
    if os.path.exists("../docs/images/logo.svg"):
        return "../docs/images/logo.svg"
    
    # Default fallback
    return "docs/images/logo.svg"

def create_sidebar_options() -> Dict[str, Any]:
    """Create sidebar with common options."""
    with st.sidebar:
        # Use the logo path finding function
        st.image(get_logo_path(), width=180)
        st.markdown("## Configuration")
        
        # Model selection
        model_options = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"]
        model_name = st.selectbox(
            "Model", 
            model_options,
            index=1,
            help="Select the LLM to use for summarization"
        )
        
        # Division strategy
        strategy_options = [
            "basic", "speaker", "boundary", "context_aware", "semantic"
        ]
        strategy_descriptions = {
            "basic": "Simple division with smart paragraph breaks",
            "speaker": "Preserves speaker attribution in conversations",
            "boundary": "Respects document structure like headings",
            "context_aware": "Maintains semantic coherence",
            "semantic": "AI-powered topic-based chunking"
        }
        
        strategy = st.selectbox(
            "Division Strategy",
            strategy_options,
            index=0,
            format_func=lambda x: f"{x.capitalize()} - {strategy_descriptions[x]}",
            help="How to divide the document for processing"
        )
        
        # Include action items
        include_action_items = st.checkbox(
            "Extract Action Items",
            value=True,
            help="Extract tasks, commitments, and follow-ups"
        )
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Lower = more deterministic, Higher = more creative"
            )
            
            min_sections = st.number_input(
                "Minimum Sections",
                min_value=1,
                max_value=10,
                value=3,
                help="Minimum number of sections to divide into"
            )
            
            section_overlap = st.slider(
                "Section Overlap",
                min_value=0.0,
                max_value=0.3,
                value=0.1,
                step=0.05,
                help="Overlap between sections (as a percentage)"
            )
            
            verbose = st.checkbox(
                "Verbose Mode",
                value=False,
                help="Show detailed processing information"
            )
    
    return {
        "model_name": model_name,
        "division_strategy": strategy,
        "include_action_items": include_action_items,
        "temperature": temperature,
        "min_sections": min_sections,
        "section_overlap": section_overlap,
        "verbose": verbose
    }

def display_file_uploader(label: str = "Upload Document", 
                        types: List[str] = ["txt", "md", "docx"],
                        help_text: str = "Upload a document to summarize") -> Optional[Dict[str, Any]]:
    """Display file uploader with document info extraction."""
    uploaded_file = st.file_uploader(label, type=types, help=help_text)
    
    if not uploaded_file:
        return None
    
    # Extract document content
    try:
        content = None
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Handle DOCX
            try:
                import docx
                doc = docx.Document(uploaded_file)
                content = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                st.error("Please install python-docx to process Word documents: `pip install python-docx`")
                return None
        else:
            # Handle text files with encoding detection
            try:
                content = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                # Try other encodings
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode("latin-1")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode("cp1252")
        
        if not content:
            st.error("Could not read file content")
            return None
        
        # Extract document info
        doc_info = {
            "filename": uploaded_file.name,
            "title": uploaded_file.name.split(".")[0].replace("_", " ").title(),
            "content": content,
            "num_characters": len(content),
            "num_words": len(content.split()),
            "file_type": uploaded_file.type
        }
        
        # Try to detect speakers
        from summarizer.division import extract_speakers
        speakers = extract_speakers(content)
        if speakers:
            doc_info["speakers"] = speakers
            doc_info["num_speakers"] = len(speakers)
        
        # Display document info
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Characters", f"{doc_info['num_characters']:,}")
        col2.metric("Words", f"{doc_info['num_words']:,}")
        if "num_speakers" in doc_info:
            col3.metric("Speakers", doc_info["num_speakers"])
        
        return doc_info
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_progress_bar(label: str = "Processing document") -> Callable:
    """Create a progress bar with update function."""
    progress = st.progress(0, text=label)
    start_time = time.time()
    
    def update_progress(percent: float, custom_text: str = None):
        elapsed = time.time() - start_time
        if custom_text:
            progress.progress(percent, text=custom_text)
        else:
            progress.progress(percent, text=f"{label}: {percent:.0%} ({elapsed:.1f}s)")
    
    return update_progress

def display_processing_result(result: Dict[str, Any], document_info: Dict[str, Any],
                             display_divisions: bool = False) -> None:
    """Display processing results in a structured way."""
    if not result or "summary" not in result:
        st.error("No summary generated.")
        return
    
    # Display processing metadata
    processing_time = result.get("metadata", {}).get("processing_time_seconds", 0)
    division_count = result.get("metadata", {}).get("division_count", 0)
    division_strategy = result.get("metadata", {}).get("division_strategy", "unknown")
    model = result.get("metadata", {}).get("model", "unknown")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processing Time", f"{processing_time:.2f}s")
    col2.metric("Divisions", division_count)
    col3.metric("Strategy", division_strategy.capitalize())
    col4.metric("Model", model)
    
    # Create tabs for different views
    tabs = st.tabs(["📝 Summary", "✅ Action Items", "🔍 Document Divisions"])
    
    # Summary tab
    with tabs[0]:
        st.markdown(result["summary"])
        
        # Add copy and download options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copy Summary"):
                try:
                    import pyperclip
                    pyperclip.copy(result["summary"])
                    st.success("Copied to clipboard!")
                except ImportError:
                    st.warning("Could not copy. Install pyperclip: `pip install pyperclip`")
        
        with col2:
            title = document_info.get("title", "Summary")
            st.download_button(
                label="💾 Download Summary",
                data=result["summary"],
                file_name=f"{title.replace(' ', '_')}_summary.md",
                mime="text/markdown"
            )
    
    # Action Items tab
    with tabs[1]:
        if "action_items" in result and result["action_items"]:
            st.markdown(result["action_items"])
            
            if st.button("📋 Copy Action Items"):
                try:
                    import pyperclip
                    pyperclip.copy(result["action_items"])
                    st.success("Copied to clipboard!")
                except ImportError:
                    st.warning("Could not copy. Install pyperclip: `pip install pyperclip`")
        else:
            st.info("No action items extracted or action item extraction was disabled.")
    
    # Document Divisions tab
    with tabs[2]:
        if display_divisions and "divisions" in result:
            st.markdown("### Document Divisions")
            st.markdown(f"The document was divided into {len(result['divisions'])} sections using the **{division_strategy}** strategy.")
            
            for i, division in enumerate(result["divisions"]):
                with st.expander(f"Division {i+1}"):
                    st.text_area(
                        f"Section {i+1} Text",
                        division["text"],
                        height=200
                    )
                    
                    if "division_summaries" in result and i < len(result["division_summaries"]):
                        st.markdown("**Section Summary:**")
                        st.markdown(result["division_summaries"][i])
        else:
            st.info("Division details are not available or display is disabled.")

