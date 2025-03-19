"""
Data Governance Assessment Page for the Note-Summarizer app.
This page analyzes documents to assess data governance maturity.
"""

import os
import streamlit as st
import time
import asyncio
from datetime import datetime
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Import from lean architecture
from lean.async_openai_adapter import AsyncOpenAIAdapter
from lean.document import DocumentAnalyzer
from lean.chunker import DocumentChunker
from lean.passes import create_pass_processor

# Page configuration
st.set_page_config(
    page_title="Data Governance Assessment - Note-Summarizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean dark theme (reusing styles from main app)
st.markdown("""
<style>
    /* Clean section headers */
    .section-header {
        margin: 1rem 0 0.75rem 0;
        font-weight: 600;
        border-bottom: 1px solid rgba(250, 250, 250, 0.2);
        padding-bottom: 0.5rem;
    }
    
    /* Maturity level colors */
    .maturity-initial { background-color: rgba(220, 53, 69, 0.3); color: white; padding: 0.2rem 0.5rem; border-radius: 12px; }
    .maturity-developing { background-color: rgba(255, 193, 7, 0.3); color: white; padding: 0.2rem 0.5rem; border-radius: 12px; }
    .maturity-defined { background-color: rgba(23, 162, 184, 0.3); color: white; padding: 0.2rem 0.5rem; border-radius: 12px; }
    .maturity-managed { background-color: rgba(0, 123, 255, 0.3); color: white; padding: 0.2rem 0.5rem; border-radius: 12px; }
    .maturity-optimized { background-color: rgba(40, 167, 69, 0.3); color: white; padding: 0.2rem 0.5rem; border-radius: 12px; }
    
    /* Container for the file upload */
    .uploader-container {
        width: 80%;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Container for the output */
    .output-container {
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* Dimension card */
    .dimension-card {
        background-color: rgba(67, 101, 236, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid rgba(67, 101, 236, 0.6);
    }
    
    .dimension-card-initial {
        border-left: 4px solid rgba(220, 53, 69, 0.6);
    }
    
    .dimension-card-developing {
        border-left: 4px solid rgba(255, 193, 7, 0.6);
    }
    
    .dimension-card-defined {
        border-left: 4px solid rgba(23, 162, 184, 0.6);
    }
    
    .dimension-card-managed {
        border-left: 4px solid rgba(0, 123, 255, 0.6);
    }
    
    .dimension-card-optimized {
        border-left: 4px solid rgba(40, 167, 69, 0.6);
    }
    
    /* Maturity scale */
    .maturity-scale {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
    }
    
    .maturity-scale-item {
        text-align: center;
        padding: 0.5rem;
        flex: 1;
        font-size: 0.9rem;
    }
    
    .maturity-scale-item.active {
        font-weight: bold;
        border-bottom: 3px solid;
    }
    
    .maturity-initial-item {
        border-bottom-color: rgba(220, 53, 69, 0.8);
    }
    
    .maturity-developing-item {
        border-bottom-color: rgba(255, 193, 7, 0.8);
    }
    
    .maturity-defined-item {
        border-bottom-color: rgba(23, 162, 184, 0.8);
    }
    
    .maturity-managed-item {
        border-bottom-color: rgba(0, 123, 255, 0.8);
    }
    
    .maturity-optimized-item {
        border-bottom-color: rgba(40, 167, 69, 0.8);
    }
    
    /* Logo styling */
    .sidebar-logo-container {
        margin: 1rem auto;
        text-align: center;
    }
    
    /* Hierarchy level indicator */
    .hierarchy-level {
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        text-align: center;
        line-height: 24px;
        margin-right: 0.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        background-color: rgba(67, 101, 236, 0.2);
    }
    
    .level-active {
        background-color: rgba(67, 101, 236, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Function to decode file with various encodings
def try_decode_file(file_content, encodings=('utf-8', 'latin-1', 'cp1252', 'ascii', 'iso-8859-1')):
    """Try to decode file content with various encodings."""
    for encoding in encodings:
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # If all decoding attempts fail, use replacement mode with utf-8
    return file_content.decode('utf-8', errors='replace')

def display_hierarchy_levels(detail_level):
    """Display hierarchical processing levels as simple dots."""
    # Define levels for each detail level
    levels = {
        "essential": 1,
        "detailed": 2,
        "detailed-complex": 3
    }
    
    max_levels = 3
    active_levels = levels.get(detail_level, 2)
    
    html = '<div style="display: flex; align-items: center; margin: 0.5rem 0;">'
    html += '<span style="margin-right: 0.5rem;">Hierarchical levels: </span>'
    
    for i in range(1, max_levels + 1):
        active_class = "level-active" if i <= active_levels else ""
        html += f'<span class="hierarchy-level {active_class}">{i}</span>'
    
    html += '</div>'
    
    return st.markdown(html, unsafe_allow_html=True)

async def analyze_document_async(text, options, progress_container=None):
    """
    Analyze a document to assess data governance maturity.
    
    Args:
        text: Document text
        options: Processing options
        progress_container: Container for progress updates
        
    Returns:
        Processing results
    """
    # Create progress bar and status if container provided
    progress_bar = None
    status_text = None
    
    if progress_container:
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
    
    # Update progress function
    def update_progress(progress, message):
        if progress_bar and status_text:
            status_text.markdown(f"**Status:** {message}")
            progress_bar.progress(progress)
    
    try:
        # Initialize components with proper settings
        llm_client = AsyncOpenAIAdapter(
            model=options["model_name"],
            temperature=options["temperature"]
        )
        document_analyzer = DocumentAnalyzer(llm_client)
        document_chunker = DocumentChunker()
        
        # Step 1: Analyze document to get context
        update_progress(0.1, "Analyzing document context...")
        document_info = await document_analyzer.analyze_preview(text)
        
        # Step 2: Create pass processor
        pass_processor = create_pass_processor("data_governance", llm_client, document_chunker)
        
        # Step 3: Determine chunk settings based on detail level
        min_chunks = options["min_chunks"]
        detail_level = options["detail_level"]
        
        # Scale min_chunks based on detail level
        if detail_level == "detailed":
            scaled_min_chunks = min_chunks * 2
        elif detail_level == "detailed-complex":
            scaled_min_chunks = min_chunks * 3
        else:  # essential
            scaled_min_chunks = min_chunks
            
        # Override document_chunker's chunk method to use our settings
        original_chunk_method = document_chunker.chunk_document
        
        # Create a wrapper for the chunk method
        def custom_chunk_method(text, min_chunks=3, max_chunk_size=None):
            return original_chunk_method(text, min_chunks=scaled_min_chunks, max_chunk_size=max_chunk_size)
        
        # Replace the method temporarily
        document_chunker.chunk_document = custom_chunk_method
        
        # Step 4: Process the document
        result = await pass_processor.process_document(text, document_info, update_progress)
        
        # Restore original chunk method
        document_chunker.chunk_document = original_chunk_method
        
        # Add processing options to the result
        result["processing_options"] = options
        
        return result
    
    except Exception as e:
        if progress_container:
            progress_container.error(f"Error processing document: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    finally:
        # Clear progress indicators
        if progress_container:
            time.sleep(0.5)  # Brief pause so user can see completion
            progress_container.empty()

def analyze_document(text, options, progress_container=None):
    """Run the async analysis function synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            analyze_document_async(text, options, progress_container)
        )
    finally:
        loop.close()

def display_maturity_scale(level):
    """
    Display a visual maturity scale with the current level highlighted.
    
    Args:
        level: Current maturity level
    """
    # Normalize the level
    level = level.lower() if level else "unknown"
    
    # Define all levels in order
    levels = ["initial", "developing", "defined", "managed", "optimized"]
    
    # Create HTML for scale
    html = '<div class="maturity-scale">'
    
    for lvl in levels:
        active_class = "active" if lvl == level else ""
        item_class = f"maturity-scale-item {active_class} maturity-{lvl}-item" if lvl == level else "maturity-scale-item"
        html += f'<div class="{item_class}">{lvl.title()}</div>'
    
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

def display_dimension_card(dim_name, dim_data):
    """
    Display a dimension assessment in a card format.
    
    Args:
        dim_name: Name of the dimension
        dim_data: Dimension assessment data
    """
    # Get dimension details
    level = dim_data.get('level', 'Unknown')
    evidence = dim_data.get('evidence', 'No evidence provided')
    recommendations = dim_data.get('recommendations', [])
    
    # Normalize the level for CSS classes
    level_lower = level.lower() if level else "unknown"
    
    # Create HTML for the card
    card_class = f"dimension-card dimension-card-{level_lower}"
    level_class = f"maturity-{level_lower}"
    
    html = f"""
    <div class="{card_class}">
        <h3>{dim_name}</h3>
        <div style="margin-bottom: 0.8rem;">
            <span class="{level_class}">{level}</span>
        </div>
        <h4>Evidence</h4>
        <p>{evidence}</p>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Display recommendations
    if recommendations:
        with st.expander(f"Recommendations for {dim_name}"):
            for rec in recommendations:
                st.markdown(f"- {rec}")

def display_results(result):
    """
    Display processing results in the UI.
    
    Args:
        result: Processing result dictionary
    """
    # Extract document info and result data
    document_info = result.get('document_info', {})
    assessment_data = result.get('result', {})
    metadata = result.get('processing_metadata', {})
    processing_options = result.get('processing_options', {})
    
    # Extract assessment details
    maturity_assessment = assessment_data.get('maturity_assessment', {})
    overall_assessment = assessment_data.get('overall_assessment', {})
    
    # Document information
    st.markdown("## Document Information")
    
    # Display document type
    doc_type = "Meeting Transcript" if document_info.get('is_meeting_transcript', False) else "Document"
    st.markdown(f"**Type**: {doc_type}")
    
    # Display client name if available
    if document_info.get('client_name'):
        st.markdown(f"**Client**: {document_info['client_name']}")
    
    # Display key topics if available
    if document_info.get('key_topics'):
        st.markdown(f"**Key Topics**: {', '.join(document_info['key_topics'])}")
    
    # Display processing info
    st.markdown("---")
    
    # Display processing options
    if processing_options:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", processing_options.get("model_name", "Unknown"))
        with col2:
            st.metric("Detail Level", processing_options.get("detail_level", "Unknown"))
        with col3:
            st.metric("Processing Time", f"{metadata.get('processing_time_seconds', 0):.2f}s")
    
    # Display overall assessment
    if overall_assessment:
        st.markdown("## Overall Data Governance Maturity")
        
        level = overall_assessment.get('level', 'Unknown')
        summary = overall_assessment.get('summary', 'No summary provided')
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"### {level}")
        with col2:
            # Display maturity scale
            display_maturity_scale(level)
        
        st.markdown(summary)
    
    # Display dimension assessments
    if maturity_assessment:
        st.markdown("## Dimension Assessments")
        
        # Define dimension display names
        dimensions = [
            ('tools_technology', 'Tools & Technology'),
            ('policies_standards', 'Policies & Standards'),
            ('communications', 'Communications'),
            ('roles_responsibilities', 'Roles & Responsibilities'),
            ('data_quality', 'Data Quality')
        ]
        
        # Create tabs for each dimension
        dimension_tabs = st.tabs([name for _, name in dimensions])
        
        # Populate each tab
        for i, (dim_key, dim_name) in enumerate(dimensions):
            if dim_key in maturity_assessment:
                with dimension_tabs[i]:
                    dim_data = maturity_assessment[dim_key]
                    display_dimension_card(dim_name, dim_data)
    
    # Display strengths and gaps
    if overall_assessment:
        st.markdown("## Strengths & Gaps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Key Strengths")
            strengths = overall_assessment.get('key_strengths', [])
            if strengths:
                for strength in strengths:
                    st.markdown(f"- {strength}")
            else:
                st.info("No specific strengths identified")
        
        with col2:
            st.markdown("### Key Gaps")
            gaps = overall_assessment.get('key_gaps', [])
            if gaps:
                for gap in gaps:
                    st.markdown(f"- {gap}")
            else:
                st.info("No specific gaps identified")
        
        # Display priority recommendations
        recommendations = overall_assessment.get('priority_recommendations', [])
        if recommendations:
            st.markdown("### Priority Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")

def main():
    # Set up the sidebar
    with st.sidebar:
        # Logo at the top
        logo_path = "docs/images/logo.svg"  # Update with your logo path
        try:
            # Use markdown to apply CSS class
            st.markdown('<div class="sidebar-logo-container">', unsafe_allow_html=True)
            st.image(logo_path, width=140)
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.title("Note-Summarizer")
        
        st.markdown("---")
        
        # Core settings
        st.markdown("### Model")
        model_choice = st.selectbox(
            "Select AI model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="GPT-4 provides higher quality analysis but is slower and more expensive"
        )
        
        st.markdown("### Processing")
        min_chunks = st.slider(
            "Minimum Chunks", 
            min_value=1, 
            max_value=10, 
            value=2,
            help="Minimum number of chunks to divide text into"
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.05,
            help="Higher values make output more creative, lower more deterministic"
        )
        
        st.markdown("### Detail Level")
        detail_level = st.radio(
            "Select detail level",
            ["essential", "detailed", "detailed-complex"],
            captions=["Brief analysis with key dimensions", "Balanced analysis with details", "Comprehensive with rich context"],
            index=1
        )
        
        # Display hierarchical levels visualization
        display_hierarchy_levels(detail_level)
        
        st.markdown("### Framework")
        framework = st.selectbox(
            "Assessment Framework",
            ["DAMA-DMBOK", "CMMI Data Management", "Data Management Capability Assessment Model (DCAM)"],
            index=0,
            help="Framework to use for assessment criteria"
        )
    
    # Title and description
    st.markdown("# 📊 Data Governance Assessment")
    st.markdown(
        """
        Analyze documents to assess data governance maturity across key dimensions.
        This specialized processor evaluates your organization's data governance practices
        using industry-standard frameworks.
        """
    )
    
    # File uploader
    st.markdown('<div class="uploader-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop a transcript or document file",
        type=["txt", "md"],
        help="Documents discussing data strategies, governance, or management practices"
    )
    
    input_text = ""
    filename_base = None
    if uploaded_file:
        try:
            # Read file content as bytes
            file_content = uploaded_file.read()
            
            # Try to decode with multiple encodings
            input_text = try_decode_file(file_content)
            filename_base = uploaded_file.name.split('.')[0]
            
            # Show success message
            st.success(f"Loaded {len(input_text):,} characters from {uploaded_file.name}")
            
            # Preview in expander
            with st.expander("Preview document"):
                st.text_area("Document content", input_text[:2000] + ("..." if len(input_text) > 2000 else ""), height=200)
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Analysis button
    if input_text:
        analyze_clicked = st.button(
            "Assess Data Governance",
            type="primary",
            use_container_width=True,
            key="analyze_button"
        )
    else:
        analyze_clicked = False
        st.button(
            "Assess Data Governance",
            type="primary",
            use_container_width=True,
            disabled=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process document if button clicked
    if analyze_clicked and input_text:
        # Create a container for the output
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        
        # Create a container for progress updates
        progress_container = st.container()
        
        # Create options dictionary from settings
        options = {
            "model_name": model_choice,
            "temperature": temperature,
            "min_chunks": min_chunks,
            "detail_level": detail_level,
            "framework": framework
        }
        
        # Analyze the document
        with st.spinner("Assessing data governance maturity..."):
            result = analyze_document(input_text, options, progress_container)
            
            # Display the results
            display_results(result)
            
            # Get the pass processor to save results
            llm_client = AsyncOpenAIAdapter(model=model_choice, temperature=temperature)
            document_chunker = DocumentChunker()
            pass_processor = create_pass_processor("data_governance", llm_client, document_chunker)
            
            # Save results if successful
            saved_paths = pass_processor.save_results(result, filename_base)
            
            # Add download buttons if files were saved
            col1, col2 = st.columns(2)
            
            with col1:
                if 'markdown' in saved_paths and saved_paths['markdown']:
                    with open(saved_paths['markdown'], 'r') as f:
                        md_content = f.read()
                    
                    st.download_button(
                        label="Download Assessment (Markdown)",
                        data=md_content,
                        file_name=saved_paths['markdown'].name,
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with col2:
                if 'json' in saved_paths and saved_paths['json']:
                    with open(saved_paths['json'], 'r') as f:
                        json_content = f.read()
                    
                    st.download_button(
                        label="Download Assessment (JSON)",
                        data=json_content,
                        file_name=saved_paths['json'].name,
                        mime="application/json",
                        key="json_download",
                        use_container_width=True
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()