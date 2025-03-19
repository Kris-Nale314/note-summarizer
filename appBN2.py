"""
Streamlit application for Note-Summarizer with enhanced features.
Includes action item extraction and summary refinement.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import time
from datetime import datetime
from pathlib import Path
import json
import asyncio

# Import from lean architecture
from lean.options import ProcessingOptions
from lean.factory import SummarizerFactory
from lean.itemizer import ActionItemExtractor
from lean.refiner import SummaryRefiner

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Note-Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean dark theme
st.markdown("""
<style>
    /* Clean section headers */
    .section-header {
        margin: 1rem 0 0.75rem 0;
        font-weight: 600;
        border-bottom: 1px solid rgba(250, 250, 250, 0.2);
        padding-bottom: 0.5rem;
    }
    
    /* Keyword badges */
    .keyword-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        margin: 0.25rem;
        font-size: 0.85rem;
        background-color: rgba(67, 101, 236, 0.2);
    }
    
    /* Importance colors */
    .importance-1 { background-color: rgba(108, 117, 125, 0.3); }
    .importance-2 { background-color: rgba(67, 101, 236, 0.2); }
    .importance-3 { background-color: rgba(0, 123, 255, 0.25); }
    .importance-4 { background-color: rgba(255, 193, 7, 0.25); }
    .importance-5 { background-color: rgba(220, 53, 69, 0.25); }
    
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
    
    /* Logo styling */
    .sidebar-logo-container {
        margin: 1rem auto;
        text-align: center;
    }
    
    /* Button styling */
    .generate-btn {
        background-color: #4b6fff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .generate-btn:hover {
        background-color: #3a5ffd;
    }
    
    /* Refinement buttons */
    .refinement-button {
        background-color: rgba(67, 101, 236, 0.1);
        border: 1px solid rgba(67, 101, 236, 0.3);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        transition: all 0.2s;
        margin-right: 0.5rem;
    }
    
    .refinement-button:hover {
        background-color: rgba(67, 101, 236, 0.2);
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
    
    /* Processing info card */
    .info-card {
        background-color: rgba(67, 101, 236, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
    }
    
    /* Smooth progress bar */
    .stProgress > div > div {
        transition: width 0.3s;
    }
    
    /* Custom text area for user instructions */
    .user-instructions {
        border: 1px solid rgba(67, 101, 236, 0.3);
        border-radius: 6px;
        padding: 0.5rem;
        font-size: 0.9rem;
        background-color: rgba(67, 101, 236, 0.05);
    }
    
    /* Badge for refined summaries */
    .refined-badge {
        display: inline-block;
        background-color: rgba(67, 101, 236, 0.2);
        color: white;
        border-radius: 12px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Ensure outputs directory exists
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def try_decode_file(file_content, encodings=('utf-8', 'latin-1', 'cp1252', 'ascii', 'iso-8859-1')):
    """Try to decode file content with various encodings."""
    for encoding in encodings:
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # If all decoding attempts fail, use replacement mode with utf-8
    return file_content.decode('utf-8', errors='replace')


def create_pipeline(options):
    """Create the lean processing pipeline with specified options."""
    # Create the pipeline using the factory
    pipeline = SummarizerFactory.create_pipeline(api_key=api_key, options=options)
    return pipeline


async def process_text_async(text, options, progress_container):
    """Process the text asynchronously and return summary with enhanced metadata."""
    # Create the pipeline
    pipeline = create_pipeline(options)
    
    # Create progress bar and status
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    # Callback for progress updates
    def update_progress(progress, message):
        status_text.markdown(f"**Status:** {message}")
        progress_bar.progress(progress)
    
    # Process the text
    orchestrator = pipeline['orchestrator']
    result = await orchestrator.process_document(text, progress_callback=update_progress)
    
    # Clear progress indicators
    time.sleep(0.5)  # Brief pause so user can see completion
    progress_container.empty()
    
    return result


def process_text(text, options, progress_container):
    """Process text synchronously by running async function in event loop."""
    # Run the async function in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            process_text_async(text, options, progress_container)
        )
        return result
    finally:
        loop.close()


async def refine_summary_async(result, refinement_type, llm_client):
    """Refine a summary asynchronously."""
    # Create the refiner
    refiner = SummaryRefiner(llm_client)
    
    # Refine the summary
    refined_result = await refiner.refine_summary(result, refinement_type)
    
    return refined_result


def refine_summary(result, refinement_type, llm_client):
    """Refine a summary synchronously."""
    # Run the async function in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        refined_result = loop.run_until_complete(
            refine_summary_async(result, refinement_type, llm_client)
        )
        return refined_result
    finally:
        loop.close()


async def apply_user_instructions_async(result, user_instructions, llm_client):
    """Apply user instructions to a summary asynchronously."""
    # Create the refiner
    refiner = SummaryRefiner(llm_client)
    
    # Apply the user instructions
    refined_result = await refiner.incorporate_user_instructions(result, user_instructions)
    
    return refined_result


def apply_user_instructions(result, user_instructions, llm_client):
    """Apply user instructions to a summary synchronously."""
    # Run the async function in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        refined_result = loop.run_until_complete(
            apply_user_instructions_async(result, user_instructions, llm_client)
        )
        return refined_result
    finally:
        loop.close()


def save_summary(result, filename_base=None, detail_level="detailed"):
    """Save the summary to the outputs directory."""
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    if filename_base:
        # Clean the filename base
        clean_base = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in filename_base)
        filename = f"{clean_base}_{timestamp}"
    else:
        filename = f"summary_{timestamp}"
    
    # Extract summary and metadata
    summary_text = result['summary']
    doc_info = result.get('document_info', {})
    metadata = result.get('metadata', {})
    hier_meta = result.get('hierarchical_metadata', {})
    
    # Extract action items if available
    action_items = ""
    if 'action_items' in result:
        if isinstance(result['action_items'], list):
            action_items = "\n".join([f"- {item}" for item in result['action_items']])
        else:
            action_items = result['action_items']
    
    # Create the markdown summary
    summary_md = f"# Document Summary\n\n"
    summary_md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} with detail level: {detail_level}*\n\n"
    
    # Add refinement info if available
    if metadata.get('refinement_applied'):
        summary_md += f"*Refinement applied: {metadata.get('refinement_applied')}*\n\n"
    
    # Add executive summary if available
    if 'executive_summary' in result:
        summary_md += f"## Executive Summary\n\n{result['executive_summary']}\n\n"
    
    # Add main summary
    summary_md += f"## Full Summary\n\n{summary_text}\n\n"
    
    # Add action items if available
    if action_items:
        summary_md += f"## Action Items\n\n{action_items}\n\n"
    
    # Add key topics if available
    if 'key_topics' in result:
        key_topics = result['key_topics']
        if isinstance(key_topics, list):
            topics_str = ", ".join(key_topics)
            summary_md += f"## Key Topics\n\n{topics_str}\n\n"
    
    # Add processing details
    summary_md += f"## Processing Details\n\n"
    summary_md += f"- **Model:** {metadata.get('model', 'Unknown')}\n"
    summary_md += f"- **Processing Time:** {metadata.get('processing_time_seconds', 0):.2f} seconds\n"
    summary_md += f"- **Chunks Processed:** {metadata.get('chunks_processed', 0)}\n"
    
    # Add hierarchical metadata if available
    if hier_meta:
        summary_md += f"- **Hierarchical Levels:** {hier_meta.get('hierarchical_levels', 1)}\n"
        summary_md += f"- **Level 1 Summaries:** {hier_meta.get('level1_summaries', 0)}\n"
        if hier_meta.get('level2_summaries', 0) > 0:
            summary_md += f"- **Level 2 Summaries:** {hier_meta.get('level2_summaries', 0)}\n"
        if hier_meta.get('level3_summaries', 0) > 0:
            summary_md += f"- **Level 3 Summaries:** {hier_meta.get('level3_summaries', 0)}\n"
    
    # Add document type if available
    is_transcript = doc_info.get('is_meeting_transcript', False)
    summary_md += f"- **Document Type:** {'Meeting Transcript' if is_transcript else 'Document'}\n"
    
    # Save markdown file
    md_path = OUTPUTS_DIR / f"{filename}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    
    # Also save the raw result as JSON for possible future use
    json_path = OUTPUTS_DIR / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    
    return md_path


def get_previous_summaries():
    """Get list of previous summaries from the outputs directory."""
    md_files = list(OUTPUTS_DIR.glob("*.md"))
    
    summaries = []
    for file_path in md_files:
        # Get the creation time
        created_time = file_path.stat().st_ctime
        created_datetime = datetime.fromtimestamp(created_time)
        
        # Get the first line (title) from the file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                title = first_line.replace("# ", "") if first_line.startswith("# ") else file_path.stem
        except:
            title = file_path.stem
        
        summaries.append({
            "path": file_path,
            "title": title,
            "created": created_datetime,
            "filename": file_path.name
        })
    
    # Sort by created time (newest first)
    summaries.sort(key=lambda x: x["created"], reverse=True)
    return summaries


def get_logo_path():
    """Get the path to the logo file."""
    # Check for logo in docs/images
    logo_path = Path("docs/images/logo.svg")
    if logo_path.exists():
        return str(logo_path)
    
    # Check other common locations
    alternative_paths = [
        Path("logo.svg"),
        Path("../docs/images/logo.svg"),
        Path("assets/logo.svg")
    ]
    
    for path in alternative_paths:
        if path.exists():
            return str(path)
    
    # Return None if no logo found
    return None


def extract_keywords_from_result(result):
    """Extract keywords from result with their importance."""
    # Try different places keywords might be stored
    keywords = []
    
    # Check key_topics
    if 'key_topics' in result:
        topics = result['key_topics']
        if isinstance(topics, list):
            # Convert simple list to dict with default importance
            keywords.extend([{"keyword": k, "importance": 3} for k in topics])
    
    # Check document_info -> domain_categories
    doc_info = result.get('document_info', {})
    if 'domain_categories' in doc_info:
        domains = doc_info['domain_categories']
        if isinstance(domains, list):
            # Add domains with higher importance
            keywords.extend([{"keyword": d, "importance": 4} for d in domains])
    
    # Check keyword_frequencies if present (from hierarchical processing)
    if 'keyword_frequencies' in result:
        freq = result['keyword_frequencies']
        if isinstance(freq, dict):
            # Map frequency to importance (1-5 scale)
            for keyword, count in freq.items():
                # Normalize count to importance (simple approach)
                importance = min(5, max(1, int(count / 2) + 1))
                keywords.append({"keyword": keyword, "importance": importance})
    
    # Remove duplicates, keeping highest importance
    keyword_map = {}
    for item in keywords:
        keyword = item["keyword"]
        importance = item["importance"]
        
        if keyword in keyword_map:
            keyword_map[keyword] = max(keyword_map[keyword], importance)
        else:
            keyword_map[keyword] = importance
    
    # Convert back to list format
    unique_keywords = [{"keyword": k, "importance": v} for k, v in keyword_map.items()]
    
    # Sort by importance (high to low)
    unique_keywords.sort(key=lambda x: x["importance"], reverse=True)
    
    return unique_keywords


def display_keywords(keywords, max_display=20):
    """Display keywords as badges with importance-based styling."""
    if not keywords:
        return
    
    # Display limited number of keywords
    display_count = min(len(keywords), max_display)
    
    # Create HTML for keyword badges
    html = ""
    for i in range(display_count):
        keyword = keywords[i]["keyword"]
        importance = keywords[i]["importance"]
        html += f'<span class="keyword-badge importance-{importance}">{keyword}</span>'
    
    st.markdown(html, unsafe_allow_html=True)


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


def display_refinement_buttons(result, llm_client):
    """Display buttons for refining the summary."""
    col1, col2 = st.columns(2)
    
    with col1:
        more_detail = st.button("➕ Add More Detail", key="more_detail_btn")
        if more_detail:
            with st.spinner("Adding more detail..."):
                # Apply more detail refinement
                refined_result = refine_summary(result, "more_detail", llm_client)
                st.session_state.result = refined_result
                st.rerun()
    
    with col2:
        more_concise = st.button("➖ Make More Concise", key="more_concise_btn")
        if more_concise:
            with st.spinner("Making more concise..."):
                # Apply more concise refinement
                refined_result = refine_summary(result, "more_concise", llm_client)
                st.session_state.result = refined_result
                st.rerun()


def main():
    # Initialize session state for storing results
    if "result" not in st.session_state:
        st.session_state.result = None
    
    # Set up the sidebar
    with st.sidebar:
        # Logo at the top
        logo_path = get_logo_path()
        if logo_path:
            # Use markdown to apply CSS class instead of directly in st.image
            st.markdown('<div class="sidebar-logo-container">', unsafe_allow_html=True)
            st.image(logo_path, width=140)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.title("Note-Summarizer")
        
        st.markdown("---")
        
        # Core settings
        st.markdown("### Model")
        model_choice = st.selectbox(
            "Select AI model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="GPT-4 provides higher quality summaries but is slower and more expensive"
        )
        
        st.markdown("### Processing")
        min_chunks = st.slider(
            "Minimum Chunks", 
            min_value=1, 
            max_value=10, 
            value=3,
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
            captions=["Brief summary with key points", "Balanced summary with details", "Comprehensive with rich context"],
            index=1
        )
        
        # Display hierarchical levels visualization
        display_hierarchy_levels(detail_level)
        
        st.markdown("### Options")
        col1, col2 = st.columns(2)
        with col1:
            extract_actions = st.toggle("Action Items", value=True, help="Extract actionable tasks")
        with col2:
            use_cache = st.toggle("Use Cache", value=True, help="Cache results for speed")
        
        # User instructions
        st.markdown("### User Instructions")
        user_instructions = st.text_area(
            "Add specific instructions (optional)",
            placeholder="E.g., Focus on technical details, Emphasize customer feedback, etc.",
            key="user_instructions",
            help="These instructions will guide the summarization process"
        )
        
        # Previous summaries dropdown
        st.markdown("---")
        st.markdown("### Previous Summaries")
        summaries = get_previous_summaries()
        if summaries:
            selected_summary = st.selectbox(
                "Load previous summary",
                options=[""] + [f"{s['title']} ({s['created'].strftime('%m/%d %H:%M')})" for s in summaries],
                format_func=lambda x: "Select a summary..." if x == "" else x
            )
            
            # If a summary is selected, load it
            if selected_summary != "":
                selected_index = [f"{s['title']} ({s['created'].strftime('%m/%d %H:%M')})" for s in summaries].index(selected_summary)
                selected_path = summaries[selected_index]["path"]
                
                if st.button("Load Selected Summary"):
                    st.session_state["view_summary"] = selected_path
                    st.rerun()
        else:
            st.info("No previous summaries found")

    # Main content area
    st.markdown("## 📝 Transform Messy Transcripts into Useful Notes")
    
    # File uploader spanning ~80% of the window
    st.markdown('<div class="uploader-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop a transcript or document file",
        type=["txt", "md"],
        help="Microsoft Teams transcript files work best"
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
    
    # Generate button
    if input_text:
        process_clicked = st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True
        )
    else:
        process_clicked = False
        st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True,
            disabled=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show selected summary if any
    if "view_summary" in st.session_state:
        summary_path = st.session_state["view_summary"]
        
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_content = f.read()
            
            # Try to load JSON version for metadata
            json_path = summary_path.with_suffix('.json')
            metadata = None
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    pass
        except:
            pass    
            # Display the summary
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            
            # Display tabs for the summary content
            summary_tabs = st.tabs(["📄 Summary", "✨ Details"])
            
            with summary_tabs[0]:
                # Display the keywords if available in metadata
                if metadata:
                    keywords = extract_keywords_from_result(metadata)
                    if keywords:
                        st.markdown("### Key Topics")
                        display_keywords(keywords)
                
                st.markdown(summary_content)
                
                # Download button
                st.download_button(
                    label="Download Summary",
                    data=summary_content,
                    file_name=summary_path.name,
                    mime="text/markdown"
                )
            
            with summary_tabs[1]:
                if metadata:
                    # Show hierarchical processing info
                    st.markdown("### Hierarchical Processing")
                    
                    if 'hierarchical_metadata' in metadata:
                        hier_data = metadata['hierarchical_metadata']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Hierarchical Levels", hier_data.get('hierarchical_levels', 1))
                        
                        with col2:
                            st.metric("Chunks Processed", metadata.get('metadata', {}).get('chunks_processed', 0))
                        
                        with col3:
                            processing_time = metadata.get('metadata', {}).get('processing_time_seconds', 0)
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                    
                    # Show document info
                    st.markdown("### Document Information")
                    
                    doc_info = metadata.get('document_info', {})
                    is_transcript = doc_info.get('is_meeting_transcript', False)
                    
                    st.info(f"Document Type: {'Meeting Transcript' if is_transcript else 'Document'}")
                    
                    # Show meeting purpose if available
                    if 'meeting_purpose' in doc_info and doc_info['meeting_purpose']:
                        st.markdown(f"**Meeting Purpose:** {doc_info['meeting_purpose']}")
                    
                    # Show client name if available
                    if 'client_name' in doc_info and doc_info['client_name']:
                        st.markdown(f"**Client:** {doc_info['client_name']}")
                    
                    # Show key topics if available in document context
                    if 'key_topics' in doc_info and doc_info['key_topics']:
                        topics = ", ".join(doc_info['key_topics'])
                        st.markdown(f"**Key Topics:** {topics}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clear the viewed summary after showing it
            if st.button("Clear Summary"):
                del st.session_state["view_summary"]
                st.rerun()
    
    # Process document if button clicked
    elif process_clicked and input_text:
        # Create a container for the output
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        
        # Create a container for progress updates
        progress_container = st.container()
        
        # Create options based on selected settings
        options = ProcessingOptions(
            model_name=model_choice,
            temperature=temperature,
            min_chunks=min_chunks,
            detail_level=detail_level,
            include_action_items=extract_actions,
            max_concurrent_chunks=3,  # Fixed for simplicity
            include_metadata=True,
            user_instructions=user_instructions if user_instructions else None
        )
        
        # Process the text
        with st.spinner("Processing document..."):
            start_time = time.time()
            result = process_text(input_text, options, progress_container)
            
            # Save result in session state for refinement
            st.session_state.result = result
            
            # Auto-save the summary
            saved_path = save_summary(result, filename_base, detail_level)
            
            # Display success message
            processing_time = result.get('metadata', {}).get('processing_time_seconds', time.time() - start_time)
            st.success(f"Summary generated in {processing_time:.2f} seconds")
            
            # Display summary tabs
            result_tabs = st.tabs(["📄 Summary", "✅ Action Items", "🔍 Insights"])
            
            with result_tabs[0]:
                # If there's an executive summary, display it first
                if 'executive_summary' in result:
                    st.markdown("### Executive Summary")
                    st.markdown(result["executive_summary"])
                    st.markdown("---")
                
                # Extract keywords from result
                keywords = extract_keywords_from_result(result)
                
                # Display keywords if available
                if keywords:
                    st.markdown("### Key Topics")
                    display_keywords(keywords)
                
                # Check if this is a refined summary
                refinement_applied = result.get('metadata', {}).get('refinement_applied')
                if refinement_applied:
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <span class="refined-badge">{refinement_applied}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### Full Summary")
                st.markdown(result["summary"])
                
                # Display refinement buttons
                pipeline = create_pipeline(options)
                llm_client = pipeline['llm_client']
                display_refinement_buttons(result, llm_client)
                
                # Download button
                st.download_button(
                    label="Download Summary",
                    data=result["summary"],
                    file_name=f"{filename_base}_summary.md" if filename_base else "summary.md",
                    mime="text/markdown"
                )
            
            with result_tabs[1]:
                if extract_actions:
                    action_items_text = ""
                    if 'action_items' in result:
                        action_items = result['action_items']
                        if isinstance(action_items, list):
                            if action_items:
                                st.markdown("### Action Items")
                                for item in action_items:
                                    st.markdown(f"- {item}")
                                action_items_text = "\n".join([f"- {item}" for item in action_items])
                            else:
                                st.info("No action items identified in this document.")
                        else:
                            st.markdown(action_items)
                            action_items_text = action_items
                    
                    if action_items_text:
                        # Download button
                        st.download_button(
                            label="Download Action Items",
                            data=action_items_text,
                            file_name=f"{filename_base}_action_items.md" if filename_base else "action_items.md",
                            mime="text/markdown"
                        )
                    else:
                        st.info("No action items were extracted.")
                else:
                    st.info("Action item extraction was disabled.")
            
            with result_tabs[2]:
                # Display hierarchical processing info
                st.markdown("### Hierarchical Processing")
                
                if 'hierarchical_metadata' in result:
                    hier_data = result['hierarchical_metadata']
                    
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Hierarchical Levels", hier_data.get('hierarchical_levels', 1))
                    
                    with col2:
                        st.metric("Level 1 Summaries", hier_data.get('level1_summaries', 0))
                    
                    with col3:
                        if hier_data.get('hierarchical_levels', 1) >= 2:
                            st.metric("Level 2 Summaries", hier_data.get('level2_summaries', 0))
                        else:
                            st.metric("Level 2 Summaries", "N/A")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show document context
                doc_info = result.get('document_info', {})
                
                # Check if this is a transcript
                is_transcript = doc_info.get('is_meeting_transcript', False)
                st.info(f"Document Type: {'Meeting Transcript' if is_transcript else 'Document'}")
                
                if is_transcript:
                    st.markdown("### Meeting Analysis")
                    
                    # Show meeting purpose if available
                    if 'meeting_purpose' in doc_info and doc_info['meeting_purpose']:
                        st.markdown(f"**Meeting Purpose:** {doc_info['meeting_purpose']}")
                    
                    # Show client name if available
                    if 'client_name' in doc_info and doc_info['client_name']:
                        st.markdown(f"**Client:** {doc_info['client_name']}")
                    
                    # Show participants if available
                    if 'participants' in doc_info and doc_info['participants']:
                        participants = ", ".join(doc_info['participants'])
                        st.markdown(f"**Participants:** {participants}")
                
                # Show main conclusions if available
                if 'main_conclusions' in result and result['main_conclusions']:
                    st.markdown("### Main Conclusions")
                    conclusions = result['main_conclusions']
                    if isinstance(conclusions, list):
                        for item in conclusions:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown(conclusions)
                
                # Show keyword frequencies if available
                if 'keyword_frequencies' in result:
                    st.markdown("### Keyword Analysis")
                    
                    # Convert to a list of (word, frequency) pairs and sort
                    freq_data = [(word, freq) for word, freq in result['keyword_frequencies'].items()]
                    freq_data.sort(key=lambda x: x[1], reverse=True)
                    
                    # Display as a table
                    freq_df = {"Keyword": [], "Frequency": []}
                    for word, freq in freq_data[:15]:  # Display top 15
                        freq_df["Keyword"].append(word)
                        freq_df["Frequency"].append(freq)
                    
                    st.dataframe(freq_df)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display existing result from previous refinement if available
    elif st.session_state.result is not None:
        result = st.session_state.result
        
        # Display the summary
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        
        # Display tabs for the summary content
        result_tabs = st.tabs(["📄 Summary", "✅ Action Items", "🔍 Insights"])
        
        with result_tabs[0]:
            # If there's an executive summary, display it first
            if 'executive_summary' in result:
                st.markdown("### Executive Summary")
                st.markdown(result["executive_summary"])
                st.markdown("---")
            
            # Extract keywords from result
            keywords = extract_keywords_from_result(result)
            
            # Display keywords if available
            if keywords:
                st.markdown("### Key Topics")
                display_keywords(keywords)
            
            # Check if this is a refined summary
            refinement_applied = result.get('metadata', {}).get('refinement_applied')
            if refinement_applied:
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <span class="refined-badge">{refinement_applied}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Full Summary")
            st.markdown(result["summary"])
            
            # Display refinement buttons
            options = ProcessingOptions(model_name="gpt-3.5-turbo")  # Default for refinement
            pipeline = create_pipeline(options)
            llm_client = pipeline['llm_client']
            display_refinement_buttons(result, llm_client)
            
            # Download button
            st.download_button(
                label="Download Summary",
                data=result["summary"],
                file_name="summary.md",
                mime="text/markdown"
            )
        
        with result_tabs[1]:
            if 'action_items' in result:
                action_items = result['action_items']
                if isinstance(action_items, list):
                    if action_items:
                        st.markdown("### Action Items")
                        for item in action_items:
                            st.markdown(f"- {item}")
                        action_items_text = "\n".join([f"- {item}" for item in action_items])
                        
                        # Download button
                        st.download_button(
                            label="Download Action Items",
                            data=action_items_text,
                            file_name="action_items.md",
                            mime="text/markdown"
                        )
                    else:
                        st.info("No action items identified in this document.")
                else:
                    st.markdown(action_items)
            else:
                st.info("No action items were extracted.")
        
        with result_tabs[2]:
            # Display hierarchical processing info
            st.markdown("### Hierarchical Processing")
            
            if 'hierarchical_metadata' in result:
                hier_data = result['hierarchical_metadata']
                
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Hierarchical Levels", hier_data.get('hierarchical_levels', 1))
                
                with col2:
                    st.metric("Level 1 Summaries", hier_data.get('level1_summaries', 0))
                
                with col3:
                    if hier_data.get('hierarchical_levels', 1) >= 2:
                        st.metric("Level 2 Summaries", hier_data.get('level2_summaries', 0))
                    else:
                        st.metric("Level 2 Summaries", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show document context
            doc_info = result.get('document_info', {})
            
            # Check if this is a transcript
            is_transcript = doc_info.get('is_meeting_transcript', False)
            st.info(f"Document Type: {'Meeting Transcript' if is_transcript else 'Document'}")
        
        # Clear button to reset state
        if st.button("Clear Results"):
            st.session_state.result = None
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display a placeholder if no file loaded and no summary
    # Simple placeholder when nothing is loaded or processed
        st.markdown('<div class="output-container" style="text-align: center; padding: 3rem 1rem;">', unsafe_allow_html=True)
        
        # Display placeholder with hierarchical processing highlight
        st.markdown("""
        <div style="opacity: 0.7; max-width: 600px; margin: 0 auto;">
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            <h3 style="margin-top: 1.5rem;">Upload a document to get started</h3>
            <p style="margin-top: 0.5rem;">
                Our lean architecture uses hierarchical processing to create summaries that are both detailed and coherent.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a simple visualization of how it works
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h4 style="opacity: 0.8;">How Hierarchical Processing Works</h4>
            <div style="display: flex; justify-content: space-between; max-width: 600px; margin: 1.5rem auto; text-align: center;">
                <div style="flex: 1;">
                    <div style="background-color: rgba(67, 101, 236, 0.2); border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                        <span style="font-size: 1.5rem;">1</span>
                    </div>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">Smart document chunking</p>
                </div>
                <div style="flex: 1;">
                    <div style="background-color: rgba(67, 101, 236, 0.3); border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                        <span style="font-size: 1.5rem;">2</span>
                    </div>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">Multi-level synthesis</p>
                </div>
                <div style="flex: 1;">
                    <div style="background-color: rgba(67, 101, 236, 0.4); border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                        <span style="font-size: 1.5rem;">3</span>
                    </div>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">Coherent final summary</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def create_app(pipeline=None):
    """
    Create a Streamlit app with the provided pipeline.
    This function allows the app to be imported and used in other applications.
    
    Args:
        pipeline: Optional pre-configured pipeline components
        
    Returns:
        The Streamlit app function
    """
    # Store the pipeline in session state if provided
    if pipeline:
        import streamlit as st
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = pipeline
    
    # Return the main function
    return main


if __name__ == "__main__":
    main()