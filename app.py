"""
Main Streamlit application for Note-Summarizer.
Clean, minimalist design with dark mode support, logo integration and auto-save feature.
Fixed for latest Streamlit API.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import time
from datetime import datetime
from pathlib import Path
import json

# Import directly from summarizer package
from summarizer import SummaryOptions, TranscriptSummarizer

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Note-Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a cleaner, more modern interface with dark mode support
st.markdown("""
<style>
    /* Main header styling - works in both light and dark mode */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Sub-header styling - works in both light and dark mode */
    .sub-header {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.8;
        text-align: center;
    }
    
    /* Card styling - works in both light and dark mode */
    .card {
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Make file uploader more prominent */
    .stFileUploader > div:first-child {
        width: 100%;
    }
    
    /* Give the file uploader a cleaner look */
    .stFileUploader > div:first-child > div:first-child {
        border-radius: 8px;
        border: 2px dashed rgba(128, 128, 128, 0.3);
        padding: 1rem;
    }
    
    /* Better button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Output area styling */
    .output-area {
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-top: 1rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        border-radius: 8px 8px 0 0;
        padding: 0 1rem;
        font-weight: 600;
    }
    
    /* Add more space to tab content */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1rem 0;
    }
    
    /* Success message styling */
    .success-message {
        background-color: rgba(0, 200, 0, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 3rem;
        text-align: center;
        opacity: 0.7;
        font-size: 0.875rem;
    }
    
    /* Make text in the app cleaner */
    p, li {
        line-height: 1.6;
    }
    
    /* Center logo */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    /* Logo size */
    .logo {
        width: 180px;
        height: auto;
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


def process_text(text, model_name, include_action_items, progress_container):
    """Process the text and return summary."""
    options = SummaryOptions(
        model_name=model_name,
        include_action_items=include_action_items
    )
    
    # Initialize the summarizer
    summarizer = TranscriptSummarizer(options)
    
    # Create progress bar
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    # Callback for progress updates
    def update_progress(message, progress):
        status_text.markdown(f"**Status:** {message}")
        progress_bar.progress(progress)
    
    # Process the text
    result = summarizer.summarize(text, progress_callback=update_progress)
    
    # Clear progress indicators
    time.sleep(0.5)  # Brief pause so user can see completion
    progress_container.empty()
    
    return result


def save_summary(result, filename_base=None):
    """Save the summary to the outputs directory."""
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    if filename_base:
        # Clean the filename base
        clean_base = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in filename_base)
        filename = f"{clean_base}_{timestamp}"
    else:
        filename = f"summary_{timestamp}"
    
    # Create the markdown summary
    summary_md = f"# Transcript Summary\n\n"
    summary_md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n\n"
    summary_md += f"## Summary\n\n{result['summary']}\n\n"
    
    if result.get('action_items'):
        summary_md += f"## Action Items\n\n{result['action_items']}\n\n"
    
    # Add metadata
    summary_md += f"## Processing Details\n\n"
    summary_md += f"- **Model:** {result['metadata'].get('model', 'Unknown')}\n"
    summary_md += f"- **Processing Time:** {result['metadata'].get('processing_time_seconds', 0):.2f} seconds\n"
    summary_md += f"- **Sections:** {result['metadata'].get('division_count', 0)}\n"
    
    # Save markdown file
    md_path = OUTPUTS_DIR / f"{filename}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    
    # Also save the raw result as JSON for possible future use
    json_path = OUTPUTS_DIR / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": result["summary"],
            "action_items": result.get("action_items"),
            "metadata": result["metadata"]
        }, f, indent=2)
    
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


def main():
    # Initialize session state for tab selection
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Create Summary"
    
    # Logo
    logo_path = get_logo_path()
    if logo_path:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image(logo_path, use_container_width=False, width=180)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Header (simplified if logo is present)
    if logo_path:
        st.markdown('<p class="sub-header">Transform meeting transcripts into clear, actionable notes</p>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 class="main-header">Note-Summarizer</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Transform meeting transcripts into clear, actionable notes</p>', unsafe_allow_html=True)

    # Create tabs for main interface
    tabs = st.tabs(["📝 Create Summary", "📚 Previous Summaries"])
    
    # Set the active tab
    tab_index = 0 if st.session_state["active_tab"] == "Create Summary" else 1
    
    # Create Summary tab
    with tabs[0]:
        if tab_index == 0:  # Only render content if this tab is active
            # Use columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                st.subheader("Upload Transcript")
                
                uploaded_file = st.file_uploader(
                    "Drag and drop a transcript file",
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
                        st.markdown(
                            f'<div class="success-message">✅ Loaded {len(input_text):,} characters from {uploaded_file.name}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Show a small preview of the file
                        with st.expander("Preview transcript"):
                            st.text(input_text[:750] + "..." if len(input_text) > 750 else input_text)
                            
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
                
                # Divider
                st.markdown("---")
                
                # Model selection
                st.subheader("Settings")
                
                model_choice = st.selectbox(
                    "Model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    index=0,
                    help="GPT-4 provides higher quality summaries but costs more and processes slower"
                )
                
                include_action_items = st.checkbox(
                    "Extract action items", 
                    value=True,
                    help="Identify tasks, commitments, and follow-ups"
                )
                
                # Process button - only enable if a file is loaded
                st.markdown("<br>", unsafe_allow_html=True)  # Add some space
                process_button = st.button(
                    "Generate Summary", 
                    type="primary",
                    disabled=(not input_text),
                    key="process_button"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add a recent summaries section if available
                recent_summaries = get_previous_summaries()[:3]  # Get latest 3
                if recent_summaries:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Recent Summaries")
                    
                    for summary in recent_summaries:
                        col_a, col_b = st.columns([4, 1])
                        with col_a:
                            st.markdown(f"**{summary['title']}**")
                            st.caption(f"{summary['created'].strftime('%b %d, %Y at %H:%M')}")
                        with col_b:
                            if st.button("View", key=f"view_{summary['filename']}"):
                                # Set this path to session state so the other tab can display it
                                st.session_state["view_summary"] = summary["path"]
                                # Switch to the other tab
                                st.session_state["active_tab"] = "Previous Summaries"
                                # Instead of experimental_rerun, use rerun
                                st.rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                output_container = st.container()
                
                if process_button and input_text:
                    with output_container:
                        st.markdown('<div class="output-area">', unsafe_allow_html=True)
                        st.subheader("Processing Transcript")
                        
                        # Create a container for progress updates
                        progress_container = st.container()
                        
                        # Process the text
                        with st.spinner():
                            start_time = time.time()
                            result = process_text(input_text, model_choice, include_action_items, progress_container)
                            
                            # Auto-save the summary
                            saved_path = save_summary(result, filename_base)
                            
                            # Display success message
                            st.success(f"Summary generated in {result['metadata']['processing_time_seconds']:.2f} seconds")
                            st.info(f"Saved to: {saved_path}")
                            
                            # Display tabs for different parts of the output
                            result_tabs = st.tabs(["📄 Summary", "✅ Action Items"])
                            
                            with result_tabs[0]:
                                st.markdown(result["summary"])
                                
                                # Download button
                                st.download_button(
                                    label="Download Summary",
                                    data=result["summary"],
                                    file_name=f"{filename_base}_summary.md" if filename_base else "summary.md",
                                    mime="text/markdown"
                                )
                            
                            with result_tabs[1]:
                                if include_action_items and result.get("action_items"):
                                    st.markdown(result["action_items"])
                                    
                                    # Download button
                                    st.download_button(
                                        label="Download Action Items",
                                        data=result["action_items"],
                                        file_name=f"{filename_base}_action_items.md" if filename_base else "action_items.md",
                                        mime="text/markdown"
                                    )
                                else:
                                    st.info("No action items were extracted or action item extraction was disabled.")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    with output_container:
                        st.markdown('<div class="output-area" style="min-height: 400px; display: flex; align-items: center; justify-content: center;">', unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div style="text-align: center; opacity: 0.7;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                            </svg>
                            <p style="margin-top: 1rem; font-size: 1.25rem;">Upload a transcript and click "Generate Summary" to begin</p>
                            <p style="margin-top: 0.5rem; font-size: 0.875rem;">Your summary will appear here</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Previous Summaries tab
    with tabs[1]:
        if tab_index == 1:  # Only render content if this tab is active
            st.subheader("Previous Summaries")
            
            # Get all available summaries
            all_summaries = get_previous_summaries()
            
            if not all_summaries:
                st.info("No previous summaries found. Generate a summary first.")
            else:
                # Create a container for the selected summary
                selected_summary_container = st.container()
                
                # Create a table of summaries
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                for i, summary in enumerate(all_summaries):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{summary['title']}**")
                    
                    with col2:
                        st.caption(f"{summary['created'].strftime('%b %d, %Y at %H:%M')}")
                    
                    with col3:
                        view_key = f"view_prev_{i}"
                        if st.button("View", key=view_key):
                            st.session_state["view_summary"] = summary["path"]
                    
                    if i < len(all_summaries) - 1:
                        st.markdown("---")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display selected summary if any
                if "view_summary" in st.session_state:
                    summary_path = st.session_state["view_summary"]
                    
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary_content = f.read()
                        
                        with selected_summary_container:
                            st.markdown('<div class="output-area">', unsafe_allow_html=True)
                            st.markdown(summary_content)
                            
                            # Download button
                            st.download_button(
                                label="Download Summary",
                                data=summary_content,
                                file_name=summary_path.name,
                                mime="text/markdown"
                            )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        with selected_summary_container:
                            st.error(f"Error loading summary: {e}")
    
    # Footer with about information
    st.markdown("""
    <div class="footer">
        <p>Note-Summarizer • Transform your transcripts into structured, actionable notes</p>
        <p>Optimized for Microsoft Teams meeting transcripts</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()