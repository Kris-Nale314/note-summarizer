"""
Streamlit app for document summarization with multiple division strategies.
"""
import streamlit as st
import time
import pyperclip
import os
import logging
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
from typing import Dict, Any, List

from summarizer import TranscriptSummarizer, SummaryOptions
from summarizer.division import extract_speakers, divide_document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_strategies(transcript_content: str, base_options: SummaryOptions) -> Dict[str, Any]:
    """
    Run summarization with all division strategies for comparison.
    
    Args:
        transcript_content: The document text
        base_options: Base options to use
        
    Returns:
        Dictionary of results for each strategy
    """
    strategies = ["basic", "speaker", "boundary", "context_aware"]
    results = {}
    
    # Set up progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, strategy in enumerate(strategies):
        # Update progress (value between 0.0 and 1.0)
        status_text.text(f"Processing with {strategy} strategy...")
        progress_bar.progress((i / len(strategies)))
        
        # Clone the options and set the strategy
        options = SummaryOptions(
            division_strategy=strategy,
            model_name=base_options.model_name,
            min_sections=base_options.min_sections,
            target_tokens_per_section=base_options.target_tokens_per_section,
            section_overlap=base_options.section_overlap,
            include_action_items=False,  # Skip action items for comparison
            temperature=base_options.temperature,
            verbose=base_options.verbose
        )
        
        try:
            # Create summarizer with this strategy
            summarizer = TranscriptSummarizer(options)
            
            # Time the execution
            start_time = time.time()
            result = summarizer.summarize(transcript_content)
            elapsed_time = time.time() - start_time
            
            # Store results
            results[strategy] = {
                "summary": result["summary"],
                "division_count": result["metadata"]["division_count"],
                "divisions": result["divisions"],
                "division_summaries": result.get("division_summaries", []),
                "processing_time": elapsed_time
            }
            
            logger.info(f"Completed {strategy} strategy in {elapsed_time:.2f}s with {result['metadata']['division_count']} divisions")
            
        except Exception as e:
            logger.error(f"Error with {strategy} strategy: {e}")
            results[strategy] = {
                "summary": f"Error: {str(e)}",
                "division_count": 0,
                "divisions": [],
                "division_summaries": [],
                "processing_time": 0,
                "error": str(e)
            }
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("Comparison complete!")
    
    return results

def generate_word_cloud(text, speakers=None):
    """Generate a word cloud visualization from text."""
    try:
        # Set up stopwords
        stopwords = set(STOPWORDS)
        
        # Add common meeting stopwords
        meeting_stopwords = {"think", "know", "going", "yeah", "um", "uh", "like", "just", "okay", "right"}
        stopwords.update(meeting_stopwords)
        
        # Add speaker names to stopwords if provided
        if speakers:
            for speaker in speakers:
                stopwords.add(speaker.lower())
        
        # Create and configure word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=150,
            stopwords=stopwords,
            contour_width=3
        ).generate(text)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        
        return fig
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        # Create a simple error figure
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, f"Could not generate word cloud: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis("off")
        return fig

def safe_copy_to_clipboard(text):
    """Safely copy text to clipboard with error handling."""
    try:
        pyperclip.copy(text)
        st.success("Copied to clipboard!")
    except Exception as e:
        logger.error(f"Error copying to clipboard: {e}")
        st.warning("Could not copy to clipboard. Please select and copy the text manually.")

def main():
    st.set_page_config(page_title="Note Summarizer", layout="wide")
    
    # App title and description
    st.title("Note Summarizer")
    st.markdown("""
    Transform meeting transcripts and long documents into comprehensive, well-organized notes.
    Compare different division strategies to see which works best for your content.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a document", type=["txt"])
    
    if uploaded_file is not None:
        # Try to read the file with different encodings
        try:
            document_content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 fails, try other common encodings
            uploaded_file.seek(0)  # Reset file pointer
            try:
                document_content = uploaded_file.read().decode("latin-1")
                st.info("Note: This file wasn't in UTF-8 format. Using latin-1 encoding instead.")
            except UnicodeDecodeError:
                # If that fails too, try Windows-1252 (common for Windows files)
                uploaded_file.seek(0)
                try:
                    document_content = uploaded_file.read().decode("cp1252")
                    st.info("Note: This file wasn't in UTF-8 format. Using Windows-1252 encoding instead.")
                except UnicodeDecodeError:
                    # Last resort
                    st.error("Unable to decode this file. Please ensure it's a valid text file.")
                    st.stop()
        
        # Initial document analysis for metadata
        num_characters = len(document_content)
        num_words = len(document_content.split())
        sentences = len(re.findall(r'[.!?]+', document_content))
        
        # Extract speakers for display
        speakers = extract_speakers(document_content)
        num_speakers = len(speakers)
        
        # Display document in expandable section
        with st.expander("View Uploaded Document"):
            st.text_area("Document Content", document_content, height=300)
        
        # Create columns for metadata display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", f"{num_characters:,}")
        with col2:
            st.metric("Words", f"{num_words:,}")
        with col3:
            st.metric("Sentences", f"{sentences:,}")
        with col4:
            st.metric("Speakers", f"{num_speakers}")
        
        # Division strategy selection
        st.sidebar.subheader("Division Strategy")
        division_strategy = st.sidebar.radio(
            "Select how to divide the document",
            ["Run all strategies", "basic", "speaker", "boundary", "context_aware"],
            index=0,
            help="""
            - Run all: Compare all strategies side by side
            - Basic: Simple division with smart breaks
            - Speaker: Division based on speaker transitions (for transcripts)
            - Boundary: Division based on document structure
            - Context-aware: Smart division for semantic coherence
            """
        )
        
        # Model selection
        st.sidebar.subheader("Model Options")
        model_options = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"]
        model_name = st.sidebar.selectbox(
            "Model", 
            model_options,
            help="Select the model to use for summarization"
        )
        
        # Division parameters
        st.sidebar.subheader("Division Parameters")
        min_sections = st.sidebar.slider(
            "Minimum Sections", 
            min_value=2, 
            max_value=10, 
            value=3,
            help="Minimum number of sections to divide the document into"
        )
        
        # Convert max tokens to more user-friendly "Size"
        section_size = st.sidebar.select_slider(
            "Section Size", 
            options=["Small", "Medium", "Large", "Extra Large"],
            value="Medium",
            help="Controls how large each section will be"
        )
        
        # Map size selection to token counts
        size_to_tokens = {
            "Small": 10000,
            "Medium": 25000,
            "Large": 40000,
            "Extra Large": 60000
        }
        target_tokens = size_to_tokens[section_size]
        
        # Advanced options
        with st.sidebar.expander("Advanced Options"):
            section_overlap = st.slider(
                "Section Overlap", 
                min_value=0.0, 
                max_value=0.3, 
                value=0.1, 
                step=0.05,
                help="Overlap between sections as a fraction of section size"
            )
            
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2, 
                step=0.1,
                help="Controls creativity: lower is more focused, higher is more creative"
            )
            
            include_action_items = st.checkbox(
                "Extract Action Items", 
                value=True,
                help="Identify and extract action items, tasks, and commitments"
            )
            
            verbose_mode = st.checkbox(
                "Show Processing Details", 
                value=False,
                help="Display detailed processing information"
            )
        
        # Process button
        process_button = st.button(
            "Process Document", 
            type="primary",
            help="Start processing the document with the selected settings"
        )
        
        if process_button:
            # Check document size
            if len(document_content) > 100000 and model_name == "gpt-3.5-turbo":
                st.warning(f"This document is quite large ({len(document_content):,} characters). Consider using a model with a larger context window like 'gpt-3.5-turbo-16k' for better results.")
            
            # Set up base options
            try:
                base_options = SummaryOptions(
                    division_strategy="basic",  # Will be overridden if needed
                    model_name=model_name,
                    min_sections=min_sections,
                    target_tokens_per_section=target_tokens,
                    section_overlap=section_overlap,
                    include_action_items=include_action_items,
                    temperature=temperature,
                    verbose=verbose_mode
                )
                
                # Process with selected strategy or all strategies
                with st.spinner("Processing document..."):
                    try:
                        if division_strategy == "Run all strategies":
                            # Run all strategies for comparison
                            results = run_all_strategies(document_content, base_options)
                            
                            # Create tabs for results
                            strategy_tabs = st.tabs(list(results.keys()) + ["📊 Analysis"])
                            
                            # Process each strategy tab
                            for i, strategy in enumerate(results.keys()):
                                with strategy_tabs[i]:
                                    result = results[strategy]
                                    
                                    # Display results for this strategy
                                    st.subheader(f"{strategy.capitalize()} Strategy")
                                    
                                    # Show metrics
                                    metrics_cols = st.columns(3)
                                    metrics_cols[0].metric("Processing Time", f"{result['processing_time']:.2f}s")
                                    metrics_cols[1].metric("Sections Created", result["division_count"])
                                    if result["division_count"] > 0:
                                        avg_words = num_words // max(1, result["division_count"])  # Prevent division by zero
                                        metrics_cols[2].metric("Avg Words/Section", f"{avg_words:,}")
                                    
                                    # Summary output
                                    st.markdown("### Summary")
                                    st.markdown(result["summary"])
                                    
                                    # Export options
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"Copy {strategy} Summary", key=f"copy_{strategy}"):
                                            safe_copy_to_clipboard(result["summary"])
                                    
                                    with col2:
                                        st.download_button(
                                            label=f"Download {strategy} Summary",
                                            data=result["summary"],
                                            file_name=f"{strategy}_summary.md",
                                            mime="text/markdown"
                                        )
                                    
                                    # Show divisions in expander
                                    with st.expander(f"View {result['division_count']} Sections"):
                                        for j, division in enumerate(result["divisions"]):
                                            st.markdown(f"**Section {j+1}** ({len(division['text'])} chars)")
                                            st.text_area(f"Section {j+1} Content", 
                                                        division["text"], 
                                                        height=100, 
                                                        key=f"{strategy}_division_{j}")
                                            
                                            # Show division summary if available
                                            if j < len(result.get("division_summaries", [])):
                                                st.markdown("**Section Summary:**")
                                                st.markdown(result["division_summaries"][j])
                                            
                                            st.markdown("---")
                            
                            # Analysis tab
                            with strategy_tabs[-1]:
                                st.subheader("Strategy Comparison")
                                
                                # Create comparison metrics
                                comparison_data = []
                                for strategy, result in results.items():
                                    comparison_data.append({
                                        "Strategy": strategy.capitalize(),
                                        "Processing Time (s)": f"{result['processing_time']:.2f}",
                                        "Sections Created": result["division_count"],
                                        "Avg Section Size (chars)": int(num_characters / max(1, result["division_count"])) if result["division_count"] > 0 else 0,
                                        "Summary Length (chars)": len(result["summary"])
                                    })
                                
                                # Display as dataframe
                                st.dataframe(comparison_data)
                                
                                # Word cloud visualization
                                st.subheader("Word Cloud Visualization")
                                fig = generate_word_cloud(document_content, speakers)
                                st.pyplot(fig)
                                
                                # Recommendation
                                st.subheader("Strategy Recommendation")
                                if num_speakers >= 3:
                                    st.info("This document has multiple speakers, suggesting it's a transcript. The 'speaker' strategy is likely to work best.")
                                elif re.search(r'\n#{1,3}\s+', document_content) or re.search(r'\n\s*\d+\.\s+', document_content):
                                    st.info("This document has clear section headers or structured lists. The 'boundary' strategy is likely to work best.")
                                else:
                                    st.info("For this document type, the 'context_aware' strategy may provide the best balance of quality and processing time.")
                        
                        else:
                            # Run single strategy
                            options = SummaryOptions(
                                division_strategy=division_strategy,
                                model_name=model_name,
                                min_sections=min_sections,
                                target_tokens_per_section=target_tokens,
                                section_overlap=section_overlap,
                                include_action_items=include_action_items,
                                temperature=temperature,
                                verbose=verbose_mode
                            )
                            
                            summarizer = TranscriptSummarizer(options)
                            
                            # Time the execution
                            start_time = time.time()
                            result = summarizer.summarize(document_content)
                            elapsed_time = time.time() - start_time
                            
                            # Create tabs for summary and details
                            tabs = st.tabs(["📝 Summary", "✅ Action Items", "📊 Analysis"])
                            
                            with tabs[0]:
                                # Show metrics
                                metrics_cols = st.columns(3)
                                metrics_cols[0].metric("Processing Time", f"{elapsed_time:.2f}s")
                                metrics_cols[1].metric("Sections Created", result["metadata"]["division_count"])
                                
                                if result["metadata"]["division_count"] > 0:
                                    avg_words = num_words // max(1, result["metadata"]["division_count"])
                                    metrics_cols[2].metric("Avg Words/Section", f"{avg_words:,}")
                                
                                # Display summary
                                st.markdown("### Summary")
                                st.markdown(result["summary"])
                                
                                # Export options
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Copy Summary"):
                                        safe_copy_to_clipboard(result["summary"])
                                
                                with col2:
                                    st.download_button(
                                        label="Download Summary",
                                        data=result["summary"],
                                        file_name="document_summary.md",
                                        mime="text/markdown"
                                    )
                                
                                # Show divisions in expander
                                with st.expander(f"View {result['metadata']['division_count']} Sections"):
                                    for j, division in enumerate(result["divisions"]):
                                        st.markdown(f"**Section {j+1}** ({len(division['text'])} chars)")
                                        st.text_area(f"Section {j+1} Content", 
                                                    division["text"], 
                                                    height=100, 
                                                    key=f"single_division_{j}")
                                        
                                        # Show division summary if available
                                        if "division_summaries" in result and j < len(result["division_summaries"]):
                                            st.markdown("**Section Summary:**")
                                            st.markdown(result["division_summaries"][j])
                                        
                                        st.markdown("---")
                            
                            with tabs[1]:
                                if include_action_items and result["action_items"]:
                                    st.markdown("### Action Items")
                                    st.markdown(result["action_items"])
                                    
                                    if st.button("Copy Action Items"):
                                        safe_copy_to_clipboard(result["action_items"])
                                        
                                    st.download_button(
                                        label="Download Action Items",
                                        data=result["action_items"],
                                        file_name="action_items.md",
                                        mime="text/markdown"
                                    )
                                else:
                                    st.info("Action item extraction was not enabled or no action items were found.")
                            
                            with tabs[2]:
                                # Word cloud visualization
                                st.subheader("Word Cloud Visualization")
                                fig = generate_word_cloud(document_content, speakers)
                                st.pyplot(fig)
                                
                                # Show metadata
                                st.subheader("Processing Information")
                                st.json(result["metadata"])
                    
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
                        logger.error(f"Processing error: {e}", exc_info=True)
            except Exception as e:
                st.error(f"Configuration error: {str(e)}")
                logger.error(f"Configuration error: {e}", exc_info=True)
    
    else:
        # Display welcome message and instructions when no file is uploaded
        st.info("Please upload a document file to begin.")
        
        # Example section
        with st.expander("How to use this tool"):
            st.markdown("""
            ### How to use Note Summarizer
            
            1. **Upload a document** using the file uploader (.txt format)
            2. **Configure options** in the sidebar:
               - Choose a division strategy or compare all strategies
               - Select the LLM model to use
               - Adjust section size and count as needed
               - Enable action item extraction if needed
            3. **Click 'Process Document'** to start analysis
            4. **View results** in the tabbed interface:
               - Summary: The comprehensive document summary
               - Action Items: Extracted tasks and commitments (if enabled)
               - Analysis: Document insights and visualizations
               
            #### Division Strategies Explained
            
            - **Basic**: Simple division with intelligent sentence and paragraph breaks
            - **Speaker**: Preserves speaker transitions (best for meeting transcripts)
            - **Boundary**: Uses natural document boundaries like headings and paragraphs
            - **Context-aware**: Smart division that preserves semantic coherence
            """)
            
            st.markdown("""
            #### Tips for Best Results
            
            - For meeting transcripts with multiple speakers, use the **speaker** strategy
            - For documents with clear sections, use the **boundary** strategy
            - For complex documents with interrelated topics, try the **context-aware** strategy
            - For very large documents, use a model with larger context window (16k or higher)
            """)

if __name__ == "__main__":
    main()