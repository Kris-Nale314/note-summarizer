"""
Streamlit app for transcript summarization with multiple division strategies.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_speakers(text):
    """Extract speaker names from transcript for display."""
    speakers = []
    
    patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+):', # Full name
        r'([A-Z][a-z]+):', # First name
        r'(Dr\. [A-Z][a-z]+):', # Dr. Name
        r'(Mr\. [A-Z][a-z]+):', # Mr. Name
        r'(Mrs\. [A-Z][a-z]+):', # Mrs. Name
        r'(Ms\. [A-Z][a-z]+):' # Ms. Name
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        speakers.extend(matches)
    
    return list(set(speakers))

def run_all_strategies(transcript_content: str, base_options: SummaryOptions) -> Dict[str, Any]:
    """
    Run summarization with all division strategies for comparison.
    
    Args:
        transcript_content: The transcript text
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
        progress_bar.progress((i / len(strategies)))  # Remove *100 here
        
        # Clone the options and set the strategy
        options = SummaryOptions(
            chunk_strategy=strategy,
            model_name=base_options.model_name,
            max_chunk_size=base_options.max_chunk_size,
            chunk_overlap=base_options.chunk_overlap,
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
                "processing_time": elapsed_time
            }
            
            logger.info(f"Completed {strategy} strategy in {elapsed_time:.2f}s with {result['metadata']['division_count']} divisions")
            
        except Exception as e:
            logger.error(f"Error with {strategy} strategy: {e}")
            results[strategy] = {
                "summary": f"Error: {str(e)}",
                "division_count": 0,
                "divisions": [],
                "processing_time": 0,
                "error": str(e)
            }
    
    # Complete progress (value between 0.0 and 1.0)
    progress_bar.progress(1.0)  # Use 1.0 instead of 100
    status_text.text("Comparison complete!")
    
    return results

def generate_word_cloud(text, speakers=None):
    """Generate a word cloud visualization from text."""
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
    uploaded_file = st.file_uploader("Upload a transcript", type=["txt"])
    
    if uploaded_file is not None:
        transcript_content = uploaded_file.read().decode("utf-8")
        
        # Initial transcript analysis for metadata
        num_characters = len(transcript_content)
        num_words = len(transcript_content.split())
        sentences = len(re.findall(r'[.!?]+', transcript_content))
        
        # Extract speakers for display
        speakers = extract_speakers(transcript_content)
        num_speakers = len(speakers)
        
        # Display transcript in expandable section
        with st.expander("View Uploaded Document"):
            st.text_area("Document Content", transcript_content, height=300)
        
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
        
        # Model selection
        st.sidebar.subheader("Model Options")
        model_options = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"]
        model_name = st.sidebar.selectbox(
            "Model", 
            model_options,
            help="Select the model to use for summarization"
        )
        
        # Division strategy selection
        st.sidebar.subheader("Processing Options")
        single_strategy = st.sidebar.radio(
            "Process with a single strategy",
            ["Run all strategies", "basic", "speaker", "boundary", "context_aware"],
            index=0,
            help="""
            - Run all: Compare all strategies side by side
            - Basic: Simple division with smart breaks
            - Speaker: Division based on speaker transitions
            - Boundary: Division based on natural document boundaries
            - Context-aware: Smart division that preserves semantic context
            """
        )
        
        # Advanced options
        st.sidebar.subheader("Advanced Options")
        include_action_items = st.sidebar.checkbox(
            "Extract Action Items", 
            value=False,
            help="Extract and organize action items from the document"
        )
        
        with st.sidebar.expander("Advanced Settings"):
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2, 
                step=0.1,
                help="Lower values = more focused output, higher values = more creative"
            )
            
            max_chunk_size = st.slider(
                "Maximum Division Size", 
                min_value=1000, 
                max_value=4000, 
                value=2000, 
                step=100,
                help="Maximum characters per division (larger divisions need more powerful models)"
            )
            
            chunk_overlap = st.slider(
                "Division Overlap", 
                min_value=0, 
                max_value=200, 
                value=50, 
                step=10,
                help="Overlap between divisions to maintain context"
            )
            
        verbose_mode = st.sidebar.checkbox(
            "Show Processing Details", 
            value=False
        )
        
        # Process button
        process_button = st.button(
            "Process Document", 
            type="primary",
            help="Start processing the document with the selected settings"
        )
        
        if process_button:
            # Set up base options
            base_options = SummaryOptions(
                chunk_strategy="basic",  # Will be overridden
                model_name=model_name,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                include_action_items=include_action_items,
                temperature=temperature,
                verbose=verbose_mode
            )
            
            # Run comparison or single strategy
            with st.spinner("Processing document..."):
                try:
                    if single_strategy == "Run all strategies":
                        # Run all strategies for comparison
                        results = run_all_strategies(transcript_content, base_options)
                        
                        # Create tabs for results
                        strategy_tabs = st.tabs(list(results.keys()) + ["📊 Analysis"])
                        
                        # Process each strategy tab
                        for i, strategy in enumerate(results.keys()):
                            with strategy_tabs[i]:
                                result = results[strategy]
                                
                                # Display results for this strategy
                                st.subheader(f"{strategy.capitalize()} Strategy Results")
                                
                                # Show metrics
                                metrics_cols = st.columns(3)
                                metrics_cols[0].metric("Processing Time", f"{result['processing_time']:.2f}s")
                                metrics_cols[1].metric("Divisions Created", result["division_count"])
                                if result["division_count"] > 0:
                                    metrics_cols[2].metric("Avg Words/Division", 
                                                         f"{num_words // result['division_count']:,}")
                                
                                # Summary output
                                st.markdown("### Summary")
                                st.markdown(result["summary"])
                                
                                # Export options
                                if st.button(f"Copy {strategy} Summary", key=f"copy_{strategy}"):
                                    pyperclip.copy(result["summary"])
                                    st.success("Copied to clipboard!")
                                
                                # Download option
                                st.download_button(
                                    label=f"Download {strategy} Summary",
                                    data=result["summary"],
                                    file_name=f"{strategy}_summary.md",
                                    mime="text/markdown"
                                )
                                
                                # Show divisions in expander
                                with st.expander(f"View {result['division_count']} Divisions"):
                                    for j, division in enumerate(result["divisions"]):
                                        st.markdown(f"**Division {j+1}** ({len(division['text'])} chars)")
                                        st.text_area(f"Division {j+1} Content", 
                                                    division["text"], 
                                                    height=100, 
                                                    key=f"{strategy}_division_{j}")
                                        st.markdown("---")
                        
                        # Analysis tab
                        with strategy_tabs[-1]:
                            st.subheader("Comparison Analysis")
                            
                            # Create comparison metrics
                            comparison_data = []
                            for strategy, result in results.items():
                                comparison_data.append({
                                    "Strategy": strategy.capitalize(),
                                    "Processing Time (s)": f"{result['processing_time']:.2f}",
                                    "Divisions Created": result["division_count"],
                                    "Avg Division Size (chars)": int(num_characters / result["division_count"]) if result["division_count"] > 0 else 0,
                                    "Summary Length (chars)": len(result["summary"])
                                })
                            
                            # Display as dataframe
                            st.dataframe(comparison_data)
                            
                            # Word cloud visualization
                            st.subheader("Word Cloud Visualization")
                            fig = generate_word_cloud(transcript_content, speakers)
                            st.pyplot(fig)
                    
                    else:
                        # Run single strategy
                        options = SummaryOptions(
                            chunk_strategy=single_strategy,
                            model_name=model_name,
                            max_chunk_size=max_chunk_size,
                            chunk_overlap=chunk_overlap,
                            include_action_items=include_action_items,
                            temperature=temperature,
                            verbose=verbose_mode
                        )
                        
                        summarizer = TranscriptSummarizer(options)
                        
                        # Time the execution
                        start_time = time.time()
                        result = summarizer.summarize(transcript_content)
                        elapsed_time = time.time() - start_time
                        
                        # Create tabs for summary and details
                        tabs = st.tabs(["📝 Summary", "✅ Action Items", "📊 Analysis"])
                        
                        with tabs[0]:
                            # Show metrics
                            metrics_cols = st.columns(3)
                            metrics_cols[0].metric("Processing Time", f"{elapsed_time:.2f}s")
                            metrics_cols[1].metric("Divisions Created", result["metadata"]["division_count"])
                            
                            if result["metadata"]["division_count"] > 0:
                                avg_words = num_words // result["metadata"]["division_count"]
                                metrics_cols[2].metric("Avg Words/Division", f"{avg_words:,}")
                            
                            # Display summary
                            st.markdown("### Summary")
                            st.markdown(result["summary"])
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Copy Summary"):
                                    pyperclip.copy(result["summary"])
                                    st.success("Copied to clipboard!")
                            
                            with col2:
                                st.download_button(
                                    label="Download Summary",
                                    data=result["summary"],
                                    file_name="document_summary.md",
                                    mime="text/markdown"
                                )
                            
                            # Show divisions in expander
                            with st.expander(f"View {result['metadata']['division_count']} Divisions"):
                                for j, division in enumerate(result["divisions"]):
                                    st.markdown(f"**Division {j+1}** ({len(division['text'])} chars)")
                                    st.text_area(f"Division {j+1} Content", 
                                                division["text"], 
                                                height=100, 
                                                key=f"single_division_{j}")
                                    st.markdown("---")
                        
                        with tabs[1]:
                            if include_action_items and result["action_items"]:
                                st.markdown("### Action Items")
                                st.markdown(result["action_items"])
                                
                                if st.button("Copy Action Items"):
                                    pyperclip.copy(result["action_items"])
                                    st.success("Copied to clipboard!")
                            else:
                                st.info("Action item extraction was not enabled or no action items were found.")
                        
                        with tabs[2]:
                            # Word cloud visualization
                            st.subheader("Word Cloud Visualization")
                            fig = generate_word_cloud(transcript_content, speakers)
                            st.pyplot(fig)
                            
                            # Show metadata
                            st.subheader("Processing Information")
                            st.json(result["metadata"])
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)
    
    else:
        # Display welcome message and instructions when no file is uploaded
        st.info("Please upload a document file to begin.")
        
        # Example section
        with st.expander("How to use this tool"):
            st.markdown("""
            ### How to use Note Summarizer
            
            1. **Upload a document** using the file uploader (.txt format)
            2. **Configure options** in the sidebar:
               - Choose a processing strategy or compare all strategies
               - Select the LLM model to use
               - Enable action item extraction if needed
            3. **Click 'Process Document'** to start analysis
            4. **View results** in the tabbed interface:
               - Summary: The comprehensive document summary
               - Action Items: Extracted tasks and commitments (if enabled)
               - Analysis: Document insights and visualizations
               
            #### Division Strategies Explained
            
            - **Basic**: Simple division with intelligent sentence and paragraph breaks
            - **Speaker**: Preserves speaker transitions (best for meeting transcripts)
            - **Boundary**: Uses natural document boundaries like paragraphs and sections
            - **Context-aware**: Smart division that preserves semantic coherence and logical flow
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