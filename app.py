# app.py (Enhanced with improved UI and chunk options)
import streamlit as st
from crew import TranscriptAnalysisCrew
import nltk
from nltk.tokenize import sent_tokenize
import pyperclip
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io
import sys
import os
import time
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="Enhanced Transcript Analyzer", layout="wide")

# App title and description
st.title("Enhanced Transcript Analyzer")
st.markdown("""
This tool analyzes meeting transcripts to generate comprehensive, well-organized notes.
It identifies key points, action items, and provides a structured synthesis of the entire discussion.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# File upload section
uploaded_file = st.file_uploader("Upload a transcript", type=["txt"])

if uploaded_file is not None:
    transcript_content = uploaded_file.read().decode("utf-8")
    
    # Initial transcript analysis for metadata
    num_characters = len(transcript_content)
    num_words = len(transcript_content.split())
    sentences = sent_tokenize(transcript_content)
    num_sentences = len(sentences)
    
    # Calculate recommended chunk count based on transcript size
    recommended_chunks = max(5, min(10, num_characters // 4000))
    
    # Display transcript in expandable section
    with st.expander("View Uploaded Transcript"):
        st.text_area("Transcript Content", transcript_content, height=300)
    
    # Create columns for metadata display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{num_characters:,}")
    with col2:
        st.metric("Words", f"{num_words:,}")
    with col3:
        st.metric("Sentences", f"{num_sentences:,}")
    with col4:
        st.metric("Recommended Chunks", recommended_chunks)
    
    # --- Configuration Options ---
    st.sidebar.subheader("Chunking Options")
    
    # Improved chunk slider with recommended value
    num_chunks = st.sidebar.slider(
        "Number of Chunks", 
        min_value=3, 
        max_value=20, 
        value=recommended_chunks,
        help="Divide transcript into this many sections for processing. More chunks may improve detail but take longer."
    )
    
    # Calculate and display estimated chunk size
    chunk_size = num_characters // num_chunks
    st.sidebar.caption(f"Approx. chunk size: {chunk_size:,} characters")
    
    # Improved overlap slider with percentage-based calculation
    overlap_percentage = st.sidebar.slider(
        "Chunk Overlap %", 
        min_value=5, 
        max_value=30, 
        value=15, 
        help="Higher overlap helps maintain context between chunks"
    )
    overlap_size = int((chunk_size * overlap_percentage) / 100)
    st.sidebar.caption(f"Overlap size: {overlap_size:,} characters")
    
    # Model selection
    st.sidebar.subheader("Model Options")
    model_options = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"]
    model_name = st.sidebar.selectbox("Select OpenAI Model", model_options)
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    verbose_mode = st.sidebar.checkbox("Show Processing Details", value=True)
    show_timing = st.sidebar.checkbox("Show Timing Information", value=True)
    
    # Analysis button with progress components
    if st.button("Analyze Transcript", type="primary"):
        # Start timer if timing is enabled
        start_time = time.time()
        
        # Create progress components
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Setup capture for verbose output
        verbose_output = st.expander("Agent Processing Details", expanded=False)
        
        status_text.text("Preparing transcript analysis...")
        progress_bar.progress(10)
        
        # Capture stdout for verbose output
        with io.StringIO() as buf:
            old_stdout = sys.stdout
            sys.stdout = buf  # Redirect stdout
            
            # Update status
            status_text.text("Analyzing transcript chunks...")
            progress_bar.progress(25)
            
            # Run the analysis
            try:
                crew = TranscriptAnalysisCrew(
                    transcript_content=transcript_content, 
                    num_chunks=num_chunks, 
                    overlap=overlap_size, 
                    verbose=verbose_mode, 
                    model_name=model_name
                )
                
                # Update progress
                status_text.text("Processing summaries and action items...")
                progress_bar.progress(50)
                
                results = crew.run()
                
                # Final progress update
                status_text.text("Finalizing results...")
                progress_bar.progress(90)
                
                # Restore stdout
                sys.stdout = old_stdout
                
                # Show verbose output if enabled
                if verbose_mode:
                    with verbose_output:
                        st.text(buf.getvalue())
                
                # Calculate and display execution time
                if show_timing:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    st.info(f"Processing completed in {execution_time:.2f} seconds")
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📝 Summary Notes", "✅ Action Items", "📊 Analysis"])
                
                with tab1:
                    st.subheader("Meeting Notes")
                    st.markdown(results)
                    if st.button("Copy Notes to Clipboard"):
                        pyperclip.copy(results)
                        st.success("Copied to clipboard!")
                    
                    # Export options
                    export_format = st.selectbox("Export format", ["Markdown (.md)", "Text (.txt)"])
                    if st.button("Download Notes"):
                        if export_format == "Markdown (.md)":
                            st.download_button(
                                label="Download Markdown",
                                data=results,
                                file_name="meeting_notes.md",
                                mime="text/markdown"
                            )
                        else:
                            st.download_button(
                                label="Download Text",
                                data=results,
                                file_name="meeting_notes.txt",
                                mime="text/plain"
                            )
                
                with tab2:
                    st.subheader("Action Items")
                    # Extract action items from the results
                    # This is a simple extraction - you may want to enhance this
                    if "Action Items" in results:
                        action_section = results.split("Action Items")[1].split("\n\n")[0]
                        st.markdown(action_section)
                    else:
                        st.markdown("No specific action items section identified in the results.")
                
                with tab3:
                    st.subheader("Content Analysis")
                    
                    # Word Cloud visualization
                    st.markdown("### Word Frequency Analysis")
                    stopwords = set(STOPWORDS)
                    remove_stopwords = st.checkbox("Remove Common Words", value=True)
                    
                    if remove_stopwords:
                        wordcloud_text = " ".join([w for w in transcript_content.split() if w.lower() not in stopwords])
                    else:
                        wordcloud_text = transcript_content
                        
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=150,
                        contour_width=3
                    ).generate(wordcloud_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                    
                    # Add more visualizations or analysis as needed
            
            except Exception as e:
                sys.stdout = old_stdout
                st.error(f"An error occurred during analysis: {str(e)}")
                with verbose_output:
                    st.text(buf.getvalue())
                    st.text(f"Error details: {str(e)}")
    
    # Display guidance if no analysis has been run yet
    else:
        st.info("Configure your options in the sidebar and click 'Analyze Transcript' to begin processing.")

else:
    # Display welcome message and instructions when no file is uploaded
    st.info("Please upload a transcript file to begin.")
    
    # Example section
    with st.expander("How to use this tool"):
        st.markdown("""
        ### How to use the Enhanced Transcript Analyzer
        
        1. **Upload a transcript** using the file uploader (.txt format)
        2. **Configure options** in the sidebar:
           - Adjust the number of chunks based on transcript length
           - Set chunk overlap percentage to maintain context
           - Select the OpenAI model to use
           - Toggle processing details and timing information
        3. **Click 'Analyze Transcript'** to begin processing
        4. **View results** in the tabbed interface:
           - Summary Notes: The complete synthesized notes
           - Action Items: Extracted actionable tasks
           - Analysis: Word frequency and other insights
           
        The analyzer works best with clear, well-formatted transcripts from meetings,
        interviews, or discussions.
        """)