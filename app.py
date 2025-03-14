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
import pandas as pd
from dotenv import load_dotenv
import re

# Import the updated utils
from utils import (
    chunk_transcript_advanced,
    extract_speakers,
    chunk_by_speaker,
    chunk_by_character
)

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
    
    # Extract speakers for display
    speakers = extract_speakers(transcript_content)
    num_speakers = len(speakers)
    
    # Calculate recommended chunk count based on transcript size
    recommended_chunks = max(5, min(10, num_characters // 4000))
    
    # Display transcript in expandable section
    with st.expander("View Uploaded Transcript"):
        st.text_area("Transcript Content", transcript_content, height=300)
    
    # Create columns for metadata display
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Characters", f"{num_characters:,}")
    with col2:
        st.metric("Words", f"{num_words:,}")
    with col3:
        st.metric("Sentences", f"{num_sentences:,}")
    with col4:
        st.metric("Speakers", f"{num_speakers}")
    with col5:
        st.metric("Recommended Chunks", recommended_chunks)
    
    # --- Configuration Options ---
    st.sidebar.subheader("Chunking Options")
    
    # Chunking strategy selection with more options
    chunking_strategy = st.sidebar.radio(
        "Chunking Strategy",
        ["speaker_aware", "boundary_aware", "fixed_size", "semantic"],
        index=0,
        help="""
        - Speaker-aware: Divides by speaker changes (best for meetings)
        - Boundary-aware: Uses natural paragraph/section breaks
        - Fixed-size: Traditional character-based chunking
        - Semantic: Uses AI to identify topic boundaries (requires OpenAI API)
        """
    )
    
    # Advanced options expander
    with st.sidebar.expander("Advanced Chunking Options"):
        # Options vary based on selected strategy
        if chunking_strategy == "fixed_size":
            # For fixed-size chunking, show chunk count and overlap sliders
            num_chunks = st.slider(
                "Number of Chunks", 
                min_value=3, 
                max_value=20, 
                value=recommended_chunks,
                help="Divide transcript into this many sections for processing"
            )
            
            # Calculate and display estimated chunk size
            chunk_size = num_characters // num_chunks
            st.caption(f"Approx. chunk size: {chunk_size:,} characters")
            
            # Overlap slider with percentage-based calculation
            overlap_percentage = st.slider(
                "Chunk Overlap %", 
                min_value=5, 
                max_value=30, 
                value=15, 
                help="Higher overlap helps maintain context between chunks"
            )
            overlap_size = int((chunk_size * overlap_percentage) / 100)
            st.caption(f"Overlap size: {overlap_size:,} characters")
        else:
            # For other strategies, we still need these values for compatibility
            num_chunks = recommended_chunks
            overlap_size = 200
            
            # Show a note about the chosen strategy
            if chunking_strategy == "speaker_aware":
                st.caption("Speaker-aware chunking preserves speaker context and creates chunks based on conversation flow.")
            elif chunking_strategy == "boundary_aware":
                st.caption("Boundary-aware chunking identifies natural topic and paragraph transitions.")
            elif chunking_strategy == "semantic":
                use_llm = st.checkbox(
                    "Use OpenAI for Enhanced Chunking", 
                    value=False,
                    help="Uses AI to identify optimal chunk boundaries (requires OpenAI API key)"
                )
                st.caption("Semantic chunking identifies topic-based chunks for optimal context preservation.")
    
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
            status_text.text("Running transcript analysis with advanced chunking...")
            progress_bar.progress(20)
            
            # Use the updated TranscriptAnalysisCrew with chunking strategy
            try:
                crew = TranscriptAnalysisCrew(
                    transcript_content=transcript_content, 
                    num_chunks=num_chunks, 
                    overlap=overlap_size, 
                    verbose=verbose_mode, 
                    model_name=model_name,
                    chunking_strategy=chunking_strategy
                )
                
                # Update progress
                status_text.text("Processing transcript chunks...")
                progress_bar.progress(30)
                
                # Run the analysis
                results = crew.run()
                
                # Update progress
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
                    if "Action Items" in results:
                        try:
                            # Try to extract the Action Items section
                            action_section = results.split("# Action Items")[1]
                            # Handle both cases: another section follows or it's the last section
                            next_section = action_section.split("#")[0] if "#" in action_section else action_section
                            st.markdown(next_section)
                        except:
                            # If the above fails, just display what we found
                            action_section = results.split("Action Items")[1].split("\n\n")[0]
                            st.markdown(action_section)
                    else:
                        st.markdown("No specific action items section identified in the results.")
                
                with tab3:
                    st.subheader("Content Analysis")
                    
                    # Basic speaker analysis
                    if speakers:
                        st.markdown("### Speaker Contributions")
                        
                        # Count speaker mentions
                        speaker_counts = {speaker: transcript_content.count(f"{speaker}:") for speaker in speakers}
                        
                        # Calculate approximate word counts per speaker
                        speaker_words = {}
                        for speaker in speakers:
                            # Find all segments for this speaker
                            # Fixed pattern that avoids the f-string nesting issue
                            escape_patterns = [re.escape(s)+':' for s in speakers] + ['$']
                            pattern = f"{re.escape(speaker)}:(.*?)(?={'|'.join(escape_patterns)})"
                            matches = re.findall(pattern, transcript_content, re.DOTALL)
                            speaker_words[speaker] = sum(len(m.split()) for m in matches)
                        
                        # Create a DataFrame for display
                        speaker_df = pd.DataFrame({
                            "Speaker": list(speaker_counts.keys()),
                            "Contributions": list(speaker_counts.values()),
                            "Words": [speaker_words.get(speaker, 0) for speaker in speaker_counts.keys()]
                        })
                        
                        # Calculate percentages
                        total_words = speaker_df["Words"].sum()
                        speaker_df["Speaking %"] = (speaker_df["Words"] / total_words * 100).round(1)
                        
                        # Sort by most talkative
                        speaker_df = speaker_df.sort_values("Words", ascending=False)
                        
                        st.dataframe(speaker_df)
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(speaker_df["Speaker"], speaker_df["Speaking %"])
                        
                        # Add percentage labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                    f'{height}%', ha='center', va='bottom')
                        
                        ax.set_ylabel("Speaking Time (%)")
                        ax.set_title("Speaker Contribution Analysis")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    
                    # Word Cloud visualization
                    st.markdown("### Word Frequency Analysis")
                    stopwords = set(STOPWORDS)
                    
                    # Add common meeting stopwords
                    meeting_stopwords = {"think", "know", "going", "yeah", "um", "uh", "like", "just", "okay", "right"}
                    stopwords.update(meeting_stopwords)
                    
                    # Add speaker names to stopwords
                    for speaker in speakers:
                        stopwords.add(speaker.lower())
                    
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
           - Choose a chunking strategy that best fits your transcript
           - Adjust settings based on transcript length and complexity
           - Select the OpenAI model to use
        3. **Click 'Analyze Transcript'** to begin processing
        4. **View results** in the tabbed interface:
           - Summary Notes: The complete synthesized notes
           - Action Items: Extracted actionable tasks
           - Analysis: Word frequency and speaker contributions
           
        #### Chunking Strategies
        
        - **Speaker-aware**: Divides based on speaker transitions (best for meetings and interviews)
        - **Boundary-aware**: Uses natural document boundaries like paragraphs and sections
        - **Fixed-size**: Divides into equal-sized pieces (traditional approach)
        - **Semantic**: Uses AI to identify topic changes (requires OpenAI API)
        
        #### Best Practices
        
        - For meeting transcripts with multiple speakers, use **speaker-aware** chunking
        - For articles or documents without clear speakers, use **boundary-aware** chunking
        - For very long documents, try **semantic** chunking with the OpenAI option enabled
        - If you experience any issues, fall back to **fixed-size** chunking which is the most reliable
        """)