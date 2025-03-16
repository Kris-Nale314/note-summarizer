"""
Streamlit app for document summarization with multiple division strategies.
Supports both single document and multi-document processing.
"""
import streamlit as st
import time
import pyperclip
import os
import logging
import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from summarizer import TranscriptSummarizer, SummaryOptions
from summarizer.division import extract_speakers, divide_document
from summarizer.multi_document_processor import MultiDocumentProcessor
from multi_document_ui import display_multi_document_ui, display_multi_document_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def auto_save_results(title: str, strategy: str, content: Dict[str, Any], document_info: Dict[str, Any] = None) -> str:
    """
    Automatically save processing results to the outputs folder.
    
    Args:
        title: Document title
        strategy: Strategy used (or 'comparison' for multiple strategies)
        content: Result content
        document_info: Optional document metadata
    
    Returns:
        Path to saved file
    """
    # Clean the title to make it file-system friendly
    safe_title = "".join(c if c.isalnum() or c in ['.', '-', '_'] else '_' for c in title)
    
    # Get current date in YYYY-MM-DD format
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Create filename in format: Title_Strategy_YYYY-MM-DD.md
    filename = f"{safe_title}_{strategy}_{date_str}.md"
    
    # Ensure outputs directory exists
    output_dir = Path("outputs")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / filename
    
    # Create markdown content based on content type
    if strategy == "comparison":
        # It's a strategy comparison
        markdown = f"# {title} - Strategy Comparison\n\n"
        markdown += f"*Generated on: {date_str}*\n\n"
        
        # Add document info if available
        if document_info:
            markdown += "## Document Information\n\n"
            markdown += f"- Characters: {document_info.get('num_characters', 'N/A'):,}\n"
            markdown += f"- Words: {document_info.get('num_words', 'N/A'):,}\n"
            markdown += f"- Speakers: {document_info.get('num_speakers', 'N/A')}\n\n"
        
        # Add comparison table
        markdown += "## Strategy Comparison\n\n"
        markdown += "| Strategy | Processing Time | Divisions | Summary Length |\n"
        markdown += "|----------|-----------------|-----------|----------------|\n"
        
        for strat, result in content.items():
            processing_time = f"{result.get('processing_time', 0):.2f}s"
            divisions = result.get('division_count', 0)
            summary_length = len(result.get('summary', ''))
            markdown += f"| {strat} | {processing_time} | {divisions} | {summary_length:,} chars |\n"
        
        markdown += "\n"
        
        # Add each strategy's summary
        markdown += "## Strategy Summaries\n\n"
        for strat, result in content.items():
            markdown += f"### {strat.capitalize()} Strategy\n\n"
            markdown += result.get("summary", "No summary available") + "\n\n"
            markdown += "---\n\n"
            
    elif strategy in ["earnings_calls_multi_company", "earnings_calls_sequential", "generic_documents"]:
        # It's a multi-document analysis
        markdown = f"# {title}\n\n"
        markdown += f"*Generated on: {date_str}*\n\n"
        
        # Add document collection info if available
        if "metadata" in content:
            markdown += "## Analysis Information\n\n"
            
            doc_type = content["metadata"].get("document_type", "Generic")
            if doc_type == "earnings_calls_multi_company":
                markdown += "**Analysis Type:** Multi-Company Earnings Call Comparison\n"
                if "companies" in content["metadata"]:
                    markdown += f"**Companies:** {', '.join(content['metadata']['companies'])}\n"
            elif doc_type == "earnings_calls_sequential":
                markdown += "**Analysis Type:** Sequential Earnings Call Analysis\n"
                if "company" in content["metadata"]:
                    markdown += f"**Company:** {content['metadata']['company']}\n"
                if "periods" in content["metadata"]:
                    markdown += f"**Periods:** {', '.join(content['metadata']['periods'])}\n"
            else:
                markdown += "**Analysis Type:** Multi-Document Synthesis\n"
            
            markdown += f"**Documents Analyzed:** {content['metadata'].get('document_count', 'N/A')}\n"
            markdown += f"**Processing Time:** {content['metadata'].get('processing_time_seconds', 'N/A'):.2f} seconds\n\n"
        
        # Add main analysis
        if "summary" in content:
            markdown += content["summary"] + "\n\n"
        
        # Add action items if available
        if "action_items" in content and content["action_items"]:
            if not content["action_items"].startswith("#"):
                markdown += "## Action Items\n\n"
            markdown += content["action_items"] + "\n\n"
            
    else:
        # It's a single strategy summary
        markdown = f"# {title} ({strategy} strategy)\n\n"
        markdown += f"*Generated on: {date_str}*\n\n"
        
        # Add document info if available
        if document_info:
            markdown += "## Document Information\n\n"
            markdown += f"- Characters: {document_info.get('num_characters', 'N/A'):,}\n"
            markdown += f"- Words: {document_info.get('num_words', 'N/A'):,}\n"
            markdown += f"- Speakers: {document_info.get('num_speakers', 'N/A')}\n\n"
        
        # Add processing info if available
        if "metadata" in content:
            markdown += "## Processing Information\n\n"
            markdown += f"- Division Strategy: {content['metadata'].get('division_strategy', 'N/A')}\n"
            markdown += f"- Model: {content['metadata'].get('model', 'N/A')}\n"
            markdown += f"- Processing Time: {content['metadata'].get('processing_time_seconds', 'N/A'):.2f} seconds\n"
            markdown += f"- Division Count: {content['metadata'].get('division_count', 'N/A')}\n\n"
        
        # Add summary
        if "summary" in content:
            markdown += content.get("summary", "") + "\n\n"
        
        # Add action items if available
        if "action_items" in content and content["action_items"]:
            if not content["action_items"].startswith("#"):
                markdown += "## Action Items\n\n"
            markdown += content["action_items"] + "\n\n"
    
    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    logger.info(f"Results saved to {file_path}")
    return str(file_path)

def safe_copy_to_clipboard(text):
    """Safely copy text to clipboard with error handling."""
    try:
        pyperclip.copy(text)
        st.success("Copied to clipboard!")
    except Exception as e:
        logger.error(f"Error copying to clipboard: {e}")
        st.warning("Could not copy to clipboard. Please select and copy the text manually.")

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

def display_strategy_performance_comparison(results, document_info=None):
    """
    Display interactive visualizations comparing division strategy performance.
    
    Args:
        results: Dictionary of results from different strategies
        document_info: Optional dictionary with document metadata
    """
    st.subheader("Performance Comparison")
    
    # Prepare data for charts
    strategies = list(results.keys())
    processing_times = [results[s]['processing_time'] for s in strategies]
    division_counts = [results[s]['division_count'] for s in strategies]
    summary_lengths = [len(results[s]['summary']) for s in strategies]
    
    # Create performance metrics dataframe
    data = {
        'Strategy': strategies,
        'Processing Time (s)': processing_times,
        'Division Count': division_counts,
        'Summary Length': summary_lengths
    }
    df = pd.DataFrame(data)
    
    # Calculate efficiency score (normalized)
    max_time = max(processing_times) if max(processing_times) > 0 else 1
    df['Efficiency Score'] = 1 - (df['Processing Time (s)'] / max_time)  # Higher is better
    
    # Create visualization tabs
    viz_tabs = st.tabs(["Processing Time", "Division Strategy", "Summary Length", "Recommendations"])
    
    # Processing time comparison
    with viz_tabs[0]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(strategies, processing_times, color='skyblue')
            ax.set_ylabel('Processing Time (seconds)')
            ax.set_title('Processing Time by Strategy')
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Show fastest strategy
            fastest = df.loc[df['Processing Time (s)'].idxmin()]
            st.metric("Fastest Strategy", fastest['Strategy'], 
                      f"{fastest['Processing Time (s)']:.1f}s")
            
            # Show efficiency metrics
            avg_time = df['Processing Time (s)'].mean()
            st.metric("Average Time", f"{avg_time:.1f}s")
    
    # Division strategy comparison
    with viz_tabs[1]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(strategies, division_counts, color='lightgreen')
            ax.set_ylabel('Number of Divisions')
            ax.set_title('Division Count by Strategy')
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Show division stats
            most_granular = df.loc[df['Division Count'].idxmax()]
            st.metric("Most Granular", most_granular['Strategy'], 
                      f"{int(most_granular['Division Count'])} divisions")
            
            least_granular = df.loc[df['Division Count'].idxmin()]
            st.metric("Least Granular", least_granular['Strategy'],
                      f"{int(least_granular['Division Count'])} divisions")
    
    # Summary length comparison
    with viz_tabs[2]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(strategies, summary_lengths, color='coral')
            ax.set_ylabel('Summary Length (characters)')
            ax.set_title('Summary Length by Strategy')
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Show summary stats
            most_detailed = df.loc[df['Summary Length'].idxmax()]
            st.metric("Most Detailed", most_detailed['Strategy'], 
                      f"{most_detailed['Summary Length']:,} chars")
            
            most_concise = df.loc[df['Summary Length'].idxmin()]
            st.metric("Most Concise", most_concise['Strategy'],
                      f"{most_concise['Summary Length']:,} chars")
    
    # Strategy recommendations
    with viz_tabs[3]:
        st.markdown("### Strategy Recommendations")
        
        # Calculate scores for different factors
        df['Speed Score'] = 1 - (df['Processing Time (s)'] / df['Processing Time (s)'].max())
        df['Detail Score'] = df['Summary Length'] / df['Summary Length'].max()
        df['Granularity Score'] = df['Division Count'] / df['Division Count'].max()
        
        # Create radar chart data
        categories = ['Speed', 'Detail', 'Granularity']
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw one axis per variable + add labels
        plt.xticks(angles[:-1], categories)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        # Plot each strategy
        for i, strategy in enumerate(strategies):
            values = [df.loc[df['Strategy'] == strategy, 'Speed Score'].values[0],
                      df.loc[df['Strategy'] == strategy, 'Detail Score'].values[0],
                      df.loc[df['Strategy'] == strategy, 'Granularity Score'].values[0]]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=strategy)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        st.pyplot(fig)
        
        # Strategy recommendations based on document type
        st.markdown("#### Based on your document characteristics:")
        
        # Get document info
        has_speakers = False
        has_structure = False
        is_large = False
        
        if document_info:
            # Check for speaker presence
            has_speakers = document_info.get('num_speakers', 0) > 2
            
            # Check for document structure
            document_content = document_info.get('content', '')
            has_structure = (re.search(r'\n#{1,3}\s+', document_content) or 
                            re.search(r'\n\s*\d+\.\s+', document_content))
            
            # Check if document is large
            is_large = document_info.get('num_characters', 0) > 50000
        
        # Generate recommendations
        if has_speakers:
            st.info("📝 **Speaker-heavy document detected**: The 'speaker' strategy generally works best for transcripts and conversations, preserving speaker attribution and flow.")
        
        if has_structure:
            st.info("📝 **Structured document detected**: The 'boundary' strategy works well for documents with clear headings, sections, or lists.")
        
        if is_large:
            st.info("📝 **Large document detected**: For this size document, consider using either:")
            st.markdown("  - 'semantic' strategy for best topic coherence (but slower processing)")
            st.markdown("  - 'context_aware' strategy for a good balance of speed and coherence")
        
        # Overall recommendation
        st.markdown("#### Overall Strategy Recommendation:")
        
        best_strategy = None
        recommendation_reason = ""
        
        if has_speakers and not is_large:
            best_strategy = "speaker"
            recommendation_reason = "preserves conversation flow and speaker attribution"
        elif has_structure and not is_large:
            best_strategy = "boundary"
            recommendation_reason = "respects document structure like headings and lists"
        elif is_large:
            best_strategy = "semantic"
            recommendation_reason = "provides best topic coherence for long documents"
        else:
            best_strategy = "context_aware"
            recommendation_reason = "provides good balance of coherence and processing time"
        
        st.success(f"**Recommended strategy: '{best_strategy}'** - This strategy {recommendation_reason}.")

def run_all_strategies(transcript_content: str, base_options: SummaryOptions, document_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run summarization with all division strategies for comparison.
    
    Args:
        transcript_content: The document text
        base_options: Base options to use
        document_info: Document metadata
        
    Returns:
        Dictionary of results for each strategy
    """
    # Determine which strategies to run based on document size
    if len(transcript_content) > 50000:  # For larger documents
        strategies = ["basic", "speaker", "boundary", "context_aware", "semantic"]
    else:
        # For smaller documents, skip semantic to save embedding API costs
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
    
    # Auto-save the comparison results
    title = document_info.get("title", "Document_Comparison")
    file_path = auto_save_results(title, "comparison", results, document_info)
    logger.info(f"Strategy comparison saved to {file_path}")
    
    return results

def process_single_document(document_content: str, options: Dict[str, Any]):
    """Process a single document with the selected options."""
    # Check document size
    if len(document_content) > 100000 and options["model_name"] == "gpt-3.5-turbo":
        st.warning(f"This document is quite large ({len(document_content):,} characters). Consider using a model with a larger context window like 'gpt-3.5-turbo-16k' for better results.")
    
    # Save document info for visualization
    document_info = {
        'content': document_content,
        'num_characters': options["num_characters"],
        'num_words': options["num_words"],
        'num_speakers': options["num_speakers"],
        'speakers': options["speakers"],
        'title': options.get("title", "Document Summary")
    }
    
    # Set up the summarization options
    summary_options = SummaryOptions(
        division_strategy=options["division_strategy"],
        model_name=options["model_name"],
        min_sections=options["min_sections"],
        target_tokens_per_section=options["target_tokens"],
        section_overlap=options["section_overlap"],
        include_action_items=options["include_action_items"],
        temperature=options["temperature"],
        verbose=options["verbose_mode"]
    )
    
    # Process with selected strategy or all strategies
    with st.spinner("Processing document..."):
        try:
            if options["division_strategy"] == "Run all strategies":
                # Run all strategies for comparison
                results = run_all_strategies(document_content, summary_options, document_info)
                
                # Create tabs for results
                strategy_tabs = st.tabs(list(results.keys()) + ["📊 Analysis"])
                
                # Show the auto-save notification
                st.success(f"✅ Comparison automatically saved to outputs folder")
                
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
                            avg_words = options["num_words"] // max(1, result["division_count"])  # Prevent division by zero
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
                    # Use the new performance comparison visualization
                    display_strategy_performance_comparison(results, document_info)
                    
                    # Word cloud visualization
                    st.subheader("Word Cloud Visualization")
                    fig = generate_word_cloud(document_content, options["speakers"])
                    st.pyplot(fig)
            
            else:
                # Run single strategy
                summarizer = TranscriptSummarizer(summary_options)
                
                # Time the execution
                start_time = time.time()
                result = summarizer.summarize(document_content)
                elapsed_time = time.time() - start_time
                
                # Auto-save the summary
                file_path = auto_save_results(options["title"], options["division_strategy"], result, document_info)
                
                # Create tabs for summary and details
                tabs = st.tabs(["📝 Summary", "✅ Action Items", "📊 Analysis"])
                
                with tabs[0]:
                    # Show metrics
                    metrics_cols = st.columns(3)
                    metrics_cols[0].metric("Processing Time", f"{elapsed_time:.2f}s")
                    metrics_cols[1].metric("Sections Created", result["metadata"]["division_count"])
                    
                    if result["metadata"]["division_count"] > 0:
                        avg_words = options["num_words"] // max(1, result["metadata"]["division_count"])
                        metrics_cols[2].metric("Avg Words/Section", f"{avg_words:,}")
                    
                    # Display summary
                    st.markdown("### Summary")
                    st.markdown(result["summary"])
                    
                    # Show auto-save notification
                    st.success(f"✅ Summary automatically saved to outputs folder")
                    
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
                    if options["include_action_items"] and result["action_items"]:
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
                    fig = generate_word_cloud(document_content, options["speakers"])
                    st.pyplot(fig)
                    
                    # Show metadata
                    st.subheader("Processing Information")
                    st.json(result["metadata"])
                    
                    # Add recommendations for this strategy
                    st.subheader("Strategy Analysis")
                    
                    if options["division_strategy"] == "semantic":
                        st.info("You're using the semantic division strategy, which provides the best topic coherence for long documents but may be slower due to embedding generation.")
                    elif options["division_strategy"] == "speaker" and options["num_speakers"] < 2:
                        st.warning("The speaker strategy works best with documents that have multiple speakers. Consider trying context_aware or boundary strategies for this document type.")
                    elif options["division_strategy"] == "boundary" and not re.search(r'\n#{1,3}\s+', document_content):
                        st.info("The boundary strategy works best with documents that have clear section headers. Your document has few or no headers, so context_aware might work better.")
                    elif options["division_strategy"] == "basic" and len(document_content) > 50000:
                        st.info("For large documents like this, semantic or context_aware strategies typically provide better results.")
        
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True)

def display_single_document_ui():
    """Display UI for single document processing."""
    st.markdown("### Single Document Processing")
    
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
        
        # Get document title - default to filename without extension
        default_title = os.path.splitext(uploaded_file.name)[0]
        document_title = st.text_input("Document Title", value=default_title)
        
        # Document display
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
            ["Run all strategies", "basic", "speaker", "boundary", "context_aware", "semantic"],
            index=0,
            help="""
            - Run all: Compare all strategies side by side
            - Basic: Simple division with smart breaks
            - Speaker: Division based on speaker transitions (for transcripts)
            - Boundary: Division based on document structure
            - Context-aware: Smart division for semantic coherence
            - Semantic: AI-powered chunking based on topic transitions (best for long documents)
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
        
        # Bundle options for processing
        options = {
            "division_strategy": division_strategy,
            "model_name": model_name,
            "min_sections": min_sections,
            "target_tokens": target_tokens,
            "section_overlap": section_overlap,
            "temperature": temperature,
            "include_action_items": include_action_items,
            "verbose_mode": verbose_mode,
            "num_characters": num_characters,
            "num_words": num_words,
            "num_speakers": num_speakers,
            "speakers": speakers,
            "title": document_title  # Include the document title
        }
        
        # Process document if button clicked
        if process_button:
            process_single_document(document_content, options)
    
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
            - **Semantic**: Advanced AI-powered chunking using embeddings to detect topic transitions
              (ideal for very long documents, but requires API calls for embedding generation)
            """)

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
    
    return sorted(files, reverse=True)  # Newest first

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

def main():
    """Main application entry point with auto-save functionality."""
    st.set_page_config(page_title="Note Summarizer", layout="wide")
    
    # App title and description
    st.title("🎙️ Note Summarizer 📝")
    st.markdown("""
    Transform meeting transcripts, earnings calls, and long documents into comprehensive, organized notes.
    Process single documents or analyze multiple related documents with comparative insights.
    """)
    
    # Create tabs for single document, multi-document, and outputs
    tabs = st.tabs(["📄 Single Document", "📚 Multiple Documents", "💾 Outputs"])
    
    # Single document tab
    with tabs[0]:
        display_single_document_ui()
    
    # Multi-document tab
    with tabs[1]:
        documents, process_button, options = display_multi_document_ui()
        
        if documents and process_button:
            with st.spinner("Processing multiple documents..."):
                try:
                    # Set up options
                    summary_options = SummaryOptions(
                        division_strategy="context_aware",  # Best strategy for multi-doc analysis
                        model_name=options["model_name"],
                        min_sections=3,
                        target_tokens_per_section=30000,  # Larger sections for multi-document
                        section_overlap=options["section_overlap"],
                        include_action_items=options["include_action_items"],
                        temperature=options["temperature"],
                        verbose=options["verbose"]
                    )
                    
                    # Initialize the multi-document processor
                    processor = MultiDocumentProcessor(summary_options)
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Override document type if specified by user
                    if options["document_type"] != "Auto-detect":
                        if "multi-company" in options["document_type"].lower():
                            # Set metadata for detecting multi-company
                            for i, doc in enumerate(documents):
                                if "company" not in doc["metadata"]:
                                    # Assign different company names if not provided
                                    doc["metadata"]["company"] = f"Company {i+1}"
                        
                        elif "sequential" in options["document_type"].lower():
                            # Ensure all documents have the same company
                            company_name = None
                            for doc in documents:
                                if "company" in doc["metadata"] and doc["metadata"]["company"]:
                                    company_name = doc["metadata"]["company"]
                                    break
                            
                            if company_name:
                                for doc in documents:
                                    doc["metadata"]["company"] = company_name
                    
                    # Process the documents
                    result = processor.process_multiple_documents(documents)
                    
                    # Calculate total processing time
                    processing_time = time.time() - start_time
                    
                    # Add original documents to result for visualization
                    result["documents"] = documents
                    
                    # Auto-save the multi-document analysis
                    try:
                        # Generate automatic title based on document type
                        if "metadata" in result and "document_type" in result["metadata"]:
                            doc_type = result["metadata"]["document_type"]
                            if doc_type == "earnings_calls_multi_company" and "companies" in result["metadata"]:
                                companies = result["metadata"]["companies"]
                                if len(companies) <= 3:
                                    auto_title = f"{'-'.join(companies)}_comparison"
                                else:
                                    auto_title = f"{len(companies)}-companies_analysis"
                            elif doc_type == "earnings_calls_sequential" and "company" in result["metadata"]:
                                company = result["metadata"]["company"]
                                auto_title = f"{company}_sequential"
                            else:
                                auto_title = "multi-document_analysis"
                        else:
                            auto_title = "multi-document_analysis"
                        
                        # Store the analysis
                        file_path = auto_save_results(auto_title, result["metadata"]["document_type"], result)
                        logger.info(f"Analysis automatically saved to {file_path}")
                        
                    except Exception as e:
                        logger.error(f"Error auto-saving analysis: {e}")
                    
                    # Display the results
                    display_multi_document_results(result, processing_time)
                    
                    # Show auto-save notification
                    st.success(f"✅ Analysis automatically saved to outputs folder")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    logger.error(f"Multi-document processing error: {e}", exc_info=True)
    
    # Outputs tab
    with tabs[2]:
        display_outputs_browser()

if __name__ == "__main__":
    main()