"""
Single document processing page for Note-Summarizer.
"""

import streamlit as st
import time
import os
import logging
from typing import Dict, Any, List, Optional

from summarizer import TranscriptSummarizer, SummaryOptions
from summarizer.division import assess_document, divide_document
from utils import ui_helpers, visualizations, output_store

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Single Document Processing | Note-Summarizer",
    page_icon="🎙️",
    layout="wide"
)

def process_document(text: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single document with the given options."""
    # Create options object
    summary_options = SummaryOptions(
        division_strategy=options["division_strategy"],
        model_name=options["model_name"],
        temperature=options["temperature"],
        min_sections=options["min_sections"],
        target_tokens_per_section=options.get("target_tokens_per_section", 25000),
        include_action_items=options["include_action_items"],
        verbose=options["verbose"]
    )
    
    # Initialize summarizer
    summarizer = TranscriptSummarizer(summary_options)
    
    # Process the document
    start_time = time.time()
    result = summarizer.summarize(text)
    
    # Ensure processing time is in metadata if not already
    if "metadata" not in result:
        result["metadata"] = {}
    
    if "processing_time_seconds" not in result["metadata"]:
        result["metadata"]["processing_time_seconds"] = time.time() - start_time
    
    return result

def compare_strategies(text: str, options: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Compare all division strategies on the same document."""
    # First, run document assessment
    doc_assessment = assess_document(text)
    
    # Get the division results for each strategy
    division_results = divide_document(
        text=text,
        compare_all=True,
        min_sections=options["min_sections"],
        target_tokens_per_section=options.get("target_tokens_per_section", 25000),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create combined results
    results = {
        "metadata": {
            "document_assessment": doc_assessment,
            "recommended_strategy": doc_assessment["recommended_strategy"],
            "text_length": len(text),
            "estimated_tokens": len(text) // 4,
            "total_processing_time": division_results["metadata"]["total_processing_time"]
        }
    }
    
    # Process each strategy
    for strategy in ["essential", "long", "complex"]:
        strategy_options = options.copy()
        strategy_options["division_strategy"] = strategy
        
        # Create a customized summarizer for this strategy
        summary_options = SummaryOptions(
            division_strategy=strategy,
            model_name=options["model_name"],
            temperature=options["temperature"],
            min_sections=options["min_sections"],
            target_tokens_per_section=options.get("target_tokens_per_section", 25000),
            include_action_items=options["include_action_items"],
            verbose=options["verbose"]
        )
        
        summarizer = TranscriptSummarizer(summary_options)
        
        with st.spinner(f"Processing with {strategy} strategy..."):
            # Get divisions for this strategy
            divisions = division_results[strategy]["divisions"]
            
            # Process divisions (without re-dividing)
            start_time = time.time()
            
            # Process divisions directly if we have a specialized method
            if hasattr(summarizer, "process_divisions"):
                result = summarizer.process_divisions(divisions, text)
            else:
                # Otherwise, manually process divisions
                doc_type = doc_assessment["doc_type"]
                division_summaries = []
                
                # Process each division
                for i, division in enumerate(divisions):
                    try:
                        prompt = f"Summarize this section ({i+1}/{len(divisions)}) of a document:\n\n{division['text']}"
                        summary = summarizer.llm_client.generate_completion(prompt)
                        division_summaries.append(summary)
                    except Exception as e:
                        logger.error(f"Error summarizing division {i+1}: {e}")
                        division_summaries.append(f"Error summarizing division {i+1}: {str(e)}")
                
                # Create synthesis prompt
                combined_summaries = "\n\n===== SECTION SEPARATOR =====\n\n".join([
                    f"SECTION {i+1}:\n{summary}" for i, summary in enumerate(division_summaries)
                ])
                
                synthesis_prompt = f"Create a cohesive, well-structured summary from these section summaries:\n\n{combined_summaries}"
                
                # Generate final summary
                final_summary = summarizer.llm_client.generate_completion(synthesis_prompt)
                
                # Create result
                result = {
                    "summary": final_summary,
                    "divisions": divisions,
                    "division_summaries": division_summaries,
                    "metadata": {
                        "division_count": len(divisions),
                        "division_strategy": strategy,
                        "model": summarizer.options.model_name,
                        "processing_time_seconds": time.time() - start_time
                    }
                }
            
            # Add division processing time from the comparison
            result["metadata"]["division_processing_time"] = division_results[strategy]["processing_time"]
            
            # Add to results
            results[strategy] = result
    
    return results

def display_comparison_results(results: Dict[str, Dict[str, Any]], doc_info: Dict[str, Any]):
    """Display comparison results between different strategies."""
    st.markdown("## Strategy Comparison Results")
    
    # Get document assessment
    doc_assessment = results.get("metadata", {}).get("document_assessment", {})
    recommended = results.get("metadata", {}).get("recommended_strategy", "")
    
    # Show recommendation
    if recommended:
        st.info(f"**Recommended Strategy**: Based on document analysis, the '**{recommended}**' strategy is recommended for this document.")
    
    # Create metrics for each strategy
    metrics_cols = st.columns(3)
    
    for i, strategy in enumerate(["essential", "long", "complex"]):
        if strategy in results:
            strategy_result = results[strategy]
            division_count = len(strategy_result.get("divisions", []))
            division_time = strategy_result.get("metadata", {}).get("division_processing_time", 0)
            total_time = strategy_result.get("metadata", {}).get("processing_time_seconds", 0)
            
            with metrics_cols[i]:
                st.metric(
                    f"{strategy.capitalize()} Strategy", 
                    f"{division_count} divisions", 
                    f"{division_time:.2f}s to divide"
                )
    
    # Create visualization comparing strategies
    with st.expander("Strategy Comparison Visualization", expanded=True):
        # Create comparison chart
        fig = visualizations.create_comparison_chart(
            {strategy: results[strategy] for strategy in ["essential", "long", "complex"] if strategy in results}
        )
        if fig:
            st.pyplot(fig)
    
    # Create tabs for each strategy
    tabs = st.tabs([f"{s.capitalize()} Strategy" for s in ["essential", "long", "complex"]])
    
    for i, (strategy, tab) in enumerate(zip(["essential", "long", "complex"], tabs)):
        with tab:
            if strategy in results:
                ui_helpers.display_processing_result(
                    results[strategy], 
                    doc_info,
                    display_divisions=True
                )
    
    # Option to save comparison
    store = output_store.OutputStore()
    st.markdown("### Save Comparison Results")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        comparison_title = st.text_input(
            "Comparison Title",
            value=f"{doc_info.get('title', 'Document')} Strategy Comparison"
        )
    
    with col2:
        if st.button("Save Comparison", type="primary"):
            if comparison_title:
                try:
                    # Prepare results for storage
                    store_results = {}
                    for strategy in ["essential", "long", "complex"]:
                        if strategy in results:
                            store_results[strategy] = {
                                "summary": results[strategy].get("summary", ""),
                                "metadata": results[strategy].get("metadata", {})
                            }
                    
                    # Store document info
                    store_info = {
                        "title": doc_info.get("title", "Document"),
                        "num_characters": doc_info.get("num_characters", len(doc_info.get("content", ""))),
                        "num_words": doc_info.get("num_words", 0)
                    }
                    
                    if "num_speakers" in doc_info:
                        store_info["num_speakers"] = doc_info["num_speakers"]
                    
                    # Store the comparison
                    file_path = store.store_comparison(
                        comparison_title, 
                        store_results,
                        store_info
                    )
                    
                    st.success(f"Saved comparison to: {file_path}")
                except Exception as e:
                    st.error(f"Error saving comparison: {str(e)}")

def main():
    """Main function for the single document processing page."""
    st.markdown("# 📄 Single Document Processing")
    
    # Get options from sidebar
    options = ui_helpers.create_sidebar_options()
    
    # Section for uploading a document
    st.markdown("## Upload Document")
    
    doc_info = ui_helpers.display_file_uploader(
        label="Upload Document",
        types=["txt", "md", "docx"],
        help_text="Upload a document to summarize"
    )
    
    if not doc_info:
        # Show sample documents if no upload
        st.markdown("### Or use a sample document")
        sample_option = st.selectbox(
            "Select a sample",
            ["None", "Earnings Call", "Meeting Transcript", "Technical Document"],
            index=0
        )
        
        if sample_option != "None":
            try:
                # Load sample document
                sample_path = os.path.join("data", f"{sample_option.lower().replace(' ', '_')}.txt")
                
                if os.path.exists(sample_path):
                    with open(sample_path, "r", encoding="utf-8") as f:
                        sample_text = f.read()
                    
                    doc_info = {
                        "filename": f"{sample_option}.txt",
                        "title": sample_option,
                        "content": sample_text,
                        "num_characters": len(sample_text),
                        "num_words": len(sample_text.split())
                    }
                    
                    st.success(f"Loaded sample: {sample_option}")
                    
                    # Try to detect speakers for display
                    from summarizer.division import extract_speakers
                    speakers = extract_speakers(sample_text)
                    if speakers:
                        doc_info["speakers"] = speakers
                        doc_info["num_speakers"] = len(speakers)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Characters", f"{doc_info['num_characters']:,}")
                    col2.metric("Words", f"{doc_info['num_words']:,}")
                    if "num_speakers" in doc_info:
                        col3.metric("Speakers", doc_info["num_speakers"])
                else:
                    st.warning(f"Sample file not found: {sample_path}")
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")
    
    # Processing section
    if doc_info and "content" in doc_info:
        st.markdown("## Processing Options")
        
        # Check if we're in compare mode
        compare_mode = options["division_strategy"] == "compare"
        
        if compare_mode:
            process_button = st.button(
                "Compare All Strategies",
                type="primary"
            )
            
            if process_button:
                with st.spinner("Comparing all strategies..."):
                    results = compare_strategies(doc_info["content"], options)
                    
                    if results:
                        display_comparison_results(results, doc_info)
        else:
            # Single strategy processing
            process_button = st.button("Process Document", type="primary")
            
            if process_button:
                with st.spinner(f"Processing with {options['division_strategy']} strategy..."):
                    result = process_document(doc_info["content"], options)
                    
                    if result:
                        st.markdown("## Processing Results")
                        
                        # Display the results
                        ui_helpers.display_processing_result(
                            result, 
                            doc_info,
                            display_divisions=True
                        )
                        
                        # Add visualizations
                        with st.expander("Visualizations", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Word cloud
                                fig = visualizations.create_wordcloud(
                                    doc_info["content"], 
                                    title="Document Word Frequency"
                                )
                                if fig:
                                    st.pyplot(fig)
                                
                            with col2:
                                # Section lengths
                                fig = visualizations.create_section_lengths_chart(result)
                                if fig:
                                    st.pyplot(fig)
                                
                                # Speaker distribution if available
                                if "speakers" in doc_info:
                                    fig = visualizations.create_speaker_distribution(
                                        doc_info["content"], 
                                        doc_info["speakers"]
                                    )
                                    if fig:
                                        st.pyplot(fig)
                        
                        # Option to save results
                        store = output_store.OutputStore()
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            summary_title = st.text_input(
                                "Summary Title",
                                value=doc_info.get("title", "Document Summary")
                            )
                        
                        with col2:
                            if st.button("Save Summary", type="primary"):
                                if summary_title:
                                    try:
                                        # Store the summary
                                        file_path = store.store_single_document_summary(
                                            summary_title, 
                                            result,
                                            doc_info
                                        )
                                        
                                        st.success(f"Saved summary to: {file_path}")
                                    except Exception as e:
                                        st.error(f"Error saving summary: {str(e)}")

# Run the page
if __name__ == "__main__":
    main()