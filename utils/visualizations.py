"""
Visualization utilities for document analysis.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import io
import base64

def create_wordcloud(text: str, title: str = "Word Frequency", 
                    max_words: int = 100, width: int = 800, height: int = 400) -> Optional[plt.Figure]:
    """Create and display a word cloud from text."""
    try:
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.pyplot as plt
        
        # Setup custom stopwords
        stopwords = set(STOPWORDS)
        custom_stopwords = {
            "would", "could", "should", "think", "going", "know", "like", "just",
            "call", "yeah", "um", "uh", "okay", "one", "thing", "way", "good", 
            "right", "want", "say", "well", "time", "get", "go", "see", "look", 
            "make", "lot", "kind", "really"
        }
        stopwords.update(custom_stopwords)
        
        # Create wordcloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color="white",
            stopwords=stopwords,
            colormap="viridis",
            contour_width=1,
            contour_color="steelblue"
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title)
        ax.axis("off")
        
        return fig
    
    except ImportError:
        st.warning("Please install wordcloud: `pip install wordcloud`")
        return None
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def create_comparison_chart(results: Dict[str, Dict[str, Any]]) -> Optional[plt.Figure]:
    """Create a comparison chart for different strategies."""
    try:
        strategies = list(results.keys())
        processing_times = [
            results[s].get("metadata", {}).get("processing_time_seconds", 0) 
            for s in strategies
        ]
        
        division_counts = [
            results[s].get("metadata", {}).get("division_count", 0)
            for s in strategies
        ]
        
        summary_lengths = [
            len(results[s].get("summary", ""))
            for s in strategies
        ]
        
        # Normalize values for comparison
        max_time = max(processing_times) if processing_times else 1
        max_divisions = max(division_counts) if division_counts else 1
        max_length = max(summary_lengths) if summary_lengths else 1
        
        normalized_times = [t/max_time for t in processing_times]
        normalized_divisions = [d/max_divisions for d in division_counts]
        normalized_lengths = [l/max_length for l in summary_lengths]
        
        # Set up a plot with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Strategy Comparison", fontsize=16)
        
        # Plot processing times
        ax1.bar(strategies, processing_times, color='skyblue')
        ax1.set_ylabel('Seconds')
        ax1.set_title('Processing Time')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot division counts
        ax2.bar(strategies, division_counts, color='lightgreen')
        ax2.set_ylabel('Count')
        ax2.set_title('Number of Divisions')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot summary lengths
        ax3.bar(strategies, summary_lengths, color='salmon')
        ax3.set_ylabel('Characters')
        ax3.set_title('Summary Length')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return None

def create_speaker_distribution(text: str, speakers: List[str]) -> Optional[plt.Figure]:
    """Create a visualization of speaker distribution."""
    try:
        # Count speaker occurrences
        speaker_counts = {}
        for speaker in speakers:
            pattern = f"{speaker}:"
            count = text.count(pattern)
            if count > 0:
                speaker_counts[speaker] = count
        
        if not speaker_counts:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        speakers = list(speaker_counts.keys())
        counts = list(speaker_counts.values())
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        speakers = [speakers[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Limit to top 10 speakers if there are many
        if len(speakers) > 10:
            speakers = speakers[:10]
            counts = counts[:10]
        
        # Create bar chart
        bars = ax.bar(speakers, counts, color='lightblue')
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count/total:.1%}',
                    ha='center', va='bottom', rotation=0)
        
        ax.set_title('Speaker Distribution')
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Number of Utterances')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating speaker distribution: {str(e)}")
        return None

def export_figure_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getbuffer()).decode("utf-8")

def create_topic_flow(result: Dict[str, Any]) -> Optional[plt.Figure]:
    """Create a topic flow visualization from division summaries."""
    try:
        if "division_summaries" not in result or not result["division_summaries"]:
            return None
            
        # Extract key topics from each division summary
        summaries = result["division_summaries"]
        
        # Use simple keyword extraction for topics
        import re
        from collections import Counter
        
        # Extract topics for each division
        division_topics = []
        for summary in summaries:
            # Simple processing to extract key noun phrases
            # Remove common stopwords
            words = re.findall(r'\b[A-Za-z]{3,}\b', summary.lower())
            stopwords = {"the", "and", "for", "that", "this", "are", "with", "was",
                        "they", "have", "from", "has", "been", "were", "summary", 
                        "section", "document", "also", "which", "their", "there"}
            filtered_words = [w for w in words if w not in stopwords]
            
            # Count occurrences
            word_counts = Counter(filtered_words)
            
            # Take top 5 words as topics
            topics = [word for word, count in word_counts.most_common(5)]
            division_topics.append(topics)
        
        # Create a flow visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a matrix for transitions between topics
        unique_topics = set()
        for topics in division_topics:
            unique_topics.update(topics)
        
        unique_topics = sorted(list(unique_topics))
        topic_idx = {topic: i for i, topic in enumerate(unique_topics)}
        
        # Create a matrix to count topic transitions
        transition_matrix = np.zeros((len(unique_topics), len(unique_topics)))
        
        # Fill the transition matrix
        for i in range(len(division_topics) - 1):
            for topic1 in division_topics[i]:
                for topic2 in division_topics[i + 1]:
                    if topic1 in topic_idx and topic2 in topic_idx:
                        transition_matrix[topic_idx[topic1], topic_idx[topic2]] += 1
        
        # Normalize for plotting
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis], 
                                  where=row_sums[:, np.newaxis] != 0)
        
        # Plot heatmap
        im = ax.imshow(transition_matrix, cmap='YlGnBu')
        
        # Add labels
        ax.set_xticks(np.arange(len(unique_topics)))
        ax.set_yticks(np.arange(len(unique_topics)))
        ax.set_xticklabels(unique_topics)
        ax.set_yticklabels(unique_topics)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Transition Probability", rotation=-90, va="bottom")
        
        ax.set_title("Topic Flow Between Document Sections")
        fig.tight_layout()
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating topic flow visualization: {str(e)}")
        return None

def create_section_lengths_chart(result: Dict[str, Any]) -> Optional[plt.Figure]:
    """Create a visualization of section lengths."""
    try:
        if "divisions" not in result or not result["divisions"]:
            return None
            
        # Extract section lengths
        sections = result["divisions"]
        section_lengths = [len(section["text"]) for section in sections]
        section_nums = list(range(1, len(sections) + 1))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot section lengths
        ax.bar(section_nums, section_lengths, color='lightblue')
        
        # Add section labels
        for i, v in enumerate(section_lengths):
            ax.text(i + 1, v + 5, f"{v:,}", ha='center')
        
        ax.set_title('Section Lengths (Characters)')
        ax.set_xlabel('Section Number')
        ax.set_ylabel('Length')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error creating section lengths chart: {str(e)}")
        return None

def create_multi_document_comparison(results: Dict[str, Dict[str, Any]], 
                                   key_metric: str = "companies") -> Optional[plt.Figure]:
    """Create a comparison visualization for multi-document analysis."""
    try:
        # Check if we have companies or periods to compare
        if key_metric not in results.get("metadata", {}):
            return None
            
        labels = results["metadata"][key_metric]
        if not labels or not isinstance(labels, list):
            return None
            
        # Extract summary for each label (company/period)
        summaries = results.get("individual_summaries", [])
        if len(summaries) != len(labels):
            return None
            
        # Extract word frequencies for each summary
        import re
        from collections import Counter
        
        word_freqs = []
        for summary in summaries:
            words = re.findall(r'\b[A-Za-z]{3,}\b', summary.lower())
            stopwords = {"the", "and", "for", "that", "this", "are", "with", "was",
                        "they", "have", "from", "has", "been", "were", "summary", 
                        "also", "which", "their", "there"}
            filtered_words = [w for w in words if w not in stopwords]
            word_freqs.append(Counter(filtered_words))
        
        # Find common top words across all documents
        all_words = Counter()
        for freq in word_freqs:
            all_words.update(freq)
            
        top_words = [word for word, _ in all_words.most_common(10)]
        
        # Create a dataframe for plotting
        data = []
        for i, label in enumerate(labels):
            if i < len(word_freqs):
                for word in top_words:
                    freq = word_freqs[i].get(word, 0)
                    data.append({"Label": label, "Word": word, "Frequency": freq})
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Pivot data for heatmap
        pivot_df = df.pivot(index="Label", columns="Word", values="Frequency")
        
        # Normalize by row
        normalized_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
        
        # Plot heatmap
        im = ax.imshow(normalized_df.values, cmap='YlGnBu')
        
        # Add labels
        ax.set_xticks(np.arange(len(pivot_df.columns)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels(pivot_df.columns)
        ax.set_yticklabels(pivot_df.index)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Normalized Frequency", rotation=-90, va="bottom")
        
        # Add title based on key metric
        title = "Company Comparison" if key_metric == "companies" else "Period Comparison"
        ax.set_title(f"{title}: Key Term Distribution")
        
        fig.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error creating multi-document comparison: {str(e)}")
        return None