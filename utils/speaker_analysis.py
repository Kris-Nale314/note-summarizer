"""
Speaker analysis utilities for transcript processing.

This module provides tools to analyze speaker patterns and contributions
in meeting transcripts, including interaction analysis, speaking time,
and key point extraction by speaker.
"""

import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
    SENTIMENT_AVAILABLE = True
except:
    SENTIMENT_AVAILABLE = False
    logger.warning("NLTK SentimentIntensityAnalyzer not available. Install with: nltk.download('vader_lexicon')")


class SpeakerAnalyzer:
    """
    Analyzer for speaker contributions and patterns in meeting transcripts.
    """
    
    def __init__(self, transcript: str):
        """
        Initialize the speaker analyzer.
        
        Args:
            transcript: The full transcript text
        """
        self.transcript = transcript
        self.speakers = self._extract_speakers()
        self.speaker_segments = self._segment_by_speaker()
        logger.info(f"Initialized SpeakerAnalyzer with {len(self.speakers)} detected speakers")
    
    def _extract_speakers(self) -> List[str]:
        """
        Extract speakers from the transcript.
        
        Returns:
            List of speaker names
        """
        # Common speaker patterns in transcripts
        patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+):', # Full name: pattern
            r'([A-Z][a-z]+):', # First name: pattern
            r'(Dr\. [A-Z][a-z]+):', # Dr. Name: pattern
            r'(Mr\. [A-Z][a-z]+):', # Mr. Name: pattern
            r'(Mrs\. [A-Z][a-z]+):', # Mrs. Name: pattern
            r'(Ms\. [A-Z][a-z]+):', # Ms. Name: pattern
            r'(Speaker \d+):', # Speaker #: pattern
            r'\[(.*?)\]' # [Speaker] pattern
        ]
        
        all_speakers = []
        for pattern in patterns:
            matches = re.findall(pattern, self.transcript)
            all_speakers.extend(matches)
        
        # Remove duplicates while preserving order
        unique_speakers = []
        seen = set()
        for speaker in all_speakers:
            if speaker not in seen:
                seen.add(speaker)
                unique_speakers.append(speaker)
        
        logger.info(f"Extracted {len(unique_speakers)} speakers from transcript")
        return unique_speakers
    
    def _segment_by_speaker(self) -> Dict[str, List[str]]:
        """
        Segment the transcript by speaker.
        
        Returns:
            Dictionary mapping speakers to their segments
        """
        segments = defaultdict(list)
        
        # Create a regex pattern to match any speaker
        speaker_pattern = '|'.join([re.escape(speaker) for speaker in self.speakers])
        pattern = f"({speaker_pattern}):(.*?)(?=(?:{speaker_pattern}:|$))"
        
        # Find all speaker segments
        for match in re.finditer(pattern, self.transcript, re.DOTALL):
            speaker = match.group(1)
            text = match.group(2).strip()
            segments[speaker].append(text)
        
        logger.info(f"Segmented transcript by {len(segments)} speakers")
        return dict(segments)
    
    def get_speaker_stats(self) -> pd.DataFrame:
        """
        Get basic statistics for each speaker.
        
        Returns:
            DataFrame with speaker statistics
        """
        stats = []
        
        for speaker, segments in self.speaker_segments.items():
            # Calculate statistics
            num_segments = len(segments)
            total_words = sum(len(s.split()) for s in segments)
            avg_words_per_segment = total_words / num_segments if num_segments > 0 else 0
            
            # Count questions asked
            questions_asked = sum(s.count('?') for s in segments)
            
            # Analyze sentiment if available
            if SENTIMENT_AVAILABLE:
                combined_text = ' '.join(segments)
                sentiment = sia.polarity_scores(combined_text)
                sentiment_compound = sentiment['compound']
                sentiment_positive = sentiment['pos']
                sentiment_negative = sentiment['neg']
            else:
                sentiment_compound = None
                sentiment_positive = None
                sentiment_negative = None
            
            stats.append({
                'Speaker': speaker,
                'Segments': num_segments,
                'Words': total_words,
                'Avg Words Per Segment': round(avg_words_per_segment, 1),
                'Questions Asked': questions_asked,
                'Sentiment Compound': sentiment_compound,
                'Sentiment Positive': sentiment_positive,
                'Sentiment Negative': sentiment_negative
            })
        
        # Create DataFrame
        df = pd.DataFrame(stats)
        
        # Calculate percentages
        total_words = df['Words'].sum()
        df['Speaking Percentage'] = (df['Words'] / total_words * 100).round(1)
        
        # Sort by most talkative
        df = df.sort_values('Words', ascending=False)
        
        return df
    
    def analyze_interaction_patterns(self) -> Dict[str, Any]:
        """
        Analyze interaction patterns between speakers.
        
        Returns:
            Dictionary with interaction analysis
        """
        # Create a turn-taking sequence
        speaker_sequence = []
        
        # Speaker pattern for detecting transitions
        speaker_pattern = '|'.join([re.escape(speaker) for speaker in self.speakers])
        pattern = f"({speaker_pattern}):"
        
        for match in re.finditer(pattern, self.transcript):
            speaker = match.group(1)
            speaker_sequence.append(speaker)
        
        # Calculate speaker transitions (who speaks after whom)
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(speaker_sequence) - 1):
            current = speaker_sequence[i]
            next_speaker = speaker_sequence[i + 1]
            transitions[current][next_speaker] += 1
        
        # Convert to DataFrame for easier analysis
        transition_data = []
        
        for speaker1 in transitions:
            for speaker2, count in transitions[speaker1].items():
                transition_data.append({
                    'From': speaker1,
                    'To': speaker2,
                    'Count': count
                })
        
        transition_df = pd.DataFrame(transition_data)
        
        # Create a conversation flow graph
        speaker_nodes = list(self.speakers)
        
        # Calculate which speakers interact most with each other
        interaction_pairs = []
        
        for i, row in transition_df.iterrows():
            interaction_pairs.append({
                'Speakers': f"{row['From']} → {row['To']}",
                'Transitions': row['Count']
            })
        
        interaction_df = pd.DataFrame(interaction_pairs).sort_values('Transitions', ascending=False)
        
        # Calculate response patterns (who responds to questions)
        question_responses = defaultdict(list)
        
        for i in range(len(speaker_sequence) - 1):
            current = speaker_sequence[i]
            next_speaker = speaker_sequence[i + 1]
            
            # Check if current speaker's last segment has a question
            current_segments = self.speaker_segments[current]
            if current_segments and '?' in current_segments[-1]:
                question_responses[current].append(next_speaker)
        
        # Summarize question response patterns
        question_data = []
        
        for asker, responders in question_responses.items():
            responder_counts = defaultdict(int)
            for responder in responders:
                responder_counts[responder] += 1
            
            for responder, count in responder_counts.items():
                question_data.append({
                    'Questioner': asker,
                    'Responder': responder,
                    'Count': count
                })
        
        question_df = pd.DataFrame(question_data).sort_values('Count', ascending=False)
        
        return {
            'speaker_sequence': speaker_sequence,
            'transitions': transition_df,
            'interactions': interaction_df,
            'question_responses': question_df
        }
    
    def extract_key_points_by_speaker(self) -> Dict[str, List[str]]:
        """
        Extract potential key points made by each speaker.
        
        Returns:
            Dictionary mapping speakers to lists of key points
        """
        key_points = {}
        
        # Key point indicators
        indicators = [
            "important", "key", "critical", "essential", "crucial",
            "main point", "takeaway", "remember", "highlight",
            "priority", "focus on", "significant", "primary",
            "let me emphasize", "to summarize", "in conclusion"
        ]
        
        for speaker, segments in self.speaker_segments.items():
            speaker_points = []
            
            for segment in segments:
                # Split into sentences
                sentences = nltk.sent_tokenize(segment)
                
                for sentence in sentences:
                    # Check if sentence contains an indicator
                    if any(indicator in sentence.lower() for indicator in indicators):
                        speaker_points.append(sentence.strip())
                    
                    # Also include statements that are emphasized
                    elif re.search(r'\*\*.*\*\*|__.*__|MUST|SHOULD|NEED TO|ALWAYS|NEVER', sentence):
                        speaker_points.append(sentence.strip())
            
            # Only include speakers with key points
            if speaker_points:
                key_points[speaker] = speaker_points
        
        return key_points
    
    def get_topic_focus_by_speaker(self, topics: List[str]) -> pd.DataFrame:
        """
        Analyze each speaker's focus on different topics.
        
        Args:
            topics: List of topics to analyze
            
        Returns:
            DataFrame with topic focus by speaker
        """
        topic_focus = []
        
        for speaker, segments in self.speaker_segments.items():
            combined_text = ' '.join(segments).lower()
            
            topic_counts = {}
            for topic in topics:
                # Count word occurrences related to this topic
                topic_words = topic.lower().split()
                count = sum(combined_text.count(word) for word in topic_words if len(word) > 3)
                topic_counts[topic] = count
            
            # Add to results
            topic_focus.append({
                'Speaker': speaker,
                **topic_counts
            })
        
        return pd.DataFrame(topic_focus)
    
    def get_sentiment_over_time(self) -> Dict[str, Any]:
        """
        Analyze how sentiment changes over time for each speaker.
        
        Returns:
            Dictionary with sentiment analysis over time
        """
        if not SENTIMENT_AVAILABLE:
            return {"error": "Sentiment analysis not available. Install nltk vader_lexicon."}
        
        sentiment_data = []
        
        for speaker, segments in self.speaker_segments.items():
            # Analyze each segment chronologically
            for i, segment in enumerate(segments):
                sentiment = sia.polarity_scores(segment)
                
                sentiment_data.append({
                    'Speaker': speaker,
                    'Segment': i + 1,
                    'Text': segment[:100] + "..." if len(segment) > 100 else segment,
                    'Compound': sentiment['compound'],
                    'Positive': sentiment['pos'],
                    'Negative': sentiment['neg'],
                    'Neutral': sentiment['neu']
                })
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Calculate rolling averages
        sentiment_by_speaker = {}
        
        for speaker in self.speakers:
            speaker_data = sentiment_df[sentiment_df['Speaker'] == speaker]
            
            if len(speaker_data) > 2:
                # Calculate 3-segment rolling average
                rolling = speaker_data['Compound'].rolling(window=min(3, len(speaker_data)), min_periods=1).mean()
                
                sentiment_by_speaker[speaker] = {
                    'segments': speaker_data['Segment'].tolist(),
                    'compound': speaker_data['Compound'].tolist(),
                    'rolling_avg': rolling.tolist()
                }
        
        return {
            'sentiment_df': sentiment_df,
            'by_speaker': sentiment_by_speaker
        }
    
    def summarize_speaker_roles(self) -> Dict[str, str]:
        """
        Attempt to summarize each speaker's role in the conversation.
        
        Returns:
            Dictionary mapping speakers to roles
        """
        roles = {}
        
        stats_df = self.get_speaker_stats()
        interaction = self.analyze_interaction_patterns()
        
        for speaker in self.speakers:
            # Get speaker's stats
            speaker_stats = stats_df[stats_df['Speaker'] == speaker]
            
            if speaker_stats.empty:
                continue
                
            words_pct = speaker_stats['Speaking Percentage'].values[0]
            questions = speaker_stats['Questions Asked'].values[0]
            
            # Determine possible role
            role = ""
            
            # Check if they speak the most
            if words_pct == stats_df['Speaking Percentage'].max():
                if questions > 5:
                    role = "Discussion Leader/Facilitator"
                else:
                    role = "Primary Speaker/Presenter"
            
            # Check if they ask many questions but don't speak much
            elif questions > 3 and words_pct < 15:
                role = "Questioner/Interviewer"
            
            # Check if they respond to questions often
            elif speaker in interaction['question_responses']['Responder'].values:
                responder_count = sum(interaction['question_responses'][interaction['question_responses']['Responder'] == speaker]['Count'])
                if responder_count > 3:
                    role = "Subject Matter Expert"
            
            # Default roles based on speaking amount
            elif words_pct > 20:
                role = "Major Contributor"
            elif words_pct > 10:
                role = "Active Participant"
            elif words_pct > 5:
                role = "Regular Participant"
            else:
                role = "Occasional Contributor"
            
            roles[speaker] = role
        
        return roles


def analyze_transcript_speakers(transcript: str) -> Dict[str, Any]:
    """
    Perform comprehensive speaker analysis on a transcript.
    
    Args:
        transcript: The transcript text
        
    Returns:
        Dictionary with various speaker analyses
    """
    analyzer = SpeakerAnalyzer(transcript)
    
    results = {
        'speakers': analyzer.speakers,
        'stats': analyzer.get_speaker_stats(),
        'interaction': analyzer.analyze_interaction_patterns(),
        'key_points': analyzer.extract_key_points_by_speaker(),
        'roles': analyzer.summarize_speaker_roles()
    }
    
    # Add sentiment analysis if available
    if SENTIMENT_AVAILABLE:
        results['sentiment'] = analyzer.get_sentiment_over_time()
    
    return results