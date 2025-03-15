# summarizer/document_analysis.py
import re
import logging
import nltk
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import math

# Try to ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """
    Analyzes document structure, complexity and content to inform summarization.
    """
    
    def __init__(self, text: str):
        """Initialize with document text."""
        self.text = text
        self.stats = {}
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self._analyze()
    
    def _analyze(self):
        """Perform basic analysis of the document."""
        # Basic metrics
        self.stats['char_count'] = len(self.text)
        self.stats['word_count'] = len(self.text.split())
        
        # Sentence analysis
        sentences = nltk.sent_tokenize(self.text)
        self.stats['sentence_count'] = len(sentences)
        
        # Average sentence length
        sentence_lengths = [len(s.split()) for s in sentences]
        self.stats['avg_sentence_length'] = sum(sentence_lengths) / len(sentences) if sentences else 0
        self.stats['max_sentence_length'] = max(sentence_lengths) if sentences else 0
        
        # Paragraph analysis
        paragraphs = [p for p in self.text.split('\n\n') if p.strip()]
        self.stats['paragraph_count'] = len(paragraphs)
        
        # Estimate complexity using readability metrics
        self.stats['readability'] = self._calculate_readability(sentences)
        
        # Speaker analysis for transcripts
        speakers = self._extract_speakers()
        self.stats['speakers'] = speakers
        self.stats['speaker_count'] = len(speakers)
        
        # Extract potential topics
        self.stats['key_terms'] = self._extract_key_terms()
        
        # Estimate token count (rough approximation)
        self.stats['estimated_tokens'] = self._estimate_tokens()
        
        # Document type detection
        self.stats['document_type'] = self._detect_document_type()
        
        logger.info(f"Analyzed document: {self.stats['word_count']} words, {self.stats['sentence_count']} sentences")
    
    def _calculate_readability(self, sentences: List[str]) -> float:
        """Calculate a simple readability score."""
        if not sentences:
            return 0
            
        word_count = sum(len(s.split()) for s in sentences)
        syllable_count = sum(self._count_syllables(word) for s in sentences for word in s.split())
        
        # Modified Flesch Reading Ease score
        if word_count == 0 or len(sentences) == 0:
            return 0
            
        words_per_sentence = word_count / len(sentences)
        syllables_per_word = syllable_count / word_count if word_count > 0 else 0
        
        # Higher score = easier to read, lower = more complex
        score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        return round(score, 2)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)."""
        word = word.lower()
        # Remove non-alpha characters
        word = re.sub(r'[^a-z]', '', word)
        
        if not word:
            return 0
            
        # Special cases
        if len(word) <= 3:
            return 1
            
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        
        # Adjust for special patterns
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
            count += 1
        if count == 0:
            count = 1
            
        return count
    
    def _extract_speakers(self) -> List[str]:
        """Extract potential speakers from transcript text."""
        speakers = []
        
        # Common speaker patterns in transcripts
        patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+):', # Full name: pattern
            r'([A-Z][a-z]+):', # First name: pattern
            r'(Dr\. [A-Z][a-z]+):', # Dr. Name: pattern
            r'(Mr\. [A-Z][a-z]+):', # Mr. Name: pattern
            r'(Mrs\. [A-Z][a-z]+):', # Mrs. Name: pattern
            r'(Ms\. [A-Z][a-z]+):' # Ms. Name: pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, self.text)
            speakers.extend(matches)
        
        # Remove duplicates
        return list(set(speakers))
    
    def _extract_key_terms(self, max_terms: int = 10) -> List[str]:
        """Extract potential key terms/topics from the document."""
        # Tokenize and normalize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', self.text.lower())
        
        # Remove stopwords and count frequencies
        content_words = [w for w in words if w not in self.stopwords]
        word_freq = Counter(content_words)
        
        # Use TF-IDF-like approach to find important terms
        total_words = len(content_words)
        
        # Calculate importance score
        word_importance = {}
        for word, count in word_freq.items():
            # Term frequency normalized by document length
            tf = count / total_words
            # Inverse document frequency approximation (higher for less common words)
            idf = math.log(total_words / (count + 1)) + 1
            word_importance[word] = tf * idf
        
        # Return top terms
        important_terms = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in important_terms[:max_terms]]
    
    def _estimate_tokens(self) -> int:
        """Estimate the number of tokens (rough approximation)."""
        # A very simple estimation - about 4 characters per token on average for English
        return int(len(self.text) / 4)
    
    def _detect_document_type(self) -> str:
        """Attempt to detect the type of document."""
        # Check for transcript patterns
        if self.stats['speaker_count'] > 1 and self.stats['speakers']:
            speaker_pattern = '|'.join([re.escape(s) for s in self.stats['speakers']])
            speaker_matches = re.findall(f"({speaker_pattern}):", self.text)
            
            # If we find enough speaker markers, it's likely a transcript
            if len(speaker_matches) > 5:
                return "transcript"
        
        # Check for markdown/structured document
        if re.search(r'#{1,6}\s+', self.text):
            return "markdown"
            
        # Check for code content
        code_patterns = [
            r'function\s+\w+\s*\(',
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'import\s+\w+',
            r'<\s*[a-zA-Z][\w.-]*[^<>]*>',  # HTML tags
            r'\{\s*"[^"]+"\s*:\s*'  # JSON patterns
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, self.text):
                return "technical_content"
        
        # Default to generic text
        return "general_text"
    
    def get_stats(self) -> Dict[str, Any]:
        """Return the document statistics."""
        return self.stats
    
    def quantize_text(self) -> str:
        """
        Produce a 'quantized' version of the text with reduced token usage.
        Removes articles, common conjunctions, and normalizes formatting.
        """
        # Articles, conjunctions, and other common words to remove
        to_remove = self.stopwords.union({
            'the', 'a', 'an', 'and', 'but', 'or', 'for', 'nor', 'so', 'yet',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having',
            'that', 'which', 'who', 'whom', 'whose', 'this', 'these', 'those'
        })
        
        # Split into sentences
        sentences = nltk.sent_tokenize(self.text)
        
        # Process each sentence
        quantized_sentences = []
        for sentence in sentences:
            # Tokenize words and filter out stop words
            words = [w for w in re.findall(r'\b\w+\b', sentence) if w.lower() not in to_remove]
            
            # Reconstruct sentence and add basic punctuation
            if words:
                quantized_sentence = ' '.join(words)
                # Ensure it ends with punctuation
                if not quantized_sentence[-1] in '.!?':
                    quantized_sentence += '.'
                quantized_sentences.append(quantized_sentence)
        
        # Join sentences
        return ' '.join(quantized_sentences)
    
    def recommend_chunking_strategy(self) -> str:
        """Recommend the best chunking strategy based on document analysis."""
        doc_type = self.stats['document_type']
        
        if doc_type == "transcript" and self.stats['speaker_count'] > 1:
            return "speaker"
        elif doc_type == "markdown" or doc_type == "technical_content":
            return "boundary"
        else:
            return "simple"
    
    def recommend_chunk_size(self) -> int:
        """Recommend an appropriate chunk size based on document complexity."""
        # Base chunk size
        base_size = 2000
        
        # Adjust for document complexity
        if self.stats['avg_sentence_length'] > 25:
            # Longer sentences = smaller chunks to maintain coherence
            base_size -= 500
        
        if self.stats['readability'] < 50:
            # More complex text = smaller chunks
            base_size -= 500
        
        # Adjust for document length
        if self.stats['word_count'] > 10000:
            # Longer documents can handle larger chunks
            base_size += 500
        
        # Ensure reasonable bounds
        return max(1000, min(base_size, 3000))
    
    def suggest_preprocessing(self) -> List[str]:
        """Suggest preprocessing steps based on document analysis."""
        suggestions = []
        
        if self.stats['estimated_tokens'] > 15000:
            suggestions.append("quantize_text")
        
        if self.stats['readability'] < 30:
            suggestions.append("simplify_complex_sentences")
        
        if self.stats['avg_sentence_length'] > 30:
            suggestions.append("break_long_sentences")
        
        return suggestions