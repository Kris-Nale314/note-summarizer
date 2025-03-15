"""
Note Summarizer package for transforming documents into comprehensive summaries.
"""

from .core import TranscriptSummarizer
from .options import SummaryOptions
from .division import divide_document, extract_speakers

__all__ = ['TranscriptSummarizer', 'SummaryOptions', 'divide_document', 'extract_speakers']