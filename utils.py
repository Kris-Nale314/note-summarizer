# utils.py
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

def chunk_transcript_by_character(transcript, chunk_size=2000, overlap=100):
    """
    Splits a transcript into chunks of approximately `chunk_size` characters,
    with an overlap of `overlap` characters between chunks.

    Args:
        transcript (str): The text of the transcript to chunk.
        chunk_size (int): The desired size of each chunk (in characters).
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of string chunks.
    """
    chunks = []
    start = 0
    while start < len(transcript):
        end = min(start + chunk_size, len(transcript))
        chunks.append(transcript[start:end])
        start += chunk_size - overlap  # Overlap to maintain context
    return chunks