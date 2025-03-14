"""
Test script for advanced transcript chunking logic.
"""

import os
import sys
from utils.chunking import (
    ChunkingProcessor,
    chunk_transcript_advanced,
    extract_speakers
)

# Sample transcript with multiple speakers
SAMPLE_TRANSCRIPT = """
John: Hi everyone, thanks for joining today's meeting. We have a lot to discuss.

Mary: Thanks John. I'm excited to go through the Q3 results together.

John: Great! Let's start with the sales numbers. We hit 105% of our target for Q3.

Bob: That's fantastic news! How did we achieve such great results?

John: Good question, Bob. Our new marketing campaign was very effective, and we saw a 30% increase in web traffic.

Mary: I'd also like to point out that customer retention improved by 15% compared to Q2.

Bob: That's impressive. What about our expansion into the European market?

John: We're on track with that project. Phase 1 is complete, and we're starting Phase 2 next month.

Mary: I'd like to discuss the budget for Phase 2. I think we need to increase it by around 10%.

Bob: I agree with Mary. The initial results are promising, and we should capitalize on that momentum.

John: Alright, let's add that to the action items. We'll review the budget and make a decision by the end of the week.

Mary: Next, I'd like to discuss our product roadmap for Q4. We need to prioritize the features for the next release.

Bob: I think we should focus on the mobile app improvements first, as we're seeing a lot of demand from our users.

John: I agree with Bob. The analytics show that 60% of our users are accessing the platform via mobile.

Mary: That makes sense. Let's plan for a mid-November release for the mobile updates.

Bob: What about the desktop redesign? Should we push that to Q1?

John: Yes, I think that's wise. Let's focus on mobile for now and revisit the desktop redesign in January.

Mary: Perfect. I'll update the roadmap and share it with the team.

John: Great! The last item on our agenda is the hiring plan for Q4. HR has identified several key positions we need to fill.

Bob: How many positions are we looking at?

John: We have 5 open roles: 2 developers, 1 designer, 1 product manager, and 1 marketing specialist.

Mary: Do we have budget approval for all these positions?

John: Yes, they're all included in the Q4 budget we approved last month.

Bob: Excellent. When do we expect to have these positions filled?

John: The goal is to have offers extended by the end of November, with start dates in early December or January.

Mary: Sounds good. Anything else we need to discuss today?

John: I think we've covered everything. Thanks, everyone, for your time and input.

Bob: Thank you for leading the meeting, John. Very productive.

Mary: Agreed. Have a great day, everyone!
"""

def print_separator():
    """Print a separator line."""
    print("\n" + "=" * 80 + "\n")

def test_all_chunking_strategies():
    """Test all available chunking strategies."""
    print("Testing all chunking strategies on sample transcript...")
    
    # Verify the sample transcript isn't empty
    print(f"Sample transcript length: {len(SAMPLE_TRANSCRIPT)} characters")
    if len(SAMPLE_TRANSCRIPT) < 100:
        print("ERROR: Sample transcript appears too short or empty!")
        return {}
    
    processor = ChunkingProcessor(
        default_chunk_size=500,  # Smaller chunk size for testing
        default_chunk_overlap=100
    )
    
    # Extract speakers and boundaries first
    speakers = extract_speakers(SAMPLE_TRANSCRIPT)
    boundaries = processor.detect_document_boundaries(SAMPLE_TRANSCRIPT)
    
    print(f"Detected {len(speakers)} speakers: {speakers}")
    print(f"Detected {len(boundaries)} potential boundaries")
    
    # Print details of detected boundaries for debugging
    print("Boundary details:")
    for i, boundary in enumerate(boundaries[:5]):  # Show first 5 boundaries
        print(f"  {i+1}. {boundary}")
    
    # Test each strategy
    strategies = [
        "fixed_size",
        "speaker_aware", 
        "boundary_aware"
    ]
    
    results = {}
    
    for strategy in strategies:
        print_separator()
        print(f"Testing {strategy} strategy...")
        
        # Custom parameters for different strategies to ensure better results
        kwargs = {}
        if strategy == "fixed_size":
            kwargs = {"chunk_size": 400, "chunk_overlap": 100}
        elif strategy == "speaker_aware": 
            kwargs = {"max_chunk_size": 800, "min_chunk_size": 300}
        elif strategy == "boundary_aware":
            kwargs = {"min_chunk_size": 200, "max_chunk_size": 800}
        
        result = processor.chunk_document(
            text=SAMPLE_TRANSCRIPT,
            strategy=strategy,
            compute_metrics=True,
            **kwargs
        )
        
        chunks = result["chunks"]
        metrics = result["metrics"]
        
        print(f"Generated {len(chunks)} chunks")
        print(f"Metrics: {metrics}")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} ({chunk.length} chars, speakers: {chunk.speakers}):")
            print(f"  Text preview: {chunk.text[:100]}...")
        
        results[strategy] = {
            "chunks": chunks,
            "metrics": metrics
        }
    
    # Print comparison
    print_separator()
    print("Strategy Comparison:")
    for strategy, result in results.items():
        chunks = result["chunks"]
        metrics = result["metrics"]
        print(f"{strategy}: {len(chunks)} chunks, avg size: {metrics['avg_chunk_size']:.1f} chars")
        if "sentence_integrity_score" in metrics:
            print(f"  Sentence integrity: {metrics['sentence_integrity_score']:.3f}")
        if "avg_speakers_per_chunk" in metrics:
            print(f"  Avg speakers per chunk: {metrics['avg_speakers_per_chunk']:.1f}")
    
    return results

def test_simple_speaker_chunking():
    """Test basic speaker chunking directly."""
    print_separator()
    print("Testing basic speaker chunking with direct method call...")
    
    processor = ChunkingProcessor()
    chunks = processor.chunk_text_speaker_aware(
        text=SAMPLE_TRANSCRIPT,
        max_chunk_size=800,
        min_chunk_size=300
    )
    
    print(f"Generated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({chunk.length} chars, speakers: {chunk.speakers}):")
        print(f"  Text preview: {chunk.text[:100]}...")
    
    return chunks

if __name__ == "__main__":
    print("Running advanced chunking tests...")
    try:
        # First test basic speaker extraction
        speakers = extract_speakers(SAMPLE_TRANSCRIPT)
        print(f"Extracted speakers: {speakers}")
        print_separator()
        
        # Test direct speaker chunking
        speaker_chunks = test_simple_speaker_chunking()
        
        # Test all strategies
        results = test_all_chunking_strategies()
        
        print_separator()
        print("All tests completed!")
    except Exception as e:
        import traceback
        print(f"Error in tests: {e}")
        traceback.print_exc()