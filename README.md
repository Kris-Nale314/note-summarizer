# 🎙️   Note-Summarizer    📝 

> **Transform Messy Transcripts into Useful Notes**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Status: Beta](https://img.shields.io/badge/status-beta-green)](https://github.com/kris-nale314/note-summarizer)

## 📝 What Is This?

`Note-Summarizer` is a practical tool that turns long meeting transcripts, especially from Microsoft Teams, into organized, actionable summaries. It's designed for busy professionals who need to quickly extract meaningful insights from lengthy meeting text.

The current version uses a lean architecture with hierarchical processing to efficiently analyze and summarize documents while maintaining both detail and coherence.

## 🌟 Key Features

- **Hierarchical Processing** - Multi-level approach that preserves both detail and coherence
- **Smart Speaker Detection** - Preserves speaker attributions for key points
- **Teams Transcript Optimization** - Specially designed for Microsoft Teams meeting transcripts
- **Customizable Detail Levels** - Choose between brief, standard, or detailed summaries
- **Automatic Action Item Extraction** - Identifies tasks, commitments, and follow-ups
- **Topic Extraction** - Highlights the most important topics across the document
- **Clean, Modern Interface** - Easy-to-use design that works in both light and dark modes
- **Lean Architecture** - Efficient processing with minimal overhead

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key

### Quick Start

```bash
# Clone this repository
git clone https://github.com/Kris-Nale314/note-summarizer.git
cd note-summarizer

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Set up your OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Launch the app
note-summarizer
```

### Using in Your Own Code

You can also use Note-Summarizer in your own Python code:

```python
import os
from lean import create_pipeline, ProcessingOptions

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create configuration options
options = ProcessingOptions(
    model_name="gpt-3.5-turbo",
    detail_level="detailed",
    include_action_items=True,
    min_chunks=3
)

# Create the pipeline
pipeline = create_pipeline(options=options)
orchestrator = pipeline['orchestrator']

# Process a document
async def process_document(text):
    # Process with progress updates
    def update_progress(progress, message):
        print(f"Progress: {progress*100:.0f}% - {message}")
    
    result = await orchestrator.process_document(text, progress_callback=update_progress)
    print(result['summary'])
    return result

# Run with asyncio
import asyncio
result = asyncio.run(process_document(my_document_text))
```

## 🛠️ The Tech Behind It

Note-Summarizer uses an innovative, lean architecture:

1. **Hierarchical Processing** - Processes text at multiple layers for both detail and coherence
2. **Smart Document Analysis** - Quickly analyzes document type and context
3. **Efficient Chunking** - Divides documents based on natural break points
4. **Parallel Processing** - Processes chunks concurrently for faster results
5. **Context-Aware Synthesis** - Hierarchically combines summaries with full context

### Component Architecture

The system is built with a modular, lean design:

- **DocumentAnalyzer**: Quick initial document analysis
- **DocumentChunker**: Smart text division with natural boundaries
- **ChunkSummarizer**: Individual chunk processing with rich metadata
- **Synthesizer**: Multi-level summary combination
- **Orchestrator**: Controls workflow and optimizes performance
- **Booster**: Enhances processing with caching and concurrency

## 🧠 How It Works

1. **Document Analysis**: The system quickly analyzes the document to determine its type, purpose, and key characteristics
2. **Smart Chunking**: The document is divided into optimal chunks based on natural boundaries
3. **Parallel Processing**: Each chunk is summarized independently with detailed metadata
4. **Hierarchical Synthesis**: 
   - **Level 1**: Combines adjacent chunks into intermediate summaries
   - **Level 2**: Combines intermediate summaries into higher-level summaries
   - **Level 3**: (For detailed-complex) Creates an additional synthesis layer
   - **Final Level**: Generates the complete document summary

5. **Topic Extraction**: Key topics are identified and ranked by importance
6. **Action Item Detection**: Tasks and commitments are automatically extracted

## 🔮 Roadmap

This is an active development project with several exciting enhancements planned:

- **AI Agent Architecture** - Transitioning to specialized agents for different processing tasks
- **Expanded Document Types** - Adding support for earnings calls, articles, and technical documents
- **Enhanced Interactive Visualizations** - Adding topic networks and importance heatmaps
- **Multi-Document Processing** - Enabling comparison across related documents

## 💼 Real-World Applications

- **Meeting Follow-ups**: Transform hour-long team meetings into concise action plans
- **Knowledge Management**: Create searchable archives of meeting insights
- **Time Savings**: Get the essence of meetings you couldn't attend
- **Team Alignment**: Ensure everyone has the same understanding of discussion outcomes

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

> "The art of communication is the language of leadership." — James Humes

Made with ❤️ to save you time and capture what matters