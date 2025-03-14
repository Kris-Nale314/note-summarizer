# crew.py (Enhanced Version with Advanced Chunking)
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os
import concurrent.futures
from dotenv import load_dotenv
from utils import chunk_transcript_advanced, extract_speakers
from utils.chunking import ChunkingProcessor

load_dotenv()

# --- Tools ---
@tool
def read_and_chunk_transcript(transcript_content: str, num_chunks: int = None, overlap: int = 200, strategy: str = "speaker_aware") -> list[str]:
    """
    Reads a transcript, divides it into chunks using the specified strategy, and returns the chunks.
    
    Args:
        transcript_content: The transcript text
        num_chunks: Target number of chunks (used for fixed-size strategy)
        overlap: Overlap between chunks in characters
        strategy: Chunking strategy ('fixed_size', 'speaker_aware', 'boundary_aware', 'semantic')
        
    Returns:
        List of text chunks
    """
    try:
        # Use the advanced chunking processor
        processor = ChunkingProcessor(
            default_chunk_size=len(transcript_content) // (num_chunks or 5),
            default_chunk_overlap=overlap
        )
        
        # Apply the selected chunking strategy
        if strategy == "fixed_size":
            chunk_size = len(transcript_content) // (num_chunks or 5)
            chunk_objects = processor.chunk_text_fixed_size(
                text=transcript_content,
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
        elif strategy == "speaker_aware":
            chunk_objects = processor.chunk_text_speaker_aware(
                text=transcript_content,
                max_chunk_size=2000,
                min_chunk_size=500
            )
        elif strategy == "boundary_aware":
            chunk_objects = processor.chunk_text_boundary_aware(
                text=transcript_content,
                min_chunk_size=500,
                max_chunk_size=2000
            )
        elif strategy == "semantic":
            chunk_objects = processor.chunk_text_semantic(
                text=transcript_content
            )
        else:
            # Default to speaker_aware if invalid strategy
            chunk_objects = processor.chunk_text_speaker_aware(transcript_content)
        
        # Convert chunk objects to text for compatibility with existing code
        chunks = [chunk.text for chunk in chunk_objects]
        
        # If we didn't get enough chunks with the advanced methods, fall back to fixed-size
        if len(chunks) < 2:
            print(f"Warning: Only got {len(chunks)} chunks with {strategy} strategy, falling back to fixed-size")
            chunk_size = len(transcript_content) // (num_chunks or 5)
            chunks = chunk_transcript_advanced(
                transcript=transcript_content,
                num_chunks=num_chunks or 5,
                overlap=overlap,
                strategy="fixed_size"
            )
            
        return chunks
    except Exception as e:
        print(f"Error in chunking: {e}")
        # Fall back to basic character-based chunking in case of error
        chunk_size = len(transcript_content) // (num_chunks or 5)
        return chunk_transcript_advanced(transcript_content, num_chunks, overlap, "fixed_size")

# --- Agents ---
class TranscriptAgents:
    def __init__(self, model_name="gpt-3.5-turbo", verbose=True):
        self.verbose = verbose
        self.model_name = model_name
        
        # Use different temperature settings for different tasks
        self.summary_llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.2  # Lower temperature for more focused summaries
        )
        
        self.action_llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.1  # Very low temperature for precise action item extraction
        )
        
        self.synthesis_llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.3  # Slightly higher for creative synthesis
        )

    def create_summarizer(self):
        return Agent(
            role='Expert Transcript Summarizer and Detail Preserver',
            goal="""Create comprehensive, detailed summaries of transcript chunks that:
            1. Identify WHO is speaking at each significant point
            2. Capture ALL key discussion points, decisions, and nuances
            3. Preserve important quotes that convey tone or intent
            4. Note any disagreements, open questions, or unresolved issues
            5. Maintain the chronological flow and context of the conversation
            6. Highlight terminology or jargon specific to the discussion
            7. Indicate emotional tones or emphasis where relevant""",
            backstory="""You are a highly skilled transcript analyst with a 
            photographic memory for details. Your summaries are known for their 
            completeness, accuracy, and ability to capture both explicit and implicit 
            information. You have a special talent for recognizing patterns, identifying 
            speaker roles, and preserving the nuance of complex discussions.""",
            verbose=self.verbose,
            llm=self.summary_llm
        )

    def create_action_item_extractor(self):
        return Agent(
            role='Comprehensive Action Item Specialist',
            goal="""Meticulously extract EVERY explicit and implied action item in the transcript chunk:
            1. Format each action item as: "[PRIORITY: High/Medium/Low] - [WHO] - [WHAT] - [WHEN] - [CONTEXT]"
            2. Infer priority based on language, urgency, or speaker emphasis
            3. Identify WHO is responsible (specific names/roles; mark "Unassigned" if unclear)
            4. Clearly articulate WHAT needs to be done in actionable terms
            5. Specify WHEN it should be completed (exact date, timeframe, or "No deadline")
            6. Include brief CONTEXT explaining why this action matters
            7. List dependencies between actions when evident
            8. Note any follow-up or verification steps mentioned""",
            backstory="""You are a renowned action item specialist with experience 
            in project management and executive assistance. You can identify not just 
            explicit assignments but also implicit commitments and necessary follow-ups. 
            Your action lists are comprehensive, clearly prioritized, and include all 
            contextual information needed for execution. You excel at recognizing 
            dependencies and ensuring nothing falls through the cracks.""",
            verbose=self.verbose,
            llm=self.action_llm
        )

    def create_synthesizer(self):
        return Agent(
            role='Master Information Architect and Detail Preserving Synthesizer',
            goal="""Create a comprehensive, highly organized synthesis that:
            1. Preserves ALL important details from individual summaries
            2. Structures information logically by topics and subtopics
            3. Creates a coherent narrative flow throughout the document
            4. Uses hierarchical organization with clear headings and subheadings
            5. Combines related points while maintaining distinct perspectives
            6. Highlights areas of consensus and disagreement
            7. Consolidates action items into a comprehensive, prioritized list
            8. Provides a high-level executive summary at the beginning
            9. Includes a section specifically for open questions or unresolved issues
            10. Maintains chronological integrity where it aids understanding""",
            backstory="""You are a master information architect and knowledge 
            synthesizer with a background in technical writing, executive communications, 
            and information design. You excel at taking fragmented information and 
            transforming it into coherent, comprehensive documents that preserve all 
            important details while eliminating redundancy. Your output is known for its 
            logical structure, readability, and completeness.""",
            verbose=self.verbose,
            llm=self.synthesis_llm
        )

    def create_detailed_context_finder(self):
        """Agent to identify key contextual elements across chunks"""
        return Agent(
            role='Contextual Intelligence Analyst',
            goal="""Analyze the entire transcript to identify key contextual elements:
            1. Recurring themes, topics, and terminology
            2. Relationships between speakers and their roles/perspectives
            3. Background information that informs the discussion
            4. Evolution of ideas and positions throughout the conversation
            5. Underlying assumptions or shared knowledge
            6. Significant shifts in topic or tone
            7. Integration points between seemingly separate discussions""",
            backstory="""You are a contextual intelligence expert with a background 
            in discourse analysis and systems thinking. You excel at seeing the "big picture" 
            connections in complex discussions and identifying the contextual elements 
            that tie different parts together. Your insights help transform fragmented 
            information into coherent, meaningful narratives.""",
            verbose=self.verbose,
            llm=self.synthesis_llm
        )

    def create_speaker_analyzer(self):
        """Agent to analyze speaker patterns and dynamics"""
        return Agent(
            role='Conversation and Speaker Dynamics Specialist',
            goal="""Analyze speaker patterns and conversational dynamics in the transcript:
            1. Identify each speaker and their apparent role in the discussion
            2. Analyze communication styles and patterns for each speaker
            3. Map the flow of conversation and turn-taking patterns
            4. Identify power dynamics and influence patterns
            5. Note alignment and disagreement between participants
            6. Track how ideas evolve through speaker contributions""",
            backstory="""You are an expert in discourse analysis and communication 
            patterns with a background in sociolinguistics and conversation analysis. 
            You can identify subtle patterns in how people interact, who leads discussions, 
            and how ideas are built collaboratively. Your insights help teams understand 
            their communication dynamics and improve collaboration.""",
            verbose=self.verbose,
            llm=self.synthesis_llm
        )

# --- Crews ---
class SummarizerCrew:
    def __init__(self, chunks, agents):
        self.chunks = chunks
        self.agents = agents
        self.tasks = []
        self._create_tasks()

    def _create_tasks(self):
        for i, chunk in enumerate(self.chunks):
            task = Task(
                description=f"""Comprehensively summarize chunk {i+1} of the transcript:
                1. Begin by identifying all speakers and their apparent roles
                2. Preserve the chronological flow of the conversation
                3. Capture ALL key points, decisions, and nuanced positions
                4. Include important quotes that convey tone or intent (use "..." for direct quotes)
                5. Note areas of agreement and disagreement
                6. Highlight any technical terms or jargon with brief explanations
                7. Flag any open questions or unresolved issues
                8. Maintain all relevant context for each discussion point
                
                IMPORTANT: Your summary must be detailed enough that someone reading ONLY your 
                summary would understand all significant content from the chunk.""",
                agent=self.agents[i % len(self.agents)],
                expected_output="""A comprehensive, detailed summary that preserves all significant 
                content, speaker identification, and conversational nuance from the chunk.""",
                context=[],
                input=chunk
            )
            self.tasks.append(task)

    def run(self):
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # Using sequential processing for reliability
            verbose=True
        )
        results = crew.kickoff()
        return results

class ActionItemCrew:
    def __init__(self, chunks, agent):
        self.chunks = chunks
        self.agent = agent
        self.tasks = []
        self.create_tasks()

    def create_tasks(self):
        for i, chunk in enumerate(self.chunks):
            task = Task(
                description=f"""Extract and format ALL action items from chunk {i+1}:
                1. Format: "[PRIORITY: High/Medium/Low] - [WHO] - [WHAT] - [WHEN] - [CONTEXT]"
                2. Identify explicit AND implied action items
                3. Infer priority from language, urgency, and emphasis
                4. Specify WHO is responsible with exact names/roles when available
                5. Clearly articulate WHAT needs to be done in actionable language
                6. Include WHEN deadline (exact date, timeframe, or "No deadline specified")
                7. Add brief CONTEXT explaining why this action matters
                8. Note any dependencies or follow-up actions
                
                IMPORTANT: Be comprehensive; don't miss any potential action items, 
                including those that might be implied rather than explicitly stated.""",
                agent=self.agent,
                expected_output="""A comprehensive, prioritized list of ALL action items, 
                with responsibility, deadlines, and context clearly indicated.""",
                input=chunk
            )
            self.tasks.append(task)

    def run(self):
        crew = Crew(
            agents=[self.agent],
            tasks=self.tasks,
            process=Process.sequential,  # Using sequential processing for reliability
            verbose=True
        )
        results = crew.kickoff()
        return results

class ContextCrew:
    """Crew to extract cross-chunk context"""
    def __init__(self, all_chunks, agent):
        self.all_chunks = all_chunks
        self.agent = agent
        self.task = self.create_task()

    def create_task(self):
        combined_chunks = "\n\n--- CHUNK SEPARATOR ---\n\n".join(self.all_chunks)
        return Task(
            description="""Analyze the entire transcript to identify key contextual elements:
            1. Main themes and how they interconnect
            2. Speaker relationships and their evolution
            3. Terminology and jargon specific to this discussion
            4. Background information that informs the conversation
            5. Assumptions or shared knowledge that may not be explicit
            6. The "story arc" of the overall discussion
            
            IMPORTANT: Focus on context that spans across multiple chunks, not details 
            within a single chunk. Look for patterns, relationships, and frameworks that 
            help organize and understand the complete transcript.""",
            agent=self.agent,
            expected_output="""A concise contextual analysis that identifies key themes, 
            relationships, terminology, and frameworks spanning the entire transcript.""",
            input=combined_chunks
        )

    def run(self):
        crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=True
        )
        result = crew.kickoff()
        return result

class SpeakerCrew:
    """Crew to analyze speaker dynamics and patterns"""
    def __init__(self, all_chunks, agent):
        self.all_chunks = all_chunks
        self.agent = agent
        self.task = self.create_task()

    def create_task(self):
        combined_chunks = "\n\n--- CHUNK SEPARATOR ---\n\n".join(self.all_chunks)
        return Task(
            description="""Analyze the speaker patterns and conversation dynamics in this transcript:
            1. Identify each speaker and their apparent role or position
            2. Analyze each speaker's communication style and patterns
            3. Map who speaks to whom and how ideas flow between participants
            4. Identify the dominant voices and how they influence the discussion
            5. Note how alignment and disagreement manifest between speakers
            6. Analyze how decisions are reached through speaker interactions
            
            IMPORTANT: Focus on the social and communication dynamics, not just the content.
            Look for patterns in how people talk, interact, and influence each other.""",
            agent=self.agent,
            expected_output="""A comprehensive analysis of speaker dynamics, including 
            roles, styles, interaction patterns, and influence structures within the conversation.""",
            input=combined_chunks
        )

    def run(self):
        crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=True
        )
        result = crew.kickoff()
        return result

class SynthesisCrew:
    def __init__(self, summaries, action_items, context_analysis, speaker_analysis=None, agent=None):
        self.summaries = summaries
        self.action_items = action_items
        self.context_analysis = context_analysis
        self.speaker_analysis = speaker_analysis  # Optional
        self.agent = agent
        self.task = self.create_task()

    def create_task(self):
        # Prepare input with all available analyses
        task_input = {
            "summaries": self.summaries, 
            "action_items": self.action_items,
            "context_analysis": self.context_analysis
        }
        
        # Add speaker analysis if available
        if self.speaker_analysis:
            task_input["speaker_analysis"] = self.speaker_analysis
        
        task = Task(
            description="""Create a comprehensive, highly organized synthesis that preserves ALL details:
            1. Begin with a concise Executive Summary of key points (2-3 paragraphs)
            2. Use the provided Context Analysis to organize information logically
            3. Create a hierarchical structure with clear headings and subheadings
            4. Preserve ALL significant details from the individual summaries
            5. Maintain distinct perspectives while eliminating pure redundancy
            6. Highlight areas of consensus and disagreement
            7. Include a dedicated "Key Decisions" section
            8. Create a consolidated, prioritized "Action Items" section
            9. Add an "Unresolved Questions" section for open issues
            10. If applicable, include a "Next Steps" section
            11. If speaker analysis is provided, incorporate insights about communication dynamics
            
            IMPORTANT: The synthesis should be MORE detailed than any individual summary, 
            not less. Your goal is to create a comprehensive record that encompasses all 
            significant content while organizing it for maximum clarity and usability.""",
            agent=self.agent,
            expected_output="""A comprehensive, logically structured synthesis that preserves all 
            important details while organizing information for maximum clarity and usability.""",
            input=task_input
        )
        return task

    def run(self):
        crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=True
        )
        results = crew.kickoff()
        return results

# --- Main Execution Class ---
class TranscriptAnalysisCrew:
    def __init__(self, transcript_content, num_chunks, overlap, verbose=True, model_name="gpt-3.5-turbo", chunking_strategy="speaker_aware"):
        self.transcript_content = transcript_content
        self.num_chunks = num_chunks
        self.overlap = overlap
        self.verbose = verbose
        self.model_name = model_name
        self.chunking_strategy = chunking_strategy
        self.agents = TranscriptAgents(verbose=self.verbose, model_name=self.model_name)

    def run(self):
        # Parse transcript and create chunks using the specified strategy
        chunks_input = {
            "transcript_content": self.transcript_content,
            "num_chunks": self.num_chunks,
            "overlap": self.overlap,
            "strategy": self.chunking_strategy
        }
        chunks = read_and_chunk_transcript.invoke(chunks_input)
        
        # Print chunking info if verbose
        if self.verbose:
            print(f"Created {len(chunks)} chunks using {self.chunking_strategy} strategy")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1}: {len(chunk)} characters")
                print(f"Preview: {chunk[:100]}...")
                print("-" * 50)

        # Create right number of summarizer agents based on chunk count
        num_summarizers = min(len(chunks), 5)  # Limit number of summarizers
        summarizer_agents = [self.agents.create_summarizer() for _ in range(num_summarizers)]

        # Use thread-based parallelism to run summarization and action item extraction concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit both tasks to the executor
            summarizer_crew = SummarizerCrew(chunks, summarizer_agents)
            action_item_crew = ActionItemCrew(chunks, self.agents.create_action_item_extractor())
            
            summaries_future = executor.submit(summarizer_crew.run)
            action_items_future = executor.submit(action_item_crew.run)
            
            # Get results from both crews
            summaries = summaries_future.result()
            action_items = action_items_future.result()
        
        # Extract cross-chunk contextual information
        context_crew = ContextCrew(chunks, self.agents.create_detailed_context_finder())
        context_analysis = context_crew.run()
        
        # Optionally analyze speaker dynamics
        include_speaker_analysis = len(extract_speakers(self.transcript_content)) > 1
        speaker_analysis = None
        
        if include_speaker_analysis:
            try:
                speaker_crew = SpeakerCrew(chunks, self.agents.create_speaker_analyzer())
                speaker_analysis = speaker_crew.run()
            except Exception as e:
                print(f"Error in speaker analysis: {e}")
                speaker_analysis = None
        
        # Synthesize everything
        synthesis_crew = SynthesisCrew(
            summaries, 
            action_items, 
            context_analysis, 
            speaker_analysis, 
            self.agents.create_synthesizer()
        )
        final_notes = synthesis_crew.run()

        return final_notes