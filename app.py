"""
Atlas - AI-Powered Content Analysis Platform

This script provides a comprehensive web-based interface for AI-powered content analysis using Gradio.
It integrates multiple analysis pipelines (YouTube processing, academic papers RAG, educational content generation) into a unified platform.

Key Features:
- YouTube video analysis pipeline with natural language search
- Automatic transcript extraction and AI-powered summarization
- Academic papers RAG system with semantic search and citations
- Educational assignment generation for hands-on learning
- AI-powered comparison analysis with parallel processing
- Real-time pipeline execution with step-by-step visualization
- Professional web interface with responsive design and custom styling
- Support for multiple concurrent workers and batch processing

Required Dependencies:
- gradio: Web interface framework
- All dependencies from the YouTube pipeline components
- OpenAI API key for summarization
- YouTube Data API key for video search

Commands to run:
# Launch Gradio web interface
python app_youtube.py

# Launch with custom configuration
python app_youtube.py --host 0.0.0.0 --port 8080 --share

# Launch with debug mode
python app_youtube.py --debug
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Import components from the YouTube pipeline
from src.youtube_pipeline import YouTubePipeline

load_dotenv()


def is_cloud_environment():
    """
    Checks if the code is running in a cloud environment (Hugging Face Spaces, Colab, etc.).

    Returns:
        bool: True if running in cloud environment, False otherwise.
    """
    # Check for various cloud environment indicators
    cloud_indicators = [
        os.environ.get("SYSTEM") == "spaces",  # Hugging Face Spaces
        os.environ.get("COLAB_GPU") is not None,  # Google Colab
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None,  # Kaggle
        os.path.exists("/.dockerenv"),  # Docker container
    ]
    return any(cloud_indicators)


def validate_api_keys():
    """
    Validate that required API keys are available.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    required_keys = {
        "OPENAI_API_KEY": "OpenAI API key for transcript summarization",
        "YOUTUBE_API_KEY": "YouTube Data API key for video search",
    }

    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"- {key}: {description}")

    if missing_keys:
        error_msg = "❌ Missing required API keys:\n" + "\n".join(missing_keys)
        error_msg += (
            "\n\nPlease set these environment variables or add them to a .env file."
        )
        return False, error_msg

    return True, ""


# Global variable to store pipeline state between functions
pipeline_state = {
    "pipeline": None,
    "videos": None,
    "transcript_paths": None,
    "search_results": "",
    "transcripts_output": "",
    "summaries_output": "",
    "comparison_table": "",
    "assignments_output": "",
    # History tracking fields
    "run_id": None,
    "db": None,
    "start_time": None,
}


def step1_search_videos(
    search_query: str,
    max_videos: int,
    transcript_language: str,
    num_workers: int,
    openai_api_key: str,
    youtube_api_key: str,
    use_env_keys: bool,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """
    Step 1: Search for YouTube videos.

    Returns:
        str: Formatted search results
    """
    try:
        # Reset pipeline state
        global pipeline_state
        pipeline_state = {
            "pipeline": None,
            "videos": None,
            "transcript_paths": None,
            "search_results": "",
            "transcripts_output": "",
            "summaries_output": "",
            "comparison_table": "",
            "assignments_output": "",
            # History tracking fields
            "run_id": None,
            "db": None,
            "start_time": time.time(),
        }

        # Validate inputs
        if not search_query or not search_query.strip():
            return "❌ Error: Please provide a search query."

        # Handle API keys
        if use_env_keys:
            # Validate environment keys
            is_valid, error_msg = validate_api_keys()
            if not is_valid:
                return error_msg
        else:
            # Use provided API keys
            if not openai_api_key or not openai_api_key.strip():
                return "❌ Error: OpenAI API key is required."
            if not youtube_api_key or not youtube_api_key.strip():
                return "❌ Error: YouTube API key is required."

            # Set the API keys as environment variables for the pipeline
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["YOUTUBE_API_KEY"] = youtube_api_key

        # Initialize the pipeline with user configuration
        pipeline = YouTubePipeline(
            max_videos=max_videos,
            transcript_language=transcript_language,
            output_folder=f"output/pipeline_output_{int(time.time())}",
            num_workers=num_workers,
        )

        # Store pipeline in global state
        pipeline_state["pipeline"] = pipeline

        # Search for videos with progress tracking
        progress(0.1, desc="🔍 Searching for YouTube videos...")
        videos = pipeline.search_videos(search_query)

        if not videos:
            return "❌ No videos found for the search query."

        # Store videos in global state
        pipeline_state["videos"] = videos

        # Initialize database connection for history tracking (after successful search)
        try:
            from src.database import get_db_or_none
            db = get_db_or_none()
            if db:
                run_id = db.create_run(
                    search_query=search_query,
                    output_folder_path=pipeline.output_folder,
                    config={
                        "max_videos": max_videos,
                        "transcript_language": transcript_language,
                        "num_workers": num_workers,
                        "use_env_keys": use_env_keys,
                    },
                )
                pipeline_state["db"] = db
                pipeline_state["run_id"] = run_id
                db.update_run_progress(run_id=run_id, video_count=len(videos))
        except Exception as db_error:
            print(f"[DB] Warning: Failed to initialize history tracking: {db_error}")

        # Format and store search results
        search_results = format_search_results(videos)
        pipeline_state["search_results"] = search_results

        progress(1.0, desc="✅ Video search completed!")
        return search_results

    except Exception as e:
        print(f"[ERROR] Video search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Video Search Error: {str(e)}"


def step2_fetch_transcripts(
    search_results_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 2: Fetch transcripts for found videos.

    Args:
        search_results_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted transcript results
    """
    try:
        global pipeline_state

        # Check if previous step failed
        if search_results_input and search_results_input.startswith("❌"):
            return search_results_input  # Pass through the error

        # Check if we have valid state from previous step
        if not pipeline_state["pipeline"] or not pipeline_state["videos"]:
            return "❌ Error: No valid pipeline state. Please run video search first."

        pipeline = pipeline_state["pipeline"]
        videos = pipeline_state["videos"]

        # Fetch transcripts with progress tracking
        progress(0.1, desc="📝 Fetching video transcripts...")
        transcript_paths, fetch_results = pipeline.fetch_transcripts(videos)

        # Store transcript paths in global state
        pipeline_state["transcript_paths"] = transcript_paths

        # Update database with transcript count
        if pipeline_state.get("db") and pipeline_state.get("run_id"):
            pipeline_state["db"].update_run_progress(
                run_id=pipeline_state["run_id"],
                transcript_count=len(transcript_paths),
            )

        # Format transcript results
        transcripts_output = format_transcript_results(
            transcript_paths, videos, pipeline.transcripts_folder
        )
        pipeline_state["transcripts_output"] = transcripts_output

        progress(1.0, desc="✅ Transcript fetching completed!")
        return transcripts_output

    except Exception as e:
        print(f"[ERROR] Transcript fetching failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Transcript Fetching Error: {str(e)}"


def step3_generate_summaries(
    transcripts_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 3: Generate AI summaries for transcripts.

    Args:
        transcripts_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted summaries results
    """
    try:
        global pipeline_state

        # Check if previous step failed
        if transcripts_input and transcripts_input.startswith("❌"):
            return transcripts_input  # Pass through the error

        # Check if we have valid state from previous steps
        if (
            not pipeline_state["pipeline"]
            or not pipeline_state["videos"]
            or not pipeline_state["transcript_paths"]
        ):
            return "❌ Error: No valid pipeline state. Please run previous steps first."

        pipeline = pipeline_state["pipeline"]
        videos = pipeline_state["videos"]
        transcript_paths = pipeline_state["transcript_paths"]

        if transcript_paths:
            # Generate summaries with progress tracking
            progress(0.1, desc="🤖 Generating AI summaries...")

            # Use the pipeline's built-in parallel summarization
            summarization_results = pipeline.summarize_transcripts(
                transcript_paths, videos
            )

            # Format summaries results
            summaries_output = format_summaries_results(
                transcript_paths, videos, pipeline.summaries_folder
            )
            pipeline_state["summaries_output"] = summaries_output

            # Update database with summary count
            if pipeline_state.get("db") and pipeline_state.get("run_id"):
                # Count successful summaries
                summary_count = 0
                for transcript_path in transcript_paths:
                    filename = os.path.basename(transcript_path)
                    video_id = filename.split(".")[0]
                    summary_filename = f"{video_id}_summary.json"
                    summary_path = os.path.join(pipeline.summaries_folder, summary_filename)
                    if os.path.exists(summary_path):
                        summary_count += 1
                pipeline_state["db"].update_run_progress(
                    run_id=pipeline_state["run_id"],
                    summary_count=summary_count,
                )

            progress(1.0, desc="✅ AI summarization completed!")
            return summaries_output
        else:
            # No transcripts available for summarization
            summaries_output = "❌ No transcripts available for summarization."
            pipeline_state["summaries_output"] = summaries_output
            return summaries_output

    except Exception as e:
        print(f"[ERROR] AI summarization failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ AI Summarization Error: {str(e)}"


def step4_generate_comparison(
    summaries_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 4: Generate comparison table.

    Args:
        summaries_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted comparison table
    """
    try:
        global pipeline_state

        # Check if previous step failed
        if summaries_input and summaries_input.startswith("❌"):
            return summaries_input  # Pass through the error

        # Check if we have valid state from previous steps
        if not pipeline_state["pipeline"]:
            return "❌ Error: No valid pipeline state. Please run previous steps first."

        pipeline = pipeline_state["pipeline"]

        # Generate comparison table with progress tracking
        progress(0.1, desc="📊 Generating comparison table...")
        progress(0.3, desc="🤖 Running parallel AI insight generation...")
        comparison_table = generate_comparison_table_with_script(pipeline.output_folder)
        pipeline_state["comparison_table"] = comparison_table

        progress(1.0, desc="✅ Comparison table completed!")
        return comparison_table

    except Exception as e:
        print(f"[ERROR] Comparison table generation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ Comparison Table Error: {str(e)}"


def step5_generate_assignments(
    comparison_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 5: Generate educational assignments.

    Args:
        comparison_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted assignments results
    """
    try:
        global pipeline_state

        # Check if previous step failed - propagate the error
        if comparison_input and comparison_input.startswith("❌"):
            return comparison_input

        # Check if we have valid state from previous steps
        if not pipeline_state["pipeline"]:
            return "❌ Error: No valid pipeline state. Please run previous steps first."

        pipeline = pipeline_state["pipeline"]

        # Generate assignments with progress tracking
        progress(0.1, desc="📝 Initializing assignment generator...")

        # Import the assignment generator
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
        from assignment_generator import YouTubeAssignmentGenerator

        # Get the current pipeline state to use the same worker configuration
        num_workers = 2  # Default fallback value

        # Use the same number of workers as configured in the pipeline for consistency
        if pipeline_state.get("pipeline") and hasattr(
            pipeline_state["pipeline"], "num_workers"
        ):
            num_workers = max(
                pipeline_state["pipeline"].num_workers, 2
            )  # Minimum 2 workers

        progress(0.3, desc="🤖 Running parallel assignment generation...")

        # Initialize the assignment generator
        generator = YouTubeAssignmentGenerator(
            pipeline_output_folder=pipeline.output_folder, num_workers=num_workers
        )

        progress(0.5, desc="📊 Loading video data and summaries...")

        # Load necessary data
        video_metadata = generator.load_video_metadata()
        summary_data = generator.load_summary_data()

        if not summary_data:
            assignments_output = "❌ No summaries available for assignment generation."
            pipeline_state["assignments_output"] = assignments_output
            return assignments_output

        progress(0.7, desc="🚀 Generating assignments in parallel...")

        # Generate assignments
        assignment_results = generator.generate_assignments(
            video_metadata, summary_data
        )

        # Format the results for display
        assignments_output = format_assignments_results(
            assignment_results,
            video_metadata,
            summary_data,
            generator.assignments_folder,
        )
        pipeline_state["assignments_output"] = assignments_output

        # Complete the run record in database
        try:
            if pipeline_state.get("db") and pipeline_state.get("run_id"):
                successful_count = sum(assignment_results.values()) if assignment_results else 0
                total_count = len(assignment_results) if assignment_results else 0
                if successful_count == total_count and total_count > 0:
                    status = "success"
                elif successful_count > 0:
                    status = "partial"
                else:
                    status = "failed"
                pipeline_state["db"].complete_run(
                    run_id=pipeline_state["run_id"],
                    status=status,
                    duration_seconds=time.time() - pipeline_state.get("start_time", time.time()),
                    error_message=None,
                )
        except Exception as db_error:
            print(f"[DB] Warning: Failed to complete run record: {db_error}")

        progress(1.0, desc="✅ Assignment generation completed!")
        return assignments_output

    except Exception as e:
        print(f"[ERROR] Assignment generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Assignment Generation Error: {str(e)}"


# Global RAG system instance for caching
_rag_system = None


def get_rag_system():
    """Get or initialize the RAG system (cached globally)."""
    global _rag_system

    if _rag_system is None:
        # Import the RAG system
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
        from papers_rag import AcademicPapersRAG

        # Initialize the RAG system
        _rag_system = AcademicPapersRAG(
            papers_folder="papers/agents",
            chunk_size=512,
            chunk_overlap=50,
            similarity_top_k=5,
        )

        # Initialize the system (this will use existing index if available)
        if not _rag_system.initialize_system(force_rebuild=False):
            print("[RAG] Failed to initialize RAG system")
            _rag_system = None
            return None

        print("[RAG] RAG system initialized successfully")

    return _rag_system


def query_papers_rag(query: str, progress: gr.Progress = gr.Progress()) -> str:
    """
    Query the papers RAG database.

    Args:
        query: The search query for papers

    Returns:
        str: Formatted RAG search results with paper citations
    """
    try:
        if not query or not query.strip():
            return "❌ Please provide a query to search the papers database."

        progress(0.1, desc="🔍 Initializing RAG system...")

        # Get the cached RAG system
        rag = get_rag_system()
        if rag is None:
            return "❌ Failed to initialize RAG system. Please ensure papers are available in the 'papers/agents' folder and the vector database is set up."

        progress(0.5, desc="🔍 Searching papers...")

        # Perform the search
        result = rag.search_papers(query, include_metadata=True)

        if not result["success"]:
            return f"❌ Search failed: {result.get('error', 'Unknown error')}"

        progress(1.0, desc="✅ Search completed!")

        # Format the results
        return format_rag_results(result)

    except Exception as e:
        print(f"[ERROR] RAG query failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ RAG Query Error: {str(e)}"


def format_rag_results(result: Dict) -> str:
    """Format the RAG search results with paper citations."""
    if not result["success"]:
        return f"❌ Search failed: {result.get('error', 'Unknown error')}"

    response = result["response"]
    sources = result["sources"]
    query = result["query"]
    search_time = result["search_time"]

    # Format the output
    formatted_result = f"🔍 **Query:** {query}\n\n"
    formatted_result += f"🤖 **AI Response:**\n\n{response}\n\n"
    formatted_result += "---\n\n"
    formatted_result += (
        f"📚 **Sources & Citations** ({len(sources)} papers referenced):\n\n"
    )

    # Group sources by paper
    papers_cited = {}
    for i, source in enumerate(sources, 1):
        file_name = source.get("file_name", "Unknown")
        title = source.get("title", "Unknown Title")
        authors = source.get("authors", "Unknown Authors")
        score = source.get("score", 0.0)
        text_excerpt = source.get("text", "")

        if file_name not in papers_cited:
            papers_cited[file_name] = {
                "title": title,
                "authors": authors,
                "excerpts": [],
                "max_score": score,
            }
        else:
            papers_cited[file_name]["max_score"] = max(
                papers_cited[file_name]["max_score"], score
            )

        papers_cited[file_name]["excerpts"].append(
            {"text": text_excerpt, "score": score}
        )

    # Display papers with their citations
    for i, (file_name, paper_info) in enumerate(papers_cited.items(), 1):
        formatted_result += f"### {i}. {paper_info['title']}\n"
        formatted_result += f"**Authors:** {paper_info['authors']}\n"
        formatted_result += f"**File:** `{file_name}.pdf`\n"
        formatted_result += f"**Relevance Score:** {paper_info['max_score']:.3f}\n\n"

        # Show relevant excerpts
        formatted_result += f"**Relevant Excerpts:**\n"
        for j, excerpt in enumerate(paper_info["excerpts"], 1):
            excerpt_text = excerpt["text"]
            if len(excerpt_text) > 300:
                excerpt_text = excerpt_text[:300] + "..."
            formatted_result += f"- *{excerpt_text}* (Score: {excerpt['score']:.3f})\n"

        formatted_result += "\n---\n\n"

    # Add search statistics
    formatted_result += f"## 📊 Search Statistics\n\n"
    formatted_result += f"- **Papers Found:** {len(papers_cited)}\n"
    formatted_result += f"- **Total Sources:** {len(sources)}\n"
    formatted_result += f"- **Search Time:** {search_time:.2f} seconds\n"
    formatted_result += f"- **Query:** {query}\n\n"

    return formatted_result


def format_assignments_results(
    assignment_results: Dict[str, bool],
    video_metadata: Dict[str, Dict],
    summary_data: Dict[str, Dict],
    assignments_folder: Path,
) -> str:
    """Format the assignment generation results."""
    if not assignment_results:
        return "❌ No assignments were generated."

    successful_count = sum(assignment_results.values())
    total_count = len(assignment_results)

    result = f"📝 **Educational Assignments Generated** ({successful_count}/{total_count} completed)\n\n"
    result += f"📁 **Assignments saved to:** `{assignments_folder}`\n\n"

    for i, (video_id, success) in enumerate(assignment_results.items(), 1):
        video_info = video_metadata.get(video_id, {})
        title = video_info.get("title", "Unknown Title")
        channel = video_info.get("channel", "Unknown Channel")
        url = video_info.get("url", "#")

        result += f"## {i}. {title}\n"
        result += f"**Channel:** {channel}\n"
        result += f"**URL:** {url}\n\n"

        if success:
            assignment_file = f"{video_id}_assignment.md"
            assignment_path = assignments_folder / assignment_file

            if assignment_path.exists():
                # Read and display the full assignment content
                try:
                    with open(assignment_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Extract the actual assignment content (skip metadata)
                    lines = content.split("\n")
                    assignment_content = []
                    in_metadata = False
                    metadata_blocks = 0

                    for line in lines:
                        if line.strip() == "---":
                            if not in_metadata:
                                in_metadata = True
                                metadata_blocks += 1
                            else:
                                in_metadata = False
                                metadata_blocks += 1
                                if metadata_blocks >= 2:  # Skip the metadata block
                                    continue
                        elif not in_metadata and metadata_blocks >= 2:
                            assignment_content.append(line)

                    # Join the assignment content and clean it up
                    full_assignment = "\n".join(assignment_content).strip()

                    result += f"✅ **Assignment Generated Successfully**\n\n"
                    result += f"### 📝 Full Assignment Content:\n\n"
                    result += full_assignment + "\n\n"

                except Exception as e:
                    result += f"✅ **Assignment Generated** - `{assignment_file}`\n"
                    result += f"⚠️ Error reading assignment content: {str(e)}\n\n"
            else:
                result += f"✅ **Assignment Generated** - `{assignment_file}`\n\n"
        else:
            result += f"❌ **Assignment Generation Failed**\n"
            result += f"Please check the logs for error details.\n\n"

        result += "---\n\n"

    # Add summary statistics
    result += f"## 📊 Generation Summary\n\n"
    result += f"- **Total Videos:** {total_count}\n"
    result += f"- **Successful Assignments:** {successful_count}\n"
    result += f"- **Success Rate:** {successful_count/total_count*100:.1f}%\n"
    result += f"- **Output Folder:** `{assignments_folder}`\n\n"

    result += f"### 🎯 Assignment Features\n\n"
    result += f"Each assignment includes:\n"
    result += (
        f"- **📋 Assignment Overview** - Clear learning objectives and deliverables\n"
    )
    result += f"- **📚 Prerequisite Knowledge** - Required background and skills\n"
    result += f"- **🔧 Core Tasks** - Progressive, hands-on implementation tasks\n"
    result += f"- **💡 Practical Exercises** - Real-world problem-solving scenarios\n"
    result += (
        f"- **🚀 Advanced Challenges** - Extension activities for deeper learning\n"
    )
    result += f"- **✅ Assessment Criteria** - Clear success metrics and rubrics\n"
    result += f"- **📖 Resources & References** - Additional learning materials\n\n"

    return result


def format_transcript_results(
    transcript_paths: List[str], videos: List[Dict], transcripts_folder: str
) -> str:
    """Format the transcript fetching results with full transcript content."""
    if not videos:
        return "❌ No videos to fetch transcripts from."

    result = (
        f"📝 **Video Transcripts** ({len(transcript_paths)}/{len(videos)} fetched)\n\n"
    )

    # Create a mapping of video IDs to transcripts
    transcript_files = {Path(tp).stem.split(".")[0]: tp for tp in transcript_paths}

    for i, video in enumerate(videos, 1):
        title = video.get("title", "Unknown Title")
        video_id = video.get("video_id", "")
        channel = video.get("channel", "Unknown Channel")

        result += f"## {i}. {title}\n"
        result += f"**Channel:** {channel}\n\n"

        if video_id in transcript_files:
            # Load and display full transcript content
            try:
                with open(transcript_files[video_id], "r", encoding="utf-8") as f:
                    transcript_content = f.read()

                # Clean up SRT format and convert to readable text
                import re

                # Remove SRT timestamp lines and sequence numbers
                clean_content = re.sub(
                    r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n",
                    "",
                    transcript_content,
                )
                clean_content = re.sub(r"^\d+$", "", clean_content, flags=re.MULTILINE)
                clean_content = re.sub(r"\n\s*\n", "\n", clean_content)
                clean_content = clean_content.strip()

                if clean_content:
                    result += f"**Full Transcript:**\n\n"
                    # Format transcript as flowing text instead of code block
                    # Split into sentences and create readable paragraphs
                    sentences = clean_content.replace("\n", " ").split(". ")
                    formatted_transcript = ""
                    current_paragraph = ""

                    for i, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if sentence:
                            # Add period back if it was removed by split (except for last sentence)
                            if i < len(sentences) - 1 and not sentence.endswith("."):
                                sentence += "."

                            current_paragraph += sentence + " "

                            # Create paragraph breaks every 3-4 sentences for readability
                            if (i + 1) % 4 == 0:
                                formatted_transcript += (
                                    current_paragraph.strip() + "\n\n"
                                )
                                current_paragraph = ""

                    # Add any remaining content
                    if current_paragraph.strip():
                        formatted_transcript += current_paragraph.strip() + "\n\n"

                    result += formatted_transcript
                else:
                    result += f"⚠️ Transcript file exists but appears to be empty or malformed.\n\n"

            except Exception as e:
                result += f"❌ Error reading transcript: {str(e)}\n\n"
        else:
            result += (
                f"❌ **Transcript not available** - Failed to fetch from YouTube\n\n"
            )

        result += "---\n\n"

    return result


def format_summaries_results(
    transcript_paths: List[str], videos: List[Dict], summaries_folder: str
) -> str:
    """Format the AI summarization results with full summary content."""
    if not transcript_paths:
        return "❌ No transcripts available for summarization."

    processed_count = len(transcript_paths)
    successful_count = 0

    # Count successful summaries
    for transcript_path in transcript_paths:
        filename = os.path.basename(transcript_path)
        video_id = filename.split(".")[0]
        summary_filename = f"{video_id}_summary.json"
        summary_path = os.path.join(summaries_folder, summary_filename)
        if os.path.exists(summary_path):
            successful_count += 1

    result = f"🤖 **AI Generated Summaries** ({successful_count}/{processed_count} completed)\n\n"

    for i, transcript_path in enumerate(transcript_paths, 1):
        filename = os.path.basename(transcript_path)
        video_id = filename.split(".")[0]
        video_info = next((v for v in videos if v["video_id"] == video_id), {})
        title = video_info.get("title", "Unknown Title")
        channel = video_info.get("channel", "Unknown Channel")
        url = video_info.get("url", "#")

        summary_filename = f"{video_id}_summary.json"
        summary_path = os.path.join(summaries_folder, summary_filename)

        result += f"## {i}. {title}\n"
        result += f"**Channel:** {channel}\n"
        result += f"**URL:** {url}\n\n"

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)

                # Handle the new summarizer_v2 JSON structure
                # High-level overview (main summary)
                high_level_overview = summary_data.get("high_level_overview", "")
                if high_level_overview:
                    result += f"### 📋 High-Level Overview\n\n{high_level_overview}\n\n"

                # Technical breakdown
                technical_breakdown = summary_data.get("technical_breakdown", [])
                if technical_breakdown:
                    result += f"### 🔧 Technical Breakdown\n\n"

                    # Group by type for better organization
                    tools = [
                        item
                        for item in technical_breakdown
                        if item.get("type") == "tool"
                    ]
                    architectures = [
                        item
                        for item in technical_breakdown
                        if item.get("type") == "architecture"
                    ]
                    processes = [
                        item
                        for item in technical_breakdown
                        if item.get("type") == "process"
                    ]

                    if tools:
                        result += f"#### 🛠️ Tools & Frameworks\n"
                        for tool in tools:
                            name = tool.get("name", "Unknown Tool")
                            purpose = tool.get("purpose", "Purpose not specified")
                            result += f"• **{name}**: {purpose}\n"
                        result += "\n"

                    if architectures:
                        result += f"#### 🏗️ Architecture & Design\n"
                        for arch in architectures:
                            description = arch.get("description", "No description")
                            result += f"• {description}\n"
                        result += "\n"

                    if processes:
                        result += f"#### 📋 Step-by-Step Process\n"
                        # Sort by step_number
                        processes.sort(key=lambda x: x.get("step_number", 0))
                        for process in processes:
                            step_num = process.get("step_number", "?")
                            description = process.get("description", "No description")
                            result += f"{step_num}. {description}\n"
                        result += "\n"

                # Key insights
                insights = summary_data.get("insights", [])
                if insights:
                    result += f"### 💡 Key Engineering Insights\n\n"
                    for i, insight in enumerate(insights, 1):
                        result += f"{i}. {insight}\n"
                    result += "\n"

                # Practical applications
                applications = summary_data.get("applications", [])
                if applications:
                    result += f"### 🎯 Practical Applications\n\n"
                    for app in applications:
                        result += f"• {app}\n"
                    result += "\n"

                # Limitations and considerations
                limitations = summary_data.get("limitations", [])
                if limitations:
                    result += f"### ⚠️ Limitations & Considerations\n\n"
                    for limitation in limitations:
                        result += f"• {limitation}\n"
                    result += "\n"

            except Exception as e:
                result += f"❌ **Error loading summary:** {str(e)}\n\n"
        else:
            result += f"⏳ **Summary in progress...** 🔄\n\n"

        result += "---\n\n"

    return result


def generate_comparison_table_with_script(pipeline_output_folder: str) -> str:
    """Generate a comparison table using the existing comparison script."""
    try:
        # Import the comparison script
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
        from compare_youtube_outputs import YouTubeOutputComparator

        # Get the current pipeline state to use the same worker configuration
        global pipeline_state
        num_workers = 4  # Default fallback value

        # Use the same number of workers as configured in the pipeline for consistency
        if pipeline_state.get("pipeline") and hasattr(
            pipeline_state["pipeline"], "num_workers"
        ):
            num_workers = max(
                pipeline_state["pipeline"].num_workers, 2
            )  # Minimum 2 workers

        # Initialize the comparator with parallel processing optimized for AI insights
        comparator = YouTubeOutputComparator(
            pipeline_output_folder=pipeline_output_folder,
            use_ai_insights=True,  # Enable AI insights for comprehensive comparison
            num_workers=num_workers,  # Use user-configured worker count for better parallel performance
        )

        print(
            f"[INSIGHTS] Using {num_workers} workers for parallel AI insight generation"
        )

        # Run the comparison
        result = comparator.run_comparison(fix_json=True, save_detailed=False)

        # Unpack results (handle different return formats)
        if len(result) >= 6:
            (
                comparison_df,
                insights_report,
                recommendations,
                video_metadata,
                summary_data,
                ai_insights,
            ) = result
        elif len(result) >= 3:
            comparison_df, insights_report, recommendations = result[:3]
        elif len(result) >= 2:
            comparison_df, insights_report = result[:2]
        else:
            return "❌ Error: Could not generate comparison data."

        if comparison_df.empty:
            return "❌ No data available for comparison."

        # Convert DataFrame to HTML table with proper styling
        html_result = f'<h1 style="color: #000000; text-align: center; margin-bottom: 20px;">📊 Video Comparison Analysis</h1>\n\n'
        html_result += f'<p style="color: #000000; text-align: center; font-weight: bold; margin-bottom: 20px;">Comparing {len(comparison_df)} videos:</p>\n\n'

        # Create HTML table with improved formatting and column widths
        html_result += '<div style="overflow-x: auto; background-color: #ffffff; padding: 15px; border-radius: 8px; border: 2px solid #333333; margin: 10px 0; max-width: 100%;">\n'
        html_result += '<table style="width: 100%; min-width: 1400px; border-collapse: collapse; margin: 0; font-size: 13px; background-color: #ffffff; color: #000000; table-layout: fixed;">\n'

        # Table header - include comprehensive columns from the comparison script
        key_columns = [
            "Title",
            "Channel",
            "Published",
            "Difficulty",
            "Teaching Style",
            "Content Depth",
            "Learning Outcome",
            "Target Audience",
            "Prerequisites",
            "Tools Count",
            "Key Technologies",
            "Complexity Score",
        ]
        available_columns = [col for col in key_columns if col in comparison_df.columns]

        # Define column widths for better formatting
        column_widths = {
            "Title": "20%",
            "Channel": "10%",
            "Published": "8%",
            "Difficulty": "8%",
            "Teaching Style": "10%",
            "Content Depth": "8%",
            "Learning Outcome": "15%",
            "Target Audience": "8%",
            "Prerequisites": "10%",
            "Tools Count": "6%",
            "Key Technologies": "12%",
            "Complexity Score": "6%",
        }

        html_result += "<thead>\n"
        html_result += '<tr style="background-color: #ffffff; color: #000000; border-bottom: 3px solid #000000;">\n'
        for col in available_columns:
            icon = {
                "Title": "📺",
                "Channel": "📻",
                "Published": "📅",
                "Difficulty": "📊",
                "Teaching Style": "🎯",
                "Content Depth": "🔍",
                "Learning Outcome": "🎓",
                "Target Audience": "👥",
                "Prerequisites": "📚",
                "Tools Count": "🔧",
                "Key Technologies": "⚙️",
                "Complexity Score": "📈",
            }.get(col, "📋")
            width = column_widths.get(col, "8%")
            html_result += f'<th style="padding: 12px; text-align: left; border: 2px solid #333333; font-weight: bold; color: #000000; font-size: 12px; background-color: #ffffff; width: {width}; word-wrap: break-word;">{icon} {col}</th>\n'
        html_result += "</tr>\n"
        html_result += "</thead>\n"

        # Table body
        html_result += "<tbody>\n"

        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            # Alternate row colors with high contrast
            row_color = "#ffffff" if i % 2 == 1 else "#f8f9fa"
            text_color = "#000000"  # Force black text

            html_result += f'<tr style="background-color: {row_color};">\n'

            for col in available_columns:
                value = str(row.get(col, "N/A"))

                # Truncate titles and adjust text based on column
                if col == "Title":
                    value = value[:80] + "..." if len(str(value)) > 80 else value
                elif col == "Learning Outcome":
                    value = value[:100] + "..." if len(str(value)) > 100 else value
                elif col == "Prerequisites":
                    value = value[:80] + "..." if len(str(value)) > 80 else value
                elif col == "Key Technologies":
                    value = value[:60] + "..." if len(str(value)) > 60 else value

                # Style categorical columns with colors
                if col in ["Difficulty", "Content Depth", "Teaching Style"]:
                    # Clean up categorical values to show only the main category
                    if col == "Content Depth":
                        # Extract just the category from longer descriptions
                        if "surface" in value.lower():
                            value = "Surface-level"
                        elif "moderate" in value.lower():
                            value = "Moderate"
                        elif "deep" in value.lower():
                            value = "Deep-dive"
                        elif ":" in value:
                            value = value.split(":")[0].strip()
                        elif "-" in value and len(value) > 15:
                            value = value.split("-")[0].strip()

                    elif col == "Difficulty":
                        # Clean up difficulty values
                        if "beginner" in value.lower():
                            value = "Beginner"
                        elif "intermediate" in value.lower():
                            value = "Intermediate"
                        elif "advanced" in value.lower():
                            value = "Advanced"
                        elif " " in value:
                            # Handle cases like "Intermediate Advanced" - take first word
                            value = value.split()[0].strip()
                        elif ":" in value:
                            value = value.split(":")[0].strip()
                        elif "-" in value and len(value) > 15:
                            value = value.split("-")[0].strip()

                    elif col == "Teaching Style":
                        # Clean up teaching style values
                        value_lower = value.lower()
                        if "code-along" in value_lower or "code along" in value_lower:
                            value = "Code-along"
                        elif "explanation" in value_lower:
                            value = "Explanation-heavy"
                        elif "project" in value_lower:
                            value = "Project-based"
                        elif "theory" in value_lower:
                            value = "Theory-focused"
                        elif "mixed" in value_lower:
                            value = "Mixed"
                        elif ":" in value:
                            value = value.split(":")[0].strip()
                        elif "-" in value and len(value) > 15:
                            value = value.split("-")[0].strip()

                    color_map = {
                        # Difficulty levels
                        "Beginner": "#28a745",
                        "Intermediate": "#ffc107",
                        "Advanced": "#dc3545",
                        # Content depth
                        "Surface-level": "#ffc107",
                        "Moderate": "#17a2b8",
                        "Deep-dive": "#6f42c1",
                        # Teaching styles
                        "Code-along": "#28a745",
                        "Explanation-heavy": "#17a2b8",
                        "Project-based": "#6f42c1",
                        "Theory-focused": "#fd7e14",
                        "Mixed": "#6c757d",
                        # Default
                        "Unknown": "#6c757d",
                    }
                    bg_color = color_map.get(value, "#6c757d")
                    html_result += f'<td style="padding: 12px; border: 1px solid #333333; text-align: center; background-color: {row_color}; vertical-align: middle; min-width: 80px;">\n'
                    html_result += f'<span style="background-color: {bg_color}; color: #ffffff; padding: 6px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; white-space: nowrap; display: inline-block;">{value}</span>\n'
                    html_result += f"</td>\n"
                elif col in ["Tools Count", "Complexity Score"]:
                    # Special styling for numeric columns
                    html_result += f'<td style="padding: 12px; border: 1px solid #333333; text-align: center; background-color: {row_color}; font-weight: bold; color: #2c5282; font-size: 14px; vertical-align: middle; min-width: 60px;">{value}</td>\n'
                elif col in [
                    "Title",
                    "Learning Outcome",
                    "Prerequisites",
                    "Key Technologies",
                ]:
                    # Text columns with word wrapping
                    html_result += f'<td style="padding: 12px; border: 1px solid #333333; color: {text_color}; vertical-align: top; background-color: {row_color}; font-size: 12px; line-height: 1.4; word-wrap: break-word; overflow-wrap: break-word;">{value}</td>\n'
                else:
                    # Other columns with standard formatting
                    html_result += f'<td style="padding: 12px; border: 1px solid #333333; color: {text_color}; vertical-align: middle; background-color: {row_color}; font-size: 12px; text-align: center;">{value}</td>\n'

            html_result += "</tr>\n"

        html_result += "</tbody>\n"
        html_result += "</table>\n"
        html_result += "</div>\n\n"

        return html_result

    except Exception as e:
        return f"❌ Error generating comparison: {str(e)}\n\nPlease ensure the pipeline has completed and output files are available."


def format_search_results(videos: List[Dict]) -> str:
    """Format the video search results."""
    if not videos:
        return "❌ No videos found for the search query."

    results = f"🔍 **Found {len(videos)} videos:**\n\n"

    for i, video in enumerate(videos, 1):
        title = video.get("title", "Unknown Title")
        channel = video.get("channel", "Unknown Channel")
        url = video.get("url", "#")
        description = video.get("description", "")
        publish_date = video.get("published_at", "Unknown Date")
        duration = video.get("duration", "Unknown")

        # Truncate description if too long
        if len(description) > 200:
            description = description[:200] + "..."

        results += f"""**{i}. {title}**
   • **Channel:** {channel}
   • **Published:** {publish_date}
   • **Duration:** {duration}
   • **URL:** {url}
   • **Description:** {description}

"""

    return results


# ==================== History Tab Helper Functions ====================


def load_history_list(
    status_filter: str = "All", page: int = 1, page_size: int = 10
) -> Tuple[List[List], int, str]:
    """
    Load pipeline run history from database.

    Returns:
        Tuple of (dataframe_data, total_pages, status_message)
    """
    from src.database import get_db_or_none

    db = get_db_or_none()
    if db is None:
        return [], 1, "Database connection unavailable"

    # Get status filter
    filter_status = None if status_filter == "All" else status_filter.lower()

    # Get paginated runs
    offset = (page - 1) * page_size
    runs = db.get_all_runs(limit=page_size, offset=offset, status_filter=filter_status)
    total_count = db.get_run_count(status_filter=filter_status)
    total_pages = max(1, (total_count + page_size - 1) // page_size)

    # Format for dataframe
    data = []
    for run in runs:
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "partial": "⚠️",
            "running": "🔄",
        }.get(run["status"], "❓")

        # Format timestamp
        timestamp = run["timestamp"]
        if timestamp:
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            timestamp_str = "N/A"

        # Format duration
        duration = run["duration_seconds"]
        if duration:
            duration_str = f"{duration:.1f}s"
        else:
            duration_str = "N/A"

        data.append(
            [
                run["run_id"][:8],  # Short ID
                run["search_query"][:50] + ("..." if len(run["search_query"]) > 50 else ""),
                timestamp_str,
                f"{status_emoji} {run['status'].title()}",
                run["video_count"],
                run["transcript_count"],
                run["summary_count"],
                duration_str,
                run["run_id"],  # Full ID (hidden column for selection)
            ]
        )

    status_msg = f"Showing {len(data)} of {total_count} runs (Page {page}/{total_pages})"
    return data, total_pages, status_msg


def load_run_details(run_id: str) -> str:
    """Load and format details for a specific run."""
    if not run_id:
        return "<div style='padding: 20px; text-align: center;'>Select a run to view details</div>"

    from src.database import get_db_or_none

    db = get_db_or_none()
    if db is None:
        return "<div style='padding: 20px; color: #dc3545;'>Database connection unavailable</div>"

    run = db.get_run(run_id)
    if not run:
        return f"<div style='padding: 20px; color: #dc3545;'>Run not found: {run_id}</div>"

    # Status styling
    status_colors = {
        "success": "#28a745",
        "failed": "#dc3545",
        "partial": "#ffc107",
        "running": "#17a2b8",
    }
    status_color = status_colors.get(run["status"], "#6c757d")

    # Format config
    config = {}
    if run["config_json"]:
        try:
            config = json.loads(run["config_json"])
        except json.JSONDecodeError:
            config = {}

    # Format timestamp
    timestamp = run["timestamp"]
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "N/A"

    # Format duration
    duration = run["duration_seconds"]
    duration_str = f"{duration:.2f} seconds" if duration else "N/A"

    # Check if output folder exists
    output_exists = os.path.exists(run["output_folder_path"])
    folder_status = "✅ Available" if output_exists else "❌ Not found"

    html = f"""
    <div class="run-details" style="background: var(--background-fill-secondary); padding: 20px; border-radius: 10px; border: 1px solid var(--border-color-primary);">
        <h3 style="margin-bottom: 15px; color: var(--body-text-color);">📋 Run Details</h3>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
            <div>
                <strong style="color: var(--body-text-color);">Run ID:</strong><br/>
                <code style="font-size: 12px;">{run["run_id"]}</code>
            </div>
            <div>
                <strong style="color: var(--body-text-color);">Status:</strong><br/>
                <span style="background-color: {status_color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;">
                    {run["status"].upper()}
                </span>
            </div>
        </div>

        <div style="margin-bottom: 15px;">
            <strong style="color: var(--body-text-color);">Search Query:</strong><br/>
            <div style="background: var(--background-fill-primary); padding: 10px; border-radius: 6px; margin-top: 5px;">
                {run["search_query"]}
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 15px;">
            <div style="text-align: center; background: var(--background-fill-primary); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; color: #667eea;">{run["video_count"]}</div>
                <div style="font-size: 12px; color: var(--body-text-color-subdued);">Videos</div>
            </div>
            <div style="text-align: center; background: var(--background-fill-primary); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; color: #28a745;">{run["transcript_count"]}</div>
                <div style="font-size: 12px; color: var(--body-text-color-subdued);">Transcripts</div>
            </div>
            <div style="text-align: center; background: var(--background-fill-primary); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; color: #764ba2;">{run["summary_count"]}</div>
                <div style="font-size: 12px; color: var(--body-text-color-subdued);">Summaries</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
            <div>
                <strong style="color: var(--body-text-color);">Timestamp:</strong><br/>
                {timestamp_str}
            </div>
            <div>
                <strong style="color: var(--body-text-color);">Duration:</strong><br/>
                {duration_str}
            </div>
        </div>

        <div style="margin-bottom: 15px;">
            <strong style="color: var(--body-text-color);">Output Folder:</strong><br/>
            <code style="font-size: 11px; word-break: break-all;">{run["output_folder_path"]}</code>
            <span style="margin-left: 10px;">{folder_status}</span>
        </div>

        <div style="margin-bottom: 15px;">
            <strong style="color: var(--body-text-color);">Configuration:</strong><br/>
            <div style="background: var(--background-fill-primary); padding: 10px; border-radius: 6px; margin-top: 5px; font-family: monospace; font-size: 12px;">
                Max Videos: {config.get("max_videos", "N/A")}<br/>
                Language: {config.get("transcript_language", "N/A")}<br/>
                Workers: {config.get("num_workers", "N/A")}
            </div>
        </div>

        {"<div style='margin-bottom: 15px;'><strong style='color: #dc3545;'>Error:</strong><br/><div style='background: #fff5f5; padding: 10px; border-radius: 6px; color: #dc3545; margin-top: 5px;'>" + run["error_message"] + "</div></div>" if run["error_message"] else ""}
    </div>
    """
    return html


def delete_run_handler(run_id: str, delete_files: bool) -> Tuple[str, List[List], str]:
    """Delete a run from the database and optionally delete output files."""
    import shutil

    if not run_id:
        return "No run selected", [], ""

    from src.database import get_db_or_none

    db = get_db_or_none()
    if db is None:
        return "Database connection unavailable", [], ""

    # Get run info first
    run = db.get_run(run_id)
    if not run:
        return f"Run not found: {run_id}", [], ""

    # Delete files if requested
    files_deleted = False
    if delete_files and run["output_folder_path"]:
        try:
            if os.path.exists(run["output_folder_path"]):
                shutil.rmtree(run["output_folder_path"])
                files_deleted = True
        except Exception as e:
            return f"Failed to delete files: {str(e)}", [], ""

    # Delete from database
    if db.delete_run(run_id):
        msg = f"✅ Run {run_id[:8]}... deleted successfully"
        if files_deleted:
            msg += " (including output files)"
        # Refresh the list
        new_data, _, status = load_history_list()
        return msg, new_data, ""
    else:
        return f"❌ Failed to delete run {run_id[:8]}...", [], ""


def rerun_pipeline_handler(run_id: str) -> Tuple[str, int, str, int, bool]:
    """
    Load configuration from a previous run for re-execution.

    Returns:
        Tuple of (search_query, max_videos, transcript_language, num_workers, use_env_keys)
    """
    if not run_id:
        return "", 2, "en", 2, True

    from src.database import get_db_or_none

    db = get_db_or_none()
    if db is None:
        return "", 2, "en", 2, True

    run = db.get_run(run_id)
    if not run:
        return "", 2, "en", 2, True

    # Parse config
    config = {}
    if run["config_json"]:
        try:
            config = json.loads(run["config_json"])
        except json.JSONDecodeError:
            config = {}

    return (
        run["search_query"],
        config.get("max_videos", 2),
        config.get("transcript_language", "en"),
        config.get("num_workers", 2),
        config.get("use_env_keys", True),
    )


def create_gradio_app():
    """Create and configure the Gradio application."""

    # Theme-aware CSS using Gradio CSS variables
    css = """
    /* Theme-aware container styling */
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 20px !important;
    }

    /* Header styling - works in both light and dark mode */
    .header-section {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .header-section h1 {
        font-size: 2.5em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white !important;
    }

    .header-section p {
        font-size: 1.2em;
        margin-bottom: 10px;
        opacity: 0.95;
        color: white !important;
    }

    .header-section div {
        color: white !important;
    }

    .header-section strong {
        color: white !important;
    }

    /* Input section styling - theme aware */
    .input-section {
        background: var(--background-fill-primary);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color-primary);
    }

    /* Results section styling - theme aware */
    .results-section {
        background: var(--background-fill-primary);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid var(--border-color-primary);
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }

    /* Section headers - theme aware */
    .section-header {
        color: #667eea;
        font-weight: bold;
        border-bottom: 2px solid #667eea;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }

    /* Button styling */
    .primary-button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        color: white !important;
    }

    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }

    /* Step boxes - theme aware */
    .step-box {
        background: var(--background-fill-primary);
        border: 2px solid var(--border-color-primary);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }

    .search-step { border-left: 4px solid #10b981; }
    .transcript-step { border-left: 4px solid #3b82f6; }
    .summary-step { border-left: 4px solid #8b5cf6; }

    /* Transcript text formatting for full width */
    .transcript-container textarea {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.6 !important;
        word-wrap: break-word !important;
        white-space: pre-wrap !important;
        text-align: justify !important;
        width: 100% !important;
    }

    /* Ensure textboxes use full width */
    .gradio-textbox {
        width: 100% !important;
    }

    .gradio-textbox textarea {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Theme-aware info boxes */
    .info-box {
        background: var(--background-fill-secondary);
        color: var(--body-text-color);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--border-color-primary);
    }

    .info-box h4, .info-box li, .info-box ul {
        color: var(--body-text-color) !important;
    }

    /* History tab styling */
    .history-list-panel {
        border-right: 1px solid var(--border-color-primary);
        padding-right: 15px;
    }

    .history-detail-panel {
        padding-left: 15px;
    }

    .history-controls {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        align-items: center;
    }

    .history-table {
        width: 100%;
        border-collapse: collapse;
    }

    .history-table th, .history-table td {
        padding: 10px;
        border-bottom: 1px solid var(--border-color-primary);
        text-align: left;
    }

    .history-table tr:hover {
        background: var(--background-fill-secondary);
        cursor: pointer;
    }

    .pagination-controls {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 15px;
        align-items: center;
    }

    .action-buttons {
        display: flex;
        gap: 10px;
        margin-top: 15px;
    }

    .status-success { color: #28a745; }
    .status-failed { color: #dc3545; }
    .status-partial { color: #ffc107; }
    .status-running { color: #17a2b8; }
    """

    with gr.Blocks(title="🎬 Atlas") as app:
        # Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML(
                    """
                <div style="text-align: center;">
                    <h1>🎬 Atlas</h1>
                    <p>AI-Powered Content Analysis Platform for Educational and Research Content</p>
                    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px; flex-wrap: wrap;">
                        <div>🔍 <strong>YouTube Pipeline</strong><br/>Video search & analysis</div>
                        <div>📚 <strong>Academic RAG</strong><br/>Papers search & citations</div>
                        <div>📝 <strong>Assignment Generator</strong><br/>Educational content creation</div>
                        <div>🤖 <strong>AI Analysis</strong><br/>Parallel content processing</div>
                        <div>📊 <strong>Comparison Engine</strong><br/>Multi-video insights</div>
                        <div>📜 <strong>History</strong><br/>Track & re-run pipelines</div>
                    </div>
                </div>
                """
                )

        # Main tabs
        with gr.Tabs() as main_tabs:
            # ==================== Pipeline Tab ====================
            with gr.TabItem("🔍 Pipeline", id="pipeline_tab"):
                # Input Configuration Section
                with gr.Column(elem_classes=["input-section"]):
                    gr.HTML('<h3 class="section-header">🔍 Search Configuration</h3>')

                    search_query = gr.Textbox(
                        label="YouTube Search Query",
                        placeholder="e.g., 'Python machine learning tutorial', 'React best practices', 'Docker deployment guide'",
                        lines=3,
                        info="Enter your search query to find relevant YouTube videos",
                    )

                    with gr.Row():
                        max_videos = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1,
                            label="Max Videos",
                            info="Maximum number of videos to process",
                        )

                        num_workers = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=2,
                            step=1,
                            label="Concurrent Workers",
                            info="Number of parallel workers for processing",
                        )

                    transcript_language = gr.Dropdown(
                        choices=[
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "ru",
                            "ja",
                            "ko",
                            "zh",
                        ],
                        value="en",
                        label="Transcript Language",
                        info="Language code for subtitle extraction",
                    )

                    gr.HTML('<h3 class="section-header">🔑 API Configuration</h3>')

                    use_env_keys = gr.Checkbox(
                        value=True,
                        label="Use Environment Variables for API Keys",
                        info="Check if you have OPENAI_API_KEY and YOUTUBE_API_KEY set as environment variables",
                    )

                    with gr.Column(visible=False) as api_key_inputs:
                        openai_api_key = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="sk-...",
                            info="Required for transcript summarization",
                        )

                        youtube_api_key = gr.Textbox(
                            label="YouTube Data API Key",
                            type="password",
                            placeholder="AIza...",
                            info="Required for video search",
                        )

                    # Show/hide API key inputs based on checkbox
                    def toggle_api_inputs(use_env):
                        return gr.update(visible=not use_env)

                    use_env_keys.change(
                        fn=toggle_api_inputs,
                        inputs=[use_env_keys],
                        outputs=[api_key_inputs],
                    )

                    process_btn = gr.Button(
                        "🚀 Start Pipeline",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"],
                    )

                # Pipeline Results - Real-time output blocks in single column
                gr.HTML(
                    '<h2 style="text-align: center; color: #667eea; margin: 30px 0 20px 0;">📊 Pipeline Results</h2>'
                )

                # Single column layout
                search_results = gr.Textbox(
                    label="🔍 1. YouTube Search Results",
                    lines=8,
                    max_lines=15,
                    info="Found videos with details",
                    interactive=False,
                )

                transcripts_output = gr.Textbox(
                    label="📝 2. Video Transcripts",
                    lines=8,
                    max_lines=15,
                    info="Extracted transcripts with previews",
                    interactive=False,
                    container=True,
                    autoscroll=False,
                    elem_classes=["transcript-container"],
                )

                summaries_output = gr.Textbox(
                    label="🤖 3. AI Summaries",
                    lines=8,
                    max_lines=15,
                    info="AI-generated summaries and key points",
                    interactive=False,
                )

                comparison_table = gr.HTML(
                    label="📊 4. Video Comparison Analysis",
                    value="<div style='padding: 20px; text-align: center; color: var(--body-text-color); background-color: var(--background-fill-primary); border: 2px solid var(--border-color-primary); border-radius: 8px; font-weight: bold;'>Comparison table will appear here after all summaries are generated...</div>",
                    visible=True,
                )

                assignments_output = gr.Textbox(
                    label="📝 5. Educational Assignments",
                    lines=8,
                    max_lines=15,
                    info="AI-generated educational assignments for hands-on learning",
                    interactive=False,
                )

                # RAG Query Section
                gr.HTML(
                    '<h2 style="text-align: center; color: #667eea; margin: 30px 0 20px 0;">📚 Academic Papers RAG Query</h2>'
                )

                with gr.Row():
                    # RAG Query Input
                    with gr.Column(scale=2):
                        rag_query_input = gr.Textbox(
                            label="🔍 Query Academic Papers",
                            placeholder="e.g., 'What are the main types of AI agents?', 'How do LLM agents work?', 'What are the challenges in autonomous agents?'",
                            lines=3,
                            info="Search through indexed academic papers using natural language queries",
                        )

                        rag_query_btn = gr.Button(
                            "🔍 Search Papers",
                            variant="secondary",
                            size="lg",
                        )

                    # RAG Instructions
                    with gr.Column(scale=1):
                        gr.HTML(
                            """
                            <div class="info-box">
                                <h4 style="margin-bottom: 10px; font-weight: bold;">🎯 RAG Query Features:</h4>
                                <ul style="margin-bottom: 15px; padding-left: 20px;">
                                    <li style="margin-bottom: 5px;">Search through academic papers using natural language</li>
                                    <li style="margin-bottom: 5px;">Get AI-generated answers with paper citations</li>
                                    <li style="margin-bottom: 5px;">View relevant excerpts from source papers</li>
                                    <li style="margin-bottom: 5px;">See relevance scores for each source</li>
                                </ul>

                                <h4 style="margin-bottom: 10px; font-weight: bold;">📖 Example Queries:</h4>
                                <ul style="padding-left: 20px;">
                                    <li style="margin-bottom: 5px;">"What are the main architectures for AI agents?"</li>
                                    <li style="margin-bottom: 5px;">"How do LLM-based agents handle planning?"</li>
                                    <li style="margin-bottom: 5px;">"What evaluation methods exist for autonomous agents?"</li>
                                    <li style="margin-bottom: 5px;">"What are the current limitations of AI agents?"</li>
                                </ul>
                            </div>
                            """
                        )

                # RAG Results
                rag_results = gr.Textbox(
                    label="📚 Academic Papers Search Results",
                    lines=10,
                    max_lines=20,
                    info="AI-generated answers with paper citations and source excerpts",
                    interactive=False,
                )

                # Sequential pipeline execution using .then() method
                # Step 1: Search for videos
                step1_event = process_btn.click(
                    fn=step1_search_videos,
                    inputs=[
                        search_query,
                        max_videos,
                        transcript_language,
                        num_workers,
                        openai_api_key,
                        youtube_api_key,
                        use_env_keys,
                    ],
                    outputs=search_results,
                    show_progress="full",
                )

                # Step 2: Fetch transcripts (triggered after step 1 completes)
                step2_event = step1_event.then(
                    fn=step2_fetch_transcripts,
                    inputs=search_results,
                    outputs=transcripts_output,
                    show_progress="full",
                )

                # Step 3: Generate summaries (triggered after step 2 completes)
                step3_event = step2_event.then(
                    fn=step3_generate_summaries,
                    inputs=transcripts_output,
                    outputs=summaries_output,
                    show_progress="full",
                )

                # Step 4: Generate comparison table (triggered after step 3 completes)
                step4_event = step3_event.then(
                    fn=step4_generate_comparison,
                    inputs=summaries_output,
                    outputs=comparison_table,
                    show_progress="full",
                )

                # Step 5: Generate assignments (triggered after step 4 completes)
                step5_event = step4_event.then(
                    fn=step5_generate_assignments,
                    inputs=comparison_table,
                    outputs=assignments_output,
                    show_progress="full",
                )

                # RAG Query button click event
                rag_query_btn.click(
                    fn=query_papers_rag,
                    inputs=rag_query_input,
                    outputs=rag_results,
                    show_progress="full",
                )

            # ==================== History Tab ====================
            with gr.TabItem("📜 History", id="history_tab"):
                gr.HTML('<h2 style="text-align: center; color: #667eea; margin: 20px 0;">📜 Pipeline Run History</h2>')

                with gr.Row():
                    # Left Panel - History List
                    with gr.Column(scale=1, elem_classes=["history-list-panel"]):
                        with gr.Row():
                            history_status_filter = gr.Dropdown(
                                choices=["All", "Success", "Failed", "Partial", "Running"],
                                value="All",
                                label="Filter by Status",
                                scale=2,
                            )
                            history_refresh_btn = gr.Button("🔄 Refresh", scale=1)

                        history_table = gr.Dataframe(
                            headers=["ID", "Query", "Date", "Status", "Videos", "Transcripts", "Summaries", "Duration", "Full ID"],
                            datatype=["str", "str", "str", "str", "number", "number", "number", "str", "str"],
                            column_count=(9, "fixed"),
                            row_count=(10, "dynamic"),
                            interactive=False,
                            label="Pipeline Runs",
                            wrap=True,
                        )

                        with gr.Row():
                            history_prev_btn = gr.Button("◀ Previous", scale=1)
                            history_page_info = gr.Textbox(
                                value="Page 1",
                                label="",
                                interactive=False,
                                scale=1,
                            )
                            history_next_btn = gr.Button("Next ▶", scale=1)

                        history_status_msg = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=1,
                        )

                    # Right Panel - Run Details
                    with gr.Column(scale=1, elem_classes=["history-detail-panel"]):
                        run_details_html = gr.HTML(
                            value="<div style='padding: 40px; text-align: center; color: var(--body-text-color-subdued);'>Select a run from the list to view details</div>",
                            label="Run Details",
                        )

                        with gr.Row():
                            rerun_btn = gr.Button("🔄 Re-run Pipeline", variant="primary", scale=1)
                            delete_btn = gr.Button("🗑️ Delete Run", variant="stop", scale=1)

                        with gr.Row():
                            delete_files_checkbox = gr.Checkbox(
                                value=False,
                                label="Also delete output files",
                                info="Check to delete the output folder along with the database record",
                            )

                        delete_result_msg = gr.Textbox(
                            label="Action Result",
                            interactive=False,
                            lines=1,
                            visible=True,
                        )

                # Hidden state for tracking
                selected_run_id = gr.State(value="")
                current_page = gr.State(value=1)
                total_pages = gr.State(value=1)

                # History event handlers
                def on_history_refresh(status_filter):
                    data, pages, status = load_history_list(status_filter, page=1)
                    return data, 1, pages, status, "Page 1 of " + str(pages)

                def on_page_change(direction, current, total, status_filter):
                    new_page = current + direction
                    if new_page < 1:
                        new_page = 1
                    elif new_page > total:
                        new_page = total
                    data, pages, status = load_history_list(status_filter, page=new_page)
                    return data, new_page, pages, status, f"Page {new_page} of {pages}"

                def on_row_select(evt: gr.SelectData, dataframe_data):
                    if evt.index is not None and len(dataframe_data) > evt.index[0]:
                        row = dataframe_data.iloc[evt.index[0]]  # Use .iloc for row access by index
                        full_run_id = row.iloc[8]  # Full ID is in the 9th column (index 8)
                        details = load_run_details(full_run_id)
                        return full_run_id, details
                    return "", "<div style='padding: 20px; text-align: center;'>Select a run to view details</div>"

                def on_delete(run_id, delete_files, status_filter):
                    if not run_id:
                        return "No run selected", gr.update(), ""
                    msg, new_data, _ = delete_run_handler(run_id, delete_files)
                    if new_data:
                        return msg, new_data, ""
                    # Refresh on failure too
                    data, _, status = load_history_list(status_filter)
                    return msg, data, ""

                def on_rerun(run_id):
                    if not run_id:
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 0
                    query, max_v, lang, workers, use_env = rerun_pipeline_handler(run_id)
                    # Return values to update Pipeline tab inputs and switch to Pipeline tab
                    return query, max_v, lang, workers, use_env, 0

                # Connect history event handlers
                history_refresh_btn.click(
                    fn=on_history_refresh,
                    inputs=[history_status_filter],
                    outputs=[history_table, current_page, total_pages, history_status_msg, history_page_info],
                )

                history_status_filter.change(
                    fn=on_history_refresh,
                    inputs=[history_status_filter],
                    outputs=[history_table, current_page, total_pages, history_status_msg, history_page_info],
                )

                history_prev_btn.click(
                    fn=lambda c, t, s: on_page_change(-1, c, t, s),
                    inputs=[current_page, total_pages, history_status_filter],
                    outputs=[history_table, current_page, total_pages, history_status_msg, history_page_info],
                )

                history_next_btn.click(
                    fn=lambda c, t, s: on_page_change(1, c, t, s),
                    inputs=[current_page, total_pages, history_status_filter],
                    outputs=[history_table, current_page, total_pages, history_status_msg, history_page_info],
                )

                history_table.select(
                    fn=on_row_select,
                    inputs=[history_table],
                    outputs=[selected_run_id, run_details_html],
                )

                delete_btn.click(
                    fn=on_delete,
                    inputs=[selected_run_id, delete_files_checkbox, history_status_filter],
                    outputs=[delete_result_msg, history_table, selected_run_id],
                )

                rerun_btn.click(
                    fn=on_rerun,
                    inputs=[selected_run_id],
                    outputs=[search_query, max_videos, transcript_language, num_workers, use_env_keys, main_tabs],
                )

    return app, css


if __name__ == "__main__":
    # Check for cloud environment
    is_cloud = is_cloud_environment()

    if is_cloud:
        print("☁️  Detected cloud environment")
        print("📋 Using default configuration for cloud deployment...")

        # Use default values for cloud environments
        class DefaultArgs:
            def __init__(self):
                self.host = "0.0.0.0"
                self.port = 7860
                self.share = True
                self.debug = False

        args = DefaultArgs()

    else:
        print("💻 Detected local environment")
        print("📋 Using command line arguments...")

        # Parse command line arguments for local development
        parser = argparse.ArgumentParser(
            description="Atlas - AI-Powered Content Analysis Platform"
        )
        parser.add_argument(
            "--host", type=str, default="127.0.0.1", help="Host to run the server on"
        )
        parser.add_argument(
            "--port", type=int, default=7860, help="Port to run the server on"
        )
        parser.add_argument("--share", action="store_true", help="Create a public link")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")

        args = parser.parse_args()

    print("🔧 Configuration:")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Share: {args.share}")
    print(f"   Debug: {args.debug}")

    print("🚀 Launching Gradio interface...")

    app, css = create_gradio_app()
    app.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        show_error=args.debug,
        css=css,
        theme="soft",
    )

    if not is_cloud:
        print("\nExample usage:")
        print("  # Launch with default settings")
        print("  python app_youtube.py")
        print("  # Launch with public sharing")
        print("  python app_youtube.py --share")
        print("  # Launch on custom port")
        print("  python app_youtube.py --port 8080")
    else:
        print("\n☁️  Running in cloud environment - configuration is automatic!")
