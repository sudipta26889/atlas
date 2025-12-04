import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from src.utils import ensure_output_folder, get_config, get_worker_count, setup_logging


class YouTubeTranscriptFetcher:
    """A class to fetch transcripts from YouTube videos.

    This class provides functionality to download transcripts (both human-generated
    and automatic) from YouTube videos. It uses youtube-transcript-api as the primary
    method (no cookies required) with yt-dlp as fallback.
    """

    def __init__(
        self,
        output_folder: Optional[str] = None,
        language: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize the YouTubeTranscriptFetcher.

        Args:
            output_folder (Optional[str]): Directory where transcripts will be saved.
                If None, uses config default.
            language (Optional[str]): Language code for subtitles. If None, uses config default.
            num_workers (Optional[int]): Number of concurrent workers for parallel processing.
                If None, auto-detects based on config and CPU count.
        """
        # Initialize configuration and logging
        setup_logging()

        # Set configuration values
        self.output_folder = output_folder or get_config(
            "processing.transcripts.output_folder", "transcripts"
        )
        self.language = language or get_config("processing.transcripts.language", "en")
        self.num_workers = get_worker_count(num_workers)

        # Ensure output folder exists
        self.output_folder = ensure_output_folder(self.output_folder)

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.

        Args:
            url (str): YouTube video URL.

        Returns:
            Optional[str]: Video ID or None if not found.
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _format_transcript_to_srt(self, transcript_data: List[dict], video_id: str) -> str:
        """Convert transcript data to SRT format.

        Args:
            transcript_data: List of transcript segments with 'text', 'start', 'duration'.
            video_id: Video ID for logging.

        Returns:
            str: Transcript in SRT format.
        """
        srt_content = []
        for i, entry in enumerate(transcript_data, 1):
            start_time = entry.get('start', 0)
            duration = entry.get('duration', 0)
            end_time = start_time + duration
            text = entry.get('text', '')

            # Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

            srt_content.append(f"{i}")
            srt_content.append(f"{format_time(start_time)} --> {format_time(end_time)}")
            srt_content.append(text)
            srt_content.append("")

        return "\n".join(srt_content)

    def _fetch_with_transcript_api(self, video_id: str) -> Tuple[bool, Optional[str]]:
        """Fetch transcript using youtube-transcript-api (no cookies required).

        Args:
            video_id (str): YouTube video ID.

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            # Create API instance (v1.2.3+ uses instance methods)
            ytt_api = YouTubeTranscriptApi()

            # Try to list available transcripts first
            try:
                transcript_list = ytt_api.list(video_id)

                # Try to find transcript in preferred language
                transcript_data = None
                for transcript in transcript_list:
                    if transcript.language_code in [self.language, 'en']:
                        transcript_data = transcript.fetch()
                        print(f"[TRANSCRIPT-API] Found transcript ({transcript.language_code}) for {video_id}")
                        break

                # If no match, use the first available
                if transcript_data is None and transcript_list:
                    transcript_data = transcript_list[0].fetch()
                    print(f"[TRANSCRIPT-API] Using first available transcript for {video_id}")

            except Exception:
                # Direct fetch as fallback
                transcript_data = ytt_api.fetch(video_id)
                print(f"[TRANSCRIPT-API] Direct fetch succeeded for {video_id}")

            if not transcript_data:
                return False, "No transcript data retrieved"

            # Convert to SRT and save
            srt_content = self._format_transcript_to_srt(transcript_data, video_id)
            output_path = os.path.join(self.output_folder, f"{video_id}.{self.language}.srt")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            print(f"[TRANSCRIPT-API] ✓ Saved transcript to {output_path}")
            return True, None

        except Exception as e:
            error_msg = str(e)
            if "disabled" in error_msg.lower():
                return False, "Transcripts are disabled for this video"
            elif "not found" in error_msg.lower() or "no transcript" in error_msg.lower():
                return False, f"No transcript found in {self.language} or en"
            elif "unavailable" in error_msg.lower():
                return False, "Video is unavailable"
            return False, error_msg

    def _fetch_with_ytdlp(self, url: str, video_id: str) -> Tuple[bool, Optional[str]]:
        """Fetch transcript using yt-dlp as fallback.

        Args:
            url (str): YouTube video URL.
            video_id (str): YouTube video ID.

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            from yt_dlp import YoutubeDL

            opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [self.language],
                "subtitlesformat": "srt",
                "outtmpl": os.path.join(self.output_folder, "%(id)s.%(ext)s"),
                "socket_timeout": 120,
                "retries": 3,
                "extractor_args": {"youtube": {"player_client": ["web_creator", "mweb", "web"]}},
                "quiet": True,
                "no_warnings": True,
                "geo_bypass": True,
            }

            # Check for cookies file
            cookies_paths = [
                "/app/cookies.txt",
                os.path.join(os.path.dirname(__file__), "..", "cookies.txt"),
                os.path.expanduser("~/cookies.txt"),
            ]
            for cookies_path in cookies_paths:
                if os.path.exists(cookies_path):
                    opts["cookiefile"] = cookies_path
                    break

            with YoutubeDL(opts) as ydl:
                ydl.download([url])

            # Check if file was created
            expected_path = os.path.join(self.output_folder, f"{video_id}.{self.language}.srt")
            if os.path.exists(expected_path):
                print(f"[YT-DLP] ✓ Saved transcript to {expected_path}")
                return True, None
            else:
                return False, "Transcript file not created"

        except Exception as e:
            return False, str(e)

    def fetch_transcript(self, url: str) -> bool:
        """Fetch transcript for a single YouTube video.

        Uses youtube-transcript-api first (no cookies needed), falls back to yt-dlp.

        Args:
            url (str): YouTube video URL.

        Returns:
            bool: True if transcript was successfully downloaded, False otherwise.
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            print(f"[ERROR] Could not extract video ID from: {url}")
            return False

        # Method 1: Try youtube-transcript-api (no cookies required)
        print(f"[FETCH] Trying youtube-transcript-api for {video_id}...")
        success, error = self._fetch_with_transcript_api(video_id)
        if success:
            return True
        print(f"[FETCH] youtube-transcript-api failed: {error}")

        # Method 2: Fall back to yt-dlp
        print(f"[FETCH] Falling back to yt-dlp for {video_id}...")
        success, error = self._fetch_with_ytdlp(url, video_id)
        if success:
            return True
        print(f"[FETCH] yt-dlp also failed: {error}")

        return False

    def _fetch_transcripts_sequential(self, urls: List[str]) -> dict:
        """Fetch transcripts sequentially (one at a time).

        Args:
            urls (List[str]): List of YouTube video URLs.

        Returns:
            dict: Dictionary with URLs as keys and success status as values.
        """
        results = {}
        for url in urls:
            print(f"Fetching transcript for: {url}")
            results[url] = self.fetch_transcript(url)
        return results

    def _fetch_transcripts_parallel(self, urls: List[str]) -> dict:
        """Fetch transcripts in parallel using ThreadPoolExecutor.

        Args:
            urls (List[str]): List of YouTube video URLs.

        Returns:
            dict: Dictionary with URLs as keys and success status as values.
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(self.fetch_transcript, url): url for url in urls
            }

            # Process completed tasks
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success = future.result()
                    results[url] = success
                    status = "Success" if success else "Failed"
                    print(f"Completed {url}: {status}")
                except Exception as e:
                    results[url] = False
                    print(f"Error processing {url}: {str(e)}")

        return results

    def fetch_transcripts(self, urls: List[str]) -> dict:
        """Fetch transcripts for multiple YouTube videos with automatic parallel/sequential fallback.

        Args:
            urls (List[str]): List of YouTube video URLs.

        Returns:
            dict: Dictionary with URLs as keys and success status as values.
        """
        if not urls:
            return {}

        # Use sequential processing if num_workers is 0 or only one URL
        if self.num_workers == 0 or len(urls) == 1:
            print(f"Using sequential processing for {len(urls)} URL(s)")
            return self._fetch_transcripts_sequential(urls)

        # Try parallel processing first
        try:
            print(
                f"Using parallel processing with {self.num_workers} workers for {len(urls)} URLs"
            )
            return self._fetch_transcripts_parallel(urls)
        except Exception as e:
            print(
                f"Parallel processing failed ({str(e)}), falling back to sequential processing"
            )
            return self._fetch_transcripts_sequential(urls)


# Example usage
if __name__ == "__main__":
    # Example URLs
    urls = [
        "https://www.youtube.com/watch?v=UV81LAb3x2g",
    ]

    # Initialize the fetcher
    fetcher = YouTubeTranscriptFetcher(output_folder="transcripts")

    # Fetch transcripts
    results = fetcher.fetch_transcripts(urls)

    # Print results
    for url, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{url}: {status}")
