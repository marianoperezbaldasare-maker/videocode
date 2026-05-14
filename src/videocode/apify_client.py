"""Apify integration for videocode.

Uses Apify for:
- Downloading YouTube videos (more reliable than yt-dlp)
- Transcribing audio with Whisper actor
- Extracting video metadata
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

APIFY_BASE_URL = "https://api.apify.com/v2"


@dataclass
class ApifyVideoResult:
    """Result from Apify video download."""
    video_path: str
    title: str
    duration: float
    description: str
    thumbnail_url: str


@dataclass
class ApifyTranscriptionResult:
    """Result from Apify transcription."""
    text: str
    segments: list[dict]
    language: str


class ApifyClient:
    """Client for Apify platform integration."""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("APIFY_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Apify API token required. Set APIFY_API_TOKEN env var "
                "or pass api_token parameter."
            )
        self.client = httpx.Client(timeout=300.0)

    def is_available(self) -> bool:
        """Check if Apify API is accessible."""
        try:
            resp = self.client.get(
                f"{APIFY_BASE_URL}/acts",
                headers={"Authorization": f"Bearer {self.api_token}"},
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning("Apify not available: %s", e)
            return False

    def download_youtube_video(self, youtube_url: str, output_dir: str = "./downloads") -> ApifyVideoResult:
        """Download a YouTube video using Apify's YouTube downloader actor.
        
        Args:
            youtube_url: Full YouTube URL
            output_dir: Where to save the downloaded video
            
        Returns:
            ApifyVideoResult with video path and metadata
        """
        logger.info("Downloading YouTube video via Apify: %s", youtube_url)
        os.makedirs(output_dir, exist_ok=True)

        # Run the YouTube downloader actor
        run_input = {
            "startUrls": [{"url": youtube_url}],
            "downloadFormats": "mp4",
            "maxQuality": 720,
            "subtitles": False,
        }

        dataset_items = self._run_actor(
            "streaming/youtube-downloader",
            run_input,
        )

        if not dataset_items:
            raise RuntimeError("Apify: No video downloaded")

        item = dataset_items[0]
        video_path = os.path.join(output_dir, f"video_{int(time.time())}.mp4")
        
        # Download the actual video file from the URL
        video_url = item.get("downloadUrl") or item.get("url")
        if video_url:
            self._download_file(video_url, video_path)

        return ApifyVideoResult(
            video_path=video_path,
            title=item.get("title", "Unknown"),
            duration=item.get("duration", 0),
            description=item.get("description", ""),
            thumbnail_url=item.get("thumbnailUrl", ""),
        )

    def transcribe_video(self, video_path: str) -> ApifyTranscriptionResult:
        """Transcribe video audio using Apify's Whisper actor.
        
        Args:
            video_path: Path to local video file
            
        Returns:
            ApifyTranscriptionResult with transcription text and segments
        """
        logger.info("Transcribing video via Apify Whisper: %s", video_path)

        # Upload video to Apify key-value store or use file URL
        # For local files, we read and send
        with open(video_path, "rb") as f:
            video_data = f.read()

        # Run Whisper actor
        run_input = {
            "audioFile": video_data,  # Apify handles file uploads
            "model": "base",
            "language": "auto",
        }

        dataset_items = self._run_actor(
            "nlp/whisper",
            run_input,
        )

        if not dataset_items:
            raise RuntimeError("Apify: Transcription failed")

        item = dataset_items[0]
        return ApifyTranscriptionResult(
            text=item.get("text", ""),
            segments=item.get("segments", []),
            language=item.get("language", "en"),
        )

    def extract_video_metadata(self, youtube_url: str) -> dict:
        """Extract metadata from a YouTube video without downloading.
        
        Args:
            youtube_url: Full YouTube URL
            
        Returns:
            Dict with title, description, duration, viewCount, etc.
        """
        logger.info("Extracting metadata via Apify: %s", youtube_url)

        run_input = {
            "startUrls": [{"url": youtube_url}],
            "maxResults": 1,
            "downloadSubtitles": False,
        }

        dataset_items = self._run_actor(
            "streaming/youtube-scraper",
            run_input,
        )

        if not dataset_items:
            return {}

        item = dataset_items[0]
        return {
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "duration": item.get("duration", 0),
            "viewCount": item.get("viewCount", 0),
            "uploadDate": item.get("uploadDate", ""),
            "channel": item.get("channelName", ""),
            "tags": item.get("tags", []),
            "thumbnail": item.get("thumbnailUrl", ""),
        }

    def _run_actor(self, actor_id: str, run_input: dict, timeout_secs: int = 300) -> list:
        """Run an Apify actor and wait for results.
        
        Args:
            actor_id: Actor identifier (e.g., 'streaming/youtube-downloader')
            run_input: Input parameters for the actor
            timeout_secs: Maximum time to wait
            
        Returns:
            List of dataset items
        """
        # Start the actor run
        resp = self.client.post(
            f"{APIFY_BASE_URL}/acts/{actor_id}/runs",
            headers={"Authorization": f"Bearer {self.api_token}"},
            json=run_input,
        )
        resp.raise_for_status()
        run_data = resp.json()
        run_id = run_data["data"]["id"]
        
        logger.info("Apify actor run started: %s (run ID: %s)", actor_id, run_id)

        # Poll for completion
        for _ in range(timeout_secs // 5):
            time.sleep(5)
            resp = self.client.get(
                f"{APIFY_BASE_URL}/acts/{actor_id}/runs/{run_id}",
                headers={"Authorization": f"Bearer {self.api_token}"},
            )
            status = resp.json()["data"]["status"]
            
            if status == "SUCCEEDED":
                break
            elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                raise RuntimeError(f"Apify actor failed with status: {status}")
            logger.debug("Apify run status: %s", status)

        # Get dataset items
        dataset_id = resp.json()["data"]["defaultDatasetId"]
        resp = self.client.get(
            f"{APIFY_BASE_URL}/datasets/{dataset_id}/items",
            headers={"Authorization": f"Bearer {self.api_token}"},
        )
        resp.raise_for_status()
        
        items = resp.json()
        logger.info("Apify actor returned %d items", len(items))
        return items

    def _download_file(self, url: str, output_path: str) -> None:
        """Download a file from URL to local path."""
        logger.info("Downloading file to %s", output_path)
        with self.client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()


def create_apify_client_from_env() -> Optional[ApifyClient]:
    """Create ApifyClient from environment variable if available."""
    token = os.environ.get("APIFY_API_TOKEN")
    if token:
        try:
            return ApifyClient(token)
        except Exception as e:
            logger.warning("Failed to create Apify client: %s", e)
    return None
