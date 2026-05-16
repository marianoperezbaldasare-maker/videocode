"""Video processing engine for videocode.

Provides :class:`VideoProcessor` which orchestrates downloading,
frame extraction, scene detection and metadata inspection for video
sources (local files or YouTube URLs).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from videocode.types import Frame, ProcessedVideo, Scene  # noqa: F401

# Persistent cache for downloaded remote videos. Keyed by SHA-256 of URL
# so repeated runs against the same source are instant.
_DOWNLOAD_CACHE_DIR = Path.home() / ".cache" / "videocode" / "downloads"

_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi")

if TYPE_CHECKING:
    from videocode.config import Config
    from videocode.frame_selector import SelectionStrategy

# Optional dependency: pyscenedetect
try:
    from scenedetect import (  # type: ignore[import-untyped]
        AdaptiveDetector,
        ContentDetector,
        detect,
    )

    _HAS_SCENEDETECT = True
except ImportError:
    AdaptiveDetector = None  # type: ignore[misc,assignment]
    ContentDetector = None  # type: ignore[misc,assignment]
    detect = None  # type: ignore[misc,assignment]
    _HAS_SCENEDETECT = False

# Optional dependency: yt-dlp
try:
    import yt_dlp  # type: ignore[import-untyped]

    _HAS_YT_DLP = True
except ImportError:
    yt_dlp = None  # type: ignore[misc,assignment]
    _HAS_YT_DLP = False

# Optional: Apify for YouTube downloading
try:
    from videocode.apify_client import create_apify_client_from_env
    _HAS_APIFY = True
except ImportError:
    _HAS_APIFY = False

logger = logging.getLogger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ``ffmpeg`` binary cannot be located on ``$PATH``."""


class VideoProcessingError(RuntimeError):
    """Raised for any unrecoverable error during video processing."""


class VideoDownloadError(RuntimeError):
    """Raised when a video download (e.g. from YouTube) fails."""


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------


class VideoProcessor:
    """Orchestrates video download, frame extraction and scene detection.

    Typical usage::

        config = Config()
        vp = VideoProcessor(config)
        result = vp.process("/path/to/video.mp4")
        # ... do work with result ...
        vp.cleanup()
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.temp_dir: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        source: str | Path,
        strategy: SelectionStrategy | None = None,
    ) -> ProcessedVideo:
        """Process a video source end-to-end.

        *source* may be a local filesystem path or a YouTube URL (any URL
        supported by ``yt-dlp``).  The method downloads the video when
        necessary, extracts basic metadata, and then delegates frame
        selection to :class:`~videocode.frame_selector.FrameSelector`.

        Args:
            source: Local file path or remote URL.
            strategy: Frame selection strategy.  When ``None`` the
                :class:`~videocode.frame_selector.FrameSelector`
                uses its own default (``AUTO``).

        Returns:
            A :class:`ProcessedVideo` with populated frames and metadata.

        Raises:
            FFmpegNotFoundError: If ``ffmpeg`` is not on ``$PATH``.
            VideoProcessingError: For corrupt or unsupported video files.
            VideoDownloadError: If a remote download fails.
        """
        self._ensure_ffmpeg()

        source_str = str(source)
        video_path = self._resolve_source(source_str)

        # Gather metadata first so we can make informed decisions.
        duration, fps, resolution = self._probe_metadata(video_path)
        logger.info(
            "Processing '%s' — %.1fs @ %.2f fps, resolution %s",
            source_str,
            duration,
            fps,
            resolution,
        )

        # Lazy import to avoid circular dependency at module load time.
        from videocode.frame_selector import FrameSelector

        selector = FrameSelector(self.config)
        frames = selector.select_frames(video_path, strategy)

        # Build result
        frame_dir = self._get_or_create_frame_dir()
        result = ProcessedVideo(
            source=source_str,
            duration=duration,
            fps=fps,
            resolution=resolution,
            frames=frames,
            frame_dir=frame_dir,
            local_path=video_path,
        )
        return result

    def extract_frames(
        self,
        video_path: Path,
        timestamps: list[float],
        output_dir: Path | None = None,
    ) -> list[Frame]:
        """Extract JPEG frames at specific timestamps using FFmpeg.

        Args:
            video_path: Path to the video file.
            timestamps: Ordered list of times (in seconds) to capture.
            output_dir: Directory to write frames into.  A temporary
                directory is created automatically when ``None``.

        Returns:
            A list of :class:`Frame` objects, one per timestamp.

        Raises:
            FFmpegNotFoundError: If ``ffmpeg`` is missing.
            VideoProcessingError: If FFmpeg exits with a non-zero code.
        """
        self._ensure_ffmpeg()
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        frame_dir = output_dir or self._get_or_create_frame_dir()
        frame_dir.mkdir(parents=True, exist_ok=True)

        frames: list[Frame] = []
        width, height = self.config.frame_resolution
        quality = max(1, min(100, self.config.frame_quality))

        for idx, ts in enumerate(timestamps):
            out_name = f"frame_{idx:04d}_{ts:06.2f}s.jpg"
            out_path = frame_dir / out_name

            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", str(max(1, min(31, int((100 - quality) / 3.23)))),  # map 0-100 -> 31-1
                "-s", f"{width}x{height}",
                "-pix_fmt", "yuvj420p",
                str(out_path),
            ]

            logger.debug("FFmpeg extract frame @ %.2fs -> %s", ts, out_path)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    "Failed to extract frame @ %.2fs: %s", ts, result.stderr.strip()
                )
                continue

            # ffmpeg can exit 0 yet write no file when -ss lands past the
            # last decodable frame (it prints "Output file is empty, nothing
            # was encoded" to stderr but still returns 0). Verify the file
            # actually materialized before trusting it.
            if not out_path.exists() or out_path.stat().st_size == 0:
                logger.warning(
                    "FFmpeg returned 0 but produced no frame @ %.2fs "
                    "(likely past end of video) — skipping",
                    ts,
                )
                continue

            frames.append(Frame(path=out_path, timestamp=ts))

        logger.info("Extracted %d/%d frames", len(frames), len(timestamps))
        return frames

    def detect_scenes(self, video_path: Path) -> list[Scene]:
        """Detect scene boundaries using *PySceneDetect*.

        The method uses the :class:`AdaptiveDetector` with the
        :attr:`~Config.scene_threshold` from the configuration.

        Args:
            video_path: Path to the video file.

        Returns:
            Ordered list of :class:`Scene` objects.  An empty list is
            returned when *pyscenedetect* is not installed or when no
            scenes are found.

        Raises:
            VideoProcessingError: If the video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")

        if not _HAS_SCENEDETECT:
            logger.warning("pyscenedetect not installed — skipping scene detection")
            return []

        logger.info("Running scene detection on %s", video_path.name)
        if not _HAS_SCENEDETECT or detect is None:
            logger.warning("pyscenedetect not installed — skipping scene detection")
            return []

        try:
            # Try AdaptiveDetector first (newer pyscenedetect versions)
            detector = AdaptiveDetector(
                adaptive_threshold=self.config.scene_threshold,
                min_scene_len=15,  # ~0.5s at 30fps
            )
        except Exception:
            # Fall back to ContentDetector for older versions
            detector = ContentDetector(threshold=self.config.scene_threshold)

        try:
            scene_list = detect(str(video_path), detector)
        except Exception as exc:
            logger.error("Scene detection failed: %s", exc)
            return []

        scenes: list[Scene] = []
        for idx, (start, end) in enumerate(scene_list):
            # FrameTimecodes have get_seconds() method
            start_s = (
                start.get_seconds()
                if hasattr(start, "get_seconds")
                else float(start)
            )
            end_s = (
                end.get_seconds() if hasattr(end, "get_seconds") else float(end)
            )
            scenes.append(Scene(start=start_s, end=end_s, index=idx))

        logger.info("Detected %d scenes", len(scenes))
        return scenes

    def cleanup(self) -> None:
        """Remove temporary files and directories created by this processor.

        Safe to call multiple times — subsequent invocations are no-ops.
        """
        if self.temp_dir is not None and self.temp_dir.exists():
            logger.debug("Cleaning up temp directory: %s", self.temp_dir)
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ffmpeg(self) -> None:
        """Raise :class:`FFmpegNotFoundError` if *ffmpeg* is not available."""
        if shutil.which("ffmpeg") is None:
            raise FFmpegNotFoundError(
                "ffmpeg not found on $PATH. Install it via your system package manager."
            )

    def _get_or_create_frame_dir(self) -> Path:
        """Return (and create if needed) the frame output directory."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="videocode-"))
        frames_dir = self.temp_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir

    def _resolve_source(self, source: str) -> Path:
        """Convert *source* into a local :class:`~pathlib.Path`.

        If *source* looks like a URL it is downloaded via ``yt-dlp``.
        """
        # Heuristic: if it contains :// treat as URL
        if "://" in source:
            return self._download_video(source)
        path = Path(source)
        if not path.exists():
            raise VideoProcessingError(f"Video file not found: {path}")
        return path

    def _download_video(self, url: str) -> Path:
        """Download a video from *url* using Apify (preferred) or yt-dlp.

        Cached under ``~/.cache/videocode/downloads/<sha256>/`` so
        repeat runs against the same URL skip the download entirely.

        Returns the path to the downloaded file.

        Raises:
            VideoDownloadError: If no downloader is available or the
                download fails.
        """
        cache_subdir = self._cache_dir_for_url(url)

        cached = self._find_cached_video(cache_subdir)
        if cached is not None:
            logger.info("Using cached download for %s -> %s", url, cached)
            return cached

        cache_subdir.mkdir(parents=True, exist_ok=True)

        # Try Apify first (more reliable for YouTube)
        if _HAS_APIFY:
            try:
                apify = create_apify_client_from_env()
                if apify and apify.is_available():
                    logger.info("Downloading video via Apify: %s", url)
                    result = apify.download_youtube_video(url, str(cache_subdir))
                    if Path(result.video_path).exists():
                        logger.info("Downloaded via Apify: %s", result.title)
                        return Path(result.video_path)
            except Exception as exc:
                logger.warning("Apify download failed, falling back to yt-dlp: %s", exc)

        # Fallback to yt-dlp
        if not _HAS_YT_DLP or yt_dlp is None:
            raise VideoDownloadError(
                "No video downloader available. Install yt-dlp: pip install yt-dlp "
                "or set APIFY_API_TOKEN for Apify integration."
            )

        # Socket timeout keeps a stalled connection from blocking
        # indefinitely (default yt-dlp behavior would hold the MCP
        # request open until the client-side request timeout fires).
        socket_timeout = float(os.environ.get("VIDEOCODE_YTDLP_SOCKET_TIMEOUT", "30"))

        ydl_opts: dict[str, Any] = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(cache_subdir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": socket_timeout,
            "retries": 2,
            "fragment_retries": 2,
        }

        logger.info("Downloading video via yt-dlp: %s", url)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return Path(filename)
        except Exception as exc:
            raise VideoDownloadError(f"Failed to download video: {exc}") from exc

    @staticmethod
    def _cache_dir_for_url(url: str) -> Path:
        """Return the deterministic cache subdirectory for *url*."""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return _DOWNLOAD_CACHE_DIR / url_hash

    @staticmethod
    def _find_cached_video(cache_dir: Path) -> Path | None:
        """Return a previously-downloaded video file in *cache_dir*, or None."""
        if not cache_dir.exists():
            return None
        for entry in sorted(cache_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in _VIDEO_EXTS:
                return entry
        return None

    def _probe_metadata(self, video_path: Path) -> tuple[float, float, tuple[int, int]]:
        """Probe video metadata using ``ffprobe``.

        Returns:
            A 3-tuple of *(duration, fps, resolution)*.

        Raises:
            VideoProcessingError: If probing fails.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,duration",
            "-of", "json",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise VideoProcessingError(
                f"ffprobe failed for {video_path}: {result.stderr.strip()}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise VideoProcessingError(f"Invalid ffprobe JSON output: {exc}") from exc

        streams = data.get("streams", [])
        video_stream = None
        for s in streams:
            if s.get("codec_type") == "video":
                video_stream = s
                break

        if video_stream is None:
            raise VideoProcessingError(f"No video stream found in {video_path}")

        # Duration
        duration = 0.0
        for key in ("duration", "tags/DURATION"):
            raw = video_stream.get(key)
            if raw is None and key == "duration":
                raw = data.get("format", {}).get("duration")
            if raw:
                try:
                    duration = float(raw)
                except (ValueError, TypeError):
                    pass
                else:
                    break

        # FPS  — ffprobe returns it as a fraction "num/den"
        fps = 30.0
        fps_str = video_stream.get("r_frame_rate", "")
        if fps_str and "/" in fps_str:
            num, den = fps_str.split("/", 1)
            try:
                fps = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                fps = 30.0
        elif fps_str:
            try:
                fps = float(fps_str)
            except ValueError:
                fps = 30.0

        # Resolution
        width = int(video_stream.get("width", 0) or 1920)
        height = int(video_stream.get("height", 0) or 1080)

        logger.debug(
            "Metadata: duration=%.2f fps=%.2f resolution=%dx%d",
            duration,
            fps,
            width,
            height,
        )
        return duration, fps, (width, height)


# Re-export so that other modules can import Frame from here as well.
__all__ = [
    "Frame",
    "Scene",
    "ProcessedVideo",
    "VideoProcessor",
    "FFmpegNotFoundError",
    "VideoProcessingError",
    "VideoDownloadError",
]
