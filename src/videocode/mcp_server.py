"""MCP server for videocode using FastMCP.

Provides MCP tools for Claude Code to analyze videos, extract code,
generate summaries, and find bugs through the Model Context Protocol.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from videocode.agent_loop import AgentLoop, Task, TaskType
from videocode.audio_extractor import AudioExtractor
from videocode.code_extractor import CodeExtractor
from videocode.config import Config
from videocode.frame_selector import FrameSelector, SelectionStrategy
from videocode.video_processor import VideoProcessor
from videocode.vlm_client import VLMClient

logger = logging.getLogger(__name__)

mcp = FastMCP("videocode")

_server: "ClaudeVisionMCPServer | None" = None


def _get_server() -> "ClaudeVisionMCPServer":
    if _server is None:
        raise RuntimeError(
            "ClaudeVisionMCPServer not initialized. "
            "This should not happen if the server was started via cli.py."
        )
    return _server


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _check_ollama(url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _check_whisper() -> bool:
    """Check if Whisper is available (faster-whisper or openai-whisper)."""
    for module in ("faster_whisper", "whisper"):
        try:
            __import__(module)  # noqa: S102
            return True
        except ImportError:
            continue
    return False


def _check_yt_dlp() -> bool:
    """Check if yt-dlp is available."""
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        return False


class ClaudeVisionMCPServer:
    """MCP server exposing videocode tools to Claude Code.

    This class initializes all processing components and provides the
    implementation that module-level @mcp.tool() functions delegate to.

    Args:
        config: Configuration for all processing components.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.video_processor = VideoProcessor(config)
        self.frame_selector = FrameSelector(config)
        self.audio_extractor = AudioExtractor(config)
        self.vlm = VLMClient(config)
        self.agent = AgentLoop(self.vlm, config)
        self.code_extractor = CodeExtractor(self.agent, config)

    @staticmethod
    def _resolve_strategy(mode: str) -> "SelectionStrategy | None":
        """Map a user-facing *mode* string to a :class:`SelectionStrategy`.

        ``"auto"`` (or anything unrecognized) returns ``None`` so the
        FrameSelector picks its own default.
        """
        mode_normalized = (mode or "auto").lower()
        mapping = {
            "auto": None,
            "tutorial": SelectionStrategy.TUTORIAL,
            "scene": SelectionStrategy.SCENE,
            "uniform": SelectionStrategy.UNIFORM,
            "keyframe": SelectionStrategy.KEYFRAME,
        }
        return mapping.get(mode_normalized, None)

    async def _process_video(
        self,
        source: str,
        task_type: TaskType,
        query: str = "",
        style: str = "detailed",
        mode: str = "auto",
        skip_audio: bool = False,
    ) -> dict[str, Any]:
        """Process a video through the full pipeline."""
        loop = asyncio.get_event_loop()
        strategy = self._resolve_strategy(mode)

        try:
            processed = await loop.run_in_executor(
                None, self.video_processor.process, source, strategy
            )
        except Exception as exc:
            logger.error("Video processing failed: %s", exc)
            return {
                "status": "error",
                "message": f"Failed to process video: {exc}",
            }

        transcription = None
        if self.config.extract_audio and not skip_audio:
            # IMPORTANT: use processed.local_path (the actual downloaded
            # file), not Path(source) — for YouTube URLs source has no
            # local presence so passing it would silently break audio.
            video_file = processed.local_path or Path(source)
            try:
                audio_path = await loop.run_in_executor(
                    None, self.audio_extractor.extract, video_file
                )
                transcription = await loop.run_in_executor(
                    None, self.audio_extractor.transcribe, audio_path
                )
            except Exception as exc:
                logger.warning("Audio extraction failed (non-fatal): %s", exc)

        try:
            task = Task(type=task_type, query=query or style)
            result = await loop.run_in_executor(
                None, self.agent.run, task, processed, transcription
            )
        except Exception as exc:
            logger.error("Agent loop failed: %s", exc)
            return {
                "status": "error",
                "message": f"Analysis failed: {exc}",
            }

        timestamps = [
            {"time": src.timestamp, "description": src.description}
            for src in result.sources
        ]

        return {
            "status": "ok",
            "confidence": result.confidence,
            "content": result.content,
            "timestamps": timestamps,
            "duration": getattr(processed, "duration", 0.0),
        }

    async def do_video_analyze(
        self, source: str, query: str = "", mode: str = "auto"
    ) -> dict[str, Any]:
        result = await self._process_video(
            source=source,
            task_type=TaskType.GENERAL,
            query=query,
            mode=mode,
        )

        if result["status"] == "error":
            return result

        return {
            "summary": result["content"][:500] if result["content"] else "",
            "details": result["content"],
            "timestamps": result.get("timestamps", []),
            "confidence": result["confidence"],
        }

    async def do_video_extract_code(self, source: str) -> dict[str, Any]:
        loop = asyncio.get_event_loop()

        try:
            processed = await loop.run_in_executor(
                None, self.video_processor.process, source
            )
        except Exception as exc:
            logger.error("Video processing failed: %s", exc)
            return {
                "status": "error",
                "message": f"Failed to process video: {exc}",
            }

        transcription = None
        if self.config.extract_audio:
            # Use the resolved local path so YouTube URLs work too.
            video_file = processed.local_path or Path(source)
            try:
                audio_path = await loop.run_in_executor(
                    None, self.audio_extractor.extract, video_file
                )
                transcription = await loop.run_in_executor(
                    None, self.audio_extractor.transcribe, audio_path
                )
            except Exception as exc:
                logger.warning("Audio extraction failed (non-fatal): %s", exc)

        try:
            code_result = await loop.run_in_executor(
                None, self.code_extractor.extract, processed, transcription
            )
        except Exception as exc:
            logger.error("Code extraction failed: %s", exc)
            return {
                "status": "error",
                "message": f"Code extraction failed: {exc}",
            }

        return {
            "files": code_result.files,
            "language": code_result.language,
            "description": code_result.description,
            "setup": code_result.setup_instructions,
            "confidence": code_result.confidence,
        }

    async def do_video_summarize(
        self, source: str, style: str = "detailed", mode: str = "auto"
    ) -> dict[str, Any]:
        result = await self._process_video(
            source=source,
            task_type=TaskType.SUMMARIZATION,
            style=style,
            mode=mode,
        )

        if result["status"] == "error":
            return result

        content = result.get("content", "")
        key_points = [
            line.strip("- ")
            for line in content.split("\n")
            if line.strip().startswith("-")
        ]

        return {
            "summary": content,
            "key_points": key_points or ([content[:200]] if content else []),
            "duration": result.get("duration", 0.0),
        }

    async def do_video_find_bugs(
        self, source: str, description: str = "", mode: str = "auto"
    ) -> dict[str, Any]:
        result = await self._process_video(
            source=source,
            task_type=TaskType.BUG_FINDING,
            query=description,
            mode=mode,
        )

        if result["status"] == "error":
            return result

        content = result.get("content", "")
        bugs = [
            line.strip("- ")
            for line in content.split("\n")
            if line.strip().startswith("-")
        ]
        recommendations = [
            line.strip("- ")
            for line in content.split("\n")
            if "recommend" in line.lower()
            or "fix" in line.lower()
            or "should" in line.lower()
        ]

        severity = "low"
        lower = content.lower()
        if "critical" in lower or "crash" in lower or "severe" in lower:
            severity = "critical"
        elif "high" in lower or "major" in lower or "error" in lower:
            severity = "high"
        elif "medium" in lower or "minor" in lower or "warning" in lower:
            severity = "medium"

        return {
            "bugs": bugs or ["No specific bugs identified"],
            "severity": severity,
            "recommendations": recommendations
            or ["Review the video manually for details"],
        }

    async def do_video_find_source_repo(self, source: str) -> dict[str, Any]:
        """Find GitHub/GitLab repos linked from a video's metadata.

        For YouTube and similar URLs only — runs yt-dlp metadata-only
        (no download), then scans description/title/channel for repo URLs.
        """
        from videocode.repo_finder import find_repos_for_url

        loop = asyncio.get_event_loop()
        try:
            candidates = await loop.run_in_executor(
                None, find_repos_for_url, source
            )
        except Exception as exc:
            logger.error("Repo discovery failed: %s", exc)
            return {"status": "error", "message": str(exc)}

        return {
            "found": len(candidates),
            "repos": [
                {
                    "url": c.url,
                    "owner": c.owner,
                    "repo": c.repo,
                    "host": c.host,
                    "found_in": c.source,
                }
                for c in candidates
            ],
        }

    async def do_video_extract_text(
        self, source: str, query: str = ""
    ) -> dict[str, Any]:
        """OCR-focused tool: read all visible text from a video.

        Uses TUTORIAL strategy (dense sampling, higher frame cap) and
        skips audio transcription — text extraction is purely visual.
        """
        ocr_query = (
            query
            or "Transcribe ALL visible text from these frames verbatim. "
            "Include: UI labels, button text, code, terminal output, slide "
            "content, captions, error messages, file names, URLs. "
            "Group by timestamp. Preserve exact wording, casing, punctuation. "
            "If text is partially obscured or unclear, mark with [unclear] "
            "but include your best guess. Return as a markdown list grouped "
            "by timestamp."
        )

        result = await self._process_video(
            source=source,
            task_type=TaskType.GENERAL,
            query=ocr_query,
            mode="tutorial",
            skip_audio=True,
        )

        if result["status"] == "error":
            return result

        return {
            "text": result["content"],
            "timestamps": result.get("timestamps", []),
            "duration": result.get("duration", 0.0),
            "confidence": result["confidence"],
        }

    def do_health_check(self) -> dict[str, Any]:
        backends: dict[str, bool] = {
            "ffmpeg": _check_ffmpeg(),
            "whisper": _check_whisper(),
            "yt_dlp": _check_yt_dlp(),
        }

        if self.config.vlm_backend == "ollama":
            backends["ollama"] = _check_ollama(self.config.ollama_url)
        else:
            backends[self.config.vlm_backend] = True

        all_ok = all(backends.values())

        return {
            "status": "ok" if all_ok else "degraded",
            "backends": backends,
        }

    def run(self) -> None:
        """Start the MCP server with stdio transport.

        Binds this instance as the module-level singleton so the
        @mcp.tool() functions below can reach it, then blocks on the
        stdio transport until terminated.
        """
        global _server
        _server = self
        logger.info("Starting videocode MCP server (stdio transport)")
        mcp.run(transport="stdio")


@mcp.tool()
async def video_analyze(
    source: str, query: str = "", mode: str = "auto"
) -> dict[str, Any]:
    """Analyze a video and answer questions about its content.

    Process the video through frame extraction, audio transcription,
    and VLM analysis to answer the given query.

    Args:
        source: Path to video file or YouTube URL.
        query: Specific question about the video content.
        mode: Frame sampling mode. One of ``"auto"`` (default),
            ``"tutorial"`` (dense sampling, ideal for screen recordings
            and coding tutorials), ``"scene"``, ``"uniform"``,
            ``"keyframe"``.

    Returns:
        Dictionary with summary, details, and relevant timestamps.
    """
    return await _get_server().do_video_analyze(source, query, mode)


@mcp.tool()
async def video_extract_code(source: str) -> dict[str, Any]:
    """Extract code from a coding tutorial video.

    Process the video to detect code frames, extract code blocks,
    and assemble them into a structured project.

    Args:
        source: Path to video file or YouTube URL.

    Returns:
        Dictionary with files, language, description, setup instructions,
        and confidence score.
    """
    return await _get_server().do_video_extract_code(source)


@mcp.tool()
async def video_summarize(
    source: str, style: str = "detailed", mode: str = "auto"
) -> dict[str, Any]:
    """Generate a summary of a video.

    Extract key points and produce a structured summary in the
    requested style.

    Args:
        source: Path to video file or YouTube URL.
        style: Summary style - "detailed", "brief", or "bullet_points".
        mode: Frame sampling mode. See :func:`video_analyze`.

    Returns:
        Dictionary with summary text, key points list, and duration.
    """
    return await _get_server().do_video_summarize(source, style, mode)


@mcp.tool()
async def video_find_bugs(
    source: str, description: str = "", mode: str = "auto"
) -> dict[str, Any]:
    """Analyze a screen recording for bugs and issues.

    Examine the video for UI bugs, error messages, crashes,
    and other issues.

    Args:
        source: Path to video file or YouTube URL.
        description: Optional context about what the video shows.
        mode: Frame sampling mode. See :func:`video_analyze`.

    Returns:
        Dictionary with found bugs, severity assessment, and
        recommendations.
    """
    return await _get_server().do_video_find_bugs(source, description, mode)


@mcp.tool()
async def video_find_source_repo(source: str) -> dict[str, Any]:
    """Find GitHub/GitLab source repos linked from a tutorial video.

    Many coding tutorials link their accompanying repo in the video
    description or pinned comment. This tool extracts those references
    so you can get exact, runnable code instead of reconstructing it
    from video frames.

    Args:
        source: YouTube URL (or any yt-dlp-supported URL).

    Returns:
        Dictionary with ``found`` count and ``repos`` list. Each repo
        entry has ``url``, ``owner``, ``repo``, ``host``, ``found_in``
        (which metadata field surfaced the URL).
    """
    return await _get_server().do_video_find_source_repo(source)


@mcp.tool()
async def video_extract_text(source: str, query: str = "") -> dict[str, Any]:
    """Extract all visible text from a video (OCR mode).

    Optimized for screen recordings, slide decks, coding tutorials,
    and dashboards. Uses dense frame sampling (TUTORIAL strategy)
    and skips audio transcription — text extraction is purely visual.

    Args:
        source: Path to video file or YouTube URL.
        query: Optional refinement (e.g. "only the SQL queries shown").
            When empty, extracts ALL visible text grouped by timestamp.

    Returns:
        Dictionary with extracted text, timestamps, duration, and
        confidence score.
    """
    return await _get_server().do_video_extract_text(source, query)


@mcp.tool()
async def health_check() -> dict[str, Any]:
    """Check if all backends are available.

    Verifies that FFmpeg, the configured VLM backend (Ollama/API),
    Whisper, and yt-dlp are accessible.

    Returns:
        Dictionary with overall status and per-backend availability.
    """
    return _get_server().do_health_check()
