"""Shared types and dataclasses for videocode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Re-export Config from the canonical location so legacy imports like
# `from videocode.types import Config` keep working.
from videocode.config import Config  # noqa: F401


# ---------------------------------------------------------------------------
# Video Processing
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """A single video frame."""

    path: Path
    timestamp: float  # seconds
    scene_index: int = 0
    is_keyframe: bool = False


@dataclass
class Scene:
    """A detected scene boundary."""

    start: float  # seconds
    end: float
    index: int


@dataclass
class ProcessedVideo:
    """Result of processing a video file."""

    source: str
    duration: float
    fps: float
    resolution: Tuple[int, int]
    frames: List[Frame]
    frame_dir: Path
    audio_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Audio / Transcription
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single transcript segment."""

    start: float
    end: float
    text: str


@dataclass
class Transcription:
    """Full transcription result."""

    text: str
    segments: List[Segment]
    language: str = "en"


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

@dataclass
class VLMResponse:
    """Response from a VLM backend."""

    content: str
    model: str = ""
    tokens_used: int = 0
    frames_analyzed: int = 0


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

class TaskType(Enum):
    """Types of tasks the agent can perform."""

    CODE_EXTRACTION = "code_extraction"
    SUMMARIZATION = "summarization"
    BUG_FINDING = "bug_finding"
    GENERAL = "general"


@dataclass
class Task:
    """A task for the agent loop."""

    type: TaskType
    query: str


@dataclass
class SourceReference:
    """Reference to a source frame in the video."""

    timestamp: float
    frame_path: Path
    description: str = ""


@dataclass
class AgentResult:
    """Result from the agent loop."""

    content: str
    confidence: float = 0.0
    sources: List[SourceReference] = field(default_factory=list)
    retries: int = 0


# ---------------------------------------------------------------------------
# Code Extraction
# ---------------------------------------------------------------------------

@dataclass
class CodeFrame:
    """A frame that contains code."""

    frame: Frame
    code: str = ""
    language: str = ""
    confidence: float = 0.0


@dataclass
class CodeBlock:
    """An extracted block of code."""

    filename: str
    content: str
    language: str = ""
    source_timestamps: List[float] = field(default_factory=list)


@dataclass
class CodeResult:
    """Result of extracting code from a video."""

    files: Dict[str, str] = field(default_factory=dict)
    language: str = ""
    description: str = ""
    setup_instructions: str = ""
    confidence: float = 0.0
