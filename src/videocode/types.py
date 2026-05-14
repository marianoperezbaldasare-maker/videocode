"""Shared types and dataclasses for videocode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Application configuration."""

    # VLM Backend: "ollama" | "gemini" | "openai" | "qwen"
    vlm_backend: str = "ollama"
    vlm_model: str = "llava"
    vlm_api_key: Optional[str] = None
    vlm_base_url: Optional[str] = None

    # Ollama
    ollama_url: str = "http://localhost:11434"

    # Video Processing
    target_fps: float = 1.5
    max_frames: int = 50
    frame_quality: int = 85
    frame_resolution: Tuple[int, int] = (1280, 720)
    scene_threshold: float = 30.0

    # Audio
    whisper_model: str = "base"
    extract_audio: bool = True

    # Agent
    enable_agent_loop: bool = True
    max_retries: int = 2
    temperature: float = 0.3

    # Output
    output_dir: str = "./output"
    output_format: str = "markdown"

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        import os

        return cls(
            vlm_backend=os.getenv("VLM_BACKEND", "ollama"),
            vlm_model=os.getenv("VLM_MODEL", "llava"),
            vlm_api_key=os.getenv("VLM_API_KEY"),
            vlm_base_url=os.getenv("VLM_BASE_URL"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            target_fps=float(os.getenv("TARGET_FPS", "1.5")),
            max_frames=int(os.getenv("MAX_FRAMES", "50")),
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            output_dir=os.getenv("OUTPUT_DIR", "./output"),
        )

    @classmethod
    def from_file(cls, path: str) -> Config:
        """Load configuration from a JSON or YAML file."""
        import json

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Only set known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


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
