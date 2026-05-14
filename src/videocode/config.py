"""Configuration management for videocode.

Provides the :class:`Config` dataclass that holds all settings for video
processing, frame selection, audio extraction, VLM integration and agent
behaviour.  Configurations can be loaded from environment variables or from
JSON files.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration for videocode.

    Each field can be overridden via environment variables (prefixed with
    ``VIDEOCODE_``) or by loading a JSON configuration file.
    """

    def __post_init__(self) -> None:
        """Load environment variable overrides after initialization."""
        self._load_env_overrides()

    def _load_env_overrides(self) -> None:
        """Override config values from environment variables."""
        annotations = self.__annotations__
        for key in annotations:
            env_name = f"VIDEOCODE_{key.upper()}"
            raw_value = os.environ.get(env_name)
            
            # Also check standard env var names for integrations
            if raw_value is None:
                if key == "apify_api_token":
                    raw_value = os.environ.get("APIFY_API_TOKEN")
                elif key == "perplexity_api_key":
                    raw_value = os.environ.get("PERPLEXITY_API_KEY")
                elif key == "vlm_api_key":
                    if self.vlm_backend == "gemini":
                        raw_value = os.environ.get("GEMINI_API_KEY")
                    elif self.vlm_backend == "openai":
                        raw_value = os.environ.get("OPENAI_API_KEY")
                    elif self.vlm_backend == "qwen":
                        raw_value = os.environ.get("QWEN_API_KEY")
            
            if raw_value is None:
                continue

            # Attempt JSON parsing first
            try:
                parsed: Any = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = raw_value

            # Convert list -> tuple for frame_resolution
            if key == "frame_resolution" and isinstance(parsed, list):
                parsed = tuple(parsed)

            if hasattr(self, key):
                setattr(self, key, parsed)
                logger.debug("Config override from env: %s = %r", env_name, parsed)

    # ------------------------------------------------------------------
    # VLM Backend
    # ------------------------------------------------------------------
    vlm_backend: str = "ollama"
    """VLM backend identifier. One of ``ollama``, ``gemini``, ``openai``, ``qwen``, ``dummy``."""

    vlm_model: str = "llava"
    """Model name to use with the selected VLM backend."""

    vlm_api_key: Optional[str] = None
    """API key for cloud-based VLM backends (Gemini, OpenAI, Qwen)."""

    vlm_base_url: Optional[str] = None
    """Custom base URL for the VLM API endpoint."""

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    """Base URL for the local Ollama instance."""

    # ------------------------------------------------------------------
    # Optional: Apify (YouTube downloading)
    # ------------------------------------------------------------------
    apify_api_token: Optional[str] = None
    """Apify API token for YouTube video downloading."""

    # ------------------------------------------------------------------
    # Optional: Perplexity (code verification)
    # ------------------------------------------------------------------
    perplexity_api_key: Optional[str] = None
    """Perplexity API key for code verification and documentation lookup."""

    # ------------------------------------------------------------------
    # Video Processing
    # ------------------------------------------------------------------
    target_fps: float = 1.5
    """Target frames per second when using uniform sampling.

    Bumped from 0.5 to 1.5 so short tutorials get enough frames for readable
    text/UI/code (5 frames in a 10s clip was empirically too sparse for OCR).
    """

    max_frames: int = 50
    """Hard upper limit on the number of frames extracted per video."""

    frame_quality: int = 85
    """JPEG quality factor for extracted frames (0-100)."""

    frame_resolution: Tuple[int, int] = (1280, 720)
    """Target frame resolution as ``(width, height)``."""

    scene_threshold: float = 30.0
    """Sensitivity threshold for PySceneDetect adaptive detector."""

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    whisper_model: str = "base"
    """Whisper model size. One of ``tiny``, ``base``, ``small``, ``medium``, ``large``."""

    extract_audio: bool = True
    """Whether to extract and transcribe audio tracks by default."""

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    enable_agent_loop: bool = True
    """Enable the 3-role agent loop (Extractor / Analyser / Verifier)."""

    max_retries: int = 2
    """Maximum number of retries for failed VLM calls."""

    temperature: float = 0.3
    """Sampling temperature for VLM inference."""

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = "./output"
    """Directory where outputs (frames, transcripts, reports) are written."""

    output_format: str = "markdown"
    """Default output format. One of ``markdown``, ``json``, ``files``."""

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from environment variables.

        Every field can be set via an environment variable prefixed with
        ``VIDEOCODE_`` and upper-cased.  For example,
        ``VIDEOCODE_MAX_FRAMES=20`` sets :attr:`max_frames` to ``20``.
        JSON values are parsed automatically so nested structures such as
        :attr:`frame_resolution` (a list of two ints) work as expected.

        Returns:
            A :class:`Config` instance with fields populated from the
            environment.  Unset variables keep their default values.
        """
        instance = cls()
        annotations = cls.__dataclass_fields__  # type: ignore[attr-defined]

        for key in annotations:
            env_name = f"VIDEOCODE_{key.upper()}"
            raw_value = os.environ.get(env_name)
            
            # Also check standard env var names for integrations
            if raw_value is None:
                if key == "apify_api_token":
                    raw_value = os.environ.get("APIFY_API_TOKEN")
                elif key == "perplexity_api_key":
                    raw_value = os.environ.get("PERPLEXITY_API_KEY")
                elif key == "vlm_api_key":
                    # Check backend-specific env vars
                    backend = os.environ.get("VIDEOCODE_VLM_BACKEND", instance.vlm_backend)
                    if backend == "gemini":
                        raw_value = os.environ.get("GEMINI_API_KEY")
                    elif backend == "openai":
                        raw_value = os.environ.get("OPENAI_API_KEY")
            
            if raw_value is None:
                continue

            # Attempt JSON parsing first (handles bool, float, list, etc.)
            try:
                parsed: Any = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = raw_value  # fall back to plain string

            # Convert list -> tuple for frame_resolution
            if key == "frame_resolution" and isinstance(parsed, list):
                parsed = tuple(parsed)

            if hasattr(instance, key):
                setattr(instance, key, parsed)
                logger.debug("Config override from env: %s = %r", env_name, parsed)

        return instance

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Build a :class:`Config` from a JSON configuration file.

        The JSON file should contain top-level keys that match the
        dataclass field names.  Missing keys keep their default values.

        Args:
            path: Filesystem path to the JSON file.

        Returns:
            A :class:`Config` instance populated from the JSON data.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)

        # Convert list -> tuple for frame_resolution if needed
        if "frame_resolution" in data and isinstance(data["frame_resolution"], list):
            data["frame_resolution"] = tuple(data["frame_resolution"])

        known_keys = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_keys}
        unknown = set(data.keys()) - set(known_keys)
        if unknown:
            logger.warning("Ignored unknown config keys: %s", ", ".join(sorted(unknown)))

        return cls(**filtered)
