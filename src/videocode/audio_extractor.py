"""Audio extraction and transcription for videocode.

:class:`AudioExtractor` uses FFmpeg to pull audio tracks from video files
and then transcribes them with Whisper (``faster-whisper`` preferred,
``openai-whisper`` as fallback).

All whisper imports are deferred and guarded so the module can be imported
even when neither whisper package is installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from videocode.types import Segment, Transcription  # noqa: F401

if TYPE_CHECKING:
    from videocode.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AudioExtractionError(RuntimeError):
    """Raised when FFmpeg fails to extract audio."""


class TranscriptionError(RuntimeError):
    """Raised when the transcription engine fails."""


class WhisperNotAvailableError(RuntimeError):
    """Raised when no Whisper backend can be found."""


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------


class AudioExtractor:
    """Extract audio from video and transcribe with Whisper.

    Usage::

        config = Config(whisper_model="base")
        extractor = AudioExtractor(config)
        audio_path = extractor.extract(Path("video.mp4"))
        transcript = extractor.transcribe(audio_path)
        extractor.cleanup()
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._temp_dir: Path | None = None
        self._audio_path: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, video_path: Path) -> Path | None:
        """Extract the audio track from *video_path* to a mono WAV file.

        The output is a 16 kHz mono WAV — the format expected by Whisper
        models.  If the video has no audio track ``None`` is returned and
        a warning is logged.

        Args:
            video_path: Path to the source video.

        Returns:
            Path to the extracted ``.wav`` file, or ``None`` when no
            audio track is present.

        Raises:
            AudioExtractionError: If FFmpeg is missing or exits with an
                error.
        """
        if not self.config.extract_audio:
            logger.info("Audio extraction disabled in config")
            return None

        if shutil.which("ffmpeg") is None:
            raise AudioExtractionError(
                "ffmpeg not found on $PATH. Install it via your system package manager."
            )

        video_path = Path(video_path)
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")

        # Check whether the video actually has an audio stream
        if not self._has_audio_stream(video_path):
            logger.warning("No audio stream found in %s", video_path.name)
            return None

        out_dir = self._get_temp_dir()
        out_path = out_dir / f"{video_path.stem}_audio.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",  # no video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16 kHz — Whisper sweet spot
            "-ac", "1",  # mono
            "-f", "wav",
            str(out_path),
        ]

        logger.info("Extracting audio from %s -> %s", video_path.name, out_path)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            # Distinguish "no audio" from genuine errors
            if "Stream map '0:a'" in err or "Output file #0 does not contain" in err:
                logger.warning("Video appears to have no audio track: %s", video_path.name)
                return None
            raise AudioExtractionError(f"ffmpeg audio extraction failed: {err}")

        self._audio_path = out_path
        logger.info("Audio extracted: %s (%.1f KiB)", out_path, out_path.stat().st_size / 1024)
        return out_path

    def transcribe(self, audio_path: Path) -> Transcription:
        """Transcribe *audio_path* using Whisper.

        **Backend priority:**

        1. ``faster-whisper`` — preferred for speed and lower memory.
        2. ``openai-whisper`` — fallback when faster-whisper is absent.

        Args:
            audio_path: Path to a ``.wav`` audio file.

        Returns:
            A :class:`Transcription` with text, segments and detected
            language.

        Raises:
            WhisperNotAvailableError: If neither whisper package is
                installed.
            TranscriptionError: If the model fails to load or transcribe.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        # Try faster-whisper first
        try:
            return self._transcribe_faster_whisper(audio_path)
        except WhisperNotAvailableError:
            pass  # try fallback
        except Exception as exc:
            logger.warning("faster-whisper failed (%s) — trying openai-whisper", exc)

        # Fallback to openai-whisper
        try:
            return self._transcribe_openai_whisper(audio_path)
        except WhisperNotAvailableError:
            raise WhisperNotAvailableError(
                "No Whisper backend found. Install one of:\n"
                "  pip install faster-whisper   # recommended\n"
                "  pip install openai-whisper   # fallback"
            ) from None

    def cleanup(self) -> None:
        """Remove temporary audio files created by this extractor.

        Safe to call multiple times.
        """
        if self._temp_dir is not None and self._temp_dir.exists():
            logger.debug("Cleaning up audio temp directory: %s", self._temp_dir)
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
            self._audio_path = None

    # ------------------------------------------------------------------
    # Whisper backends
    # ------------------------------------------------------------------

    def _transcribe_faster_whisper(self, audio_path: Path) -> Transcription:
        """Transcribe with *faster-whisper*.

        Raises:
            WhisperNotAvailableError: If ``faster-whisper`` is not installed.
            TranscriptionError: On transcription failure.
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise WhisperNotAvailableError("faster-whisper not installed") from exc

        model_name = self.config.whisper_model
        logger.info("Loading faster-whisper model: %s", model_name)

        try:
            # Use CPU with int8 quantization for broad compatibility
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as exc:
            raise TranscriptionError(f"Failed to load faster-whisper model: {exc}") from exc

        logger.info("Transcribing %s …", audio_path.name)
        try:
            segments, info = model.transcribe(str(audio_path), beam_size=5)
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        detected_language = info.language if hasattr(info, "language") else "en"
        segments_list: list[Segment] = []
        full_text_parts: list[str] = []

        for seg in segments:
            seg_obj = Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            )
            segments_list.append(seg_obj)
            full_text_parts.append(seg.text.strip())

        return Transcription(
            text=" ".join(full_text_parts),
            segments=segments_list,
            language=detected_language,
        )

    def _transcribe_openai_whisper(self, audio_path: Path) -> Transcription:
        """Transcribe with *openai-whisper*.

        Raises:
            WhisperNotAvailableError: If ``openai-whisper`` is not installed.
            TranscriptionError: On transcription failure.
        """
        try:
            import whisper  # type: ignore[import-untyped]
        except ImportError as exc:
            raise WhisperNotAvailableError("openai-whisper not installed") from exc

        model_name = self.config.whisper_model
        logger.info("Loading openai-whisper model: %s", model_name)

        try:
            model = whisper.load_model(model_name)
        except Exception as exc:
            raise TranscriptionError(f"Failed to load openai-whisper model: {exc}") from exc

        logger.info("Transcribing %s …", audio_path.name)
        try:
            result = model.transcribe(str(audio_path))
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        detected_language = result.get("language", "en")
        segments_list: list[Segment] = []

        for seg in result.get("segments", []):
            segments_list.append(
                Segment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                )
            )

        return Transcription(
            text=result.get("text", "").strip(),
            segments=segments_list,
            language=detected_language,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_temp_dir(self) -> Path:
        """Return (and lazily create) a temporary directory."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="videocode-audio-"))
        return self._temp_dir

    def _has_audio_stream(self, video_path: Path) -> bool:
        """Probe whether *video_path* contains at least one audio stream."""
        if shutil.which("ffprobe") is None:
            # Be optimistic — let ffmpeg fail later if truly no audio
            return True

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and "audio" in result.stdout.lower()


__all__ = [
    "AudioExtractor",
    "Transcription",
    "Segment",
    "AudioExtractionError",
    "TranscriptionError",
    "WhisperNotAvailableError",
]
