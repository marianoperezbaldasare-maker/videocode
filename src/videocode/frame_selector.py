"""Intelligent frame selection for videocode.

:class:`FrameSelector` implements multiple strategies for choosing which
frames to extract from a video so that they fit within a token budget
while still covering the most visually interesting moments.
"""

from __future__ import annotations

import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from videocode.config import Config

from videocode.video_processor import Frame, VideoProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class SelectionStrategy(Enum):
    """Available frame selection strategies."""

    AUTO = "auto"
    """Automatically choose between *scene* and *uniform* based on content."""

    SCENE = "scene"
    """Extract one representative frame per detected scene."""

    UNIFORM = "uniform"
    """Sample frames at a fixed time interval (``1 / target_fps`` seconds)."""

    KEYFRAME = "keyframe"
    """Extract only FFmpeg keyframes (I-frames)."""

    TUTORIAL = "tutorial"
    """Dense uniform sampling (2 fps, cap 80) tuned for tutorials.

    UI/code in tutorials changes frequently — the default 1.5 fps is
    still too sparse for reliable OCR of state transitions. Use this
    explicitly when the source is a screen recording or coding tutorial.
    """


# ---------------------------------------------------------------------------
# Token budgeting constants
# ---------------------------------------------------------------------------

# Approximate vision tokens per frame at 720p (reference resolution).
_TOKENS_PER_FRAME_720P: int = 150
_REFERENCE_HEIGHT: int = 720


class FrameSelector:
    """Chooses an optimal set of frames from a video.

    The selector respects both a hard frame limit and an estimated token
    budget so that the resulting frames can be fed to a VLM without
    exceeding its context window.

    Typical usage::

        config = Config(max_frames=30)
        selector = FrameSelector(config)
        frames = selector.select_frames(Path("video.mp4"))
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_frames(
        self,
        video_path: Path,
        strategy: SelectionStrategy | None = None,
    ) -> list[Frame]:
        """Select frames from *video_path* using the given *strategy*.

        The returned list is sorted by timestamp and never exceeds
        :attr:`~Config.max_frames`.

        Args:
            video_path: Path to the video file.
            strategy: Selection strategy.  Defaults to
                :attr:`SelectionStrategy.AUTO` when ``None``.

        Returns:
            Ordered list of :class:`Frame` objects.

        Raises:
            RuntimeError: If FFmpeg is not available.
            FileNotFoundError: If *video_path* does not exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        strategy = strategy or SelectionStrategy.AUTO
        logger.info("Frame selection strategy: %s", strategy.value)

        # Gather basic metadata
        duration, fps, resolution = self._probe_metadata(video_path)

        # Resolve AUTO strategy
        if strategy == SelectionStrategy.AUTO:
            strategy = self._resolve_auto_strategy(video_path)
            logger.info("AUTO resolved to: %s", strategy.value)

        # Estimate how many frames we can afford
        optimal_count = self.estimate_optimal_frame_count(duration, resolution)

        if strategy == SelectionStrategy.TUTORIAL:
            # Tutorial mode: denser sampling, higher cap, ignore the
            # normal optimal_count gate so short clips still get many frames.
            tutorial_count = min(80, max(8, int(duration * 2.0)))
            frames = self._select_uniform_frames(video_path, duration, tutorial_count)
        elif strategy == SelectionStrategy.SCENE:
            frames = self._select_scene_frames(video_path, optimal_count)
        elif strategy == SelectionStrategy.KEYFRAME:
            frames = self._select_keyframe_frames(video_path, optimal_count)
        else:
            # UNIFORM
            frames = self._select_uniform_frames(video_path, duration, optimal_count)

        # Safety: sort and enforce hard cap.
        # TUTORIAL strategy gets a higher cap (80) since it intentionally
        # samples densely to catch UI/code state changes.
        frames.sort(key=lambda f: f.timestamp)
        effective_cap = 80 if strategy == SelectionStrategy.TUTORIAL else self.config.max_frames
        if len(frames) > effective_cap:
            logger.info(
                "Down-sampling from %d to %d frames (cap=%d, strategy=%s)",
                len(frames),
                effective_cap,
                effective_cap,
                strategy.value,
            )
            frames = self._evenly_downsample(frames, effective_cap)

        token_budget = self.calculate_token_budget(len(frames))
        logger.info(
            "Selected %d frames, estimated token budget: %d",
            len(frames),
            token_budget,
        )
        return frames

    def calculate_token_budget(self, frame_count: int) -> int:
        """Estimate the total vision-token cost for *frame_count* frames.

        The heuristic assumes ~150 tokens per frame at 720p resolution.
        The cost scales with the square root of the area ratio to keep
        the estimate conservative for larger frames.

        Args:
            frame_count: Number of frames that will be sent to the VLM.

        Returns:
            Estimated token count.
        """
        width, height = self.config.frame_resolution
        area_ratio = (width * height) / (1280 * _REFERENCE_HEIGHT)
        scaling = max(1.0, area_ratio ** 0.5)
        return int(frame_count * _TOKENS_PER_FRAME_720P * scaling)

    def estimate_optimal_frame_count(
        self,
        duration: float,
        resolution: tuple[int, int],
    ) -> int:
        """Calculate the optimal number of frames for a video.

        The result respects:
        1. The configured :attr:`~Config.max_frames` hard limit.
        2. A practical minimum of ``min(3, max_frames)`` so short clips
           still get a few frames.
        3. A token-budget ceiling so we do not blow the VLM context.

        Args:
            duration: Video length in seconds.
            resolution: Video resolution as ``(width, height)``.

        Returns:
            Recommended frame count.
        """
        max_frames = self.config.max_frames

        # Start with target-fps based estimate
        fps_based = max(1, int(duration * self.config.target_fps))

        # Token-budget based ceiling
        # Heuristic: assume a ~16k token budget for frames
        token_budget_ceiling = 16384
        tokens_per_frame = max(1, self.calculate_token_budget(1))
        token_based = max(1, token_budget_ceiling // tokens_per_frame)

        count = min(fps_based, token_based, max_frames)
        count = max(count, min(3, max_frames))  # at least a few frames
        return count

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _resolve_auto_strategy(self, video_path: Path) -> SelectionStrategy:
        """Decide whether to use SCENE or UNIFORM for *video_path*.

        We detect scenes and pick :attr:`SelectionStrategy.SCENE` when
        more than 3 scenes are found; otherwise we fall back to
        :attr:`SelectionStrategy.UNIFORM` which tends to work better for
        content with few visual changes (e.g. a single talking-head
        scene).
        """
        try:
            processor = VideoProcessor(self.config)
            scenes = processor.detect_scenes(video_path)
            if len(scenes) > 3:
                logger.info(
                    "AUTO: %d scenes detected — using SCENE strategy", len(scenes)
                )
                return SelectionStrategy.SCENE
            logger.info(
                "AUTO: %d scenes detected — using UNIFORM strategy", len(scenes)
            )
            return SelectionStrategy.UNIFORM
        except Exception:
            logger.warning("AUTO: scene detection failed — falling back to UNIFORM")
            return SelectionStrategy.UNIFORM

    def _select_scene_frames(
        self, video_path: Path, target_count: int
    ) -> list[Frame]:
        """Select one frame per detected scene.

        When there are more scenes than *target_count*, the list is
        down-sampled uniformly across the scene list.
        """
        processor = VideoProcessor(self.config)
        scenes = processor.detect_scenes(video_path)

        if not scenes:
            logger.warning("No scenes detected — falling back to UNIFORM")
            duration = self._probe_metadata(video_path)[0]
            return self._select_uniform_frames(video_path, duration, target_count)

        # Pick representative timestamp from each scene (midpoint)
        timestamps = [round((s.start + s.end) / 2.0, 3) for s in scenes]

        # Downsample if we have too many scenes
        if len(timestamps) > target_count:
            timestamps = self._evenly_downsample_list(timestamps, target_count)

        return processor.extract_frames(video_path, timestamps)

    def _select_uniform_frames(
        self, video_path: Path, duration: float, target_count: int
    ) -> list[Frame]:
        """Sample *target_count* frames evenly across the video duration."""
        if duration <= 0:
            logger.warning("Invalid duration (%.2f) — using single frame", duration)
            timestamps = [0.0]
        elif target_count <= 1:
            timestamps = [duration / 2.0]
        else:
            step = duration / target_count
            timestamps = [round(i * step + step / 2, 3) for i in range(target_count)]
            # Clamp last timestamp to avoid overrunning
            timestamps[-1] = min(timestamps[-1], max(0.0, duration - 0.1))

        processor = VideoProcessor(self.config)
        return processor.extract_frames(video_path, timestamps)

    def _select_keyframe_frames(
        self, video_path: Path, target_count: int
    ) -> list[Frame]:
        """Extract FFmpeg keyframe (I-frame) timestamps, then sample."""
        timestamps = self._probe_keyframes(video_path)
        if not timestamps:
            logger.warning("No keyframes found — falling back to UNIFORM")
            duration = self._probe_metadata(video_path)[0]
            return self._select_uniform_frames(video_path, duration, target_count)

        logger.info("Found %d keyframes", len(timestamps))
        if len(timestamps) > target_count:
            timestamps = self._evenly_downsample_list(timestamps, target_count)

        processor = VideoProcessor(self.config)
        return processor.extract_frames(video_path, timestamps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _probe_metadata(self, video_path: Path) -> tuple[float, float, tuple[int, int]]:
        """Delegate to :class:`VideoProcessor` for metadata probing."""
        processor = VideoProcessor(self.config)
        return processor._probe_metadata(video_path)

    def _probe_keyframes(self, video_path: Path) -> list[float]:
        """Return a list of keyframe timestamps using ffprobe.

        Returns an empty list on any error.
        """
        import shutil

        if shutil.which("ffprobe") is None:
            return []

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_pts_time,pict_type",
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
            logger.debug("Keyframe probing failed: %s", result.stderr.strip())
            return []

        try:
            import json

            data = json.loads(result.stdout)
            timestamps: list[float] = []
            for frame in data.get("frames", []):
                if frame.get("pict_type") == "I":
                    ts_raw = frame.get("pkt_pts_time") or frame.get("pkt_dts_time")
                    if ts_raw is not None:
                        timestamps.append(float(ts_raw))
            return timestamps
        except Exception as exc:
            logger.debug("Keyframe parsing error: %s", exc)
            return []

    @staticmethod
    def _evenly_downsample(frames: list[Frame], target: int) -> list[Frame]:
        """Return *target* frames spread uniformly across *frames*."""
        if len(frames) <= target:
            return frames
        step = len(frames) / target
        return [frames[int(i * step)] for i in range(target)]

    @staticmethod
    def _evenly_downsample_list(items: list, target: int) -> list:
        """Return *target* items spread uniformly across *items*."""
        if len(items) <= target:
            return items
        step = len(items) / target
        return [items[int(i * step)] for i in range(target)]


__all__ = [
    "FrameSelector",
    "SelectionStrategy",
    "Frame",
]
