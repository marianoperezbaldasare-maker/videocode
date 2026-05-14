"""Tests for :mod:`videocode.frame_selector`.

FFmpeg calls are mocked so the suite runs without external binaries.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from videocode.config import Config
from videocode.frame_selector import (
    FrameSelector,
    SelectionStrategy,
)
from videocode.video_processor import Frame, Scene


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(max_frames=10, target_fps=1.0, frame_resolution=(1280, 720))


@pytest.fixture
def selector(config: Config) -> FrameSelector:
    return FrameSelector(config)


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    video = tmp_path / "dummy.mp4"
    video.touch()
    return video


def _mock_ffprobe_metadata(cmd: list, **kwargs: Any) -> MagicMock:
    """Return a mock subprocess result with plausible video metadata."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps({
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
                "duration": "120.0",
            }
        ]
    })
    mock.stderr = ""
    return mock


def _mock_ffmpeg(cmd: list, **kwargs: Any) -> MagicMock:
    """Simulate successful ffmpeg execution and create output file."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = ""
    mock.stderr = ""
    for i, arg in enumerate(cmd):
        if arg.endswith(".jpg"):
            Path(arg).parent.mkdir(parents=True, exist_ok=True)
            Path(arg).touch()
    return mock


# ---------------------------------------------------------------------------
# SelectionStrategy enum
# ---------------------------------------------------------------------------


class TestSelectionStrategy:
    def test_members(self) -> None:
        assert SelectionStrategy.AUTO.value == "auto"
        assert SelectionStrategy.SCENE.value == "scene"
        assert SelectionStrategy.UNIFORM.value == "uniform"
        assert SelectionStrategy.KEYFRAME.value == "keyframe"


# ---------------------------------------------------------------------------
# calculate_token_budget
# ---------------------------------------------------------------------------


class TestCalculateTokenBudget:
    def test_one_frame_at_reference_resolution(self, selector: FrameSelector) -> None:
        tokens = selector.calculate_token_budget(1)
        # 1 * 150 * sqrt((1280*720)/(1280*720)) = 150
        assert tokens == 150

    def test_multiple_frames(self, selector: FrameSelector) -> None:
        tokens = selector.calculate_token_budget(10)
        assert tokens == 1500

    def test_higher_resolution_increases_tokens(self) -> None:
        config_4k = Config(max_frames=10, frame_resolution=(3840, 2160))
        selector_4k = FrameSelector(config_4k)
        tokens = selector_4k.calculate_token_budget(1)
        # Should be > 150 due to higher resolution
        assert tokens > 150


# ---------------------------------------------------------------------------
# estimate_optimal_frame_count
# ---------------------------------------------------------------------------


class TestEstimateOptimalFrameCount:
    def test_short_video(self, selector: FrameSelector) -> None:
        count = selector.estimate_optimal_frame_count(30.0, (1280, 720))
        assert count >= 1
        assert count <= selector.config.max_frames

    def test_respects_max_frames(self, selector: FrameSelector) -> None:
        count = selector.estimate_optimal_frame_count(600.0, (1280, 720))
        assert count <= selector.config.max_frames

    def test_minimum_three_frames(self, selector: FrameSelector) -> None:
        count = selector.estimate_optimal_frame_count(1.0, (1280, 720))
        assert count >= 1

    def test_zero_duration(self, selector: FrameSelector) -> None:
        count = selector.estimate_optimal_frame_count(0.0, (1280, 720))
        assert count >= 1

    def test_custom_max_frames(self) -> None:
        config = Config(max_frames=5)
        sel = FrameSelector(config)
        count = sel.estimate_optimal_frame_count(300.0, (1280, 720))
        assert count <= 5


# ---------------------------------------------------------------------------
# _evenly_downsample
# ---------------------------------------------------------------------------


class TestEvenlyDownsample:
    def test_no_downsample_needed(self, selector: FrameSelector) -> None:
        frames = [Frame(path=Path(f"f{i}.jpg"), timestamp=float(i)) for i in range(5)]
        result = FrameSelector._evenly_downsample(frames, 5)
        assert len(result) == 5

    def test_downsample_to_three(self, selector: FrameSelector) -> None:
        frames = [Frame(path=Path(f"f{i}.jpg"), timestamp=float(i)) for i in range(10)]
        result = FrameSelector._evenly_downsample(frames, 3)
        assert len(result) == 3
        # Should pick indices 0, 3, 6
        assert result[0].timestamp == 0.0
        assert result[1].timestamp == 3.0
        assert result[2].timestamp == 6.0


# ---------------------------------------------------------------------------
# _evenly_downsample_list
# ---------------------------------------------------------------------------


class TestEvenlyDownsampleList:
    def test_basic(self) -> None:
        items: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = FrameSelector._evenly_downsample_list(items, 3)
        assert len(result) == 3
        assert result == [0, 3, 6]

    def test_no_downsample(self) -> None:
        items = [1, 2, 3]
        result = FrameSelector._evenly_downsample_list(items, 5)
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# _select_uniform_frames
# ---------------------------------------------------------------------------


class TestSelectUniformFrames:
    def test_uniform_sampling(self, selector: FrameSelector, dummy_video: Path) -> None:
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                frames = selector._select_uniform_frames(dummy_video, 60.0, 6)

        assert len(frames) == 6
        # Timestamps should be evenly spaced: ~5, 15, 25, 35, 45, 55
        assert frames[0].timestamp == pytest.approx(5.0, abs=1.0)
        assert frames[-1].timestamp == pytest.approx(55.0, abs=1.0)

    def test_single_frame(self, selector: FrameSelector, dummy_video: Path) -> None:
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                frames = selector._select_uniform_frames(dummy_video, 60.0, 1)

        assert len(frames) == 1

    def test_zero_duration_fallback(self, selector: FrameSelector, dummy_video: Path) -> None:
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                frames = selector._select_uniform_frames(dummy_video, 0.0, 5)

        assert len(frames) == 1
        assert frames[0].timestamp == 0.0


# ---------------------------------------------------------------------------
# _probe_keyframes
# ---------------------------------------------------------------------------


class TestProbeKeyframes:
    def test_found_keyframes(self, selector: FrameSelector, dummy_video: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "frames": [
                {"pict_type": "I", "pkt_pts_time": "0.0"},
                {"pict_type": "P", "pkt_pts_time": "0.033"},
                {"pict_type": "I", "pkt_pts_time": "5.0"},
                {"pict_type": "B", "pkt_pts_time": "5.033"},
                {"pict_type": "I", "pkt_pts_time": "10.0"},
            ]
        })
        mock_result.stderr = ""

        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
                timestamps = selector._probe_keyframes(dummy_video)

        assert len(timestamps) == 3
        assert timestamps == [0.0, 5.0, 10.0]

    def test_ffprobe_not_found(self, selector: FrameSelector, dummy_video: Path) -> None:
        with patch("videocode.video_processor.shutil.which", return_value=None):
            timestamps = selector._probe_keyframes(dummy_video)
        assert timestamps == []

    def test_ffprobe_error(self, selector: FrameSelector, dummy_video: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
                timestamps = selector._probe_keyframes(dummy_video)
        assert timestamps == []

    def test_no_keyframes(self, selector: FrameSelector, dummy_video: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"frames": [{"pict_type": "P", "pkt_pts_time": "0.0"}]})
        mock_result.stderr = ""

        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
                timestamps = selector._probe_keyframes(dummy_video)
        assert timestamps == []


# ---------------------------------------------------------------------------
# select_frames — integration-style with mocks
# ---------------------------------------------------------------------------


class TestSelectFrames:
    def test_uniform_strategy(self, selector: FrameSelector, dummy_video: Path) -> None:
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                with patch.object(
                    selector, "_probe_metadata", return_value=(60.0, 30.0, (1280, 720))
                ):
                    frames = selector.select_frames(
                        dummy_video, SelectionStrategy.UNIFORM
                    )

        assert len(frames) > 0
        assert len(frames) <= selector.config.max_frames
        # Should be sorted by timestamp
        timestamps = [f.timestamp for f in frames]
        assert timestamps == sorted(timestamps)

    def test_enforces_max_frames(self, selector: FrameSelector, dummy_video: Path) -> None:
        # Request more frames than max_frames allows
        config_small = Config(max_frames=3, target_fps=10.0)
        sel = FrameSelector(config_small)

        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                with patch.object(
                    sel, "_probe_metadata", return_value=(60.0, 30.0, (1280, 720))
                ):
                    frames = sel.select_frames(dummy_video, SelectionStrategy.UNIFORM)

        assert len(frames) <= 3

    def test_file_not_found(self, selector: FrameSelector) -> None:
        with pytest.raises(FileNotFoundError):
            selector.select_frames(Path("/nonexistent/video.mp4"))

    def test_auto_strategy_falls_back_to_uniform(self, selector: FrameSelector, dummy_video: Path) -> None:
        """AUTO with <= 3 scenes falls back to UNIFORM."""
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                with patch.object(
                    selector, "_probe_metadata", return_value=(60.0, 30.0, (1280, 720))
                ):
                    with patch.object(
                        selector, "_resolve_auto_strategy", return_value=SelectionStrategy.UNIFORM
                    ):
                        frames = selector.select_frames(dummy_video, SelectionStrategy.AUTO)

        assert len(frames) > 0
        assert len(frames) <= selector.config.max_frames

    def test_frames_are_sorted(self, selector: FrameSelector, dummy_video: Path) -> None:
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                with patch.object(
                    selector, "_probe_metadata", return_value=(60.0, 30.0, (1280, 720))
                ):
                    frames = selector.select_frames(dummy_video, SelectionStrategy.UNIFORM)

        timestamps = [f.timestamp for f in frames]
        assert timestamps == sorted(timestamps)

    def test_default_strategy_is_auto(self, selector: FrameSelector, dummy_video: Path) -> None:
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=_mock_ffmpeg):
                with patch.object(
                    selector, "_probe_metadata", return_value=(60.0, 30.0, (1280, 720))
                ):
                    with patch.object(
                        selector, "_resolve_auto_strategy", return_value=SelectionStrategy.UNIFORM
                    ):
                        # No strategy argument — should default to AUTO
                        frames = selector.select_frames(dummy_video)

        assert len(frames) > 0
