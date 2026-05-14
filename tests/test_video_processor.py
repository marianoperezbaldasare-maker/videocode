"""Tests for :mod:`videocode.video_processor`.

All FFmpeg and ffprobe calls are mocked so the tests run without
requiring external binaries.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from videocode.config import Config
from videocode.video_processor import (
    FFmpegNotFoundError,
    Frame,
    ProcessedVideo,
    Scene,
    VideoProcessingError,
    VideoProcessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(max_frames=10, target_fps=1.0)


@pytest.fixture
def processor(config: Config) -> VideoProcessor:
    return VideoProcessor(config)


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Create a zero-byte file that stands in for a video."""
    video = tmp_path / "dummy.mp4"
    video.touch()
    return video


# ---------------------------------------------------------------------------
# _ensure_ffmpeg
# ---------------------------------------------------------------------------


class TestEnsureFFmpeg:
    def test_raises_when_ffmpeg_missing(self, processor: VideoProcessor) -> None:
        with patch.object(shutil, "which", return_value=None):
            with pytest.raises(FFmpegNotFoundError):
                processor._ensure_ffmpeg()

    def test_passes_when_ffmpeg_present(self, processor: VideoProcessor) -> None:
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            # Should not raise
            processor._ensure_ffmpeg()


# ---------------------------------------------------------------------------
# _probe_metadata
# ---------------------------------------------------------------------------


class TestProbeMetadata:
    def test_probe_parses_ffprobe_json(self, processor: VideoProcessor, dummy_video: Path) -> None:
        ffprobe_output = json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30000/1001",
                    "duration": "120.5",
                }
            ]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        mock_result.stderr = ""

        with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
            duration, fps, resolution = processor._probe_metadata(dummy_video)

        assert duration == 120.5
        assert pytest.approx(fps, 0.01) == 29.97
        assert resolution == (1920, 1080)

    def test_probe_integer_fps(self, processor: VideoProcessor, dummy_video: Path) -> None:
        ffprobe_output = json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "30",
                    "duration": "60.0",
                }
            ]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        mock_result.stderr = ""

        with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
            duration, fps, resolution = processor._probe_metadata(dummy_video)

        assert fps == 30.0
        assert resolution == (1280, 720)

    def test_probe_fallback_to_format_duration(self, processor: VideoProcessor, dummy_video: Path) -> None:
        """Duration can live in the 'format' section too."""
        ffprobe_output = json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "25/1",
                }
            ],
            "format": {"duration": "45.0"},
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        mock_result.stderr = ""

        with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
            duration, fps, resolution = processor._probe_metadata(dummy_video)

        assert duration == 45.0

    def test_probe_handles_missing_stream(self, processor: VideoProcessor, dummy_video: Path) -> None:
        ffprobe_output = json.dumps({"streams": []})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        mock_result.stderr = ""

        with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
            with pytest.raises(VideoProcessingError, match="No video stream"):
                processor._probe_metadata(dummy_video)

    def test_probe_ffprobe_error(self, processor: VideoProcessor, dummy_video: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "some error"

        with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
            with pytest.raises(VideoProcessingError):
                processor._probe_metadata(dummy_video)

    def test_probe_invalid_json(self, processor: VideoProcessor, dummy_video: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not json"
        mock_result.stderr = ""

        with patch("videocode.video_processor.subprocess.run", return_value=mock_result):
            with pytest.raises(VideoProcessingError):
                processor._probe_metadata(dummy_video)


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------


class TestExtractFrames:
    def test_extracts_expected_number_of_frames(
        self, processor: VideoProcessor, dummy_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "frames"
        timestamps = [0.0, 5.0, 10.0]

        # Each ffmpeg call succeeds and "creates" a file
        def side_effect(cmd: list, **kwargs: Any) -> MagicMock:
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = ""
            mock.stderr = ""
            # Simulate ffmpeg creating the output file
            for i, arg in enumerate(cmd):
                if arg == "-i":
                    pass
                elif arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).touch()
            return mock

        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=side_effect):
                frames = processor.extract_frames(dummy_video, timestamps, output_dir)

        assert len(frames) == 3
        assert all(isinstance(f, Frame) for f in frames)
        assert frames[0].timestamp == 0.0
        assert frames[1].timestamp == 5.0
        assert frames[2].timestamp == 10.0

    def test_skips_failed_extractions(
        self, processor: VideoProcessor, dummy_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "frames"
        timestamps = [0.0, 5.0]
        call_count = [0]

        def side_effect(cmd: list, **kwargs: Any) -> MagicMock:
            mock = MagicMock()
            call_count[0] += 1
            if call_count[0] == 1:
                mock.returncode = 0
                for arg in cmd:
                    if arg.endswith(".jpg"):
                        Path(arg).parent.mkdir(parents=True, exist_ok=True)
                        Path(arg).touch()
            else:
                mock.returncode = 1
                mock.stderr = "codec error"
            mock.stdout = ""
            return mock

        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("videocode.video_processor.subprocess.run", side_effect=side_effect):
                frames = processor.extract_frames(dummy_video, timestamps, output_dir)

        assert len(frames) == 1
        assert frames[0].timestamp == 0.0

    def test_raises_for_missing_video(self, processor: VideoProcessor, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.mp4"
        with patch("videocode.video_processor.shutil.which", return_value="/usr/bin/ffmpeg"):
            with pytest.raises(VideoProcessingError, match="not found"):
                processor.extract_frames(missing, [0.0])

    def test_raises_when_ffmpeg_missing(self, processor: VideoProcessor, dummy_video: Path) -> None:
        with patch.object(shutil, "which", return_value=None):
            with pytest.raises(FFmpegNotFoundError):
                processor.extract_frames(dummy_video, [0.0])


# ---------------------------------------------------------------------------
# detect_scenes (mocked pyscenedetect)
# ---------------------------------------------------------------------------


class TestDetectScenes:
    def test_detect_scenes_success(self, processor: VideoProcessor, dummy_video: Path) -> None:
        """Scene detection delegates to pyscenedetect and returns scenes."""
        mock_frame_start = MagicMock()
        mock_frame_start.get_seconds.return_value = 10.0
        mock_frame_end = MagicMock()
        mock_frame_end.get_seconds.return_value = 25.0

        # Mock the detect function and AdaptiveDetector
        with patch("videocode.video_processor._HAS_SCENEDETECT", True):
            with patch("videocode.video_processor.detect", return_value=[
                (mock_frame_start, mock_frame_end),
            ]) as mock_detect:
                with patch("videocode.video_processor.AdaptiveDetector") as mock_detector_cls:
                    mock_detector_cls.return_value = MagicMock()
                    scenes = processor.detect_scenes(dummy_video)

        assert len(scenes) == 1
        assert scenes[0].start == 10.0
        assert scenes[0].end == 25.0
        assert scenes[0].index == 0

    def test_detect_scenes_import_error(self, processor: VideoProcessor, dummy_video: Path) -> None:
        """When pyscenedetect is not installed, return an empty list."""
        with patch.dict("sys.modules", {"scenedetect": None}):
            with patch(
                "videocode.video_processor.AdaptiveDetector",
                side_effect=ImportError("No module named 'scenedetect'"),
            ):
                scenes = processor.detect_scenes(dummy_video)
        assert scenes == []

    def test_detect_scenes_file_not_found(self, processor: VideoProcessor, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.mp4"
        with pytest.raises(VideoProcessingError, match="not found"):
            processor.detect_scenes(missing)


# ---------------------------------------------------------------------------
# _resolve_source
# ---------------------------------------------------------------------------


class TestResolveSource:
    def test_local_file(self, processor: VideoProcessor, dummy_video: Path) -> None:
        resolved = processor._resolve_source(str(dummy_video))
        assert resolved == dummy_video

    def test_missing_local_file(self, processor: VideoProcessor) -> None:
        with pytest.raises(VideoProcessingError, match="not found"):
            processor._resolve_source("/nonexistent/video.mp4")

    def test_url_triggers_download(self, processor: VideoProcessor, tmp_path: Path) -> None:
        with patch.object(processor, "_download_video", return_value=tmp_path / "down.mp4") as mock_dl:
            resolved = processor._resolve_source("https://youtube.com/watch?v=abc")
            mock_dl.assert_called_once_with("https://youtube.com/watch?v=abc")


# ---------------------------------------------------------------------------
# _download_video (mocked yt_dlp)
# ---------------------------------------------------------------------------


class TestDownloadVideo:
    def test_download_success(self, processor: VideoProcessor, tmp_path: Path) -> None:
        mock_info = {"title": "Test Video", "ext": "mp4"}
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl.prepare_filename.return_value = str(tmp_path / "Test Video.mp4")

        mock_youtube_dl_cls = MagicMock()
        mock_youtube_dl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_youtube_dl_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_yt_dlp_module = MagicMock()
        mock_yt_dlp_module.YoutubeDL = mock_youtube_dl_cls

        with patch("videocode.video_processor.yt_dlp", mock_yt_dlp_module):
            with patch("videocode.video_processor._HAS_YT_DLP", True):
                path = processor._download_video("https://youtube.com/watch?v=abc123")
                assert path.name == "Test Video.mp4"

    def test_download_yt_dlp_not_installed(self, processor: VideoProcessor) -> None:
        from videocode.video_processor import VideoDownloadError

        with patch("videocode.video_processor._HAS_YT_DLP", False):
            with pytest.raises(VideoDownloadError, match="No video downloader"):
                processor._download_video("https://youtube.com/watch?v=abc")


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_removes_temp_dir(self, processor: VideoProcessor, tmp_path: Path) -> None:
        temp = tmp_path / "videocode-test"
        processor.temp_dir = temp
        temp.mkdir()
        (temp / "dummy.txt").touch()

        processor.cleanup()
        assert not temp.exists()
        assert processor.temp_dir is None

    def test_cleanup_without_temp_dir(self, processor: VideoProcessor) -> None:
        # Should be a no-op
        processor.temp_dir = None
        processor.cleanup()
        assert processor.temp_dir is None

    def test_cleanup_idempotent(self, processor: VideoProcessor, tmp_path: Path) -> None:
        processor.temp_dir = tmp_path / "videocode-test2"
        processor.temp_dir.mkdir()

        processor.cleanup()
        processor.cleanup()  # second call should not raise
        assert processor.temp_dir is None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_frame_defaults(self, tmp_path: Path) -> None:
        frame = Frame(path=tmp_path / "frame.jpg", timestamp=5.0)
        assert frame.scene_index == -1
        assert frame.is_keyframe is False

    def test_scene_creation(self) -> None:
        scene = Scene(start=0.0, end=10.0, index=0)
        assert scene.start == 0.0
        assert scene.end == 10.0
        assert scene.index == 0

    def test_processed_video_defaults(self, tmp_path: Path) -> None:
        pv = ProcessedVideo(
            source="test.mp4",
            duration=60.0,
            fps=30.0,
            resolution=(1280, 720),
            frames=[],
            frame_dir=tmp_path,
        )
        assert pv.audio_path is None
        assert pv.scenes == []
