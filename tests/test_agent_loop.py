"""Tests for the 3-role agent loop."""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from videocode.agent_loop import (
    AgentLoop,
    Analyzer,
    Extractor,
    Verifier,
    _extract_json,
)
from videocode.types import (
    AgentResult,
    Config,
    Frame,
    ProcessedVideo,
    SourceReference,
    Task,
    TaskType,
    Transcription,
    VLMResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(vlm_backend="ollama", vlm_model="llava", max_retries=2)


@pytest.fixture
def mock_vlm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def frames(tmp_path: Path) -> List[Frame]:
    return [
        Frame(path=tmp_path / f"frame_{i:03d}.png", timestamp=float(i), scene_index=0)
        for i in range(5)
    ]


@pytest.fixture
def video(frames: List[Frame], tmp_path: Path) -> ProcessedVideo:
    return ProcessedVideo(
        source="test.mp4",
        duration=10.0,
        fps=1.0,
        resolution=(1280, 720),
        frames=frames,
        frame_dir=tmp_path,
    )


@pytest.fixture
def transcription() -> Transcription:
    return Transcription(
        text="In this video we write a hello world program.",
        segments=[],
        language="en",
    )


# ---------------------------------------------------------------------------
# _extract_json helper
# ---------------------------------------------------------------------------


def test_extract_json_from_markdown() -> None:
    text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
    result = _extract_json(text)
    assert result == '{"key": "value"}'


def test_extract_json_bare() -> None:
    text = 'prefix {"key": "value"} suffix'
    result = _extract_json(text)
    assert result == '{"key": "value"}'


def test_extract_json_no_json() -> None:
    text = "just plain text"
    result = _extract_json(text)
    assert result == "just plain text"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class TestExtractor:
    def test_run_selects_relevant_frames(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        extractor = Extractor(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(
            content='{"relevant_frames": [{"index": 0, "reason": "shows code"}, {"index": 2, "reason": "shows terminal"}]}'
        )

        task = Task(type=TaskType.CODE_EXTRACTION, query="extract code")
        result = extractor.run(task, video)

        assert len(result) == 2
        assert result[0].timestamp == 0.0
        assert result[1].timestamp == 2.0

    def test_run_empty_frames(self, mock_vlm: MagicMock, config: Config, tmp_path: Path) -> None:
        extractor = Extractor(mock_vlm, config)
        empty_video = ProcessedVideo(
            source="empty.mp4",
            duration=0.0,
            fps=0.0,
            resolution=(0, 0),
            frames=[],
            frame_dir=tmp_path,
        )
        result = extractor.run(Task(type=TaskType.CODE_EXTRACTION, query=""), empty_video)
        assert result == []

    def test_run_fallback_to_all_frames_on_bad_json(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        extractor = Extractor(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(content="not json at all")

        task = Task(type=TaskType.CODE_EXTRACTION, query="")
        result = extractor.run(task, video)
        assert len(result) == len(video.frames)

    def test_run_with_general_task_query(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        extractor = Extractor(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(
            content='{"relevant_frames": [{"index": 1, "reason": "relevant"}]}'
        )

        task = Task(type=TaskType.GENERAL, query="What is the main topic?")
        result = extractor.run(task, video)
        assert len(result) == 1
        assert result[0].timestamp == 1.0


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class TestAnalyzer:
    def test_run_code_extraction(self, mock_vlm: MagicMock, config: Config, frames: List[Frame], transcription: Transcription) -> None:
        analyzer = Analyzer(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(content="Extracted code: ```python\nprint('hello')\n```")

        task = Task(type=TaskType.CODE_EXTRACTION, query="")
        result = analyzer.run(task, frames, transcription)
        assert "Extracted code" in result

    def test_run_summarization(self, mock_vlm: MagicMock, config: Config, frames: List[Frame], transcription: Transcription) -> None:
        analyzer = Analyzer(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(content="## Summary\n\nThis video covers...")

        task = Task(type=TaskType.SUMMARIZATION, query="")
        result = analyzer.run(task, frames, transcription)
        assert "Summary" in result

    def test_run_bug_finding(self, mock_vlm: MagicMock, config: Config, frames: List[Frame], transcription: Transcription) -> None:
        analyzer = Analyzer(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(content="## Bug 1\n\nCritical error at 5s")

        task = Task(type=TaskType.BUG_FINDING, query="")
        result = analyzer.run(task, frames, transcription)
        assert "Bug" in result

    def test_run_general_with_query(self, mock_vlm: MagicMock, config: Config, frames: List[Frame], transcription: Transcription) -> None:
        analyzer = Analyzer(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(content="The answer is 42.")

        task = Task(type=TaskType.GENERAL, query="What is shown?")
        result = analyzer.run(task, frames, transcription)
        assert "42" in result

    def test_run_without_transcription(self, mock_vlm: MagicMock, config: Config, frames: List[Frame]) -> None:
        analyzer = Analyzer(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(content="No transcript analysis")

        task = Task(type=TaskType.SUMMARIZATION, query="")
        result = analyzer.run(task, frames, None)
        assert "No transcript" in result


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TestVerifier:
    def test_high_confidence(self, mock_vlm: MagicMock, config: Config) -> None:
        verifier = Verifier(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(
            content='{"confidence": 0.92, "issues": [], "improved_analysis": "Great analysis."}'
        )

        analysis, confidence = verifier.run(Task(type=TaskType.CODE_EXTRACTION, query=""), "raw analysis")
        assert confidence == 0.92
        assert analysis == "Great analysis."

    def test_low_confidence(self, mock_vlm: MagicMock, config: Config) -> None:
        verifier = Verifier(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(
            content='{"confidence": 0.45, "issues": ["missing imports"], "improved_analysis": "Better analysis with imports."}'
        )

        analysis, confidence = verifier.run(Task(type=TaskType.CODE_EXTRACTION, query=""), "raw")
        assert confidence == 0.45
        assert "Better analysis" in analysis

    def test_invalid_json_fallback(self, mock_vlm: MagicMock, config: Config) -> None:
        verifier = Verifier(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(content="not json")

        analysis, confidence = verifier.run(Task(type=TaskType.CODE_EXTRACTION, query=""), "fallback text")
        assert confidence == 0.5
        assert analysis == "fallback text"

    def test_clamped_confidence(self, mock_vlm: MagicMock, config: Config) -> None:
        verifier = Verifier(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(
            content='{"confidence": 1.5, "issues": [], "improved_analysis": "ok"}'
        )

        _, confidence = verifier.run(Task(type=TaskType.CODE_EXTRACTION, query=""), "raw")
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class TestAgentLoop:
    def test_successful_run_no_retries(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo, transcription: Transcription) -> None:
        """Happy path: extractor selects frames, analyzer produces result, verifier passes."""
        loop = AgentLoop(mock_vlm, config)
        # Extractor
        mock_vlm.analyze_frames.side_effect = [
            VLMResponse(content='{"relevant_frames": [{"index": 0, "reason": "code"}]}'),
            VLMResponse(content="analysis result"),
            VLMResponse(content='{"confidence": 0.9, "issues": [], "improved_analysis": "final result"}'),
        ]

        task = Task(type=TaskType.CODE_EXTRACTION, query="extract code")
        result = loop.run(task, video, transcription)

        assert result.content == "final result"
        assert result.confidence == 0.9
        assert result.retries == 0
        assert len(result.sources) == 1

    def test_run_with_retry(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo, transcription: Transcription) -> None:
        """Verifier triggers one retry before succeeding."""
        loop = AgentLoop(mock_vlm, config)
        mock_vlm.analyze_frames.side_effect = [
            VLMResponse(content='{"relevant_frames": [{"index": 0}]}'),
            VLMResponse(content="weak analysis"),
            VLMResponse(content='{"confidence": 0.5, "issues": ["incomplete"], "improved_analysis": "better"}'),
            VLMResponse(content="better analysis"),
            VLMResponse(content='{"confidence": 0.85, "issues": [], "improved_analysis": "final"}'),
        ]

        result = loop.run(Task(type=TaskType.CODE_EXTRACTION, query=""), video, transcription)
        assert result.confidence == 0.85
        assert result.content == "final"
        assert result.retries == 1

    def test_run_extractor_failure_fallback(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        """If extractor fails, fall back to all frames."""
        loop = AgentLoop(mock_vlm, config)
        mock_vlm.analyze_frames.side_effect = [
            ConnectionError("extractor failed"),
            VLMResponse(content="analysis"),
            VLMResponse(content='{"confidence": 0.8, "issues": [], "improved_analysis": "result"}'),
        ]

        result = loop.run(Task(type=TaskType.SUMMARIZATION, query=""), video)
        assert result.content == "result"
        assert len(result.sources) == len(video.frames)

    def test_run_no_frames_after_extraction(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        """If extractor returns no frames, return empty result."""
        loop = AgentLoop(mock_vlm, config)
        mock_vlm.analyze_frames.return_value = VLMResponse(
            content='{"relevant_frames": []}'
        )

        result = loop.run(Task(type=TaskType.CODE_EXTRACTION, query=""), video)
        assert "No relevant frames" in result.content
        assert result.confidence == 0.0
        assert result.retries == 0

    def test_run_analyzer_failure(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        """If analyzer fails after retries, return error result."""
        loop = AgentLoop(mock_vlm, config)
        mock_vlm.analyze_frames.side_effect = [
            VLMResponse(content='{"relevant_frames": [{"index": 0}]}'),
            ConnectionError("analyzer fail"),
        ]

        result = loop.run(Task(type=TaskType.CODE_EXTRACTION, query=""), video)
        assert "Analysis failed" in result.content
        assert result.confidence == 0.0

    def test_run_verifier_failure_continues(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        """If verifier fails, use the analysis with neutral confidence."""
        loop = AgentLoop(mock_vlm, config)
        mock_vlm.analyze_frames.side_effect = [
            VLMResponse(content='{"relevant_frames": [{"index": 0}]}'),
            VLMResponse(content="good analysis"),
            ConnectionError("verifier failed"),
        ]

        result = loop.run(Task(type=TaskType.CODE_EXTRACTION, query=""), video)
        assert result.content == "good analysis"
        assert result.confidence == 0.5

    def test_run_exhausts_retries(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        """If confidence never reaches threshold, return best effort after max_retries."""
        loop = AgentLoop(mock_vlm, config)
        # Extractor + (analyzer, verifier) * (max_retries + 1)
        mock_vlm.analyze_frames.side_effect = [
            VLMResponse(content='{"relevant_frames": [{"index": 0}]}'),
            # Attempt 1
            VLMResponse(content="v1"),
            VLMResponse(content='{"confidence": 0.3, "issues": ["bad"], "improved_analysis": "v1-fixed"}'),
            # Attempt 2
            VLMResponse(content="v2"),
            VLMResponse(content='{"confidence": 0.4, "issues": ["still bad"], "improved_analysis": "v2-fixed"}'),
            # Attempt 3
            VLMResponse(content="v3"),
            VLMResponse(content='{"confidence": 0.5, "issues": [], "improved_analysis": "v3-fixed"}'),
        ]

        result = loop.run(Task(type=TaskType.CODE_EXTRACTION, query=""), video)
        assert result.confidence == 0.5
        assert result.retries == config.max_retries

    def test_source_references_populated(self, mock_vlm: MagicMock, config: Config, video: ProcessedVideo) -> None:
        """SourceReference objects should reference the correct frames."""
        loop = AgentLoop(mock_vlm, config)
        mock_vlm.analyze_frames.side_effect = [
            VLMResponse(content='{"relevant_frames": [{"index": 0}, {"index": 3}]}'),
            VLMResponse(content="analysis"),
            VLMResponse(content='{"confidence": 0.9, "issues": [], "improved_analysis": "final"}'),
        ]

        result = loop.run(Task(type=TaskType.SUMMARIZATION, query=""), video)
        assert len(result.sources) == 2
        assert result.sources[0].timestamp == 0.0
        assert result.sources[1].timestamp == 3.0
