"""Tests for the tutorial-to-code extractor pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from videocode.code_extractor import (
    CodeExtractor,
    _extract_code_blocks,
    _extract_json_block,
    _guess_language_from_code,
)
from videocode.types import (
    CodeBlock,
    CodeFrame,
    CodeResult,
    Config,
    Frame,
    ProcessedVideo,
    Transcription,
    VLMResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(vlm_backend="ollama", vlm_model="llava")


@pytest.fixture
def mock_vlm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_agent(config: Config, mock_vlm: MagicMock) -> MagicMock:
    """Create a mock AgentLoop that exposes the VLM."""
    from videocode.agent_loop import AgentLoop

    agent = AgentLoop(mock_vlm, config)
    return agent


@pytest.fixture
def frames(tmp_path: Path) -> List[Frame]:
    return [
        Frame(path=tmp_path / f"frame_{i:03d}.png", timestamp=float(i), scene_index=0)
        for i in range(5)
    ]


@pytest.fixture
def video(frames: List[Frame], tmp_path: Path) -> ProcessedVideo:
    return ProcessedVideo(
        source="tutorial.mp4",
        duration=50.0,
        fps=1.0,
        resolution=(1920, 1080),
        frames=frames,
        frame_dir=tmp_path,
    )


@pytest.fixture
def transcription() -> Transcription:
    return Transcription(
        text="Today we build a todo app in Python with Flask.",
        segments=[],
        language="en",
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExtractCodeBlocks:
    def test_single_block(self) -> None:
        text = "```python\nprint('hello')\n```"
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert "print('hello')" in blocks[0]

    def test_multiple_blocks(self) -> None:
        text = (
            "```python\nprint('a')\n```\n\n"
            "Some text\n\n"
            "```javascript\nconsole.log('b');\n```"
        )
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 2
        assert "print('a')" in blocks[0]
        assert "console.log" in blocks[1]

    def test_no_blocks(self) -> None:
        assert _extract_code_blocks("no code here") == []

    def test_with_language_tag(self) -> None:
        text = "```rust\nfn main() {}\n```"
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert "fn main" in blocks[0]


class TestExtractJsonBlock:
    def test_json_in_markdown(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json_block(text) == '{"key": "value"}'

    def test_json_no_tag(self) -> None:
        text = '```\n{"key": "value"}\n```'
        assert _extract_json_block(text) == '{"key": "value"}'

    def test_bare_json(self) -> None:
        text = 'prefix {"key": "value"} suffix'
        assert _extract_json_block(text) == '{"key": "value"}'


class TestGuessLanguageFromCode:
    def test_python(self) -> None:
        assert _guess_language_from_code("def hello():\n    pass") == "python"

    def test_javascript(self) -> None:
        assert _guess_language_from_code("const x = 1;") == "javascript"

    def test_typescript(self) -> None:
        assert _guess_language_from_code("const x: number = 1;") == "typescript"

    def test_rust(self) -> None:
        assert _guess_language_from_code("fn main() {}") == "rust"

    def test_go(self) -> None:
        assert _guess_language_from_code("package main") == "go"

    def test_java(self) -> None:
        assert _guess_language_from_code("public class Main {}") == "java"

    def test_cpp(self) -> None:
        assert _guess_language_from_code("#include <iostream>") == "cpp"

    def test_html(self) -> None:
        assert _guess_language_from_code("<!DOCTYPE html>") == "html"

    def test_unknown(self) -> None:
        assert _guess_language_from_code("random text without code patterns") == ""


# ---------------------------------------------------------------------------
# CodeExtractor — detect_code_frames
# ---------------------------------------------------------------------------


class TestDetectCodeFrames:
    def test_detects_code_frames(self, mock_agent: MagicMock, frames: List[Frame], config: Config) -> None:
        extractor = CodeExtractor(mock_agent, config)
        mock_agent.vlm.analyze_frames.return_value = VLMResponse(
            content='[{"index": 0, "language": "python", "has_code": true}, {"index": 2, "language": "python", "has_code": true}]'
        )

        result = extractor.detect_code_frames(frames)
        assert len(result) == 2
        assert result[0].frame.timestamp == 0.0
        assert result[0].language == "python"
        assert result[1].frame.timestamp == 2.0

    def test_fallback_on_invalid_json(self, mock_agent: MagicMock, frames: List[Frame], config: Config) -> None:
        extractor = CodeExtractor(mock_agent, config)
        mock_agent.vlm.analyze_frames.return_value = VLMResponse(content="not json")

        result = extractor.detect_code_frames(frames)
        assert len(result) == len(frames)

    def test_empty_frames(self, mock_agent: MagicMock, config: Config) -> None:
        extractor = CodeExtractor(mock_agent, config)
        result = extractor.detect_code_frames([])
        assert result == []


# ---------------------------------------------------------------------------
# CodeExtractor — extract_code_from_frame
# ---------------------------------------------------------------------------


class TestExtractCodeFromFrame:
    def test_extracts_code(self, mock_agent: MagicMock, frames: List[Frame], config: Config) -> None:
        extractor = CodeExtractor(mock_agent, config)
        mock_agent.vlm.analyze_single.return_value = (
            "```python\nprint('hello')\n```"
        )

        result = extractor.extract_code_from_frame(frames[0])
        assert "print('hello')" in result
        mock_agent.vlm.analyze_single.assert_called_once()

    def test_no_code_found(self, mock_agent: MagicMock, frames: List[Frame], config: Config) -> None:
        extractor = CodeExtractor(mock_agent, config)
        mock_agent.vlm.analyze_single.return_value = "NO_CODE_FOUND"

        result = extractor.extract_code_from_frame(frames[0])
        assert result == "NO_CODE_FOUND"


# ---------------------------------------------------------------------------
# CodeExtractor — assemble_project
# ---------------------------------------------------------------------------


class TestAssembleProject:
    def test_successful_assembly(self, mock_agent: MagicMock, config: Config) -> None:
        extractor = CodeExtractor(mock_agent, config)
        mock_agent.vlm.analyze_frames.return_value = VLMResponse(
            content='```json\n{"files": {"main.py": "print('"'"'hello'"'"')", "README.md": "# Project"}, '
            '"language": "python", "description": "A hello app", '
            '"setup_instructions": "pip install -r requirements\\npython main.py", '
            '"dependencies": ["flask"]}\n```'
        )

        blocks = [CodeBlock(filename="", content="print('hello')", language="python")]
        result = extractor.assemble_project(blocks, "some context")

        assert "main.py" in result.files
        assert result.language == "python"
        assert result.description == "A hello app"
        assert result.confidence > 0

    def test_fallback_on_bad_json(self, mock_agent: MagicMock, config: Config) -> None:
        extractor = CodeExtractor(mocklm := mock_agent, config)
        mocklm.vlm.analyze_frames.return_value = VLMResponse(content="not valid json")

        blocks = [
            CodeBlock(filename="", content="print('a')", language="python"),
            CodeBlock(filename="", content="print('b')", language="python"),
        ]
        result = extractor.assemble_project(blocks, "")
        assert len(result.files) == 2
        assert result.confidence == 0.3

    def test_empty_blocks(self, mock_agent: MagicMock, config: Config) -> None:
        extractor = CodeExtractor(mock_agent, config)
        result = extractor.assemble_project([], "context")
        assert result.files == {}
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# CodeExtractor — full pipeline (extract)
# ---------------------------------------------------------------------------


class TestExtractPipeline:
    def test_full_pipeline(self, mock_agent: MagicMock, video: ProcessedVideo, transcription: Transcription, config: Config) -> None:
        """End-to-end pipeline with mocked VLM responses."""
        extractor = CodeExtractor(mock_agent, config)

        # Set up side effects for the three VLM calls:
        # 1. detect_code_frames -> frame selection
        # 2. extract_code_from_frame (called for each code frame)
        # 3. assemble_project -> project structure
        def analyze_frames_side_effect(frames, prompt, transcription=None):
            if "identify which frames" in prompt.lower() or "code frame" in prompt.lower():
                return VLMResponse(
                    content='[{"index": 0, "language": "python", "has_code": true, "confidence": 0.9}, '
                            '{"index": 1, "language": "python", "has_code": true, "confidence": 0.8}]'
                )
            elif "organize them into a coherent project" in prompt.lower() or "software architect" in prompt.lower():
                return VLMResponse(
                    content='```json\n{"files": {"app.py": "from flask import Flask", "README.md": "# Flask App"}, '
                            '"language": "python", "description": "Flask todo app", '
                            '"setup_instructions": "pip install flask\\npython app.py"}\n```'
                )
            return VLMResponse(content="")

        def analyze_single_side_effect(frame, prompt):
            return "```python\nfrom flask import Flask\napp = Flask(__name__)\n```"

        mock_agent.vlm.analyze_frames.side_effect = analyze_frames_side_effect
        mock_agent.vlm.analyze_single.side_effect = analyze_single_side_effect

        result = extractor.extract(video, transcription)

        assert result.language == "python"
        assert "app.py" in result.files
        assert result.confidence > 0
        assert result.setup_instructions

    def test_no_code_frames(self, mock_agent: MagicMock, video: ProcessedVideo, transcription: Transcription, config: Config) -> None:
        """Pipeline returns empty result when no code frames are found."""
        extractor = CodeExtractor(mock_agent, config)
        mock_agent.vlm.analyze_frames.return_value = VLMResponse(content='[]')

        result = extractor.extract(video, transcription)
        assert result.files == {}
        assert result.confidence == 0.0
        assert "No code frames" in result.description

    def test_code_frames_but_no_extraction(self, mock_agent: MagicMock, video: ProcessedVideo, transcription: Transcription, config: Config) -> None:
        """When frames have code but extraction returns nothing."""
        extractor = CodeExtractor(mock_agent, config)
        mock_agent.vlm.analyze_frames.return_value = VLMResponse(
            content='[{"index": 0, "language": "python", "has_code": true}]'
        )
        mock_agent.vlm.analyze_single.return_value = "NO_CODE_FOUND"

        result = extractor.extract(video, transcription)
        assert result.confidence == 0.0
        assert "no code could be extracted" in result.description.lower() or result.files == {}
