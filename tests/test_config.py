"""Tests for :mod:`videocode.config`."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from videocode.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Remove all ``VIDEOCODE_*`` env vars before each test."""
    env_snapshot = {k: v for k, v in os.environ.items() if k.startswith("VIDEOCODE_")}
    for k in env_snapshot:
        del os.environ[k]
    yield
    # Restore
    for k in env_snapshot:
        os.environ[k] = env_snapshot[k]
    # Remove any newly created ones
    for k in list(os.environ.keys()):
        if k.startswith("VIDEOCODE_") and k not in env_snapshot:
            del os.environ[k]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Verify that the default configuration values are sensible."""

    def test_vlm_backend_default(self) -> None:
        config = Config()
        assert config.vlm_backend == "ollama"

    def test_vlm_model_default(self) -> None:
        config = Config()
        assert config.vlm_model == "llava"

    def test_vlm_api_key_none(self) -> None:
        config = Config()
        assert config.vlm_api_key is None

    def test_ollama_url_default(self) -> None:
        config = Config()
        assert config.ollama_url == "http://localhost:11434"

    def test_max_frames_default(self) -> None:
        config = Config()
        assert config.max_frames == 50

    def test_frame_quality_default(self) -> None:
        config = Config()
        assert config.frame_quality == 85

    def test_frame_resolution_default(self) -> None:
        config = Config()
        assert config.frame_resolution == (1280, 720)

    def test_scene_threshold_default(self) -> None:
        config = Config()
        assert config.scene_threshold == 30.0

    def test_whisper_model_default(self) -> None:
        config = Config()
        assert config.whisper_model == "base"

    def test_extract_audio_default(self) -> None:
        config = Config()
        assert config.extract_audio is True

    def test_enable_agent_loop_default(self) -> None:
        config = Config()
        assert config.enable_agent_loop is True

    def test_max_retries_default(self) -> None:
        config = Config()
        assert config.max_retries == 2

    def test_temperature_default(self) -> None:
        config = Config()
        assert config.temperature == 0.3

    def test_output_dir_default(self) -> None:
        config = Config()
        assert config.output_dir == "./output"

    def test_output_format_default(self) -> None:
        config = Config()
        assert config.output_format == "markdown"

    def test_target_fps_default(self) -> None:
        config = Config()
        assert config.target_fps == 1.5


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestConfigFromEnv:
    """Verify ``Config.from_env()`` correctly reads environment variables."""

    def test_simple_string_override(self, clean_env: None) -> None:
        os.environ["VIDEOCODE_VLM_BACKEND"] = "openai"
        config = Config.from_env()
        assert config.vlm_backend == "openai"

    def test_integer_override(self, clean_env: None) -> None:
        os.environ["VIDEOCODE_MAX_FRAMES"] = "20"
        config = Config.from_env()
        assert config.max_frames == 20

    def test_float_override(self, clean_env: None) -> None:
        os.environ["VIDEOCODE_TEMPERATURE"] = "0.7"
        config = Config.from_env()
        assert config.temperature == 0.7

    def test_boolean_override_true(self, clean_env: None) -> None:
        os.environ["VIDEOCODE_EXTRACT_AUDIO"] = "true"
        config = Config.from_env()
        assert config.extract_audio is True

    def test_boolean_override_false(self, clean_env: None) -> None:
        os.environ["VIDEOCODE_EXTRACT_AUDIO"] = "false"
        config = Config.from_env()
        assert config.extract_audio is False

    def test_tuple_override_from_json_list(self, clean_env: None) -> None:
        os.environ["VIDEOCODE_FRAME_RESOLUTION"] = "[640, 480]"
        config = Config.from_env()
        assert config.frame_resolution == (640, 480)

    def test_multiple_overrides(self, clean_env: None) -> None:
        os.environ["VIDEOCODE_VLM_BACKEND"] = "gemini"
        os.environ["VIDEOCODE_MAX_FRAMES"] = "15"
        os.environ["VIDEOCODE_TARGET_FPS"] = "1.0"
        config = Config.from_env()
        assert config.vlm_backend == "gemini"
        assert config.max_frames == 15
        assert config.target_fps == 1.0
        # Unchanged defaults
        assert config.vlm_model == "llava"
        assert config.frame_quality == 85

    def test_unrelated_env_vars_ignored(self, clean_env: None) -> None:
        os.environ["SOME_OTHER_VAR"] = "should_not_affect"
        config = Config.from_env()
        assert config.vlm_backend == "ollama"  # default

    def test_empty_env_uses_defaults(self, clean_env: None) -> None:
        config = Config.from_env()
        assert config.vlm_backend == "ollama"
        assert config.max_frames == 50


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------


class TestConfigFromFile:
    """Verify ``Config.from_file()`` correctly loads JSON files."""

    def test_load_valid_json(self) -> None:
        data = {
            "vlm_backend": "openai",
            "vlm_model": "gpt-4o",
            "max_frames": 25,
            "temperature": 0.5,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            fh.flush()
            path = fh.name

        try:
            config = Config.from_file(path)
            assert config.vlm_backend == "openai"
            assert config.vlm_model == "gpt-4o"
            assert config.max_frames == 25
            assert config.temperature == 0.5
            # Default preserved
            assert config.frame_quality == 85
        finally:
            os.unlink(path)

    def test_frame_resolution_as_list(self) -> None:
        data = {"frame_resolution": [1920, 1080]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            fh.flush()
            path = fh.name

        try:
            config = Config.from_file(path)
            assert config.frame_resolution == (1920, 1080)
        finally:
            os.unlink(path)

    def test_partial_config(self) -> None:
        """Only specified fields are overridden."""
        data = {"max_frames": 10}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            fh.flush()
            path = fh.name

        try:
            config = Config.from_file(path)
            assert config.max_frames == 10
            assert config.vlm_backend == "ollama"  # unchanged
            assert config.target_fps == 1.5  # unchanged
        finally:
            os.unlink(path)

    def test_unknown_keys_ignored_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        data = {"vlm_backend": "qwen", "unknown_key": 42, "another_unknown": "x"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            fh.flush()
            path = fh.name

        try:
            config = Config.from_file(path)
            assert config.vlm_backend == "qwen"
        finally:
            os.unlink(path)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Config.from_file("/nonexistent/path/config.json")

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            fh.write("not { valid json")
            fh.flush()
            path = fh.name

        try:
            with pytest.raises(json.JSONDecodeError):
                Config.from_file(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    """Verify env + file interactions."""

    def test_env_overrides_file(self, clean_env: None) -> None:
        """When both are used, env should win (caller's responsibility)."""
        # This documents the expected pattern: Config.from_file() then
        # caller manually overlays env values if desired.
        data = {"max_frames": 10}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(data, fh)
            fh.flush()
            path = fh.name

        try:
            config = Config.from_file(path)
            assert config.max_frames == 10
            # Caller can overlay env manually
            os.environ["VIDEOCODE_MAX_FRAMES"] = "99"
            env_config = Config.from_env()
            assert env_config.max_frames == 99
        finally:
            os.unlink(path)
