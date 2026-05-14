"""Tests for the VLM client and its backends."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from videocode.types import Config, Frame, Transcription, VLMResponse
from videocode.vlm_client import (
    GeminiBackend,
    OllamaBackend,
    OpenAIBackend,
    QwenBackend,
    VLMBackend,
    VLMClient,
    _frame_to_base64,
    _retry_with_backoff,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_frame(tmp_path: Path) -> Frame:
    """Create a fake frame with a real 1x1 PNG on disk."""
    # Create a minimal valid PNG (1x1 black pixel)
    import struct
    import zlib

    # PNG signature
    png_sig = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    # IDAT chunk (compressed empty image data for 1x1 RGB)
    raw = b"\x00\x00\x00\x00"  # filter byte + 1 pixel RGB
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    # IEND chunk
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    img_path = tmp_path / "frame_001.png"
    img_path.write_bytes(png_sig + ihdr + idat + iend)

    return Frame(path=img_path, timestamp=1.0, scene_index=0, is_keyframe=True)


@pytest.fixture
def config_ollama() -> Config:
    return Config(vlm_backend="ollama", vlm_model="llava")


@pytest.fixture
def config_gemini() -> Config:
    return Config(vlm_backend="gemini", vlm_model="gemini-1.5-pro", vlm_api_key="test-key")


@pytest.fixture
def config_openai() -> Config:
    return Config(vlm_backend="openai", vlm_model="gpt-4o", vlm_api_key="sk-test")


@pytest.fixture
def config_qwen() -> Config:
    return Config(vlm_backend="qwen", vlm_model="qwen2.5-vl", vlm_api_key="test-key")


# ---------------------------------------------------------------------------
# _frame_to_base64
# ---------------------------------------------------------------------------


def test_frame_to_base64(fake_frame: Frame) -> None:
    """A frame should round-trip through base64 encoding."""
    b64 = _frame_to_base64(fake_frame)
    raw = base64.b64decode(b64)
    assert raw[:4] == b"\x89PNG"


def test_frame_to_base64_resizing(fake_frame: Frame) -> None:
    """Frames larger than max_size should be resized."""
    b64 = _frame_to_base64(fake_frame, max_size=(1, 1))
    raw = base64.b64decode(b64)
    assert raw[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# OllamaBackend
# ---------------------------------------------------------------------------


class TestOllamaBackend:
    def test_init(self) -> None:
        be = OllamaBackend("http://localhost:11434", "llava")
        assert be.base_url == "http://localhost:11434"
        assert be.model == "llava"

    def test_chat(self) -> None:
        be = OllamaBackend("http://localhost:11434", "llava")
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Hello from Ollama"
        mock_response = MagicMock()
        mock_response.message = mock_msg
        mock_client.chat.return_value = mock_response
        be._client = mock_client

        resp = be.chat(["fakeb64"], "describe this")
        assert resp.content == "Hello from Ollama"
        assert resp.model == "llava"
        assert resp.frames_analyzed == 1

        mock_client.chat.assert_called_once_with(
            model="llava",
            messages=[{"role": "user", "content": "describe this", "images": ["fakeb64"]}],
            options={"temperature": 0.3},
        )

    def test_is_available_true(self) -> None:
        be = OllamaBackend("http://localhost:11434", "llava")
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        be._client = mock_client
        assert be.is_available() is True

    def test_is_available_false(self) -> None:
        be = OllamaBackend("http://localhost:11434", "llava")
        mock_client = MagicMock()
        mock_client.list.side_effect = ConnectionError("refused")
        be._client = mock_client
        assert be.is_available() is False


# ---------------------------------------------------------------------------
# GeminiBackend
# ---------------------------------------------------------------------------


class TestGeminiBackend:
    def _mock_genai_module(self) -> Any:
        """Create a mock google.generativeai module."""
        mock_genai = MagicMock()
        mock_model_cls = MagicMock()
        mock_genai.GenerativeModel = mock_model_cls
        mock_genai.configure = MagicMock()
        return mock_genai, mock_model_cls

    def test_init(self) -> None:
        mock_genai, _ = self._mock_genai_module()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            be = GeminiBackend("api-key", "gemini-1.5-pro")
            assert be.model_name == "gemini-1.5-pro"

    def test_chat(self) -> None:
        mock_genai, mock_model_cls = self._mock_genai_module()
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini"
        mock_usage = MagicMock()
        mock_usage.total_token_count = 42
        mock_response.usage_metadata = mock_usage
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            be = GeminiBackend("api-key", "gemini-1.5-pro")
            # Use valid base64 (1x1 PNG pixel)
            valid_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            resp = be.chat([valid_b64], "describe this")
            assert resp.content == "Hello from Gemini"
            assert resp.tokens_used == 42
            assert resp.frames_analyzed == 1

    def test_is_available(self) -> None:
        mock_genai, _ = self._mock_genai_module()
        mock_model_info = MagicMock()
        mock_model_info.name = "models/gemini-1.5-pro"
        mock_genai.list_models.return_value = [mock_model_info]

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            be = GeminiBackend("api-key", "gemini-1.5-pro")
            assert be.is_available() is True


# ---------------------------------------------------------------------------
# OpenAIBackend
# ---------------------------------------------------------------------------


class TestOpenAIBackend:
    def _mock_openai(self) -> Any:
        """Create a mock openai module with OpenAI class."""
        mock_openai_mod = MagicMock()
        mock_openai_cls = MagicMock()
        mock_openai_mod.OpenAI = mock_openai_cls
        return mock_openai_mod, mock_openai_cls

    def test_init(self) -> None:
        mock_openai_mod, mock_openai_cls = self._mock_openai()
        mock_openai_cls.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            be = OpenAIBackend("sk-test", "gpt-4o")
            assert be.model == "gpt-4o"

    def test_chat(self) -> None:
        mock_openai_mod, mock_openai_cls = self._mock_openai()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from OpenAI"
        mock_usage = MagicMock()
        mock_usage.total_tokens = 100
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            be = OpenAIBackend("sk-test", "gpt-4o")
            resp = be.chat(["fakeb64"], "describe this")
            assert resp.content == "Hello from OpenAI"
            assert resp.tokens_used == 100
            assert resp.frames_analyzed == 1

    def test_is_available(self) -> None:
        mock_openai_mod, mock_openai_cls = self._mock_openai()
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_cls.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            be = OpenAIBackend("sk-test", "gpt-4o")
            assert be.is_available() is True


# ---------------------------------------------------------------------------
# QwenBackend
# ---------------------------------------------------------------------------


class TestQwenBackend:
    def _mock_openai(self) -> Any:
        mock_openai_mod = MagicMock()
        mock_openai_cls = MagicMock()
        mock_openai_mod.OpenAI = mock_openai_cls
        return mock_openai_mod, mock_openai_cls

    def test_init(self) -> None:
        mock_openai_mod, mock_openai_cls = self._mock_openai()
        mock_openai_cls.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            be = QwenBackend("api-key", "qwen2.5-vl")
            assert be.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_chat(self) -> None:
        mock_openai_mod, mock_openai_cls = self._mock_openai()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Qwen"
        mock_usage = MagicMock()
        mock_usage.total_tokens = 80
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            be = QwenBackend("api-key", "qwen2.5-vl")
            resp = be.chat(["fakeb64"], "describe this")
            assert resp.content == "Hello from Qwen"
            assert resp.tokens_used == 80

    def test_custom_base_url(self) -> None:
        mock_openai_mod, mock_openai_cls = self._mock_openai()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            be = QwenBackend("key", "qwen2.5-vl", base_url="https://custom.example.com/v1")
            assert be.base_url == "https://custom.example.com/v1"
            # Trigger lazy init via is_available
            be.is_available()
            mock_openai_cls.assert_called_once_with(
                api_key="key",
                base_url="https://custom.example.com/v1",
            )


# ---------------------------------------------------------------------------
# VLMClient
# ---------------------------------------------------------------------------


class TestVLMClient:
    def test_create_backend_ollama(self, config_ollama: Config) -> None:
        client = VLMClient(config_ollama)
        assert isinstance(client._backend, OllamaBackend)

    def test_create_backend_gemini(self, config_gemini: Config) -> None:
        mock_genai = MagicMock()
        mock_genai.GenerativeModel = MagicMock
        mock_genai.configure = MagicMock()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            client = VLMClient(config_gemini)
            assert isinstance(client._backend, GeminiBackend)

    def test_create_backend_openai(self, config_openai: Config) -> None:
        client = VLMClient(config_openai)
        assert isinstance(client._backend, OpenAIBackend)

    def test_create_backend_qwen(self, config_qwen: Config) -> None:
        client = VLMClient(config_qwen)
        assert isinstance(client._backend, QwenBackend)

    def test_create_backend_unknown(self) -> None:
        config = Config(vlm_backend="unknown")
        with pytest.raises(ValueError, match="Unknown VLM backend"):
            VLMClient(config)

    def test_create_backend_missing_key_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = Config(vlm_backend="gemini")
        with pytest.raises(ValueError, match="Gemini backend requires API key"):
            VLMClient(config)

    def test_analyze_single(self, config_ollama: Config, fake_frame: Frame) -> None:
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "chat") as mock_chat:
            mock_chat.return_value = VLMResponse(content="code analysis result")
            result = client.analyze_single(fake_frame, "extract code")
            assert result == "code analysis result"
            mock_chat.assert_called_once()

    def test_analyze_frames(self, config_ollama: Config, fake_frame: Frame) -> None:
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "chat") as mock_chat:
            mock_chat.return_value = VLMResponse(content="multi-frame result")
            result = client.analyze_frames([fake_frame], "describe")
            assert result.content == "multi-frame result"

    def test_cache_hit(self, config_ollama: Config, fake_frame: Frame) -> None:
        """The second identical call should hit the cache."""
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "chat") as mock_chat:
            mock_chat.return_value = VLMResponse(content="cached")
            r1 = client.analyze_frames([fake_frame], "same prompt")
            r2 = client.analyze_frames([fake_frame], "same prompt")
            assert r1.content == r2.content == "cached"
            mock_chat.assert_called_once()

    def test_cache_with_transcription(self, config_ollama: Config, fake_frame: Frame) -> None:
        """Cache key should differentiate by transcription."""
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "chat") as mock_chat:
            mock_chat.side_effect = [
                VLMResponse(content="without tx"),
                VLMResponse(content="with tx"),
            ]
            tx = Transcription(text="hello world", segments=[], language="en")
            r1 = client.analyze_frames([fake_frame], "prompt")
            r2 = client.analyze_frames([fake_frame], "prompt", tx)
            assert r1.content == "without tx"
            assert r2.content == "with tx"
            assert mock_chat.call_count == 2

    def test_transcription_in_prompt(self, config_ollama: Config, fake_frame: Frame) -> None:
        """Transcription text should be appended to the prompt."""
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "chat") as mock_chat:
            mock_chat.return_value = VLMResponse(content="ok")
            tx = Transcription(text="hello", segments=[])
            client.analyze_frames([fake_frame], "extract", tx)
            call_args = mock_chat.call_args[0]
            prompt = call_args[1]
            assert "hello" in prompt
            assert "Transcription" in prompt

    def test_is_available(self, config_ollama: Config) -> None:
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "is_available", return_value=True):
            assert client.is_available() is True

    def test_retry_success_after_failure(self, config_ollama: Config, fake_frame: Frame) -> None:
        """analyze_frames should retry and eventually succeed."""
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "chat") as mock_chat:
            mock_chat.side_effect = [
                ConnectionError("first fail"),
                ConnectionError("second fail"),
                VLMResponse(content="success"),
            ]
            result = client.analyze_frames([fake_frame], "test")
            assert result.content == "success"
            assert mock_chat.call_count == 3

    def test_retry_exhaustion(self, config_ollama: Config, fake_frame: Frame) -> None:
        """analyze_frames should raise the last error after retries are exhausted."""
        config_ollama.max_retries = 1
        client = VLMClient(config_ollama)
        with patch.object(client._backend, "chat") as mock_chat:
            mock_chat.side_effect = [
                ConnectionError("first fail"),
                ConnectionError("second fail"),
                ConnectionError("third fail"),
            ]
            with pytest.raises(ConnectionError, match="second fail"):
                client.analyze_frames([fake_frame], "test")
            assert mock_chat.call_count == config_ollama.max_retries + 1


# ---------------------------------------------------------------------------
# Retry decorator unit tests
# ---------------------------------------------------------------------------


def test_retry_with_backoff_success() -> None:
    """Decorator should return result on success without retrying."""
    call_count = 0

    @_retry_with_backoff(max_retries=2, base_delay=0.01)
    def _ok() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    assert _ok() == "ok"
    assert call_count == 1


def test_retry_with_backoff_eventual_success() -> None:
    """Decorator should retry and return result on eventual success."""
    call_count = 0

    @_retry_with_backoff(max_retries=3, base_delay=0.01)
    def _eventually() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("fail")
        return "ok"

    assert _eventually() == "ok"
    assert call_count == 3


def test_retry_with_backoff_exhaustion() -> None:
    """Decorator should raise the last exception after retries are exhausted."""
    call_count = 0

    @_retry_with_backoff(max_retries=2, base_delay=0.01)
    def _never() -> str:
        nonlocal call_count
        call_count += 1
        raise ConnectionError(f"fail {call_count}")

    with pytest.raises(ConnectionError, match="fail 3"):
        _never()
    assert call_count == 3
