"""Tests for the MCP server module."""

from __future__ import annotations

import pytest

from videocode.config import Config
from videocode.mcp_server import ClaudeVisionMCPServer


class TestClaudeVisionMCPServer:
    """Tests for ClaudeVisionMCPServer."""

    @pytest.fixture
    def config(self) -> Config:
        """Create a test configuration."""
        return Config(
            vlm_backend="ollama",
            vlm_model="llava",
            max_frames=10,
            extract_audio=False,
        )

    @pytest.fixture
    def server(self, config: Config) -> ClaudeVisionMCPServer:
        """Create a test server instance."""
        return ClaudeVisionMCPServer(config)

    def test_init(self, server: ClaudeVisionMCPServer, config: Config) -> None:
        """Test server initialization."""
        assert server.config == config
        assert server.video_processor is not None
        assert server.frame_selector is not None
        assert server.audio_extractor is not None
        assert server.vlm is not None
        assert server.agent is not None
        assert server.code_extractor is not None

    def test_health_check(self, server: ClaudeVisionMCPServer) -> None:
        """Test health check returns expected structure."""
        result = server.do_health_check()
        assert "status" in result
        assert "backends" in result
        assert isinstance(result["backends"], dict)
        assert "ffmpeg" in result["backends"]

    @pytest.mark.asyncio
    async def test_video_summarize(self, server: ClaudeVisionMCPServer) -> None:
        """Test video_summarize returns expected structure."""
        result = await server.do_video_summarize("nonexistent.mp4", style="brief")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_video_find_bugs(self, server: ClaudeVisionMCPServer) -> None:
        """Test video_find_bugs surfaces a structured error for a missing file.

        Real bug analysis is exercised end-to-end in integration tests; this
        unit test only verifies the error contract returned by _process_video.
        """
        result = await server.do_video_find_bugs("nonexistent.mp4", description="test")
        assert isinstance(result, dict)
        assert result.get("status") == "error"
        assert "not found" in result.get("message", "").lower()
