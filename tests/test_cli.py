"""Tests for the CLI module."""

from __future__ import annotations

from typer.testing import CliRunner

from videocode.cli import app

runner = CliRunner()


class TestCLI:
    """Tests for the CLI application."""

    def test_help(self) -> None:
        """Test that --help shows usage information."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "videocode" in result.output
        assert "Give Claude Code eyes" in result.output

    def test_status(self) -> None:
        """Test the status command."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Backend Status" in result.output

    def test_mcp_help(self) -> None:
        """Test the mcp command help."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP server" in result.output

    def test_process_help(self) -> None:
        """Test the process command help."""
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process a video" in result.output

    def test_code_help(self) -> None:
        """Test the code command help."""
        result = runner.invoke(app, ["code", "--help"])
        assert result.exit_code == 0
        assert "Extract code" in result.output

    def test_summarize_help(self) -> None:
        """Test the summarize command help."""
        result = runner.invoke(app, ["summarize", "--help"])
        assert result.exit_code == 0
        assert "Summarize" in result.output
