"""CLI interface for videocode using Typer and Rich.

Provides commands for processing videos, extracting code, summarizing,
and running the MCP server with beautiful terminal output.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from videocode.agent_loop import AgentLoop, Task, TaskType
from videocode.audio_extractor import AudioExtractor
from videocode.code_extractor import CodeExtractor
from videocode.config import Config
from videocode.mcp_server import ClaudeVisionMCPServer
from videocode.video_processor import VideoProcessor
from videocode.vlm_client import VLMClient

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="videocode",
    help="Give Claude Code eyes -- turn coding videos into working code",
    no_args_is_help=True,
)
console = Console()


def _print_banner() -> None:
    """Print the videocode banner."""
    banner = (
        "[bold blue]videocode[/bold blue] "
        "[dim]-- Give Claude Code eyes[/dim]"
    )
    console.print(Panel(banner, expand=False))


def _create_config(backend: str, output: str, no_audio: bool = False) -> Config:
    """Create a Config from CLI parameters.

    Args:
        backend: VLM backend name.
        output: Output directory path.
        no_audio: Whether to skip audio extraction.

    Returns:
        Config instance.
    """
    config = Config.from_env()
    config.vlm_backend = backend
    config.output_dir = output
    if no_audio:
        config.extract_audio = False
    return config


def _process_with_progress(
    source: str,
    task_type: TaskType,
    config: Config,
    query: str = "",
) -> dict:
    """Process a video with a Rich progress spinner.

    Args:
        source: Video file path or URL.
        task_type: Type of processing task.
        config: Processing configuration.
        query: Optional query string.

    Returns:
        Processing result dictionary.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Processing video...", total=None)

        video_processor = VideoProcessor(config)
        audio_extractor = AudioExtractor(config)
        vlm = VLMClient(config)
        agent = AgentLoop(vlm, config)

        try:
            progress.update(task_id, description="Extracting frames...")
            processed = video_processor.process(source)

            transcription = None
            if config.extract_audio:
                try:
                    progress.update(task_id, description="Extracting audio...")
                    audio_path = audio_extractor.extract(Path(source))
                    progress.update(task_id, description="Transcribing audio...")
                    transcription = audio_extractor.transcribe(audio_path)
                except Exception as exc:
                    console.print(
                        f"[yellow]Warning: Audio extraction skipped: {exc}[/yellow]"
                    )

            progress.update(task_id, description="Analyzing with VLM...")
            task = Task(type=task_type, query=query)
            result = agent.run(task, processed, transcription)

            return {
                "status": "ok",
                "content": result.content,
                "confidence": result.confidence,
                "sources": result.sources,
                "duration": getattr(processed, "duration", 0.0),
            }

        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            logger.exception("Processing failed")
            return {"status": "error", "message": str(exc)}

        finally:
            video_processor.cleanup()
            audio_extractor.cleanup()


@app.command()
def process(
    source: str = typer.Argument(..., help="Video file path or YouTube URL"),
    query: str = typer.Option("", "--query", "-q", help="Question or task for the video"),
    backend: str = typer.Option("ollama", "--backend", "-b", help="VLM backend: ollama, gemini, openai, qwen, dummy"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio extraction"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Process a video and extract insights.

    Extract frames and audio from the video, analyze with the configured
    VLM backend, and print formatted results to the console.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    _print_banner()
    config = _create_config(backend, output, no_audio)

    result = _process_with_progress(
        source=source,
        task_type=TaskType.GENERAL,
        config=config,
        query=query,
    )

    if result["status"] == "error":
        console.print(f"[red]Processing failed: {result.get('message')}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        result.get("content", "No results"),
        title="[bold green]Analysis Result[/bold green]",
        subtitle=f"Confidence: {result.get('confidence', 0.0):.0%}",
    ))

    sources = result.get("sources", [])
    if sources:
        console.print()
        table = Table(title="Source References")
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        for src in sources:
            table.add_row(
                f"{src.timestamp:.1f}s",
                src.description,
            )
        console.print(table)

    console.print()
    console.print(f"[dim]Duration: {result.get('duration', 0):.1f}s[/dim]")


@app.command()
def code(
    source: str = typer.Argument(..., help="Coding tutorial video"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    backend: str = typer.Option("ollama", "--backend", "-b", help="VLM backend: ollama, gemini, openai, qwen, dummy"),
) -> None:
    """Extract code from a coding tutorial video.

    Detect code frames, extract code blocks, and assemble them into
    a structured project saved to the output directory.
    """
    _print_banner()
    config = _create_config(backend, output)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Processing video...", total=None)

        video_processor = VideoProcessor(config)
        audio_extractor = AudioExtractor(config)
        vlm = VLMClient(config)
        agent = AgentLoop(vlm, config)
        code_extractor = CodeExtractor(agent, config)

        try:
            progress.update(task_id, description="Extracting frames...")
            processed = video_processor.process(source)

            transcription = None
            if config.extract_audio:
                try:
                    progress.update(task_id, description="Transcribing audio...")
                    audio_path = audio_extractor.extract(Path(source))
                    transcription = audio_extractor.transcribe(audio_path)
                except Exception as exc:
                    console.print(
                        f"[yellow]Warning: Audio extraction skipped: {exc}[/yellow]"
                    )

            progress.update(task_id, description="Extracting code...")
            code_result = code_extractor.extract(processed, transcription)

        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            logger.exception("Code extraction failed")
            raise typer.Exit(1) from exc

        finally:
            video_processor.cleanup()
            audio_extractor.cleanup()

    if not code_result.files:
        console.print("[yellow]No code files were extracted.[/yellow]")
        raise typer.Exit(0)

    console.print()
    console.print(
        f"[bold green]Extracted {len(code_result.files)} file(s) "
        f"({code_result.language})[/bold green]"
    )
    console.print()

    tree = Tree(f"[bold]{output_path.name}/[/bold]")
    for filename in sorted(code_result.files.keys()):
        tree.add(f"[cyan]{filename}[/cyan]")
    console.print(tree)

    if code_result.description:
        console.print()
        console.print(Panel(code_result.description, title="Description"))

    for filename, content in code_result.files.items():
        file_path = output_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    console.print()
    console.print(f"[green]Files saved to {output_path.resolve()}[/green]")


@app.command()
def summarize(
    source: str = typer.Argument(..., help="Video file path or YouTube URL"),
    style: str = typer.Option("detailed", "--style", "-s", help="Summary style"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
) -> None:
    """Summarize a video.

    Generate a structured summary of the video in the requested style.
    """
    _print_banner()
    config = _create_config("ollama", output)

    result = _process_with_progress(
        source=source,
        task_type=TaskType.SUMMARIZATION,
        config=config,
        query=style,
    )

    if result["status"] == "error":
        console.print(f"[red]Summarization failed: {result.get('message')}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        result.get("content", "No summary generated"),
        title=f"[bold green]Summary ({style})[/bold green]",
    ))

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_file = output_path / "summary.md"
    summary_file.write_text(result.get("content", ""))
    console.print(f"[dim]Summary saved to {summary_file}[/dim]")


@app.command(name="mcp")
def mcp_server(
    config_file: str | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Run in MCP server mode for Claude Code integration.

    Starts the ClaudeVisionMCPServer with stdio transport for
    communication with Claude Code via the Model Context Protocol.
    """
    config = Config.from_file(config_file) if config_file else Config.from_env()

    from rich.console import Console

    err_console = Console(file=sys.stderr)
    err_console.print("[bold green]Starting MCP server for Claude Code...[/bold green]")
    err_console.print("[dim]Transport: stdio[/dim]")
    err_console.print("[dim]Press Ctrl+C to stop[/dim]")
    err_console.print()

    server = ClaudeVisionMCPServer(config)
    server.run()


@app.command()
def status() -> None:
    """Check backend status.

    Display the availability of all backends: FFmpeg, Ollama,
    Whisper, and API keys.
    """
    _print_banner()
    console.print()

    table = Table(title="Backend Status")
    table.add_column("Backend", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        table.add_row("FFmpeg", "[green]Available[/green]", ffmpeg_path)
    else:
        table.add_row("FFmpeg", "[red]Missing[/red]", "Install ffmpeg")

    ollama_url = "http://localhost:11434"
    try:
        import urllib.request

        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2):
            table.add_row("Ollama", "[green]Running[/green]", ollama_url)
    except Exception:
        table.add_row("Ollama", "[red]Unavailable[/red]", f"{ollama_url} unreachable")

    try:
        import faster_whisper  # noqa: F401

        table.add_row("Whisper", "[green]Installed[/green]", "faster-whisper")
    except ImportError:
        table.add_row("Whisper", "[red]Missing[/red]", "pip install faster-whisper")

    api_keys = []
    for name, env_var in [
        ("OpenAI", "OPENAI_API_KEY"),
        ("Gemini", "GEMINI_API_KEY"),
        ("Apify", "APIFY_API_TOKEN"),
        ("Perplexity", "PERPLEXITY_API_KEY"),
    ]:
        import os

        if os.environ.get(env_var):
            api_keys.append(name)

    if api_keys:
        table.add_row("API Keys", "[green]Configured[/green]", ", ".join(api_keys))
    else:
        table.add_row("API Keys", "[yellow]None[/yellow]", "Set OPENAI_API_KEY, GEMINI_API_KEY, APIFY_API_TOKEN, or PERPLEXITY_API_KEY")

    # Apify
    if os.environ.get("APIFY_API_TOKEN"):
        table.add_row("Apify", "[green]Configured[/green]", "YouTube download + metadata")
    else:
        table.add_row("Apify", "[yellow]Optional[/yellow]", "Set APIFY_API_TOKEN for YouTube")

    # Perplexity
    if os.environ.get("PERPLEXITY_API_KEY"):
        table.add_row("Perplexity", "[green]Configured[/green]", "Code verification + docs")
    else:
        table.add_row("Perplexity", "[yellow]Optional[/yellow]", "Set PERPLEXITY_API_KEY for code verification")

    try:
        import yt_dlp  # noqa: F401

        table.add_row("yt-dlp", "[green]Installed[/green]", "YouTube support enabled")
    except ImportError:
        table.add_row("yt-dlp", "[yellow]Missing[/yellow]", "YouTube URLs unavailable")

    console.print(table)


if __name__ == "__main__":
    app()
