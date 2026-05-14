# 👁️ videocode

[![CI](https://github.com/marianoperezbaldasare-maker/videocode/actions/workflows/ci.yml/badge.svg)](https://github.com/marianoperezbaldasare-maker/videocode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![Runs locally](https://img.shields.io/badge/runs-100%25%20locally-success)](https://ollama.com)
[![No API key required](https://img.shields.io/badge/API%20key-not%20required-brightgreen)](#backends)

> **The missing video layer for Claude Code.** Turn any coding video into copy-pasteable code — or extract text, summaries, and bugs from any screen recording.
>
> 🔒 **Runs 100% locally with Ollama. No API key. No data leaves your machine.** Cloud backends (Gemini, OpenAI) are optional.

```
    ╔═══════════════════════════════════════════════════════════╗
    ║  📺  Coding Video  →  🔍  videocode  →  📁  Code    ║
    ║                                                           ║
    ║  "Build a React app"  →  ./output/src/App.tsx            ║
    ║                        ./output/package.json              ║
    ║                        ./output/README.md                 ║
    ╚═══════════════════════════════════════════════════════════╝
```

## The Problem

Claude Code can read your files. It can see your images. But it **can't watch videos**.

Meanwhile:
- 📺 **44% of developers** learn from video tutorials
- 🐛 **Bug reports** increasingly include screen recordings
- 📹 **Code reviews** happen over video walkthroughs

Every day, thousands of hours of coding knowledge are locked behind play buttons. Developers pause, rewind, squint at screens, and manually type code — introducing typos, missing context, and wasting hours.

## The Solution

```bash
# 1. Install (Python 3.10+)
pip install videocode

# 2. Install a local vision model (free, no key)
ollama pull llama3.2-vision

# 3. Point it at a video
videocode code "https://youtube.com/watch?v=..."
# → Generates working code files in ./output/
```

That's it. No accounts. No keys. No quota. The whole pipeline runs on your laptop.

> Want higher accuracy? Set `GEMINI_API_KEY` and add `--backend gemini`. The free tier handles ~1500 requests/day.

### What videocode does:

- 🔍 **Intelligent frame selection** — extracts key moments, not random frames
- 🎯 **Scene detection** — understands video structure (intro, coding, demo, outro)
- 🎙️ **Audio transcription** — captures the narrator's explanation for context (Whisper)
- 🤖 **VLM analysis** — reads code from video frames using vision-language models
- 🔎 **Smart repo discovery** — finds the tutorial's actual GitHub repo from video metadata (no need to OCR if the real code is one link away)
- 📁 **Project generation** — assembles extracted code into runnable files
- 🐛 **Bug finding** — surfaces errors in screen recordings
- 📝 **OCR mode** — pure text extraction from slides, dashboards, code
- 🔌 **MCP server** — native Claude Code integration with 7 tools

## Quick Start

### 1. Install

```bash
pip install videocode
```

### 2. Set up Ollama (free, local)

```bash
ollama pull llava
```

### 3. Extract code from any video

```bash
# From a file
videocode code tutorial.mp4

# From YouTube
videocode code "https://youtube.com/watch?v=dQw4w9WgXcQ"

# With a specific backend
videocode code tutorial.mp4 --backend gemini

# Or use with Claude Code
videocode mcp
# Then in Claude Code: "Extract the code from this tutorial: tutorial.mp4"
```

## Demo

### Before: Watching a 20-minute React tutorial

```
[20 min video] → [pause at 3:24] → [squint at screen] → [type code] → [typos]
   ↓
[rewind to 3:15] → [pause again] → [type more] → [wrong import]
   ↓
[frustration] → [abandon] → [Google the repo instead]

⏱️ Time wasted: 45 minutes
😤 Satisfaction: 0/10
```

### After: videocode

```bash
$ videocode code react-tutorial.mp4
✓ Analyzed 20:34 video
✓ Detected 12 scenes (intro → setup → components → hooks → demo → outro)
✓ Selected 8 key frames with code
✓ Extracted 8 code blocks
✓ Resolved 3 dependencies
✓ Assembled project in ./output/

📁 output/
├── src/
│   ├── App.tsx              ← Main application
│   ├── components/
│   │   └── Counter.tsx      ← Reusable counter
│   └── hooks/
│       └── useCounter.ts    ← Custom hook
├── package.json             ← Dependencies resolved
├── tsconfig.json            ← TypeScript config
└── README.md                ← Generated docs

⏱️ Time spent: 30 seconds
🚀 Ready to run: npm install && npm run dev
```

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🎬 Video processing | FFmpeg-powered extraction and scene detection | ✅ |
| 🖼️ Smart frame selection | Adaptive sampling — more frames during coding segments | ✅ |
| 🎙️ Audio transcription | Whisper integration for narrator context | ✅ |
| 🤖 Multi-backend VLM | Ollama (local), Gemini, OpenAI, Qwen | ✅ |
| 📦 Code extraction | Tutorial → working, runnable code | ✅ |
| 🔌 MCP server | Native Claude Code integration — just ask | ✅ |
| 🐛 Bug detection | Find issues in screen recordings | ✅ |
| 📺 YouTube support | Direct URL processing | ✅ |
| 🏗️ Project assembly | Generates full project structure, not just snippets | ✅ |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Video URL  │────▶│   FFmpeg    │────▶│  Scene Detection │
│  or File    │     │  Extraction │     │  (content-based)  │
└─────────────┘     └─────────────┘     └─────────────────┘
                                                │
                    ┌─────────────┐              ▼
                    │   Whisper   │     ┌─────────────────┐
                    │Transcription│◀────│  Frame Selector  │
                    │   (audio)   │     │ (adaptive pick)  │
                    └──────┬──────┘     └─────────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────────────────────────────┐
                    │           VLM Analysis               │
                    │  (Ollama / Gemini / OpenAI / Qwen)  │
                    │                                     │
                    │  Frame + Transcription → Code       │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │           Agent Loop                  │
                    │  Orchestrates extraction & assembly  │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │        Code Extractor & Assembler    │
                    │  → Syntax validation                 │
                    │  → Dependency resolution             │
                    │  → File structure generation         │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │  ./output/     │
                              │  Runnable code │
                              └────────────────┘
```

## 🔗 Optional Integrations

### Apify — Reliable YouTube Downloads

Set `APIFY_API_TOKEN` for more reliable YouTube video downloading and metadata extraction:

```bash
export APIFY_API_TOKEN="your-apify-token"
videocode code "https://youtube.com/watch?v=..."
```

**What Apify adds:**
- More reliable YouTube downloads (handles age-restricted, geo-blocked videos)
- Automatic video metadata extraction (title, description, duration, tags)
- Better thumbnail and subtitle support

Get your token at [apify.com](https://console.apify.com/actors)

### Perplexity — Code Verification & Docs

Set `PERPLEXITY_API_KEY` to verify extracted code and get better setup instructions:

```bash
export PERPLEXITY_API_KEY="your-perplexity-key"
videocode code tutorial.mp4
```

**What Perplexity adds:**
- ✅ Code verification — checks for errors and suggests fixes
- 📚 Documentation lookup — finds official docs for detected frameworks
- 📝 Better setup instructions — generates accurate install steps
- 🔍 Framework info — detects versions, breaking changes, migration notes

Get your key at [perplexity.ai/settings](https://www.perplexity.ai/settings/api)

---

## Backends

### Ollama (default — free, local, private)

Everything runs on your machine. No API keys. No data leaves your computer.

```bash
# Pull a vision model
ollama pull llava              # fast, good baseline
ollama pull llama3.2-vision    # better code recognition
ollama pull minicpm-v          # excellent for UI/screenshots

# Use it
videocode code tutorial.mp4
```

### Cloud APIs (for higher accuracy)

When you need the best possible extraction quality.

```bash
# Set your API key
export GEMINI_API_KEY="..."     # or OPENAI_API_KEY

# Use cloud backend
videocode code tutorial.mp4 --backend gemini
```

| Backend | Speed | Accuracy | Cost | Privacy |
|---------|-------|----------|------|---------|
| Ollama (llava) | ⚡ Fast | Good | Free | 100% local |
| Ollama (llama3.2-vision) | ⚡ Fast | Better | Free | 100% local |
| Gemini 2.5 Flash | 🌐 Network | Best | Free tier | Cloud |
| OpenAI GPT-4o | 🌐 Network | Best | Paid | Cloud |
| Qwen2.5-VL | ⚡ Fast | Good | Free (self-host) | Configurable |

## MCP Server Setup (for Claude Code)

The real magic: use videocode *inside* Claude Code.

Add to your Claude Code config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "videocode": {
      "command": "videocode",
      "args": ["mcp"]
    }
  }
}
```

Then just ask:

```
> Extract the code from this React tutorial: https://youtube.com/...
> Find the bugs in this screen recording: bug-repro.mp4
> Summarize this tech talk for me: talk.mp4
> Read every line of code shown in this video: demo.mp4
> Find the GitHub repo for this tutorial: https://youtube.com/...
> Convert this UI demo into a Next.js app: demo-video.mp4
```

Claude Code handles the rest. It calls videocode, gets the results, and integrates them into your project.

### Available MCP tools

| Tool | What it does |
|------|--------------|
| `video_analyze(source, query, mode)` | Q&A over the video; `mode="tutorial"` for dense sampling |
| `video_summarize(source, style, mode)` | Detailed/brief/bullet summary |
| `video_extract_code(source)` | Tutorial → runnable project |
| `video_extract_text(source, query)` | OCR-focused; reads ALL visible text (slides, code, UI) |
| `video_find_source_repo(source)` | Find the tutorial's actual GitHub repo from video metadata |
| `video_find_bugs(source, description, mode)` | Surface errors, crashes, glitches |
| `health_check()` | Verify all backends (FFmpeg / Whisper / yt-dlp / VLM) are reachable |

## Installation

```bash
pip install videocode
```

**Requirements:**
- Python 3.10+
- FFmpeg (`brew install ffmpeg` or `apt-get install ffmpeg`)

**Optional (for local VLM):**
- [Ollama](https://ollama.com) — free, local vision models

## CLI Reference

```bash
# Extract code from video
videocode code <video> [options]

Options:
  --backend {ollama,gemini,openai,qwen}   VLM backend (default: ollama)
  --output-dir PATH                       Output directory (default: ./output/)
  --model NAME                            Specific model to use
  --no-transcription                      Skip audio transcription
  --scene-threshold FLOAT                 Scene detection sensitivity
  --max-frames INT                        Maximum frames to analyze

# Run MCP server
videocode mcp

# Check installation
videocode --version
videocode --help
```

## Project Status

**Current version: v0.1.0** — Early release, core extraction pipeline working.

This is a new project. The core pipeline (video → frames → VLM → code) is solid. We're actively improving accuracy and adding backends. Expect rough edges — and please [open issues](https://github.com/marianoperezbaldasare-maker/videocode/issues) when you hit them.

**What's working:**
- ✅ Video input (file, YouTube URL with persistent local cache)
- ✅ Scene detection and smart frame selection
- ✅ Tutorial-tuned dense sampling mode (`mode="tutorial"`)
- ✅ VLM-based code extraction (Ollama, Gemini, OpenAI, Qwen)
- ✅ Audio transcription (Whisper) — works for local files AND YouTube URLs
- ✅ Project assembly (file structure generation)
- ✅ Video summarization (`video_summarize`) with detailed/brief/bullet styles
- ✅ Bug finding from screen recordings (`video_find_bugs`)
- ✅ Pure OCR mode (`video_extract_text`) for slides, dashboards, code on screen
- ✅ MCP server for Claude Code integration with 6 tools

**Coming soon:**
- 🔄 Frame deduplication via perceptual hashing (cuts cost on static screen recordings)
- 🔄 Streaming progress notifications for long-running MCP requests
- 🔄 Better dependency inference and version detection
- 🔄 Support for more video platforms (Vimeo, Loom, direct upload)

## Documentation

- [Architecture](docs/architecture.md) — Deep dive into the system design
- [Tutorials](docs/tutorials.md) — Step-by-step guides for common workflows
- [Configuration](docs/configuration.md) — Backend setup and options
- [Contributing](CONTRIBUTING.md) — Help us make this better

## Why This Exists

I was tired of pausing YouTube tutorials, squinting at code on screen, and typing it out manually. I wanted to point Claude at a video and say "give me that code." So I built it.

If you've ever:
- ⏸️ Paused a tutorial 50 times to copy code
- 🤬 Hit a bug because you missed a semicolon from the screen
- 🙏 Wished the tutorial had a GitHub repo (it didn't)

...this tool is for you.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=marianoperezbaldasare-maker/videocode&type=Date)](https://star-history.com/#marianoperezbaldasare-maker/videocode&Date)

## Related Projects

If you're building AI-driven tooling on top of unstructured data, these are worth knowing:

- [**ScrapeGraphAI**](https://github.com/ScrapeGraphAI/Scrapegraph-ai) — AI-driven web scraping with LLM + graph logic. Same philosophy as videocode (local-first via Ollama, no keys required) but for **web pages** instead of video.
- [**Whisper**](https://github.com/openai/whisper) — what powers our audio transcription.
- [**Ollama**](https://github.com/ollama/ollama) — local LLM/VLM runtime; the foundation of our zero-key default.
- [**Anthropic MCP servers**](https://github.com/modelcontextprotocol/servers) — reference implementations for the Model Context Protocol.

If you find one of these useful, give them a star — open-source thrives on visibility.

## License

MIT — free for personal and commercial use.

---

<p align="center">
  Built with 👁️ by developers who were tired of pausing videos
</p>
