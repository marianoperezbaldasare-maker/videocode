# Configuration

Configure videocode to match your workflow.

## Configuration Sources

Settings are loaded in this order (later sources override earlier ones):

1. **Defaults** — built into the application
2. **Config file** — `~/.config/videocode/config.yaml`
3. **Environment variables** — `VIDEOCODE_*`
4. **CLI flags** — `--backend`, `--output-dir`, etc.

## Config File

Create `~/.config/videocode/config.yaml`:

```yaml
# Default backend for VLM analysis
# Options: ollama, gemini, openai, qwen
default_backend: ollama

# Default output directory
output_dir: ./output/

# Scene detection sensitivity (0.0 - 1.0)
# Higher = more sensitive to changes = more scenes detected
scene_threshold: 0.35

# Maximum frames to analyze per video
# Increase for better accuracy, decrease for speed
max_frames: 20

# Frame selection preferences
code_bias: 0.3          # Prefer frames with text (0.0 - 1.0)
similarity_threshold: 0.85  # Skip similar frames (0.0 - 1.0)

# Ollama backend settings
ollama:
  host: http://localhost:11434
  model: llama3.2-vision  # or llava, minicpm-v
  timeout: 120  # seconds

# Gemini backend settings
gemini:
  model: gemini-2.5-flash
  timeout: 60

# OpenAI backend settings
openai:
  model: gpt-4o
  timeout: 60

# Whisper settings (for audio transcription)
whisper:
  model: base  # tiny, base, small, medium, large
  language: auto  # auto-detect, or specify: en, zh, es, etc.
```

## Environment Variables

All config values can be set via environment variables:

```bash
# Core settings
export VIDEOCODE_BACKEND=gemini
export VIDEOCODE_OUTPUT_DIR=./output/

# API keys (used by cloud backends)
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Ollama settings
export OLLAMA_HOST=http://localhost:11434

# FFmpeg path (if not in PATH)
export FFMPEG_PATH=/usr/local/bin/ffmpeg
```

## CLI Flags Reference

```
videocode code [VIDEO] [OPTIONS]

Options:
  --backend {ollama,gemini,openai,qwen}
                        VLM backend to use
  --output-dir PATH     Output directory for generated files
  --model NAME          Specific model to use (overrides config)
  --no-transcription    Skip audio transcription
  --no-scene-detection  Use uniform frame sampling instead
  --scene-threshold FLOAT
                        Scene detection sensitivity (0.0-1.0)
  --max-frames INT      Maximum frames to analyze
  --frame-rate FLOAT    Frames per second to extract
  --code-bias FLOAT     Prefer text-dense frames (0.0-1.0)
  --language CODE       Audio language (e.g., en, zh, es)
  --whisper-model NAME  Whisper model size (tiny/base/small/medium/large)
  --verbose, -v         Enable verbose output
  --help                Show help message
```

## Backend-Specific Setup

### Ollama

See the [Ollama tutorial](tutorials.md#set-up-local-ollama-backend).

Key config:
```yaml
ollama:
  host: http://localhost:11434
  model: llama3.2-vision
```

### Gemini

Get API key: [Google AI Studio](https://aistudio.google.com/app/apikey)

```yaml
gemini:
  model: gemini-2.5-flash
```

Set `GEMINI_API_KEY` environment variable or add to `~/.config/videocode/.env`.

### OpenAI

Get API key: [OpenAI Platform](https://platform.openai.com/api-keys)

```yaml
openai:
  model: gpt-4o
```

Set `OPENAI_API_KEY` environment variable.

### Qwen

Qwen is self-hosted. You'll need to run the model server separately.

```yaml
qwen:
  host: http://localhost:8000  # your Qwen server
  model: Qwen2.5-VL-7B
```

## Per-Project Configuration

Create a `.videocode.yaml` in your project root:

```yaml
# Project-specific videocode settings
default_backend: gemini
output_dir: ./generated-from-videos/
max_frames: 30
scene_threshold: 0.25
```

This is useful when your team prefers a specific backend or output location.
