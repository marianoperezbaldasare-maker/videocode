# Tutorials

Step-by-step guides for common videocode workflows.

---

## Extract Code from a YouTube Tutorial

Turn a YouTube coding tutorial into a working project.

### Prerequisites

- videocode installed (`pip install videocode`)
- FFmpeg installed
- Ollama running (for local) or API key set (for cloud)

### Steps

**1. Find the video URL**

Copy the YouTube URL. Any of these formats work:
```
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/embed/VIDEO_ID
```

**2. Run videocode**

```bash
videocode code "https://www.youtube.com/watch?v=EXAMPLE"
```

You'll see progress output:
```
🎬 Processing: How to Build a Todo App with React
   Duration: 18:42
   → Downloading video... ✓
   → Detecting scenes... ✓ (found 14 scenes)
   → Extracting frames... ✓ (selected 23 frames)
   → Transcribing audio... ✓
   → Analyzing frames with llama3.2-vision...
     Frame 1/23 ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░
```

**3. Check the output**

```bash
ls ./output/
# → src/  package.json  README.md

cd output && npm install && npm run dev
```

### Tips

- **Long videos (>30 min):** Use `--max-frames 30` to analyze more frames
- **Poor extraction:** Try `--backend gemini` for better accuracy
- **Wrong file structure:** Check the generated README.md — it documents the layout

---

## Analyze a Bug Report Video

Got a screen recording of a bug? Extract the error, stack trace, and reproduction steps.

### Steps

**1. Run with bug detection mode**

```bash
videocode code bug-repro.mp4 --output-dir ./bug-analysis
```

**2. Review the output**

videocode will extract:
- Error messages visible on screen
- Stack traces from terminal/console
- Code shown during the recording
- UI state at the time of the bug

**3. Use with Claude Code MCP**

Even better, use the MCP server:

```
> Analyze this bug recording and find the issue: ./bug-repro.mp4
```

Claude Code will:
1. Call videocode to extract the error and code
2. Search your codebase for the relevant files
3. Identify the root cause
4. Suggest a fix

---

## Use with Claude Code MCP

The most powerful way to use videocode — integrated directly into Claude Code.

### Setup

**1. Install videocode**

```bash
pip install videocode
```

**2. Add to Claude Code config**

Open `~/.claude/settings.json` and add:

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

**3. Restart Claude Code**

The `videocode` tools will now be available.

### Example Workflows

**Extract and integrate code:**

```
> Extract the code from this tutorial into my project: https://youtube.com/...

Claude Code will:
1. Call videocode to extract the code
2. Read your existing project structure
3. Integrate the new code appropriately
4. Update dependencies if needed
```

**Find bugs from a recording:**

```
> I recorded a bug in my app: ./repro.mp4. What's going wrong?

Claude Code will:
1. Extract the error and relevant code from the video
2. Search your codebase
3. Trace the issue
4. Suggest a fix with a diff
```

**Learn from a tech talk:**

```
> Summarize this conference talk for me: https://youtube.com/...
> What were the key architectural decisions they discussed?
```

---

## Set Up Local Ollama Backend

Run everything locally — no data leaves your machine.

### Install Ollama

**macOS:**
```bash
brew install ollama
ollama serve
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

### Pull a Vision Model

```bash
# Good baseline — fast, decent accuracy
ollama pull llava

# Better for code — recommended default
ollama pull llama3.2-vision

# Excellent for UI screenshots and complex layouts
ollama pull minicpm-v
```

### Verify It Works

```bash
# Should list your models
ollama list

# Test with a simple image
videocode code tutorial.mp4 --backend ollama
```

### Performance Tips

- **GPU acceleration:** Ollama auto-detects CUDA (NVIDIA) and Metal (Apple Silicon)
- **Model size tradeoff:** llava is ~4GB, llama3.2-vision is ~8GB. More parameters = better accuracy but slower.
- **RAM:** You'll need ~2x the model size in RAM. 16GB recommended for llama3.2-vision.

---

## Switch to Cloud API Backend

Use cloud APIs when you need maximum accuracy.

### Gemini (Google)

**1. Get an API key**

Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create a key.

**2. Set the key**

```bash
export GEMINI_API_KEY="your-key-here"
```

Or add to your shell profile (`~/.bashrc`, `~/.zshrc`):
```bash
echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.bashrc
```

**3. Use Gemini**

```bash
videocode code tutorial.mp4 --backend gemini
```

Gemini has a generous free tier (1,500 requests/day as of early 2025).

### OpenAI

**1. Get an API key**

Visit [OpenAI Platform](https://platform.openai.com/api-keys).

**2. Set the key**

```bash
export OPENAI_API_KEY="your-key-here"
```

**3. Use OpenAI**

```bash
videocode code tutorial.mp4 --backend openai
```

### Make Cloud Your Default

```bash
# Add to ~/.config/videocode/config.yaml
echo "default_backend: gemini" >> ~/.config/videocode/config.yaml
```

### Which Cloud Backend?

| | Gemini | OpenAI |
|--|--------|--------|
| **Free tier** | 1,500 req/day | No free tier |
| **Price** | $0.0005/1K tokens | $0.005/1K tokens |
| **Speed** | Fast | Fast |
| **Code accuracy** | Excellent | Slightly better |
| **Small text** | Excellent | Excellent |

Choose Gemini for the free tier and great performance. Choose OpenAI if you need the absolute best accuracy and cost isn't a concern.

---

## Common Issues

### "No code found in video"

- Check that the video actually shows code (not just slides/talking)
- Try `--backend gemini` for better text detection
- Increase frames: `--max-frames 30`

### "Extracted code has errors"

- This is expected — the VLM captures what's on screen, which may include typos
- Use `--backend openai` for higher accuracy
- The generated README.md notes any detected issues

### "Slow processing"

- Use Ollama with GPU: check `ollama ps` shows GPU usage
- Reduce frames: `--max-frames 10`
- Skip transcription: `--no-transcription`

### "YouTube download fails"

- Update yt-dlp: `pip install -U yt-dlp`
- Some videos are restricted (age-gated, private) — download manually and use the file path
