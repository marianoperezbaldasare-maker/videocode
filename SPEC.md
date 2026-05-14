# SPEC: videocode — Give Claude Code Eyes

## Overview
videocode is an MCP server and CLI tool that enables Claude Code to understand and code from video content. It processes video files or YouTube URLs, extracts key frames using intelligent scene detection, transcribes audio with Whisper, analyzes visuals with a VLM (Ollama local or cloud API), and produces structured output: code files, summaries, or bug reports.

## Architecture

```
Input (video file / YouTube URL)
  └── Video Processor (FFmpeg + PySceneDetect)
         ├── Frame Selector (adaptive sampling)
         └── Audio Extractor (Whisper)
                └── VLM Client (Ollama / API)
                       └── Agent Loop (3-role pattern)
                              ├── Extractor → relevant frames
                              ├── Analyzer → code/text extraction
                              └── Verifier → consistency check
                                     └── Output (code / summary / bugs)
```

## Module Specs

### 1. `config.py` — Configuration
```python
@dataclass
class Config:
    # VLM Backend: "ollama" | "gemini" | "openai" | "qwen"
    vlm_backend: str = "ollama"
    vlm_model: str = "llava"  # or "gemini-1.5-pro", "gpt-4o", "qwen2.5-vl"
    vlm_api_key: Optional[str] = None
    vlm_base_url: Optional[str] = None
    
    # Ollama
    ollama_url: str = "http://localhost:11434"
    
    # Video Processing
    target_fps: float = 0.5  # frames per second (adaptive override possible)
    max_frames: int = 50  # hard limit per video
    frame_quality: int = 85  # JPEG quality
    frame_resolution: Tuple[int, int] = (1280, 720)
    scene_threshold: float = 30.0  # PySceneDetect sensitivity
    
    # Audio
    whisper_model: str = "base"  # "tiny" | "base" | "small" | "medium" | "large"
    extract_audio: bool = True
    
    # Agent
    enable_agent_loop: bool = True
    max_retries: int = 2
    temperature: float = 0.3
    
    # Output
    output_dir: str = "./output"
    output_format: str = "markdown"  # "markdown" | "json" | "files"
    
    @classmethod
    def from_env(cls) -> "Config": ...
    @classmethod
    def from_file(cls, path: str) -> "Config": ...
```

### 2. `video_processor.py` — Video Processing Engine
```python
class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.temp_dir = None
    
    def process(self, source: Union[str, Path]) -> ProcessedVideo:
        """
        Main entry point. Takes a video file path or YouTube URL.
        Returns a ProcessedVideo object with frames and metadata.
        """
    
    def extract_frames(self, video_path: Path, strategy: FrameStrategy) -> List[Frame]:
        """Extract frames using the specified strategy."""
    
    def detect_scenes(self, video_path: Path) -> List[Scene]:
        """Use PySceneDetect to find scene boundaries."""
    
    def cleanup(self) -> None:
        """Remove temporary files."""

@dataclass
class Frame:
    path: Path
    timestamp: float  # seconds
    scene_index: int
    is_keyframe: bool

@dataclass  
class Scene:
    start: float  # seconds
    end: float
    index: int

@dataclass
class ProcessedVideo:
    source: str
    duration: float
    fps: float
    resolution: Tuple[int, int]
    frames: List[Frame]
    frame_dir: Path
    audio_path: Optional[Path]
```

### 3. `frame_selector.py` — Intelligent Frame Selection
```python
class FrameSelector:
    def __init__(self, config: Config):
        self.config = config
    
    def select_frames(self, video_path: Path, strategy: SelectionStrategy = SelectionStrategy.AUTO) -> List[Frame]:
        """
        Select frames using the specified strategy:
        - AUTO: scene-based for scene-rich videos, uniform for others
        - SCENE: one keyframe per detected scene
        - UNIFORM: fixed FPS sampling
        - KEYFRAME: FFmpeg keyframe extraction only
        Returns a list of Frame objects, respecting max_frames and token budget.
        """
    
    def calculate_token_budget(self, frame_count: int) -> int:
        """Estimate token usage for the selected frames."""
    
    def estimate_optimal_frame_count(self, duration: float, resolution: Tuple[int, int]) -> int:
        """Calculate how many frames fit in context window."""

class SelectionStrategy(Enum):
    AUTO = "auto"
    SCENE = "scene"
    UNIFORM = "uniform"
    KEYFRAME = "keyframe"
```

### 4. `audio_extractor.py` — Audio Processing
```python
class AudioExtractor:
    def __init__(self, config: Config):
        self.config = config
    
    def extract(self, video_path: Path) -> Path:
        """Extract audio from video using FFmpeg. Returns WAV path."""
    
    def transcribe(self, audio_path: Path) -> Transcription:
        """Transcribe audio using Whisper. Returns transcript with timestamps."""
    
    def cleanup(self) -> None:
        """Remove temporary audio files."""

@dataclass
class Transcription:
    text: str
    segments: List[Segment]
    language: str

@dataclass
class Segment:
    start: float
    end: float
    text: str
```

### 5. `vlm_client.py` — Vision Language Model Client
```python
class VLMClient:
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_frames(self, frames: List[Frame], prompt: str, transcription: Optional[Transcription] = None) -> VLMResponse:
        """
        Send frames to the configured VLM with the given prompt.
        Returns structured analysis.
        """
    
    def analyze_single(self, frame: Frame, prompt: str) -> str:
        """Analyze a single frame."""
    
    def is_available(self) -> bool:
        """Check if the VLM backend is reachable."""

@dataclass
class VLMResponse:
    content: str
    model: str
    tokens_used: int
    frames_analyzed: int

# Backends
class OllamaBackend:
    def __init__(self, base_url: str, model: str): ...
    
class GeminiBackend:
    def __init__(self, api_key: str, model: str): ...
    
class OpenAIBackend:
    def __init__(self, api_key: str, model: str): ...
```

### 6. `agent_loop.py` — 3-Role Agent Loop
```python
class AgentLoop:
    def __init__(self, vlm: VLMClient, config: Config):
        self.vlm = vlm
        self.config = config
    
    def run(self, task: Task, video: ProcessedVideo, transcription: Optional[Transcription] = None) -> AgentResult:
        """
        Execute the 3-role agent loop:
        1. Extractor: Identify relevant frames for the task
        2. Analyzer: Extract information from those frames + transcript
        3. Verifier: Check consistency and assemble final output
        """

@dataclass
class Task:
    type: TaskType  # CODE_EXTRACTION | SUMMARIZATION | BUG_FINDING | GENERAL
    query: str  # User's specific question or instruction

@dataclass
class AgentResult:
    content: str
    confidence: float
    sources: List[SourceReference]
    retries: int

@dataclass
class SourceReference:
    timestamp: float
    frame_path: Path
    description: str
```

### 7. `code_extractor.py` — Tutorial-to-Code Pipeline
```python
class CodeExtractor:
    def __init__(self, agent: AgentLoop, config: Config):
        self.agent = agent
        self.config = config
    
    def extract(self, video: ProcessedVideo, transcription: Transcription) -> CodeResult:
        """
        Convert a coding tutorial video into structured code files.
        Pipeline:
        1. Detect code frames (screenshots showing code)
        2. Extract code from each frame using VLM + OCR
        3. Cross-reference with audio transcript for context
        4. Assemble into files with proper structure
        5. Generate README with setup instructions
        """
    
    def detect_code_frames(self, frames: List[Frame]) -> List[CodeFrame]:
        """Identify which frames contain code."""
    
    def extract_code_from_frame(self, frame: Frame) -> str:
        """Use VLM to extract code text from a frame."""
    
    def assemble_project(self, code_blocks: List[CodeBlock], context: str) -> CodeResult:
        """Assemble extracted code into a project structure."""

@dataclass
class CodeFrame:
    frame: Frame
    code: str
    language: str
    confidence: float

@dataclass
class CodeBlock:
    filename: str
    content: str
    language: str
    source_timestamps: List[float]

@dataclass
class CodeResult:
    files: Dict[str, str]  # filename -> content
    language: str
    description: str
    setup_instructions: str
    confidence: float
```

### 8. `mcp_server.py` — MCP Server Mode
```python
class ClaudeVisionMCPServer:
    def __init__(self, config: Config):
        self.config = config
        self.video_processor = VideoProcessor(config)
        self.frame_selector = FrameSelector(config)
        self.audio_extractor = AudioExtractor(config)
        self.vlm = VLMClient(config)
        self.agent = AgentLoop(self.vlm, config)
        self.code_extractor = CodeExtractor(self.agent, config)
    
    # MCP Tools (exposed to Claude Code)
    async def video_analyze(self, source: str, query: str = "") -> dict:
        """Analyze a video and answer questions about it."""
    
    async def video_extract_code(self, source: str) -> dict:
        """Extract code from a coding tutorial video. Returns file structure."""
    
    async def video_summarize(self, source: str, style: str = "detailed") -> dict:
        """Generate a summary of a video."""
    
    async def video_find_bugs(self, source: str, description: str = "") -> dict:
        """Analyze a screen recording for bugs/issues."""
    
    async def health_check(self) -> dict:
        """Check if all backends are available."""
    
    def run(self) -> None:
        """Start the MCP server with stdio transport."""

# Tool schemas for MCP
VIDEO_ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Path to video file or YouTube URL"},
        "query": {"type": "string", "description": "Specific question about the video"}
    },
    "required": ["source"]
}

VIDEO_EXTRACT_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Path to coding tutorial video or YouTube URL"}
    },
    "required": ["source"]
}
```

### 9. `cli.py` — Command Line Interface
```python
app = typer.Typer(name="videocode", help="Give Claude Code eyes")

@app.command()
def process(
    source: str = typer.Argument(..., help="Video file path or YouTube URL"),
    query: str = typer.Option("", "--query", "-q", help="Question or task for the video"),
    backend: str = typer.Option("ollama", "--backend", "-b", help="VLM backend"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio extraction"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Process a video and extract insights."""

@app.command()
def code(
    source: str = typer.Argument(..., help="Coding tutorial video"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    backend: str = typer.Option("ollama", "--backend", "-b", help="VLM backend"),
):
    """Extract code from a coding tutorial video."""

@app.command()
def summarize(
    source: str = typer.Argument(..., help="Video file path or YouTube URL"),
    style: str = typer.Option("detailed", "--style", "-s", help="Summary style"),
    output: str = typer.Option("./output", "--output", "-o"),
):
    """Summarize a video."""

@app.command()
def mcp(
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Run in MCP server mode for Claude Code integration."""

@app.command()
def status():
    """Check backend status (Ollama, Whisper, FFmpeg)."""

if __name__ == "__main__":
    app()
```

## Data Flow

### Tutorial-to-Code Flow
```
User: videocode code tutorial.mp4
  → CLI validates source
  → VideoProcessor.process() 
    → FrameSelector.select_frames(SCENE) → 12 keyframes
    → AudioExtractor.transcribe() → transcript with timestamps
  → CodeExtractor.extract()
    → detect_code_frames() → 8 frames have code
    → extract_code_from_frame() × 8 → code blocks
    → assemble_project() → project structure
  → Output: ./output/src/... , ./output/README.md
```

### MCP Server Flow
```
Claude Code: calls video_extract_code(source="bug_repro.mp4")
  → MCP Server receives call
  → Same pipeline as CLI
  → Returns JSON with file structure
  → Claude Code creates files in workspace
```

## Dependencies
```
# Core
typer>=0.12.0
rich>=13.0
pydantic>=2.0
pillow>=10.0

# Video
opencv-python>=4.9.0
scenedetect>=0.6.0

# Audio
openai-whisper>=20231117

# VLM
ollama>=0.1.0
openai>=1.0
google-generativeai>=0.5

# MCP
mcp>=1.0.0

# Utils
aiohttp>=3.9
httpx>=0.27
python-dotenv>=1.0

# Dev
pytest>=8.0
pytest-asyncio>=0.23
ruff>=0.4.0
mypy>=1.9
```

## Testing Strategy
- Unit tests for each module
- Integration test: process a 30-sec test video end-to-end
- MCP protocol test: verify tool schemas

## Packaging
- Python package: `videocode`
- CLI entry point: `videocode`
- MCP entry point: `videocode mcp`
- Published to PyPI
