"""VLM client with pluggable backends for Ollama, Gemini, OpenAI, and Qwen."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, cast

from videocode.types import Config, Frame, Transcription, VLMResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

# Maps backend name → (env var name, api key field name)
_API_KEY_ENVS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "qwen": "QWEN_API_KEY",
}


def _load_env_key(backend: str) -> str | None:
    """Load an API key from a standard environment variable for *backend*."""
    import os
    env_name = _API_KEY_ENVS.get(backend.lower())
    if env_name:
        key = os.environ.get(env_name)
        if key:
            logger.debug("Loaded %s for %s backend", env_name, backend)
            return key
    return None


# ---------------------------------------------------------------------------
# Base64 / image helpers
# ---------------------------------------------------------------------------


def _frame_to_base64(frame: Frame, max_size: tuple[int, int] = (1024, 1024)) -> str:
    """Convert a frame to a base64-encoded PNG.

    Resizes the image if either dimension exceeds *max_size* to keep token
    usage reasonable.
    """
    try:
        from PIL import Image
    except ImportError as err:  # pragma: no cover
        raise ImportError("Pillow is required for image encoding: pip install Pillow") from err

    img = Image.open(frame.path)  # type: ignore[assignment]
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")  # type: ignore[assignment]

    # Resize if too large
    w, h = img.size
    max_w, max_h = max_size
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_size = (int(w * scale), int(h * scale))
        resampling = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]
        img = img.resize(new_size, resampling)  # type: ignore[assignment]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Simple in-memory response cache
# ---------------------------------------------------------------------------


class _ResponseCache:
    """LRU-ish cache for VLM responses keyed by (frame_hashes + prompt)."""

    def __init__(self, max_size: int = 128) -> None:
        self._store: dict[str, VLMResponse] = {}
        self._order: list[str] = []
        self._max = max_size

    def _key(
        self,
        frames: list[Frame],
        prompt: str,
        transcription: Transcription | None = None,
    ) -> str:
        hasher = hashlib.sha256()
        for f in frames:
            hasher.update(str(f.path).encode())
            hasher.update(str(f.timestamp).encode())
        hasher.update(prompt.encode())
        if transcription:
            hasher.update(transcription.text.encode())
        return hasher.hexdigest()

    def get(
        self,
        frames: list[Frame],
        prompt: str,
        transcription: Transcription | None = None,
    ) -> VLMResponse | None:
        k = self._key(frames, prompt, transcription)
        return self._store.get(k)

    def put(
        self,
        frames: list[Frame],
        prompt: str,
        transcription: Transcription | None,
        response: VLMResponse,
    ) -> None:
        k = self._key(frames, prompt, transcription)
        if k in self._store:
            self._order.remove(k)
        self._order.append(k)
        self._store[k] = response
        # Evict oldest
        while len(self._order) > self._max:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class VLMBackend(ABC):
    """Abstract base for VLM backends."""

    @abstractmethod
    def chat(self, images_b64: list[str], prompt: str) -> VLMResponse:
        """Send images + prompt to the model and return the response."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the backend is reachable."""
        ...


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------


class OllamaBackend(VLMBackend):
    """Backend using Ollama's local HTTP API.

    Expects the ``ollama`` package to be installed.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Any = None
        self._timeout = 120

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.base_url)
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "ollama package required for Ollama backend: pip install ollama"
                ) from err
        return self._client

    def chat(self, images_b64: list[str], prompt: str) -> VLMResponse:
        client = self._get_client()
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ]
        response = client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.3},
        )
        content = response.message.content if hasattr(response, "message") else str(response)
        return VLMResponse(
            content=content,
            model=self.model,
            tokens_used=0,
            frames_analyzed=len(images_b64),
        )

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception:
            return False


class GeminiBackend(VLMBackend):
    """Backend using Google's Generative AI (Gemini) API.

    Expects the ``google-generativeai`` package.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model_name = model
        self._configure()

    def _configure(self) -> None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "google-generativeai required for Gemini backend: "
                "pip install google-generativeai"
            ) from err

    def _b64_to_pil(self, b64: str) -> Any:
        from PIL import Image

        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data))

    def chat(self, images_b64: list[str], prompt: str) -> VLMResponse:
        contents: list[Any] = [prompt]
        for b64 in images_b64:
            contents.append(self._b64_to_pil(b64))

        response = self._model.generate_content(
            contents,
            generation_config={"temperature": 0.3, "max_output_tokens": 4096},
        )
        text = response.text if hasattr(response, "text") else str(response)
        token_count = 0
        with contextlib.suppress(Exception):
            token_count = response.usage_metadata.total_token_count
        return VLMResponse(
            content=text,
            model=self.model_name,
            tokens_used=token_count,
            frames_analyzed=len(images_b64),
        )

    def is_available(self) -> bool:
        try:
            import google.generativeai as genai

            # A lightweight call to verify the key works
            return any("gemini" in m.name for m in genai.list_models())
        except Exception:
            return False


class OpenAIBackend(VLMBackend):
    """Backend using OpenAI's vision-capable chat completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "openai package required for OpenAI backend: pip install openai"
                ) from err
        return self._client

    def chat(self, images_b64: list[str], prompt: str) -> VLMResponse:
        client = self._get_client()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"},
                }
            )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=4096,
        )
        message = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return VLMResponse(
            content=message,
            model=self.model,
            tokens_used=tokens,
            frames_analyzed=len(images_b64),
        )

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False


class QwenBackend(VLMBackend):
    """Backend for Qwen2.5-VL using the OpenAI-compatible API.

    Most cloud providers that host Qwen expose an OpenAI-compatible chat
    completions endpoint.  The constructor accepts *api_key* and infers the
    base URL from ``Config.vlm_base_url`` when ``VLMClient`` instantiates it.
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "openai package required for Qwen backend: pip install openai"
                ) from err
        return self._client

    def chat(self, images_b64: list[str], prompt: str) -> VLMResponse:
        client = self._get_client()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=4096,
        )
        message = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return VLMResponse(
            content=message,
            model=self.model,
            tokens_used=tokens,
            frames_analyzed=len(images_b64),
        )

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False


class DummyBackend(VLMBackend):
    """Dummy backend for demos — simulates a VLM without any external service.

    Analyzes the *prompt* to infer what kind of response is expected and
    returns realistic-looking output.  Useful for testing the pipeline end-to-
    end when no real VLM (Ollama / API) is available.
    """

    # Keyword → response template mappings
    _CODE_DETECTION_PROMPT = "contain code"
    _CODE_EXTRACTION_PROMPT = "Extract all code"
    _PROJECT_ASSEMBLY_PROMPT = "organize these code blocks"
    _SUMMARY_PROMPT = "summarize"
    _BUG_PROMPT = "bug"

    def __init__(self) -> None:
        self._call_count = 0

    def chat(self, images_b64: list[str], prompt: str) -> VLMResponse:
        self._call_count += 1
        logger.info("DummyBackend: call #%d (prompt: %s...)", self._call_count, prompt[:80])

        p_lower = prompt.lower()

        # ── Code-frame detection ──────────────────────────────────────
        if self._CODE_DETECTION_PROMPT in p_lower:
            return self._detect_code_frames(len(images_b64))

        # ── Code extraction from a single frame ───────────────────────
        if self._CODE_EXTRACTION_PROMPT in p_lower:
            return self._extract_code_from_frame(prompt)

        # ── Project assembly ──────────────────────────────────────────
        if self._PROJECT_ASSEMBLY_PROMPT in p_lower:
            return self._assemble_project(prompt)

        # ── Summary ───────────────────────────────────────────────────
        if any(k in p_lower for k in ["summarize", "summary", "resum"]):
            return self._summarize(len(images_b64))

        # ── Bug finding ───────────────────────────────────────────────
        if any(k in p_lower for k in ["bug", "issue", "error", "crash"]):
            return self._find_bugs(len(images_b64))

        # ── Generic fallback ──────────────────────────────────────────
        return self._generic_response(len(images_b64), prompt)

    # -- Response generators ------------------------------------------

    def _detect_code_frames(self, n_frames: int) -> VLMResponse:
        """Return which frames contain code."""
        frames_with_code = list(range(n_frames))  # Assume all have code for demo
        items = [
            {"frame_index": i, "has_code": True, "language": "javascript"}
            for i in frames_with_code
        ]
        content = json.dumps({"frames": items}, indent=2)
        return VLMResponse(content=content, model="dummy", tokens_used=100, frames_analyzed=n_frames)

    def _extract_code_from_frame(self, prompt: str) -> VLMResponse:
        """Extract code from a single frame."""
        # Try to infer language from context
        lang = "javascript"
        if "python" in prompt.lower():
            lang = "python"
        elif "react" in prompt.lower() or "jsx" in prompt.lower():
            lang = "jsx"
        elif "css" in prompt.lower():
            lang = "css"

        code_blocks = {
            "javascript": [
                "function calculateSum(a, b) {",
                "  const result = a + b;",
                '  console.log(`Sum: ${result}`);',
                "  return result;",
                "}",
            ],
            "python": [
                "def calculate_sum(a, b):",
                "    result = a + b",
                "    print(f'Sum: {result}')",
                "    return result",
            ],
            "jsx": [
                "import React, { useState } from 'react';",
                "",
                "function Counter() {",
                "  const [count, setCount] = useState(0);",
                "",
                "  return (",
                '    <div className="counter">',
                "      <h1>Count: {count}</h1>",
                '      <button onClick={() => setCount(c => c + 1)}>',
                "        Increment",
                "      </button>",
                "    </div>",
                "  );",
                "}",
                "",
                "export default Counter;",
            ],
            "css": [
                ".counter {",
                "  display: flex;",
                "  flex-direction: column;",
                "  align-items: center;",
                "  padding: 2rem;",
                "  background: #f5f5f5;",
                "  border-radius: 8px;",
                "}",
                "",
                ".counter button {",
                "  padding: 0.5rem 1rem;",
                "  font-size: 1rem;",
                "  background: #007bff;",
                "  color: white;",
                "  border: none;",
                "  border-radius: 4px;",
                "  cursor: pointer;",
                "}",
            ],
        }

        code = "\n".join(code_blocks.get(lang, code_blocks["javascript"]))
        content = json.dumps({"code": code, "language": lang}, indent=2)
        return VLMResponse(content=content, model="dummy", tokens_used=200, frames_analyzed=1)

    def _assemble_project(self, prompt: str) -> VLMResponse:
        """Assemble extracted code blocks into a project."""
        project = {
            "files": {
                "src/App.jsx": "import React from 'react';\nimport Counter from './components/Counter';\n\nfunction App() {\n  return (\n    <div className=\"App\">\n      <Counter />\n    </div>\n  );\n}\n\nexport default App;",
                "src/components/Counter.jsx": "import React, { useState } from 'react';\nimport './Counter.css';\n\nfunction Counter() {\n  const [count, setCount] = useState(0);\n\n  return (\n    <div className=\"counter\">\n      <h1>Count: {count}</h1>\n      <button onClick={() => setCount(c => c + 1)}>\n        Increment\n      </button>\n    </div>\n  );\n}\n\nexport default Counter;",
                "src/components/Counter.css": ".counter {\n  display: flex;\n  flex-direction: column;\n  align-items: center;\n  padding: 2rem;\n  background: #f5f5f5;\n  border-radius: 8px;\n}\n\n.counter button {\n  padding: 0.5rem 1rem;\n  font-size: 1rem;\n  background: #007bff;\n  color: white;\n  border: none;\n  border-radius: 4px;\n  cursor: pointer;\n}",
                "src/index.js": "import React from 'react';\nimport ReactDOM from 'react-dom/client';\nimport App from './App';\n\nconst root = ReactDOM.createRoot(document.getElementById('root'));\nroot.render(<App />);",
                "package.json": '{\n  "name": "counter-app",\n  "version": "0.1.0",\n  "dependencies": {\n    "react": "^18.2.0",\n    "react-dom": "^18.2.0"\n  }\n}',
                "README.md": "# Counter App\n\nSimple React counter application.\n\n## Setup\n\n```bash\nnpm install\nnpm start\n```\n",
            },
            "language": "javascript",
            "description": "A simple React counter application with a single Counter component.",
            "setup_instructions": "1. Run `npm install` to install dependencies.\n2. Run `npm start` to start the development server.",
        }
        content = json.dumps(project, indent=2)
        return VLMResponse(content=content, model="dummy", tokens_used=500, frames_analyzed=3)

    def _summarize(self, n_frames: int) -> VLMResponse:
        summary = (
            "This video is a coding tutorial that demonstrates how to build "
            "a simple React counter application. The instructor covers:\n\n"
            "1. Setting up a React project\n"
            "2. Creating a Counter component with useState\n"
            "3. Adding CSS styling\n"
            "4. Exporting and using the component\n\n"
            "The tutorial is beginner-friendly and takes about 10 minutes."
        )
        return VLMResponse(content=summary, model="dummy", tokens_used=150, frames_analyzed=n_frames)

    def _find_bugs(self, n_frames: int) -> VLMResponse:
        bugs = [
            {"severity": "medium", "description": "Missing error handling in async operations", "timestamp": "0:15"},
            {"severity": "low", "description": "Hardcoded API endpoint URL", "timestamp": "0:32"},
        ]
        content = json.dumps({"bugs": bugs, "severity": "medium", "recommendations": ["Add try/catch blocks", "Use environment variables for URLs"]}, indent=2)
        return VLMResponse(content=content, model="dummy", tokens_used=200, frames_analyzed=n_frames)

    def _generic_response(self, n_frames: int, prompt: str) -> VLMResponse:
        content = (
            f"Analyzed {n_frames} frames from the video. "
            f"The content appears to be a coding tutorial. "
            f"Prompt was: {prompt[:100]}..."
        )
        return VLMResponse(content=content, model="dummy", tokens_used=50, frames_analyzed=n_frames)

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def _retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> Any:
    """Decorator that retries a function with exponential backoff."""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        break
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "VLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# VLMClient public API
# ---------------------------------------------------------------------------


class VLMClient:
    """Unified client for vision-language model backends.

    Usage::

        config = Config(vlm_backend="openai", vlm_api_key="sk-...")
        client = VLMClient(config)
        resp = client.analyze_single(frame, "Describe this image.")
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._backend = self._create_backend(config)
        self._cache = _ResponseCache()

    # ------------------------------------------------------------------
    # Backend factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_backend(config: Config) -> VLMBackend:
        backend = config.vlm_backend.lower()
        if backend == "ollama":
            return OllamaBackend(
                base_url=config.ollama_url,
                model=config.vlm_model,
            )
        if backend == "gemini":
            api_key = config.vlm_api_key or _load_env_key("gemini")
            if not api_key:
                raise ValueError(
                    "Gemini backend requires API key. Set GEMINI_API_KEY env var "
                    "or pass vlm_api_key in Config."
                )
            return GeminiBackend(
                api_key=api_key,
                model=config.vlm_model,
            )
        if backend == "openai":
            api_key = config.vlm_api_key or _load_env_key("openai")
            if not api_key:
                raise ValueError(
                    "OpenAI backend requires API key. Set OPENAI_API_KEY env var "
                    "or pass vlm_api_key in Config."
                )
            return OpenAIBackend(
                api_key=api_key,
                model=config.vlm_model,
            )
        if backend == "qwen":
            api_key = config.vlm_api_key or _load_env_key("qwen")
            if not api_key:
                raise ValueError(
                    "Qwen backend requires API key. Set QWEN_API_KEY env var "
                    "or pass vlm_api_key in Config."
                )
            return QwenBackend(
                api_key=api_key,
                model=config.vlm_model,
                base_url=config.vlm_base_url,
            )
        if backend == "dummy":
            return DummyBackend()
        raise ValueError(f"Unknown VLM backend: {config.vlm_backend}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frames(
        self,
        frames: list[Frame],
        prompt: str,
        transcription: Transcription | None = None,
    ) -> VLMResponse:
        """Send *frames* + *prompt* to the VLM and return a structured response.

        The call is retried with exponential backoff on failure and cached
        so that duplicate (frames, prompt) pairs return instantly.
        """
        # Check cache first
        cached = self._cache.get(frames, prompt, transcription)
        if cached is not None:
            logger.debug("VLM cache hit for %d frames", len(frames))
            return cached

        # Build the full prompt including transcript if provided
        full_prompt = prompt
        if transcription and transcription.text:
            full_prompt = (
                f"{prompt}\n\n"
                f"--- Video Transcription ---\n"
                f"{transcription.text}\n"
                f"--- End Transcription ---"
            )

        images_b64 = [_frame_to_base64(f) for f in frames]

        @_retry_with_backoff(max_retries=self.config.max_retries, base_delay=1.0)
        def _call() -> VLMResponse:
            return self._backend.chat(images_b64, full_prompt)

        response = cast(VLMResponse, _call())
        self._cache.put(frames, prompt, transcription, response)
        return response

    def analyze_single(self, frame: Frame, prompt: str) -> str:
        """Analyze a single *frame* with *prompt*.

        Returns the raw text content of the response.
        """
        resp = self.analyze_frames([frame], prompt)
        return resp.content

    def is_available(self) -> bool:
        """Check whether the configured backend is reachable."""
        return self._backend.is_available()
