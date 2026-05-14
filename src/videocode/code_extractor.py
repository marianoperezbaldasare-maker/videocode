"""Tutorial-to-code extraction pipeline.

Converts coding-tutorial videos into structured code files using the 3-role
agent loop for frame-level detection and code extraction.
"""

from __future__ import annotations

import json
import logging
import re

from videocode.agent_loop import AgentLoop
from videocode.types import (
    CodeBlock,
    CodeFrame,
    CodeResult,
    Config,
    Frame,
    ProcessedVideo,
    Task,
    TaskType,
    Transcription,
)

logger = logging.getLogger(__name__)

# Optional integrations
try:
    from videocode.perplexity_client import PerplexityClient, create_perplexity_client_from_env
    _HAS_PERPLEXITY = True
except ImportError:
    _HAS_PERPLEXITY = False

try:
    from videocode.apify_client import create_apify_client_from_env  # noqa: F401
    _HAS_APIFY = True
except ImportError:
    _HAS_APIFY = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DETECT_CODE_PROMPT = (
    "You are an expert at identifying code in video frames. "
    "Look at each frame and determine whether it contains code, a code editor, "
    "a terminal, or any programming-related content.\n\n"
    "For each frame that contains code, respond with a JSON array of objects:\n"
    '[{"index": 0, "language": "python", "has_code": true, "description": "shows a function definition"}, '
    '{"index": 3, "language": "javascript", "has_code": true, "description": "shows React component"}]\n\n'
    "Only include frames that clearly show code. Use standard language names "
    "(python, javascript, typescript, rust, go, java, cpp, etc.). "
    "If no frames contain code, return an empty array []."
)

_EXTRACT_CODE_PROMPT = (
    "You are a code extraction expert. Look at this frame from a programming tutorial "
    "and extract ALL code visible in the image.\n\n"
    "Rules:\n"
    "1. Extract the code EXACTLY as shown — preserve indentation, spelling, and comments\n"
    "2. If the code is partially visible, extract what you can see and note [truncated]\n"
    "3. Output ONLY the code, wrapped in a markdown code block with the language\n"
    "4. Do NOT add explanations or descriptions outside the code block\n\n"
    "Format your response like this:\n"
    "```python\n"
    "def hello():\n"
    "    print('world')\n"
    "```\n\n"
    "If you cannot see any code in this frame, respond with: NO_CODE_FOUND"
)

_ASSEMBLE_PROJECT_PROMPT = (
    "You are a senior software architect. You have extracted code blocks from a programming tutorial video. "
    "Your task is to organize them into a coherent project structure.\n\n"
    "Context from video transcription:\n<<CONTEXT>>\n\n"
    "Extracted code blocks:\n<<BLOCKS>>\n\n"
    "For each code block, determine:\n"
    "1. The correct filename (with extension)\n"
    "2. Any dependencies or imports needed\n"
    "3. The correct order/placement in the project\n\n"
    "Respond with a JSON object in this exact format:\n"
    "```json\n"
    "{\n"
    '  "files": {\n'
    '    "main.py": "import os\\n\\ndef main():...",\n'
    '    "README.md": "# Project\\n\\nDescription..."\n'
    "  },\n"
    '  "language": "python",\n'
    '  "description": "A brief project description",\n'
    '  "setup_instructions": "pip install -r requirements\\npython main.py",\n'
    '  "dependencies": ["requests", "numpy"]\n'
    "}\n"
    "```\n\n"
    "Ensure:\n"
    "- Each file has complete, working code\n"
    "- Include a README.md with setup and usage instructions\n"
    "- Group related code into the same file\n"
    "- The project should be runnable"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_code_blocks(text: str) -> list[str]:
    """Extract all markdown code blocks from *text*."""
    pattern = r"```(?:\w+)?\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL)


def _extract_json_block(text: str) -> str:
    """Extract a JSON object or array, with or without markdown fences."""
    # Try JSON in code fence (object or array)
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try bare JSON array first (starts with [)
    stripped = text.strip()
    if stripped.startswith("["):
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            return match.group(1)
    # Try bare JSON object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _guess_language_from_code(code: str) -> str:
    """Make a best-effort guess at the programming language from code content."""
    patterns = {
        "python": [
            r"^\s*def\s+\w+\s*\(",
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import",
            r"^\s*class\s+\w+.*:\s*$",
            r"print\s*\(",
        ],
        "javascript": [
            r"^\s*const\s+\w+\s*=",
            r"^\s*let\s+\w+\s*=",
            r"^\s*var\s+\w+\s*=",
            r"function\s+\w+\s*\(",
            r"=>\s*\{",
            r"require\s*\(",
        ],
        "typescript": [
            r":\s*(string|number|boolean|any|void)\s*[=;)]",
            r"interface\s+\w+",
            r"type\s+\w+\s*=",
        ],
        "rust": [
            r"^\s*fn\s+\w+\s*\(",
            r"^\s*let\s+mut\s",
            r"^\s*use\s+\w+::",
            r"^\s*mod\s+\w+",
        ],
        "go": [
            r"^\s*func\s+\w+\s*\(",
            r"^\s*package\s+\w+",
            r"^\s*import\s+\(",
        ],
        "java": [
            r"^\s*public\s+class\s",
            r"^\s*private\s+\w+",
            r"^\s*import\s+java\.",
            r"System\.out\.println",
        ],
        "cpp": [
            r"#include\s*[<\"]",
            r"std::",
            r"int\s+main\s*\(",
        ],
        "html": [
            r"<!DOCTYPE\s+html",
            r"<html>",
            r"<div\s+",
        ],
        "css": [
            r"^\s*\.\w+\s*\{",
            r"^\s*#\w+\s*\{",
            r":\s*(flex|grid|block|none)",
        ],
    }
    scores: dict[str, int] = dict.fromkeys(patterns, 0)
    for lang, regexes in patterns.items():
        for pat in regexes:
            if re.search(pat, code, re.MULTILINE):
                scores[lang] += 1
    if scores:
        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best
    return ""


# ---------------------------------------------------------------------------
# CodeExtractor
# ---------------------------------------------------------------------------


class CodeExtractor:
    """Extract code from a coding-tutorial video and assemble a project.

    The pipeline is:
    1. **detect_code_frames** — ask the VLM which frames show code.
    2. **extract_code_from_frame** — pull raw code out of each code frame.
    3. **assemble_project** — organise blocks into files and generate a README.
    """

    def __init__(self, agent: AgentLoop, config: Config) -> None:
        self.agent = agent
        self.config = config
        self.perplexity = self._init_perplexity()

    def _init_perplexity(self) -> PerplexityClient | None:
        """Initialize Perplexity client if API key is available."""
        if not _HAS_PERPLEXITY:
            return None
        client = create_perplexity_client_from_env()
        if client:
            logger.info("Perplexity integration enabled")
        return client

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def extract(self, video: ProcessedVideo, transcription: Transcription) -> CodeResult:
        """Run the full tutorial-to-code pipeline.

        Parameters
        ----------
        video:
            The processed video with extracted frames.
        transcription:
            The audio transcription for context.

        Returns
        -------
        CodeResult
            The assembled project with files, language info, and setup
            instructions.  On failure an empty result with zero confidence
            is returned.
        """
        logger.info("CodeExtractor: starting extraction from %s", video.source)

        # 1. Detect code frames
        code_frames = self.detect_code_frames(video.frames)
        if not code_frames:
            logger.warning("CodeExtractor: no code frames detected")
            return CodeResult(
                files={},
                language="",
                description="No code frames detected in the video.",
                setup_instructions="",
                confidence=0.0,
            )

        # 2. Extract code from each frame
        code_blocks: list[CodeBlock] = []
        for cf in code_frames:
            try:
                raw = self.extract_code_from_frame(cf.frame)
                if raw and "NO_CODE_FOUND" not in raw:
                    lang = cf.language or _guess_language_from_code(raw)
                    blocks = _extract_code_blocks(raw)
                    if blocks:
                        for b in blocks:
                            code_blocks.append(
                                CodeBlock(
                                    filename="",  # filled in during assembly
                                    content=b,
                                    language=lang,
                                    source_timestamps=[cf.frame.timestamp],
                                )
                            )
                    else:
                        # No markdown fence — treat whole text as code
                        code_blocks.append(
                            CodeBlock(
                                filename="",
                                content=raw,
                                language=lang,
                                source_timestamps=[cf.frame.timestamp],
                            )
                        )
            except Exception as exc:
                logger.error("Failed to extract code from frame %s: %s", cf.frame.path, exc)

        logger.info("CodeExtractor: extracted %d code blocks", len(code_blocks))

        if not code_blocks:
            return CodeResult(
                files={},
                language="",
                description="Code frames found but no code could be extracted.",
                setup_instructions="",
                confidence=0.0,
            )

        # 3. Assemble project
        context = transcription.text if transcription else ""
        return self.assemble_project(code_blocks, context)

    # ------------------------------------------------------------------
    # Step 1: detect code frames
    # ------------------------------------------------------------------

    def detect_code_frames(self, frames: list[Frame]) -> list[CodeFrame]:
        """Ask the VLM which *frames* contain code.

        Returns a list of :class:`CodeFrame` objects with inferred language
        and confidence.
        """
        if not frames:
            return []

        Task(type=TaskType.CODE_EXTRACTION, query="detect code frames")
        resp = self.agent.vlm.analyze_frames(frames, _DETECT_CODE_PROMPT)

        return self._parse_code_frame_response(resp.content, frames)

    def _parse_code_frame_response(self, content: str, frames: list[Frame]) -> list[CodeFrame]:
        """Parse the JSON array returned by the VLM."""
        code_frames: list[CodeFrame] = []
        try:
            json_str = _extract_json_block(content)
            # If it's wrapped in ```json … ``` the regex above already extracted
            # just the JSON.  But the response might be an array directly.
            data = json.loads(json_str)
            # The response may be an array or wrapped in an object
            items: list[dict] = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("relevant_frames", data.get("frames", []))

            for item in items:
                idx = item.get("index", -1)
                if 0 <= idx < len(frames) and item.get("has_code", True):
                    code_frames.append(
                        CodeFrame(
                            frame=frames[idx],
                            language=item.get("language", ""),
                            confidence=item.get("confidence", 0.8),
                        )
                    )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Code frame detection returned invalid JSON (%s): %s", exc, content[:200])
            # Fallback: assume all frames might contain code
            for f in frames:
                code_frames.append(CodeFrame(frame=f, language="", confidence=0.3))

        return code_frames

    # ------------------------------------------------------------------
    # Step 2: extract code from a single frame
    # ------------------------------------------------------------------

    def extract_code_from_frame(self, frame: Frame) -> str:
        """Use the VLM to extract code text from a single *frame*.

        Returns the raw response text which may contain markdown code blocks.
        """
        return self.agent.vlm.analyze_single(frame, _EXTRACT_CODE_PROMPT)

    # ------------------------------------------------------------------
    # Step 3: assemble project
    # ------------------------------------------------------------------

    def assemble_project(self, code_blocks: list[CodeBlock], context: str) -> CodeResult:
        """Organise *code_blocks* into files and generate a README.

        Uses the VLM to infer filenames, project structure, and setup
        instructions.
        """
        if not code_blocks:
            return CodeResult(
                files={},
                language="",
                description="No code blocks to assemble.",
                setup_instructions="",
                confidence=0.0,
            )

        # Build a compact representation of the blocks
        block_texts: list[str] = []
        for i, block in enumerate(code_blocks):
            lang_tag = block.language or ""
            block_texts.append(
                f"--- Block {i} ({lang_tag}) ---\n"
                f"Source: {block.source_timestamps}\n"
                f"```\n{block.content[:1000]}\n```"
            )

        prompt = (
            _ASSEMBLE_PROJECT_PROMPT
            .replace("<<CONTEXT>>", context or "No transcription available.")
            .replace("<<BLOCKS>>", "\n\n".join(block_texts))
        )

        # Use the VLM to assemble (text-only, no frames needed)
        resp = self.agent.vlm.analyze_frames([], prompt)
        result = self._parse_assembly_response(resp.content, code_blocks)

        # Verify with Perplexity if available
        if self.perplexity and result.files:
            result = self._verify_with_perplexity(result)

        return result

    def _verify_with_perplexity(self, result: CodeResult) -> CodeResult:
        """Verify extracted code with Perplexity and improve setup instructions."""
        try:
            logger.info("Verifying code with Perplexity...")

            # Verify each significant file
            verified_files = dict(result.files)
            for fname, content in list(verified_files.items()):
                if len(content) < 50 or fname.endswith((".json", ".txt", ".md")):
                    continue

                verification = self.perplexity.verify_code(content, result.language)
                if verification.fixed_code and not verification.is_valid:
                    logger.info("Perplexity fixed issues in %s", fname)
                    verified_files[fname] = verification.fixed_code

            # Generate better setup instructions
            if not result.setup_instructions or len(result.setup_instructions) < 100:
                deps = self._extract_dependencies(verified_files, result.language)
                if deps:
                    setup = self.perplexity.generate_setup_instructions(
                        result.language, deps
                    )
                    return CodeResult(
                        files=verified_files,
                        language=result.language,
                        description=result.description,
                        setup_instructions=setup,
                        confidence=min(result.confidence + 0.1, 1.0),
                    )

            return CodeResult(
                files=verified_files,
                language=result.language,
                description=result.description,
                setup_instructions=result.setup_instructions,
                confidence=min(result.confidence + 0.05, 1.0),
            )
        except Exception as e:
            logger.warning("Perplexity verification failed: %s", e)
            return result

    def _extract_dependencies(self, files: dict[str, str], language: str) -> list[str]:
        """Extract dependency names from source files."""
        deps = set()
        if language == "python":
            for content in files.values():
                for match in re.finditer(r"^(?:import|from)\s+(\w+)", content, re.M):
                    deps.add(match.group(1))
        elif language in ("javascript", "typescript"):
            for content in files.values():
                for match in re.finditer(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", content):
                    dep = match.group(1)
                    if not dep.startswith((".", "/")):
                        deps.add(dep.split("/")[0])
        return sorted(deps)

    def _parse_assembly_response(self, content: str, code_blocks: list[CodeBlock]) -> CodeResult:
        """Parse the JSON project structure from the VLM response."""
        try:
            json_str = _extract_json_block(content)
            data = json.loads(json_str)

            files: dict[str, str] = dict(data.get("files", {}))
            language = data.get("language", code_blocks[0].language if code_blocks else "")
            description = data.get("description", "")
            setup = data.get("setup_instructions", "")
            deps = data.get("dependencies", [])

            if deps and "requirements.txt" not in files and "package.json" not in files and language == "python":
                files["requirements.txt"] = "\n".join(deps)

            confidence = 0.85 if len(files) > 1 else 0.6

            return CodeResult(
                files=files,
                language=language,
                description=description,
                setup_instructions=setup,
                confidence=confidence,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Assembly response parse failed (%s): %s", exc, content[:200])

            # Fallback: dump all blocks into a single file
            fallback_files: dict[str, str] = {}
            for i, block in enumerate(code_blocks):
                ext = block.language or "txt"
                fname = f"extracted_{i}.{ext}"
                fallback_files[fname] = block.content

            return CodeResult(
                files=fallback_files,
                language=code_blocks[0].language if code_blocks else "",
                description="Auto-assembled (parser fallback).",
                setup_instructions="",
                confidence=0.3,
            )
