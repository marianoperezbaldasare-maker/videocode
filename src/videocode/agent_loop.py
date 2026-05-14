"""3-role agent loop: Extractor -> Analyzer -> Verifier."""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from videocode.types import (
    AgentResult,
    Config,
    Frame,
    ProcessedVideo,
    SourceReference,
    Task,
    TaskType,
    Transcription,
)
from videocode.vlm_client import VLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXTRACTOR_PROMPTS: dict[TaskType, str] = {
    TaskType.CODE_EXTRACTION: (
        "You are an expert video analyst. Your job is to identify which frames "
        "from this video contain code, programming tutorials, or terminal/IDE content.\n\n"
        "Look at the provided frames and respond with a JSON object in this exact format:\n"
        '{"relevant_frames": [{"index": 0, "reason": "shows Python code"}, {"index": 5, "reason": "shows terminal output"}]}\n\n'
        "Only include frames that clearly show code, code editors, terminals, or programming-related content. "
        "If no frames contain code, return an empty list: {\"relevant_frames\": []}"
    ),
    TaskType.BUG_FINDING: (
        "You are a QA analyst. Identify which frames show errors, bugs, crashes, "
        "or unexpected behavior in an application.\n\n"
        "Respond with JSON:\n"
        '{"relevant_frames": [{"index": 0, "reason": "shows error message"}]}\n\n'
        "Only include frames with clear error indicators: red error text, stack traces, "
        "crash dialogs, assertion failures, or visual glitches."
    ),
    TaskType.SUMMARIZATION: (
        "You are a video summarizer. For a general summary, all frames are relevant. "
        "However, identify the most informative frames that represent key moments.\n\n"
        "Respond with JSON:\n"
        '{"relevant_frames": [{"index": 0, "reason": "title slide"}, {"index": 10, "reason": "main content"}]}\n\n'
        "Select up to the most diverse and informative frames."
    ),
    TaskType.GENERAL: (
        "You are a video analyst. Given the user's query, identify which frames "
        "are most relevant to answering it.\n\n"
        "Respond with JSON:\n"
        '{"relevant_frames": [{"index": 0, "reason": "relevant to query"}]}\n\n'
        "User query: <<QUERY>>"
    ),
}

_ANALYZER_PROMPTS: dict[TaskType, str] = {
    TaskType.CODE_EXTRACTION: (
        "You are an expert programmer watching a coding tutorial. "
        "Extract ALL code shown in these frames. For each frame:\n"
        "1. Identify the programming language\n"
        "2. Extract the complete code block\n"
        "3. Note any file names or paths mentioned\n\n"
        "Use the transcription for additional context about what the code does.\n\n"
        "Transcription:\n<<TRANSCRIPTION>>\n\n"
        "Respond in this format:\n"
        "## Frame at <timestamp>s\n"
        "**Language:** <language>\n"
        "**Filename:** <filename>\n"
        "```\n<code>\n```\n\n"
        "Extract ALL code completely and accurately."
    ),
    TaskType.BUG_FINDING: (
        "You are a bug analyst reviewing a screen recording. "
        "Analyze these frames and describe any bugs, errors, or issues you find.\n\n"
        "Transcription:\n<<TRANSCRIPTION>>\n\n"
        "For each issue found, provide:\n"
        "1. A clear description of the bug\n"
        "2. The timestamp where it appears\n"
        "3. Severity (critical / major / minor)\n"
        "4. Steps to reproduce if discernible\n"
        "5. A suggested fix if possible\n\n"
        "Format each bug as a markdown section."
    ),
    TaskType.SUMMARIZATION: (
        "You are a video summarizer. Create a comprehensive summary of this video "
        "based on the frames and transcription provided.\n\n"
        "Transcription:\n<<TRANSCRIPTION>>\n\n"
        "Provide:\n"
        "1. A concise overview (2-3 sentences)\n"
        "2. Key topics covered (bullet list)\n"
        "3. Important details or timestamps\n"
        "4. Any actionable takeaways\n\n"
        "Write in clear, well-structured markdown."
    ),
    TaskType.GENERAL: (
        "You are an AI assistant with video understanding capabilities.\n\n"
        "The user asks: <<QUERY>>\n\n"
        "Transcription:\n<<TRANSCRIPTION>>\n\n"
        "Analyze the provided frames and transcript, then answer the user's question "
        "as thoroughly and accurately as possible. Cite specific timestamps when referencing "
        "visual content."
    ),
}

_VERIFIER_PROMPT = (
    "You are a quality verifier. Review the following analysis for consistency, "
    "completeness, and accuracy.\n\n"
    "Original task: <<TASK_TYPE>>\n"
    "User query: <<QUERY>>\n\n"
    "Analysis to verify:\n<<ANALYSIS>>\n\n"
    "Evaluate the analysis and respond with a JSON object:\n"
    '{"confidence": 0.85, "issues": ["issue1", "issue2"], "improved_analysis": "..."}\n\n'
    "confidence should be a float between 0 and 1.\n"
    "If there are no issues, return an empty issues list.\n"
    "The improved_analysis field should contain the corrected/enhanced version of the analysis."
)


# ---------------------------------------------------------------------------
# Helper: safely parse JSON from VLM response
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Extract a JSON object from a possibly-markdown-wrapped string."""
    # Try to find JSON between triple backticks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try to find bare JSON object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


# ---------------------------------------------------------------------------
# Agent loop components
# ---------------------------------------------------------------------------


class Extractor:
    """Role 1: Identifies relevant frames for the given task."""

    def __init__(self, vlm: VLMClient, config: Config) -> None:
        self.vlm = vlm
        self.config = config

    def run(self, task: Task, video: ProcessedVideo) -> List[Frame]:
        """Return the subset of *video.frames* relevant to *task*."""
        if not video.frames:
            return []

        prompt_template = _EXTRACTOR_PROMPTS.get(task.type, _EXTRACTOR_PROMPTS[TaskType.GENERAL])
        prompt = prompt_template.replace("<<QUERY>>", task.query)

        logger.info("Extractor: analyzing %d frames for task %s", len(video.frames), task.type.value)

        resp = self.vlm.analyze_frames(video.frames, prompt)
        relevant = self._parse_frame_selection(resp.content, video.frames)

        logger.info("Extractor: selected %d/%d frames", len(relevant), len(video.frames))
        return relevant

    def _parse_frame_selection(self, content: str, all_frames: List[Frame]) -> List[Frame]:
        """Parse the extractor JSON response and return matching frames."""
        try:
            json_str = _extract_json(content)
            data = json.loads(json_str)
            indices = {item["index"] for item in data.get("relevant_frames", [])}
            return [f for i, f in enumerate(all_frames) if i in indices]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Extractor returned invalid JSON (%s), using all frames", exc)
            return all_frames


class Analyzer:
    """Role 2: Extracts detailed information from relevant frames."""

    def __init__(self, vlm: VLMClient, config: Config) -> None:
        self.vlm = vlm
        self.config = config

    def run(
        self,
        task: Task,
        frames: List[Frame],
        transcription: Optional[Transcription],
    ) -> str:
        """Analyze *frames* and return raw analysis text."""
        prompt_template = _ANALYZER_PROMPTS.get(task.type, _ANALYZER_PROMPTS[TaskType.GENERAL])

        tx = transcription.text if transcription else "No transcription available."
        prompt = prompt_template.replace("<<QUERY>>", task.query).replace("<<TRANSCRIPTION>>", tx)

        logger.info("Analyzer: analyzing %d frames", len(frames))
        resp = self.vlm.analyze_frames(frames, prompt, transcription)
        return resp.content


class Verifier:
    """Role 3: Checks analysis quality and improves if needed."""

    def __init__(self, vlm: VLMClient, config: Config) -> None:
        self.vlm = vlm
        self.config = config

    def run(self, task: Task, analysis: str) -> tuple[str, float]:
        """Verify *analysis* and return (improved_analysis, confidence).

        If confidence is below the threshold, a retry is triggered by the
        caller (``AgentLoop``).
        """
        prompt = (
            _VERIFIER_PROMPT
            .replace("<<TASK_TYPE>>", task.type.value)
            .replace("<<QUERY>>", task.query)
            .replace("<<ANALYSIS>>", analysis)
        )

        logger.info("Verifier: checking analysis quality")
        # Use an empty frame list — verification is text-only
        resp = self.vlm.analyze_frames([], prompt)

        confidence, improved = self._parse_verification(resp.content, analysis)
        logger.info("Verifier: confidence=%.2f", confidence)
        return improved, confidence

    def _parse_verification(self, content: str, fallback: str) -> tuple[float, str]:
        """Parse verifier JSON response."""
        try:
            json_str = _extract_json(content)
            data = json.loads(json_str)
            confidence = float(data.get("confidence", 0.5))
            improved = data.get("improved_analysis", fallback)
            return max(0.0, min(1.0, confidence)), improved
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Verifier returned invalid JSON (%s): %s", exc, content[:200])
            return 0.5, fallback


# ---------------------------------------------------------------------------
# Agent loop orchestrator
# ---------------------------------------------------------------------------


class AgentLoop:
    """3-role agent pipeline: Extractor -> Analyzer -> Verifier.

    The pipeline is:
    1. **Extractor** narrows down the frame set to those relevant to the task.
    2. **Analyzer** performs the actual information extraction from those frames.
    3. **Verifier** checks consistency / completeness and can trigger a re-run.

    If the verifier's confidence is below ``0.7`` the analyzer re-runs (up to
    ``config.max_retries`` times).
    """

    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, vlm: VLMClient, config: Config) -> None:
        self.vlm = vlm
        self.config = config
        self._extractor = Extractor(vlm, config)
        self._analyzer = Analyzer(vlm, config)
        self._verifier = Verifier(vlm, config)

    def extract_relevant_frames(
        self, task: Task, video: ProcessedVideo
    ) -> tuple[List[Frame], List[SourceReference]]:
        """Run the Extractor phase only: select frames relevant to *task*.

        Exposed separately so callers can run this in parallel with audio
        transcription (which has no dependency on the extractor's output).
        Returns ``(relevant_frames, sources)``; both empty if extraction
        produced no usable frames.
        """
        try:
            relevant_frames = self._extractor.run(task, video)
        except Exception as exc:
            logger.error("Extractor failed: %s", exc)
            relevant_frames = video.frames  # fall back to all frames

        sources = [
            SourceReference(
                timestamp=f.timestamp,
                frame_path=f.path,
                description=f"frame at {f.timestamp:.1f}s",
            )
            for f in relevant_frames
        ]
        return relevant_frames, sources

    def analyze_and_verify(
        self,
        task: Task,
        relevant_frames: List[Frame],
        sources: List[SourceReference],
        transcription: Optional[Transcription] = None,
    ) -> AgentResult:
        """Run the Analyzer + Verifier loop on pre-extracted frames.

        Caller is responsible for having run :meth:`extract_relevant_frames`
        first.  Retries up to ``config.max_retries`` times if the verifier
        confidence is below :attr:`CONFIDENCE_THRESHOLD`.
        """
        retries = 0
        analysis = ""
        confidence = 0.0

        for attempt in range(self.config.max_retries + 1):
            try:
                analysis = self._analyzer.run(task, relevant_frames, transcription)
            except Exception as exc:
                logger.error("Analyzer failed (attempt %d): %s", attempt, exc)
                return AgentResult(
                    content=f"Analysis failed after {attempt} attempts: {exc}",
                    confidence=0.0,
                    sources=sources,
                    retries=attempt,
                )

            try:
                analysis, confidence = self._verifier.run(task, analysis)
            except Exception as exc:
                logger.error("Verifier failed (attempt %d): %s", attempt, exc)
                confidence = 0.5  # neutral confidence, use analysis as-is
                break  # can't verify, return what we have

            if confidence >= self.CONFIDENCE_THRESHOLD:
                break

            if attempt < self.config.max_retries:
                retries += 1
                logger.info(
                    "Confidence %.2f below threshold %.2f — retrying (%d/%d)",
                    confidence,
                    self.CONFIDENCE_THRESHOLD,
                    retries,
                    self.config.max_retries,
                )

        return AgentResult(
            content=analysis,
            confidence=confidence,
            sources=sources,
            retries=retries,
        )

    def run(
        self,
        task: Task,
        video: ProcessedVideo,
        transcription: Optional[Transcription] = None,
    ) -> AgentResult:
        """Execute the full 3-role pipeline sequentially.

        Thin wrapper around :meth:`extract_relevant_frames` and
        :meth:`analyze_and_verify` for callers that don't need to overlap
        the extractor phase with other work.
        """
        relevant_frames, sources = self.extract_relevant_frames(task, video)
        if not relevant_frames:
            return AgentResult(
                content="No relevant frames found for the given task.",
                confidence=0.0,
                sources=[],
                retries=0,
            )
        return self.analyze_and_verify(task, relevant_frames, sources, transcription)
