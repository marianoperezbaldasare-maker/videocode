#!/usr/bin/env python3
"""
Demo script for videocode.
Runs the video processing pipeline with a mock VLM (no API key needed).

Usage:
    python demo.py                    # Uses demo/test_tutorial.mp4
    python demo.py path/to/video.mp4  # Uses your video

This demo:
1. Processes a video (extracts frames + detects scenes)
2. Shows the extracted frames
3. Runs frame selection (adaptive strategy)
4. Simulates VLM analysis (mock, no AI needed)
5. Shows the code extraction pipeline
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
from videocode.config import Config
from videocode.video_processor import VideoProcessor
from videocode.frame_selector import FrameSelector, SelectionStrategy

# ANSI colors
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


def print_step(num: int, text: str) -> None:
    print(f"{CYAN}Step {num}:{RESET} {text}")
    print(f"{CYAN}{'─' * 50}{RESET}")


def main() -> int:
    # Determine video path
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        video_path = Path(__file__).parent / "demo" / "test_tutorial.mp4"

    if not video_path.exists():
        print(f"{RED}Error: Video not found: {video_path}{RESET}")
        print(f"Run: python demo.py [path/to/video.mp4]")
        return 1

    print_header("👁️  CLAUDE-VISION DEMO — Pipeline Test")
    print(f"Video: {BLUE}{video_path.absolute()}{RESET}")
    print(f"Size:  {video_path.stat().st_size / 1024:.1f} KB\n")

    # Step 1: Configuration
    print_step(1, "Configuration")
    config = Config(
        target_fps=0.5,
        max_frames=10,
        frame_resolution=(640, 360),
        extract_audio=False,
    )
    print(f"  Backend:     {config.vlm_backend}")
    print(f"  Max frames:  {config.max_frames}")
    print(f"  Resolution:  {config.frame_resolution}")
    print(f"  Audio:       {config.extract_audio}")
    print(f"  {GREEN}✓ Config loaded{RESET}")

    # Step 2: Video Processing
    print_step(2, "Video Processing (FFmpeg)")
    processor = VideoProcessor(config)
    try:
        processed = processor.process(str(video_path))
        print(f"  Duration:    {processed.duration:.1f}s")
        print(f"  FPS:         {processed.fps:.2f}")
        print(f"  Resolution:  {processed.resolution}")
        print(f"  Frame dir:   {processed.frame_dir}")
        print(f"  {GREEN}✓ Video processed successfully{RESET}")
    except Exception as e:
        print(f"  {RED}✗ Error: {e}{RESET}")
        return 1

    # Step 3: Frame Selection
    print_step(3, "Intelligent Frame Selection")
    selector = FrameSelector(config)

    for strategy in SelectionStrategy:
        try:
            frames = selector.select_frames(video_path, strategy)
            tokens = selector.calculate_token_budget(len(frames))
            print(
                f"  {strategy.value:12s} → "
                f"{len(frames)} frames, ~{tokens:,} tokens"
            )
        except Exception as e:
            print(f"  {strategy.value:12s} → {YELLOW}skipped ({e}){RESET}")

    # Use AUTO strategy for demo (pass video path, not frame dir)
    frames = selector.select_frames(video_path, SelectionStrategy.AUTO)
    print(f"\n  {GREEN}✓ Selected {len(frames)} frames with AUTO strategy{RESET}")

    # Show frame details
    for i, frame in enumerate(frames):
        size_kb = frame.path.stat().st_size / 1024
        print(
            f"    Frame {i}: t={frame.timestamp:.1f}s, "
            f"scene={frame.scene_index}, "
            f"keyframe={'✓' if frame.is_keyframe else '○'}, "
            f"size={size_kb:.1f}KB"
        )

    # Step 4: Mock VLM Analysis
    print_step(4, "VLM Analysis (MOCK — no API key needed)")
    print(f"  {YELLOW}⚠ Using mock VLM (no AI backend configured){RESET}")
    print(f"  In production, this would send frames to:")
    print(f"    • Ollama (local, free)  ← default")
    print(f"    • Gemini 1.5 Pro (cloud)")
    print(f"    • OpenAI GPT-4o (cloud)")
    print(f"    • Qwen2.5-VL (open source)")
    print()
    print(f"  Simulated analysis of {len(frames)} frames:")

    # Simulate code detection
    mock_results = [
        ("Frame 0", "JavaScript function definition detected", "calculateSum(a, b)"),
        ("Frame 1", "React component with useState hook", "Counter() component"),
        ("Frame 2", "CSS styling rules detected", ".counter styles"),
    ]

    for frame_name, desc, code in mock_results:
        print(f"\n    {BLUE}{frame_name}{RESET}: {desc}")
        print(f"    Code: {GREEN}{code}{RESET}")

    # Step 5: Code Extraction (simulated)
    print_step(5, "Code Extraction Pipeline")
    print(f"  {GREEN}✓ Detected 3 code blocks across {len(frames)} frames{RESET}")
    print(f"\n  {BOLD}Extracted files:{RESET}")

    extracted_files = {
        "utils.js": "function calculateSum(a, b) {\n  const result = a + b;\n  console.log(`Sum: ${result}`);\n  return result;\n}",
        "Counter.jsx": "import React, { useState } from 'react';\n\nfunction Counter() {\n  const [count, setCount] = useState(0);\n  // ...\n}",
        "styles.css": ".counter {\n  display: flex;\n  flex-direction: column;\n  align-items: center;\n}",
    }

    for filename, content in extracted_files.items():
        lines = content.count("\n") + 1
        print(f"\n    📄 {CYAN}{filename}{RESET} ({lines} lines)")
        # Show first 3 lines
        for line in content.split("\n")[:3]:
            print(f"       {line}")
        if content.count("\n") > 2:
            print(f"       {YELLOW}... ({lines - 3} more lines){RESET}")

    # Summary
    print_header("🎉 DEMO COMPLETE")
    print(f"  Video:        {processed.duration:.1f}s processed")
    print(f"  Frames:       {len(frames)} extracted")
    print(f"  Code blocks:  {len(extracted_files)} detected")
    print(f"  Strategy:     {SelectionStrategy.AUTO.value} (adaptive)")
    print()
    print(f"  {GREEN}✓ Pipeline works correctly!{RESET}")
    print()
    print(f"{BOLD}Next steps:{RESET}")
    print(f"  1. Install Ollama:       {CYAN}ollama pull llava{RESET}")
    print(f"  2. Run with real AI:     {CYAN}videocode code your_video.mp4{RESET}")
    print(f"  3. Or use cloud API:     {CYAN}export GEMINI_API_KEY=...{RESET}")
    print()

    # Cleanup
    processor.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
