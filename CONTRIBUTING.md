# Contributing to videocode

Thanks for your interest! This is an early-stage project and all contributions are welcome.

## Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/videocode.git
cd videocode

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install in dev mode
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify setup
pytest tests/ -x
```

## Development Setup

### Requirements

- Python 3.10+
- FFmpeg (`brew install ffmpeg` or `apt-get install ffmpeg`)
- Ollama (for local VLM testing) — `ollama pull llava`

### Project Structure

```
videocode/
├── src/
│   └── videocode/
│       ├── __init__.py
│       ├── __main__.py          # CLI entry point
│       ├── video_processor.py   # FFmpeg + scene detection
│       ├── frame_selector.py    # Smart frame selection
│       ├── vlm_client.py        # Multi-backend VLM interface
│       ├── agent_loop.py        # Orchestration pipeline
│       ├── code_extractor.py    # Code assembly & dedup
│       ├── mcp_server.py        # MCP protocol server
│       ├── models.py            # Pydantic data models
│       └── config.py            # Configuration management
├── tests/
│   ├── unit/                    # Unit tests (fast, no VLM)
│   ├── integration/             # Integration tests (needs VLM)
│   └── fixtures/                # Sample videos, frames
├── docs/
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
# All tests (requires VLM backend)
pytest

# Unit tests only (fast, no external deps)
pytest tests/unit/ -x

# Integration tests only
pytest tests/integration/ -x

# With coverage
pytest --cov=videocode --cov-report=html

# Specific test
pytest tests/unit/test_frame_selector.py -xvs
```

### Code Style

We use `ruff` for linting/formatting and `mypy` for type checking.

```bash
# Check everything
make check

# Or run individually
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/

# Auto-fix issues
ruff check --fix src/ tests/
ruff format src/ tests/
```

**Key style rules:**
- Line length: 100 characters
- Docstrings: Google style
- Type hints: required for all function signatures
- Imports: sorted with `isort` (enforced by ruff)

## PR Process

1. **Open an issue first** (for significant changes)
   - Describe the problem or feature
   - Discuss approach before investing time

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Make your changes**
   - Add tests for new functionality
   - Update docs if needed
   - Keep changes focused — one PR per concern

4. **Run checks locally**
   ```bash
   make check          # ruff + mypy
   pytest tests/unit/  # unit tests
   ```

5. **Submit PR**
   - Fill out the PR template
   - Reference any related issues
   - Keep the description clear and concise

### PR Template

```markdown
## Summary
One-line description of what this PR does.

## Changes
- Bullet list of changes
- Be specific but concise

## Testing
- How you tested this
- What you verified

## Related Issues
Fixes #123
```

### What We Look For

- **Tests:** New code should have tests. Bug fixes should include a regression test.
- **Documentation:** Update README, docstrings, or docs/ if behavior changes.
- **Focused scope:** One logical change per PR. Don't refactor unrelated code.
- **Clear commits:** Use conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`).

## Feature Requests

Have an idea? We'd love to hear it.

**Before requesting:** Check [existing issues](https://github.com/videocode/videocode/issues) to avoid duplicates.

**Good feature requests include:**
- Clear description of the problem you're solving
- How you'd use it (concrete workflow)
- Any implementation ideas you have

Open an issue with the `enhancement` label.

## Bug Reports

**Good bug reports include:**
- videocode version (`videocode --version`)
- Python version (`python --version`)
- Operating system
- Steps to reproduce
- Expected vs. actual behavior
- Error message (full traceback if applicable)
- Video that triggers the issue (if shareable)

## Areas We Need Help

Priority areas for contributions:

- **🤖 New VLM backends** — Add support for more vision models (Anthropic Claude Vision, etc.)
- **🎬 Video platform support** — Vimeo, Twitch, local screen recordings
- **🧪 Test coverage** — More unit tests, especially for edge cases
- **📖 Documentation** — Tutorials, examples, API docs
- **⚡ Performance** — Faster scene detection, parallel VLM calls
- **🌍 i18n** — Support for non-English tutorial videos

## Questions?

- Open a [Discussion](https://github.com/videocode/videocode/discussions) for general questions
- Join the conversation in existing issues

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to build something useful.

---

**Thank you for contributing!** 🙏
