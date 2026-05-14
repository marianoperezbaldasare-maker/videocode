#!/usr/bin/bash
# Setup script for videocode
# Run: bash setup.sh

set -e

BOLD='\033[1m'
GREEN='\033[92m'
YELLOW='\033[93m'
CYAN='\033[96m'
RED='\033[91m'
RESET='\033[0m'

echo -e "${BOLD}👁️  Claude Vision Setup${RESET}\n"

# 1. Check Python
echo -e "${CYAN}Step 1:${RESET} Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo -e "${RED}✗ Python 3.10+ required. Install from https://python.org${RESET}"
    exit 1
fi
PYVER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  Found: Python $PYVER"

# 2. Check FFmpeg
echo -e "\n${CYAN}Step 2:${RESET} Checking FFmpeg..."
if command -v ffmpeg &> /dev/null && command -v ffprobe &> /dev/null; then
    FFVER=$(ffmpeg -version | head -1 | awk '{print $3}')
    echo -e "  Found: FFmpeg $FFVER"
else
    echo -e "  ${YELLOW}⚠ FFmpeg not found${RESET}"
    echo -e "  Install with:"
    echo -e "    macOS:  ${BOLD}brew install ffmpeg${RESET}"
    echo -e "    Ubuntu: ${BOLD}sudo apt-get install ffmpeg${RESET}"
    echo -e "    Windows: ${BOLD}choco install ffmpeg${RESET}"
    exit 1
fi

# 3. Install Python package
echo -e "\n${CYAN}Step 3:${RESET} Installing videocode..."
pip install -e "." 2>&1 | tail -3
echo -e "  ${GREEN}✓ Installed${RESET}"

# 4. Check Ollama
echo -e "\n${CYAN}Step 4:${RESET} Checking Ollama (optional, for local AI)..."
if command -v ollama &> /dev/null; then
    echo -e "  Found: $(ollama --version 2>/dev/null || echo 'Ollama')"
    echo -e "  Pulling vision model..."
    ollama pull llava 2>&1 | tail -2 || echo -e "  ${YELLOW}⚠ Could not pull model (Ollama may not be running)${RESET}"
else
    echo -e "  ${YELLOW}⚠ Ollama not found${RESET}"
    echo -e "  Install: ${BOLD}https://ollama.com/download${RESET}"
    echo -e "  Then run: ${BOLD}ollama pull llava${RESET}"
fi

# 5. Verify installation
echo -e "\n${CYAN}Step 5:${RESET} Verifying installation..."
if videocode --help &> /dev/null; then
    echo -e "  ${GREEN}✓ CLI working${RESET}"
else
    echo -e "  ${RED}✗ CLI not found. Try: pip install -e .${RESET}"
fi

# Done
echo -e "\n${GREEN}${BOLD}✅ Setup complete!${RESET}\n"
echo -e "${BOLD}Quick start:${RESET}"
echo -e "  ${CYAN}videocode --help${RESET}              Show all commands"
echo -e "  ${CYAN}python demo.py${RESET}                  Run demo (no AI needed)"
echo -e "  ${CYAN}videocode code video.mp4${RESET}    Extract code from video"
echo -e "  ${CYAN}videocode mcp${RESET}                Start MCP server for Claude Code"
echo -e "\n${BOLD}Docs:${RESET} https://github.com/videocode/videocode#readme"
