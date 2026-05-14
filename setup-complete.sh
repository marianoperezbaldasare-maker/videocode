#!/usr/bin/env bash
# Setup completo para videocode con Gemini
# Uso: bash setup-complete.sh

set -e

echo "=========================================="
echo "  👁️  CLAUDE-VISION — Setup Completo"
echo "=========================================="
echo ""

# Verificar Python
python3 --version || python --version || { echo "❌ Python 3.10+ requerido"; exit 1; }

# Verificar FFmpeg
ffmpeg -version 2>/dev/null | head -1 || { echo "❌ FFmpeg requerido. brew install ffmpeg"; exit 1; }
echo "✅ FFmpeg: $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"

# Instalar paquete
echo ""
echo "📦 Instalando videocode..."
pip install -e "." -q
echo "✅ videocode instalado"

# Verificar que funciona
echo ""
echo "🔧 Verificando instalación..."
videocode status 2>/dev/null || PYTHONPATH=src python -m videocode.cli status
echo "✅ Todo verde!"

# Configurar API key si no está seteada
if [ -z "$GEMINI_API_KEY" ]; then
    echo ""
    echo "⚠️  GEMINI_API_KEY no está configurada"
    echo "   Para usar Gemini, ejecuta:"
    echo "   export GEMINI_API_KEY='tu-api-key'"
else
    echo ""
    echo "✅ GEMINI_API_KEY configurada"
fi

# Crear config.json para Claude Code
echo ""
echo "📝 Creando config.json para Claude Code..."

CONFIG_DIR=""
if [ -d "$HOME/Library/Application Support/Claude" ]; then
    CONFIG_DIR="$HOME/Library/Application Support/Claude"
elif [ -d "$HOME/.config/Claude" ]; then
    CONFIG_DIR="$HOME/.config/Claude"
elif [ -d "$APPDATA/Claude" 2>/dev/null ]; then
    CONFIG_DIR="$APPDATA/Claude"
fi

if [ -n "$CONFIG_DIR" ]; then
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/config.json" << 'EOF'
{
  "mcpServers": {
    "videocode": {
      "command": "videocode",
      "args": ["mcp"]
    }
  }
}
EOF
    echo "✅ Config creado en: $CONFIG_DIR/config.json"
    echo ""
    echo "   Para usar Gemini, edita el archivo y agrega:"
    echo "   {\"env\": {\"GEMINI_API_KEY\": \"tu-key\"}}"
else
    echo "⚠️  No se encontró el directorio de Claude Code"
    echo "   Crea manualmente el archivo config.json"
fi

echo ""
echo "=========================================="
echo "  ✅ Setup completo!"
echo "=========================================="
echo ""
echo "Comandos disponibles:"
echo "  videocode code video.mp4 --backend gemini"
echo "  videocode summarize video.mp4"
echo "  videocode mcp"
echo ""
echo "Para conectar con Claude Code:"
echo "  1. Copia config.json al directorio de Claude"
echo "  2. Reinicia Claude Code"
echo "  3. Dile: 'Extract code from this video: ...'"
echo ""
