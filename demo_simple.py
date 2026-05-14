#!/usr/bin/env python3
"""
Demo SIMPLE de videocode con backend Dummy.
Muestra el flujo completo: video → frames → VLM → código.

Uso:
    PYTHONPATH=src python demo_simple.py
"""

import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from videocode.config import Config
from videocode.video_processor import VideoProcessor
from videocode.frame_selector import FrameSelector, SelectionStrategy
from videocode.vlm_client import VLMClient

B, G, R, Y, C, M, _ = "\033[1m", "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[95m", "\033[0m"

def banner(t): print(f"\n{B}{'='*58}{_}\n  {t}\n{B}{'='*58}{_}\n")
def step(n,t): print(f"\n  {C}▶{_} {B}Paso {n}:{_} {t}\n  {C}{'─'*50}{_}")
def ok(t): print(f"    {G}✓{_} {t}")
def info(t): print(f"    {C}•{_} {t}")
def warn(t): print(f"    {Y}⚠{_} {t}")

banner(f"{B}👁️  CLAUDE-VISION — Demo Completo (Dummy Backend){_}")
print(f"  {Y}Este demo simula un VLM. No necesitas API key ni Ollama.{_}\n")

# Config
config = Config(vlm_backend="dummy", target_fps=0.5, max_frames=10,
                frame_resolution=(640, 360), extract_audio=False)

video = os.path.join(os.path.dirname(__file__), "demo", "test_tutorial.mp4")
if not os.path.exists(video):
    print(f"  {R}✗{_} Video no encontrado. Ejecuta: python demo.py")
    sys.exit(1)

# ── Paso 1: Video ──────────────────────────────────────────────
step(1, "Procesando video con FFmpeg")
processor = VideoProcessor(config)
processed = processor.process(video)
ok(f"Video: {processed.duration:.1f}s, {processed.resolution}, {len(processed.frames)} frames")

# ── Paso 2: Frames ─────────────────────────────────────────────
step(2, "Selección inteligente de frames")
frames = FrameSelector(config).select_frames(video, SelectionStrategy.AUTO)
ok(f"{len(frames)} frames seleccionados")
for i,f in enumerate(frames):
    info(f"Frame {i}: t={f.timestamp:.1f}s")

# ── Paso 3: VLM Dummy ──────────────────────────────────────────
step(3, "Inicializando VLM (Dummy Backend)")
vlm = VLMClient(config)
ok(f"Backend: dummy | Disponible: {vlm.is_available()}")

# ── Paso 4: Análisis ───────────────────────────────────────────
step(4, "Analizando frames con VLM")

# 4a. Detectar código
info("Detectando frames con código...")
resp = vlm.analyze_frames(frames, "Which frames contain code?")
detection = json.loads(resp.content)
ok(f"Frames con código: {len(detection['frames'])}")
for f in detection['frames']:
    info(f"  Frame {f['frame_index']}: {f['language']}")

# 4b. Extraer código de cada frame
info("Extrayendo código...")
codes = []
for i,frame in enumerate(frames[:3]):
    text = vlm.analyze_single(frame, "Extract all code visible")
    try:
        data = json.loads(text)
        codes.append(data)
        ok(f"Frame {i}: {data['language']} ({len(data['code'].split())} palabras)")
    except:
        codes.append({"code": text, "language": "unknown"})
        ok(f"Frame {i}: código extraído")

# 4c. Ensamblar proyecto
info("Ensamblando proyecto...")
resp = vlm.analyze_frames(frames[:3], "Organize these code blocks into a project")
project = json.loads(resp.content)
files = project.get("files", {})
ok(f"Proyecto ensamblado: {len(files)} archivos")

# ── Paso 5: Mostrar resultados ─────────────────────────────────
step(5, "Archivos generados")
print()
for fn, content in files.items():
    lines = content.count("\n") + 1
    print(f"    {G}📄{_} {B}{fn:30s}{_} ({lines} líneas)")
    for line in content.split("\n")[:3]:
        if line.strip():
            print(f"       {C}│{_} {line[:65]}")
    if lines > 3:
        print(f"       {C}│{_} {Y}... ({lines-3} más){_}")
    print()

# Guardar
out = config.output_dir
os.makedirs(out, exist_ok=True)
for fn, content in files.items():
    fp = os.path.join(out, fn)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as f: f.write(content)
with open(f"{out}/README.md", "w") as f:
    f.write(f"# {project.get('description', 'App')}\n\n## Setup\n\n{project.get('setup_instructions', '')}\n")
ok(f"{len(files)+1} archivos guardados en {out}/")

# Cleanup
processor.cleanup()

# ── Footer ──────────────────────────────────────────────────────
banner(f"{G}🎉 Demo completo!{_}")
print(f"  {B}Resumen:{_}")
print(f"    {C}•{_} Video: {processed.duration:.1f}s → {len(frames)} frames")
print(f"    {C}•{_} Código: {len(codes)} bloques extraídos")
print(f"    {C}•{_} Archivos: {len(files)+1} generados")
print(f"    {C}•{_} Todo con backend {Y}DUMMY{_} (sin API key!)")
print(f"\n  {B}Para AI real:{_}")
print(f"    1. Ollama: {C}ollama pull llava && ollama serve{_}")
print(f"    2. Gemini: {C}export GEMINI_API_KEY=...{_}")
print(f"    3. OpenAI: {C}export OPENAI_API_KEY=...{_}")
print()
