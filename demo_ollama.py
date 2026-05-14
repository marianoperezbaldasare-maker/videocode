#!/usr/bin/env python3
"""
Demo de videocode con Ollama (backend REAL).
Procesa un video y extrae código usando llava.

Uso:
    PYTHONPATH=src python demo_ollama.py
"""

import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from videocode.config import Config
from videocode.video_processor import VideoProcessor
from videocode.frame_selector import FrameSelector, SelectionStrategy
from videocode.vlm_client import VLMClient

B = "\033[1m"
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
C = "\033[96m"
M = "\033[95m"
_ = "\033[0m"

def banner(t): print(f"\n{B}{'='*58}{_}\n  {t}\n{B}{'='*58}{_}\n")
def step(n,t): print(f"\n  {C}▶{_} {B}Paso {n}:{_} {t}\n  {C}{'─'*50}{_}")
def ok(t): print(f"    {G}✓{_} {t}")
def info(t): print(f"    {C}•{_} {t}")
def warn(t): print(f"    {Y}⚠{_} {t}")

banner(f"{B}👁️  CLAUDE-VISION + OLLAMA (llava){_}")
print(f"  {Y}Usando Ollama con llava — AI local y gratuita{_}\n")

video = os.path.join(os.path.dirname(__file__), "demo", "test_tutorial.mp4")
if not os.path.exists(video):
    print(f"  {R}✗{_} Video no encontrado")
    sys.exit(1)

# Config con Ollama
config = Config(
    vlm_backend="ollama",
    vlm_model="moondream",
    ollama_url="http://localhost:11434",
    target_fps=0.5,
    max_frames=4,
    frame_resolution=(640, 360),
    extract_audio=False,
)

# ── Paso 1: Video ──────────────────────────────────────────────
step(1, "Procesando video con FFmpeg")
processor = VideoProcessor(config)
processed = processor.process(video)
ok(f"Video: {processed.duration:.1f}s, {len(processed.frames)} frames extraídos")

# ── Paso 2: Selección de frames ────────────────────────────────
step(2, "Seleccionando frames")
frames = FrameSelector(config).select_frames(video, SelectionStrategy.AUTO)
ok(f"{len(frames)} frames seleccionados")
for i, f in enumerate(frames):
    info(f"Frame {i}: t={f.timestamp:.1f}s")

# ── Paso 3: VLM (Ollama + llava) ───────────────────────────────
step(3, "Conectando a Ollama (llava)")
vlm = VLMClient(config)
ok(f"Ollama disponible: {vlm.is_available()}")
ok(f"Modelo: {config.vlm_model}")

# ── Paso 4: Análisis con llava ─────────────────────────────────
step(4, "Analizando frames con llava (puede tardar ~30s)...")
print(f"    {Y}⏳ Enviando {len(frames)} frames a llava...{_}")

t0 = time.time()
resp = vlm.analyze_frames(frames, "Describe what code or programming content you see in these images. List any code snippets, programming languages, or UI components visible.")
t1 = time.time()

ok(f"Análisis completo en {t1-t0:.1f}s")
ok(f"Tokens usados: {resp.tokens_used}")
ok(f"Frames analizados: {resp.frames_analyzed}")
print(f"\n    {B}Respuesta de llava:{_}")
print(f"    {M}{'─'*46}{_}")
for line in resp.content.split("\n")[:15]:
    print(f"    {line[:60]}")
if len(resp.content.split("\n")) > 15:
    print(f"    {Y}... ({len(resp.content.split(chr(10)))-15} líneas más){_}")
print(f"    {M}{'─'*46}{_}")

# ── Paso 5: Extraer código ─────────────────────────────────────
step(5, "Extrayendo código de cada frame")
for i, frame in enumerate(frames[:2]):
    info(f"Frame {i}...")
    t0 = time.time()
    code = vlm.analyze_single(frame, "Extract only the code visible in this image. Output just the code, no explanations.")
    t1 = time.time()
    ok(f"Código extraído en {t1-t0:.1f}s")
    for line in code.split("\n")[:4]:
        if line.strip():
            print(f"       {C}│{_} {line[:55]}")
    if len(code.split("\n")) > 4:
        print(f"       {C}│{_} {Y}...{_}")

# Cleanup
processor.cleanup()

# ── Footer ──────────────────────────────────────────────────────
banner(f"{G}🎉 Demo con Ollama completado!{_}")
print(f"  {B}Resumen:{_}")
print(f"    {C}•{_} Video: {processed.duration:.1f}s → {len(frames)} frames")
print(f"    {C}•{_} Backend: {G}Ollama + llava{_} (AI local)")
print(f"    {C}•{_} Análisis: Completado en {t1-t0:.1f}s")
print(f"\n  {B}Para usar en producción:{_}")
print(f"    {C}videocode code video.mp4 --backend ollama{_}")
print()
