#!/usr/bin/env python3
"""
Demo COMPLETO de videocode con backend Dummy (sin API key, sin Ollama).
Muestra el flujo completo: video → frames → VLM → código.

Uso:
    PYTHONPATH=src python demo_full.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from videocode.config import Config
from videocode.video_processor import VideoProcessor
from videocode.frame_selector import FrameSelector, SelectionStrategy
from videocode.vlm_client import VLMClient
from videocode.agent_loop import AgentLoop, Task, TaskType
from videocode.code_extractor import CodeExtractor
from videocode.types import ProcessedVideo

# Colores
B = "\033[1m"
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
C = "\033[96m"
_ = "\033[0m"


def banner(text):
    print(f"\n{B}{'='*60}{_}")
    print(f"  {text}")
    print(f"{B}{'='*60}{_}\n")


def step(n, text):
    print(f"\n  {C}▶{_} {B}Paso {n}:{_} {text}")
    print(f"  {C}{'─'*50}{_}")


def ok(text):
    print(f"    {G}✓{_} {text}")


def info(text):
    print(f"    {C}•{_} {text}")


def warn(text):
    print(f"    {Y}⚠{_} {text}")


def main():
    banner(f"{B}👁️  CLAUDE-VISION — Demo Completo (Dummy Backend){_}")
    print(f"  {Y}Este demo simula un VLM para mostrar el flujo completo.{_}")
    print(f"  {Y}No necesitas API key ni Ollama.{_}\n")

    video_path = os.path.join(os.path.dirname(__file__), "demo", "test_tutorial.mp4")
    if not os.path.exists(video_path):
        print(f"  {R}✗{_} Video no encontrado: {video_path}")
        print(f"  Ejecuta primero: python demo.py")
        return 1

    # Configuración con backend dummy
    config = Config(
        vlm_backend="dummy",
        target_fps=0.5,
        max_frames=10,
        frame_resolution=(640, 360),
        extract_audio=False,
        output_dir="./demo_output",
    )

    # ── Paso 1: Procesar video ──────────────────────────────────────
    step(1, "Procesando video con FFmpeg")
    info(f"Video: {video_path}")
    processor = VideoProcessor(config)
    processed = processor.process(video_path)
    ok(f"Duración: {processed.duration:.1f}s")
    ok(f"Resolución: {processed.resolution}")
    ok(f"FPS: {processed.fps:.2f}")
    ok(f"Frames extraídos: {len(processed.frames)}")

    # ── Paso 2: Selección inteligente de frames ─────────────────────
    step(2, "Selección inteligente de frames")
    selector = FrameSelector(config)
    frames = selector.select_frames(video_path, SelectionStrategy.AUTO)
    ok(f"Frames seleccionados: {len(frames)}")
    for i, f in enumerate(frames):
        info(f"Frame {i}: t={f.timestamp:.1f}s, keyframe={'✓' if f.is_keyframe else '○'}")

    # ── Paso 3: Inicializar VLM (Dummy) ─────────────────────────────
    step(3, "Inicializando VLM (Dummy Backend)")
    vlm = VLMClient(config)
    ok(f"Backend: {config.vlm_backend}")
    ok(f"Disponible: {vlm.is_available()}")

    # ── Paso 4: Análisis con VLM ────────────────────────────────────
    step(4, "Analizando frames con VLM")

    # 4a. Detectar frames con código
    info("Detectando qué frames contienen código...")
    resp = vlm.analyze_frames(frames, "Which of these frames contain code? For each, tell me the programming language.")
    try:
        detection = json.loads(resp.content)
        code_frames = [f for f in detection.get("frames", []) if f.get("has_code")]
        ok(f"Frames con código: {len(code_frames)}")
        for cf in code_frames:
            info(f"  Frame {cf['frame_index']}: {cf['language']}")
    except (json.JSONDecodeError, KeyError):
        ok("Detección completada (ver respuesta en output)")

    # 4b. Extraer código de cada frame
    info("Extrayendo código de cada frame...")
    extracted_codes = []
    for i, frame in enumerate(frames[:3]):  # Primeros 3 frames
        resp_text = vlm.analyze_single(frame, "Extract all code visible in this image. Output only the code.")
        try:
            data = json.loads(resp_text)
            code = data.get("code", "")
            lang = data.get("language", "unknown")
            extracted_codes.append({"code": code, "language": lang})
            ok(f"Frame {i}: {lang} ({len(code.split(chr(10)))} líneas)")
        except (json.JSONDecodeError, KeyError):
            extracted_codes.append({"code": resp_text, "language": "unknown"})
            ok(f"Frame {i}: código extraído")

    # ── Paso 5: Agent Loop ──────────────────────────────────────────
    step(5, "Agent Loop — 3 roles (Extract → Analyze → Verify)")
    agent = AgentLoop(vlm, config)
    task = Task(type=TaskType.CODE_EXTRACTION, query="Extract all code from this tutorial video")
    result = agent.run(task, processed, transcription=None)
    ok(f"Confianza: {result.confidence:.0%}")
    ok(f"Reintentos: {result.retries}")
    ok(f"Fuentes: {len(result.sources)}")

    # ── Paso 6: Code Extractor ──────────────────────────────────────
    step(6, "Code Extractor — Ensamblando proyecto")
    extractor = CodeExtractor(agent, config)
    # Simular transcripción vacía
    from videocode.types import Transcription
    trans = Transcription(text="", segments=[], language="en")
    code_result = extractor.extract(processed, trans)
    ok(f"Archivos generados: {len(code_result.files)}")
    ok(f"Lenguaje: {code_result.language}")
    ok(f"Confianza: {code_result.confidence:.0%}")

    # ── Paso 7: Mostrar resultados ──────────────────────────────────
    step(7, "Resultados — Archivos generados")
    print()
    for filename, content in code_result.files.items():
        lines = content.count("\n") + 1
        print(f"    {G}📄{_} {B}{filename:30s}{_} ({lines:3d} líneas)")
        # Primeras 3 líneas
        for line in content.split("\n")[:3]:
            if line.strip():
                print(f"       {C}│{_} {line[:70]}")
        if lines > 3:
            print(f"       {C}│{_} {Y}... ({lines - 3} líneas más){_}")
        print()

    # Guardar archivos
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for filename, content in code_result.files.items():
        filepath = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)

    # README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# {code_result.description}\n\n")
        f.write("## Setup\n\n")
        f.write(f"{code_result.setup_instructions}\n")
    ok(f"README guardado: {readme_path}")

    ok(f"{len(code_result.files) + 1} archivos guardados en {output_dir}/")

    # ── Footer ──────────────────────────────────────────────────────
    banner(f"{G}🎉 Demo completo!{_}")
    print(f"  {B}Resumen:{_}")
    print(f"    {C}•{_} Video procesado: {processed.duration:.1f}s → {len(frames)} frames")
    print(f"    {C}•{_} Código extraído: {len(extracted_codes)} bloques")
    print(f"    {C}•{_} Archivos generados: {len(code_result.files) + 1}")
    print(f"    {C}•{_} Todo con backend {Y}DUMMY{_} (sin API key!)")
    print()
    print(f"  {B}Para usar con AI real:{_}")
    print(f"    1. Ollama (gratis):  {C}ollama pull llava && ollama serve{_}")
    print(f"    2. Gemini:           {C}export GEMINI_API_KEY=...{_}")
    print(f"    3. OpenAI:           {C}export OPENAI_API_KEY=...{_}")
    print()

    # Cleanup
    processor.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
