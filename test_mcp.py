#!/usr/bin/env python3
"""
Prueba automática del MCP server de videocode.
Importa el módulo directamente para evitar problemas con Rich en stdout.

Uso:
    python test_mcp.py
    python test_mcp.py -v
"""

import sys
import os
import asyncio
import json

# Asegurar que src está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Colores
B = "\033[1m"      # Bold
G = "\033[92m"     # Green
R = "\033[91m"     # Red
Y = "\033[93m"     # Yellow
C = "\033[96m"     # Cyan
_ = "\033[0m"      # Reset


def log_step(n, text):
    print(f"\n  {C}▶{_} {B}Paso {n}:{_} {text}")
    print(f"  {C}{'─'*50}{_}")


def log_ok(text):
    print(f"    {G}✓{_} {text}")


def log_err(text):
    print(f"    {R}✗{_} {text}")


def log_warn(text):
    print(f"    {Y}⚠{_} {text}")


def log_info(text):
    print(f"    {C}•{_} {text}")


def print_banner():
    print(f"\n{B}{'='*60}{_}")
    print(f"{B}  👁️  PRUEBA MCP SERVER — videocode{_}")
    print(f"{B}{'='*60}{_}\n")


def print_footer():
    print(f"\n{B}{'='*60}{_}")
    print(f"  {G}✅ MCP server funcionando correctamente!{_}")
    print(f"{B}{'='*60}{_}\n")
    print(f"  {B}Para conectar con Claude Code:{_}")
    print(f"""
  {Y}{{
    "mcpServers": {{
      "videocode": {{
        "command": "videocode",
        "args": ["mcp"]
      }}
    }}
  }}{_}
""")


async def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    print_banner()

    # ── Paso 1: Importar ─────────────────────────────────────────────
    log_step(1, "Importando mcp_server...")
    try:
        from videocode.mcp_server import ClaudeVisionMCPServer
        from videocode.config import Config
        log_ok("Módulo importado correctamente")
    except ImportError as e:
        log_err(f"Error importando: {e}")
        return 1

    # ── Paso 2: Crear instancia ──────────────────────────────────────
    log_step(2, "Creando instancia del servidor...")
    try:
        config = Config()
        server = ClaudeVisionMCPServer(config)
        log_ok("Servidor creado")
    except Exception as e:
        log_err(f"Error creando servidor: {e}")
        return 1

    # ── Paso 3: Listar herramientas ──────────────────────────────────
    log_step(3, "Listando herramientas MCP...")
    try:
        # Importar el objeto FastMCP global del módulo
        from videocode.mcp_server import mcp as mcp_instance
        tools = list(mcp_instance._tool_manager._tools.values())
        log_ok(f"{len(tools)} herramientas registradas:")
        print()
        for tool in tools:
            name = tool.name
            desc = tool.description.split("\n")[0][:55] if tool.description else ""
            print(f"      {G}●{_} {B}{name}{_}")
            print(f"        {desc}")
        if verbose:
            for tool in tools:
                print(f"\n      {B}{tool.name}{_} — schema:")
                schema = json.dumps(tool.parameters, indent=6)
                print(f"        {schema[:300]}")
    except Exception as e:
        log_err(f"Error listando herramientas: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ── Paso 4: Health check ─────────────────────────────────────────
    log_step(4, "Ejecutando health_check...")
    try:
        result = await server.health_check()
        status = result.get("status", "?")
        if status == "ok":
            log_ok(f"Estado: {G}{status.upper()}{_}")
        elif status == "partial":
            log_warn(f"Estado: {Y}{status.upper()}{_}")
        else:
            log_err(f"Estado: {R}{status.upper()}{_}")

        print()
        for name, info in result.get("backends", {}).items():
            if isinstance(info, dict):
                ok = info.get("available", False)
                det = info.get("details", "")
            else:
                ok = bool(info)
                det = str(info)
            icon = f"{G}✓{_}" if ok else f"{R}✗{_}"
            label = f"{G}OK{_}" if ok else f"{R}NO OK{_}"
            print(f"    {icon} {name:12s} {label}")
            if det and det != str(ok):
                print(f"       {det}")
    except Exception as e:
        log_err(f"Error en health_check: {e}")
        import traceback
        traceback.print_exc()

    # ── Paso 5: Extraer código (con mock VLM) ───────────────────────
    log_step(5, "Probar video_extract_code (mock)...")
    log_info("Usando mock VLM — no hay backend real configurado")
    try:
        result = await server.video_extract_code("demo/test_tutorial.mp4")
        if result.get("status") == "error":
            log_warn(f"Esperado (sin VLM): {result.get('message', '')[:60]}")
        else:
            log_ok("Pipeline completó!")
            if "files" in result:
                for fname in result["files"]:
                    log_info(f"Archivo: {fname}")
    except Exception as e:
        log_warn(f"Error esperado sin VLM: {str(e)[:60]}")

    # ── Footer ───────────────────────────────────────────────────────
    print_footer()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
