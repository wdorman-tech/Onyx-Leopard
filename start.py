#!/usr/bin/env python3
"""Onyx Leopard - one-command setup and launch."""

import socket
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = BACKEND / ".env"
WIN = sys.platform == "win32"


def fail(msg: str) -> None:
    print(f"\n  ERROR: {msg}\n")
    sys.exit(1)


def step(msg: str) -> None:
    print(f"\n  > {msg}")


def check_prerequisites() -> None:
    if sys.version_info < (3, 12):
        fail(f"Python 3.12+ required (you have {sys.version})")
    if not shutil.which("node"):
        fail("Node.js not found. Download it from https://nodejs.org/")
    if not shutil.which("pnpm"):
        fail(
            "pnpm not found.\n"
            "    Install it with:  npm install -g pnpm\n"
            "    Or see: https://pnpm.io/installation"
        )


def setup_env() -> None:
    if ENV_FILE.exists():
        text = ENV_FILE.read_text()
        for line in text.splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and val != "sk-ant-your-key-here":
                    return  # Already configured
    else:
        if ENV_EXAMPLE.exists():
            shutil.copy(ENV_EXAMPLE, ENV_FILE)
        else:
            ENV_FILE.write_text("ANTHROPIC_API_KEY=\n")

    step("Anthropic API key required")
    print("    Get one at https://console.anthropic.com/\n")
    key = input("    Paste your API key (or Enter to skip): ").strip()
    if key:
        ENV_FILE.write_text(f"ANTHROPIC_API_KEY={key}\n")
        print("    Saved!")
    else:
        print(f"    Skipped -- edit {ENV_FILE} before using AI features.")


def install_backend() -> None:
    check = subprocess.run(
        [sys.executable, "-m", "pip", "show", "black-jaguar-backend"],
        capture_output=True,
    )
    if check.returncode == 0:
        return
    step("Installing backend dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(BACKEND)],
        check=True,
    )


def install_frontend() -> None:
    if (FRONTEND / "node_modules").exists():
        return
    step("Installing frontend dependencies...")
    subprocess.run(
        ["pnpm", "install"],
        cwd=str(FRONTEND),
        check=True,
        shell=WIN,
    )


def free_port(port: int) -> None:
    """Kill whatever is occupying a port so servers start cleanly."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return  # Port is free
    step(f"Port {port} is already in use -- freeing it...")
    if WIN:
        # netstat to find PID, then taskkill
        out = subprocess.run(
            f'netstat -ano | findstr ":{port} "',
            capture_output=True, text=True, shell=True,
        )
        pids: set[str] = set()
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.add(parts[-1])
        for pid in pids:
            subprocess.run(f"taskkill /PID {pid} /F", shell=True, capture_output=True)
    else:
        subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, capture_output=True)
    time.sleep(1)


def start_servers() -> None:
    free_port(8000)
    free_port(3000)
    step("Starting servers...\n")
    print("    Backend  : http://localhost:8000")
    print("    Frontend : http://localhost:3000")
    print()
    print("    Open http://localhost:3000 in your browser.")
    print("    Press Ctrl+C to stop.\n")

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app", "--reload", "--port", "8000"],
        cwd=str(BACKEND),
    )
    frontend = subprocess.Popen(
        ["pnpm", "dev"],
        cwd=str(FRONTEND),
        shell=WIN,
    )

    try:
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(1)
        if backend.poll() is not None:
            print("\n  Backend exited unexpectedly.")
        if frontend.poll() is not None:
            print("\n  Frontend exited unexpectedly.")
    except KeyboardInterrupt:
        print("\n  Shutting down...")
    finally:
        for proc in [backend, frontend]:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def main() -> None:
    print()
    print("  ========================================")
    print("     Onyx Leopard -- Setup & Launch")
    print("  ========================================")

    check_prerequisites()
    setup_env()
    install_backend()
    install_frontend()
    start_servers()


if __name__ == "__main__":
    main()
