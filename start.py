#!/usr/bin/env python3
"""Onyx Leopard - one-command setup and launch."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
ENV_FILE = BACKEND / ".env"
KEY_NAME = "OPENROUTER_API_KEY"
KEY_CHECK_URL = "https://openrouter.ai/api/v1/key"
WIN = sys.platform == "win32"
USE_UV = shutil.which("uv") is not None


def fail(msg: str) -> None:
    print(f"\n  ERROR: {msg}\n")
    sys.exit(1)


def step(msg: str) -> None:
    print(f"\n  > {msg}")


def check_prerequisites() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036 -- this guard is the point
        fail(f"Python 3.12+ required (you have {sys.version.split()[0]})")
    if not shutil.which("node"):
        fail("Node.js not found. Download it from https://nodejs.org/")
    if not shutil.which("pnpm"):
        fail("pnpm not found. Install it with:  npm install -g pnpm")


def read_env_key() -> str:
    if os.environ.get(KEY_NAME):
        return os.environ[KEY_NAME].strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == KEY_NAME and value.strip():
                return value.strip()
    return ""


def save_env_key(key: str) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [line for line in lines if line.partition("=")[0].strip() != KEY_NAME]
    lines.append(f"{KEY_NAME}={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_key(key: str) -> None:
    """Ask OpenRouter about the key. Warns only, so offline launches still work."""
    request = urllib.request.Request(KEY_CHECK_URL, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            data = json.load(resp).get("data", {})
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("    WARNING: OpenRouter rejected this key. AI CEOs will fail until it is fixed.")
            print(f"    Edit {ENV_FILE} or delete the {KEY_NAME} line and rerun start.py.")
        else:
            print(f"    Could not verify the key (HTTP {exc.code}); continuing.")
        return
    except (urllib.error.URLError, TimeoutError, ValueError):
        print("    Could not reach OpenRouter to verify the key; continuing.")
        return
    remaining = data.get("limit_remaining")
    credit = f"${remaining:,.2f} credit left" if isinstance(remaining, int | float) else "no limit"
    print(f"    OpenRouter key OK ({credit}).")


def setup_key() -> None:
    key = read_env_key()
    if not key:
        step("AI CEOs run on OpenRouter (cheap Chinese models by default, Claude optional).")
        print("    Get a key at https://openrouter.ai/keys")
        try:
            key = input(f"    Paste your {KEY_NAME} (or press Enter to skip): ").strip()
        except EOFError:  # non-interactive launch
            key = ""
        if not key:
            print("    Skipped. Heuristic CEOs only until you add the key to backend/.env.")
            return
        save_env_key(key)
        print(f"    Saved to {ENV_FILE.relative_to(ROOT)}")
    check_key(key)


def install_backend() -> None:
    if USE_UV:  # idempotent and fast; keeps the venv in sync with uv.lock
        step("Syncing backend dependencies (uv)...")
        subprocess.run(["uv", "sync", "--quiet"], cwd=BACKEND, check=True)
        return
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "show", "onyx-leopard-backend"], capture_output=True
    )
    if installed.returncode != 0:
        step("Installing backend dependencies (pip)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(BACKEND)], check=True)


def install_frontend() -> None:
    if (FRONTEND / "node_modules").exists():
        return
    step("Installing frontend dependencies...")
    subprocess.run(["pnpm", "install"], cwd=FRONTEND, check=True, shell=WIN)


def free_port(port: int) -> None:
    """Kill whatever holds a port so the servers start cleanly."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return
    step(f"Port {port} is in use -- freeing it...")
    if WIN:
        out = subprocess.run(
            f'netstat -ano | findstr ":{port} "', capture_output=True, text=True, shell=True
        )
        pids = {parts[-1] for line in out.stdout.splitlines() if (parts := line.split())}
        for pid in {p for p in pids if p.isdigit()} - {"0"}:
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
    print("\n    Open http://localhost:3000 in your browser. Press Ctrl+C to stop.\n")

    uvicorn = ["uvicorn", "src.main:app", "--reload", "--port", "8000"]
    backend_cmd = ["uv", "run", *uvicorn] if USE_UV else [sys.executable, "-m", *uvicorn]
    backend = subprocess.Popen(backend_cmd, cwd=BACKEND)
    frontend = subprocess.Popen(["pnpm", "dev"], cwd=FRONTEND, shell=WIN)
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
        for proc in (backend, frontend):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def main() -> None:
    print("\n  ========================================")
    print("     Onyx Leopard -- Setup & Launch")
    print("  ========================================")
    check_prerequisites()
    setup_key()
    install_backend()
    install_frontend()
    start_servers()


if __name__ == "__main__":
    main()
