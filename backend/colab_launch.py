#!/usr/bin/env python3
"""One-command launcher for Google Colab or a Linux GPU host.

It starts Uvicorn, waits for the health endpoint, then opens a public tunnel
with cloudflared (default) or ngrok and prints the URL to paste into the
Picto3D web studio.

Usage:
    python colab_launch.py                 # cloudflared quick tunnel
    python colab_launch.py --no-tunnel     # local only
    python colab_launch.py --tunnel ngrok  # needs NGROK_AUTHTOKEN + pyngrok
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN_DIR = ROOT / ".bin"
URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def log(message: str) -> None:
    print(f"[picto3d] {message}", flush=True)


def banner(title: str, lines: list[str]) -> None:
    width = max([len(title), *(len(line) for line in lines)]) + 4
    print("\n" + "╔" + "═" * width + "╗")
    print("║ " + title.ljust(width - 2) + " ║")
    print("╠" + "═" * width + "╣")
    for line in lines:
        print("║ " + line.ljust(width - 2) + " ║")
    print("╚" + "═" * width + "╝\n", flush=True)


def check_dependencies() -> None:
    missing: list[str] = []
    for module in ("fastapi", "uvicorn", "torch", "rembg", "trimesh"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        log("Thiếu thư viện: " + ", ".join(missing))
        log("Chạy trước:  pip install -r requirements-colab.txt")
        log("Sau đó:      pip install git+https://github.com/tatsy/torchmcubes.git")
        raise SystemExit(1)


def wait_for_health(port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def cloudflared_asset() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        return (
            "cloudflared-linux-arm64"
            if machine in {"aarch64", "arm64"}
            else "cloudflared-linux-amd64"
        )
    if system == "darwin":
        return (
            "cloudflared-darwin-arm64"
            if machine in {"arm64", "aarch64"}
            else "cloudflared-darwin-amd64"
        )
    raise RuntimeError(
        "Tự tải cloudflared chỉ hỗ trợ Linux/macOS. Trên Windows hãy dùng "
        "`cloudflared tunnel --url http://localhost:8000` sau khi cài cloudflared."
    )


def ensure_cloudflared() -> Path:
    existing = shutil.which("cloudflared")
    if existing:
        return Path(existing)

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "cloudflared"
    if target.exists():
        return target

    asset = cloudflared_asset()
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
    log(f"Đang tải cloudflared ({asset})…")
    urllib.request.urlretrieve(url, target)
    target.chmod(0o755)
    log("Đã tải cloudflared.")
    return target


def start_cloudflared(port: int) -> tuple[subprocess.Popen[str], str]:
    binary = ensure_cloudflared()
    process = subprocess.Popen(
        [
            str(binary),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{port}",
            "--no-autoupdate",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    public_url = ""
    ready = threading.Event()

    def reader() -> None:
        nonlocal public_url
        assert process.stdout is not None
        for line in process.stdout:
            match = URL_PATTERN.search(line)
            if match and not public_url:
                public_url = match.group(0)
                ready.set()
            if "ERR" in line or "error" in line.lower():
                print(f"[cloudflared] {line.rstrip()}", flush=True)

    threading.Thread(target=reader, name="cloudflared-log", daemon=True).start()
    ready.wait(timeout=45)
    return process, public_url


def start_ngrok(port: int) -> tuple[object, str]:
    try:
        from pyngrok import ngrok
    except Exception as exc:
        raise RuntimeError(
            "Chưa cài pyngrok. Chạy: pip install pyngrok, "
            "và đặt NGROK_AUTHTOKEN nếu tài khoản yêu cầu."
        ) from exc

    token = os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)
    tunnel = ngrok.connect(port, "http")
    public_url = str(tunnel.public_url)
    if public_url.startswith("http://"):
        public_url = "https://" + public_url[len("http://") :]
    return tunnel, public_url


def stream_process(process: subprocess.Popen[str], prefix: str) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Picto3D backend + public tunnel")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tunnel", choices=["cloudflared", "ngrok", "none"], default="cloudflared"
    )
    parser.add_argument(
        "--no-tunnel", action="store_true", help="Shortcut for --tunnel none"
    )
    args = parser.parse_args()
    if args.no_tunnel:
        args.tunnel = "none"

    check_dependencies()

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    extra_paths = [str(ROOT)]
    triposr = env.get("PICTO3D_TRIPOSR_PATH") or "/content/TripoSR"
    if Path(triposr, "tsr").is_dir():
        extra_paths.append(triposr)
        env.setdefault("PICTO3D_TRIPOSR_PATH", triposr)
    env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + ([python_path] if python_path else [])
    )

    log(f"Khởi động API tại http://127.0.0.1:{args.port} …")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--proxy-headers",
            "--forwarded-allow-ips=*",
            "--log-level",
            "info",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_process, args=(server, "uvicorn"), name="uvicorn-log", daemon=True
    ).start()

    if not wait_for_health(args.port):
        log("Server không khởi động được. Xem log [uvicorn] phía trên.")
        server.terminate()
        return 1

    log("Server đã sẵn sàng (model AI tiếp tục nạp nền nếu là lần đầu).")

    tunnel_process: subprocess.Popen[str] | None = None
    ngrok_tunnel: object | None = None
    public_url = f"http://127.0.0.1:{args.port}"

    if args.tunnel == "cloudflared":
        try:
            tunnel_process, public_url = start_cloudflared(args.port)
        except Exception as exc:
            log(f"Không mở được cloudflared: {exc}")
            log(f"API vẫn chạy cục bộ tại {public_url}")
    elif args.tunnel == "ngrok":
        try:
            ngrok_tunnel, public_url = start_ngrok(args.port)
        except Exception as exc:
            log(f"Không mở được ngrok: {exc}")
            log(f"API vẫn chạy cục bộ tại {public_url}")

    if public_url.startswith("http"):
        (ROOT / "public_url.txt").write_text(public_url + "\n", encoding="utf-8")
        banner(
            "PICTO3D API ĐÃ SẴN SÀNG",
            [
                f"API URL : {public_url}",
                f"Health  : {public_url}/api/health",
                f"Docs    : {public_url}/docs",
                "",
                "Dán API URL vào ô 'Backend API' của Picto3D Studio.",
                "Nhấn Ctrl+C để dừng server và đóng tunnel.",
            ],
        )

    try:
        while True:
            time.sleep(1.0)
            if server.poll() is not None:
                log("Server đã dừng.")
                break
            if tunnel_process is not None and tunnel_process.poll() is not None:
                log("Tunnel cloudflared đã dừng; API vẫn chạy cục bộ.")
                tunnel_process = None
    except KeyboardInterrupt:
        log("Đang dừng…")
    finally:
        if tunnel_process is not None:
            tunnel_process.terminate()
        if ngrok_tunnel is not None:
            try:
                from pyngrok import ngrok

                ngrok.disconnect(str(getattr(ngrok_tunnel, "public_url", "")))
            except Exception:
                pass
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
