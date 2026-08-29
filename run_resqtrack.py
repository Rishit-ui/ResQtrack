"""Start the whole ResQTrack prototype with one command.

    python run_resqtrack.py                       # backend + detector
    python run_resqtrack.py --backend-only        # just the API and the UIs
    python run_resqtrack.py --source data/test.mp4 --loop
    python run_resqtrack.py --source 0            # live webcam

It launches the FastAPI backend, waits for it to answer, then starts the
detector pointed at it.  Ctrl-C stops both cleanly.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent


def wait_for_backend(url: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/api/health", timeout=3.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.6)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ResQTrack prototype")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", default="dataset/accident/accident02.mp4")
    parser.add_argument("--sensitivity", default="balanced",
                        choices=("balanced", "high", "strict"))
    parser.add_argument("--camera-id", default="CAM-001")
    parser.add_argument("--lat", type=float, default=12.9719)
    parser.add_argument("--lon", type=float, default=77.5937)
    parser.add_argument("--location-name", default="Trinity Junction, MG Road")
    parser.add_argument("--loop", action="store_true", help="replay the clip forever")
    parser.add_argument("--headless", action="store_true", help="no detector window")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--stream-port", type=int, default=8001)
    args = parser.parse_args(argv)

    backend_url = f"http://localhost:{args.port}"
    processes: list[subprocess.Popen] = []

    print("\n" + "=" * 62)
    print("  Starting ResQTrack")
    print("=" * 62)

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app",
         "--host", args.host, "--port", str(args.port)],
        cwd=ROOT,
    )
    processes.append(backend)

    if not wait_for_backend(backend_url):
        print("  Backend did not start. Check the log above.")
        backend.terminate()
        return 1

    print(f"\n  Control room : {backend_url}/dashboard")
    print(f"  Responder app: {backend_url}/responder")
    print(f"  API docs     : {backend_url}/docs")

    if not args.backend_only:
        command = [
            sys.executable, "main.py",
            "--source", args.source,
            "--backend", backend_url,
            "--camera-id", args.camera_id,
            "--lat", str(args.lat),
            "--lon", str(args.lon),
            "--location-name", args.location_name,
            "--sensitivity", args.sensitivity,
            "--stream-port", str(args.stream_port),
        ]
        if args.loop:
            command.append("--loop")
        if args.headless:
            command.append("--headless")
        print(f"  Detector feed: http://localhost:{args.stream_port}/stream\n")
        processes.append(subprocess.Popen(command, cwd=ROOT))
    else:
        print("\n  Detector not started (--backend-only).\n")

    print("  Press Ctrl-C to stop everything.\n" + "=" * 62 + "\n")

    def shutdown(*_args):
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        print("\nResQTrack stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            for process in processes:
                if process.poll() is not None and process is backend:
                    print("Backend exited; shutting down.")
                    shutdown()
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
