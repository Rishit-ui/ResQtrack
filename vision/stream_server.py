"""Tiny MJPEG server that publishes the annotated detector view.

The control-room dashboard embeds this as a plain ``<img>`` tag, so the judges
see the same YOLO11 view the detector sees, live, in the browser - with no
plugin, no WebRTC and no extra dependency.

Only the most recent frame is kept.  A slow browser therefore drops frames
instead of delaying the detector, which must always run at video speed.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

BOUNDARY = "resqtrackframe"


class FrameBuffer:
    """Latest-frame-wins buffer shared between the detector and HTTP threads."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0

    def publish(self, frame: np.ndarray, quality: int = 75) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def wait(self, last_sequence: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        with self._condition:
            if self._sequence == last_sequence:
                self._condition.wait(timeout)
            return self._jpeg, self._sequence

    @property
    def sequence(self) -> int:
        return self._sequence


class _Handler(BaseHTTPRequestHandler):
    buffer: FrameBuffer
    status_provider: Any = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:  # keep the console clean
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = self.path.split("?")[0]
        if path in ("/", "/stream", "/stream.mjpg"):
            self._serve_stream()
        elif path == "/frame.jpg":
            self._serve_single_frame()
        elif path == "/status":
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self._cors()
        self.end_headers()
        sequence = -1
        try:
            while True:
                jpeg, sequence = self.buffer.wait(sequence)
                if jpeg is None:
                    continue
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser tab closed

    def _serve_single_frame(self) -> None:
        jpeg, _ = self.buffer.wait(-1, timeout=2.0)
        if jpeg is None:
            self.send_error(503, "no frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self._cors()
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_status(self) -> None:
        import json

        payload = self.status_provider() if self.status_provider else {}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StreamServer:
    """Serves ``/stream`` (MJPEG), ``/frame.jpg`` and ``/status`` (JSON)."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.buffer = FrameBuffer()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self.status_provider: Any = None

    def start(self) -> str | None:
        handler = type("ResQTrackStreamHandler", (_Handler,), {"buffer": self.buffer})
        handler.status_provider = staticmethod(lambda: self.status_provider() if self.status_provider else {})
        try:
            self._server = _Server((self.host, self.port), handler)
        except OSError as error:
            print(f"[stream] could not bind {self.host}:{self.port} ({error})")
            return None
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://localhost:{self.port}/stream"

    def publish(self, frame: np.ndarray, quality: int = 75) -> None:
        self.buffer.publish(frame, quality)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
