"""Ships confirmed incidents from the detector to the ResQTrack backend.

The video loop must never block on the network, so every alert is queued and
sent from a background worker.  If the backend is down the alert is retried
with a backoff and, once retries are exhausted, written to ``alerts_offline/``
so nothing is silently lost.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

OFFLINE_DIR = Path("alerts_offline")


@dataclass
class AlertPayload:
    """One confirmed incident, ready for POST /api/incidents."""

    camera_id: str
    latitude: float
    longitude: float
    kind: str
    label: str
    severity: str
    confidence: float
    reason: str
    frame: int
    evidence_types: list[str]
    signals: list[dict[str, Any]]
    involved: list[dict[str, Any]]
    vehicles: list[dict[str, Any]]
    ml_probability: float
    detected_at: str
    snapshot: bytes | None = field(default=None, repr=False)

    def as_json(self) -> dict[str, Any]:
        data = {
            key: value
            for key, value in self.__dict__.items()
            if key != "snapshot"
        }
        return data


class AlertClient:
    """Background sender with retry, backoff and an offline spool."""

    def __init__(
        self,
        base_url: str,
        *,
        enabled: bool = True,
        timeout: float = 8.0,
        max_attempts: int = 4,
        on_result: Callable[[str, str], None] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        # The backend is an internal service.  Honouring HTTP_PROXY/HTTPS_PROXY
        # for it breaks the detector on any machine behind a corporate proxy,
        # so those variables are deliberately ignored here.
        self.trust_env = False
        self.enabled = enabled
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.on_result = on_result
        self.queue: "queue.Queue[AlertPayload | None]" = queue.Queue()
        self.state = "idle"
        self.last_incident_id: str | None = None
        self.sent_count = 0
        self.failed_count = 0
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, name="resqtrack-alerts", daemon=True)
        self._worker.start()

    def stop(self, drain_seconds: float = 6.0) -> None:
        if self._worker is None:
            return
        deadline = time.time() + drain_seconds
        while not self.queue.empty() and time.time() < deadline:
            time.sleep(0.1)
        self._stop.set()
        self.queue.put(None)
        self._worker.join(timeout=3.0)
        self._worker = None

    # ------------------------------------------------------------------
    def send(self, payload: AlertPayload) -> None:
        if not self.enabled:
            self.state = "disabled"
            return
        self.queue.put(payload)
        self.state = "queued"

    def check_backend(self) -> bool:
        """One-shot reachability probe used at start-up."""
        if not self.enabled:
            return False
        try:
            response = httpx.get(
                f"{self.base_url}/api/health", timeout=4.0, trust_env=self.trust_env
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                break
            self._deliver(payload)
            self.queue.task_done()

    def _deliver(self, payload: AlertPayload) -> None:
        self.state = "sending"
        delay = 1.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout, trust_env=self.trust_env) as client:
                    response = client.post(
                        f"{self.base_url}/api/incidents",
                        json=payload.as_json(),
                    )
                    response.raise_for_status()
                    body = response.json()
                    incident = body.get("incident", {})
                    incident_id = incident.get("incident_id", "?")
                    self.last_incident_id = incident_id
                    self.sent_count += 1
                    self.state = f"sent {incident_id}"

                    if payload.snapshot:
                        self._upload_snapshot(client, incident_id, payload.snapshot)

                    notified = len(body.get("dispatched_to", []))
                    self._report(
                        "sent",
                        f"{incident_id} accepted by backend, {notified} responder(s) alerted",
                    )
                    return
            except httpx.HTTPError as error:
                self._report(
                    "retry",
                    f"attempt {attempt}/{self.max_attempts} failed: {type(error).__name__}",
                )
                if attempt < self.max_attempts and not self._stop.is_set():
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)

        self.failed_count += 1
        self.state = "offline spool"
        self._spool(payload)

    def _upload_snapshot(self, client: httpx.Client, incident_id: str, snapshot: bytes) -> None:
        try:
            client.post(
                f"{self.base_url}/api/incidents/{incident_id}/snapshot",
                files={"file": (f"{incident_id}.jpg", snapshot, "image/jpeg")},
                timeout=self.timeout,
            )
        except httpx.HTTPError:
            # The incident itself is already recorded; the image is a bonus.
            self._report("warn", f"snapshot upload failed for {incident_id}")

    def _spool(self, payload: AlertPayload) -> None:
        OFFLINE_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = OFFLINE_DIR / f"incident-{stamp}-frame{payload.frame}.json"
        path.write_text(json.dumps(payload.as_json(), indent=2), encoding="utf-8")
        if payload.snapshot:
            path.with_suffix(".jpg").write_bytes(payload.snapshot)
        self._report("offline", f"backend unreachable, alert written to {path}")

    def _report(self, level: str, message: str) -> None:
        if self.on_result:
            self.on_result(level, message)
        else:
            print(f"[alert:{level}] {message}")
