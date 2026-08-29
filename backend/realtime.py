"""WebSocket hub - the part that makes ResQTrack a real-time system.

Every client (a responder's phone, the control-room dashboard, a hospital desk)
opens one socket and declares who it is.  The backend then pushes events to
exactly the clients that need them:

* ``control``  sees everything - the dashboard's live feed
* ``responder`` sees alerts addressed to it, plus updates on its own case
* ``hospital``  sees inbound-patient notices

Nothing polls.  A confirmed accident reaches a responder's screen in the time
it takes one JSON frame to cross the network.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from backend.database import now_utc


@dataclass
class Client:
    socket: WebSocket
    role: str                      # control | responder | hospital
    identity: str                  # responder id, or a dashboard label
    incidents: set[str] = field(default_factory=set)   # cases it is following

    def follows(self, incident_id: str | None) -> bool:
        return incident_id is not None and incident_id in self.incidents


class RealtimeHub:
    """Fan-out for live events, with per-role and per-incident targeting."""

    def __init__(self) -> None:
        self._clients: list[Client] = []
        self._lock = asyncio.Lock()
        self.history: list[dict[str, Any]] = []   # last events, for late joiners

    # ------------------------------------------------------------------
    async def connect(self, socket: WebSocket, role: str, identity: str) -> Client:
        await socket.accept()
        client = Client(socket=socket, role=role or "control", identity=identity or "anonymous")
        async with self._lock:
            self._clients.append(client)
        await self._send(
            client,
            {
                "type": "connected",
                "role": client.role,
                "identity": client.identity,
                "server_time": now_utc(),
                "recent": self.history[-15:],
            },
        )
        return client

    async def disconnect(self, client: Client) -> None:
        async with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    async def follow(self, client: Client, incident_id: str) -> None:
        client.incidents.add(incident_id)

    async def unfollow(self, client: Client, incident_id: str) -> None:
        client.incidents.discard(incident_id)

    # ------------------------------------------------------------------
    async def broadcast(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        roles: set[str] | None = None,
        responders: set[str] | None = None,
        incident_id: str | None = None,
    ) -> None:
        """Publish one event.

        ``roles`` and ``responders`` are additive filters: a client receives the
        event if its role is listed, or its id is listed, or it is following the
        incident.  With no filters at all the event goes to everyone.
        """
        message = {
            "type": event_type,
            "incident_id": incident_id,
            "sent_at": now_utc(),
            "data": payload,
        }
        self.history.append(message)
        del self.history[:-60]

        async with self._lock:
            targets = list(self._clients)

        stale: list[Client] = []
        for client in targets:
            if not self._matches(client, roles, responders, incident_id):
                continue
            try:
                await self._send(client, message)
            except Exception:  # noqa: BLE001 - a dead socket must not stop the fan-out
                stale.append(client)

        for client in stale:
            await self.disconnect(client)

    @staticmethod
    def _matches(
        client: Client,
        roles: set[str] | None,
        responders: set[str] | None,
        incident_id: str | None,
    ) -> bool:
        if roles is None and responders is None:
            return True
        if roles and client.role in roles:
            return True
        if responders and client.role == "responder" and client.identity in responders:
            return True
        return client.follows(incident_id)

    @staticmethod
    async def _send(client: Client, message: dict[str, Any]) -> None:
        await client.socket.send_text(json.dumps(message, default=str))

    # ------------------------------------------------------------------
    @property
    def connections(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for client in self._clients:
            counts[client.role] = counts.get(client.role, 0) + 1
        return counts

    def online_responders(self) -> set[str]:
        return {client.identity for client in self._clients if client.role == "responder"}


hub = RealtimeHub()
