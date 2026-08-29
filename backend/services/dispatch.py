"""Choosing who to send to an incident.

Selection is not simply "the closest ambulance".  A responder that is already
on another case must not be pulled off it, a responder whose phone has not
reported a position for ten minutes is probably not really available, and the
type of responder matters: a critical casualty needs an ambulance, a blocked
carriageway needs a patrol.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.services.geo import haversine_km

# Severity decides how wide the net is cast and how many crews are alerted.
DISPATCH_PLAN = {
    "CRITICAL": {"radius_km": 25.0, "count": 4},
    "HIGH": {"radius_km": 18.0, "count": 3},
    "MODERATE": {"radius_km": 12.0, "count": 2},
    "LOW": {"radius_km": 8.0, "count": 1},
}

# Which responder types suit which incident, best first.
PREFERRED_TYPES = {
    "vehicle_pedestrian_collision": ("AMBULANCE", "PARAMEDIC", "POLICE"),
    "pedestrian_down": ("AMBULANCE", "PARAMEDIC"),
    "possible_hit_and_run": ("AMBULANCE", "POLICE", "PARAMEDIC"),
    "multi_vehicle_pileup": ("AMBULANCE", "FIRE", "POLICE"),
    "vehicle_rollover": ("FIRE", "AMBULANCE", "POLICE"),
    "vehicle_immobilised": ("POLICE", "TOW", "AMBULANCE"),
}
DEFAULT_TYPES = ("AMBULANCE", "PARAMEDIC", "POLICE", "FIRE")

STALE_AFTER_MINUTES = 12
AVERAGE_RESPONSE_SPEED_KMH = 38.0


def _minutes_since(timestamp: str | None) -> float:
    if not timestamp:
        return 1e6
    try:
        seen = datetime.fromisoformat(timestamp)
    except ValueError:
        return 1e6
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - seen).total_seconds() / 60.0


def rank_responders(
    connection: sqlite3.Connection,
    latitude: float,
    longitude: float,
    *,
    severity: str = "HIGH",
    kind: str = "",
    online: set[str] | None = None,
    include_busy: bool = False,
) -> list[dict[str, Any]]:
    """Rank every plausible responder for one incident, best first."""
    plan = DISPATCH_PLAN.get(severity.upper(), DISPATCH_PLAN["HIGH"])
    preferred = PREFERRED_TYPES.get(kind, DEFAULT_TYPES)

    statuses = ("AVAILABLE",) if not include_busy else tuple(("AVAILABLE", "ALERTED"))
    placeholders = ",".join("?" for _ in statuses)
    rows = connection.execute(
        f"SELECT * FROM responders WHERE status IN ({placeholders})", statuses
    ).fetchall()

    ranked: list[dict[str, Any]] = []
    for row in rows:
        responder = dict(row)
        if responder.get("active_incident"):
            continue

        distance = haversine_km(latitude, longitude, responder["latitude"], responder["longitude"])
        if distance > plan["radius_km"]:
            continue

        idle_minutes = _minutes_since(responder.get("last_seen"))
        responder_type = (responder.get("responder_type") or "AMBULANCE").upper()
        type_rank = preferred.index(responder_type) if responder_type in preferred else len(preferred)

        # Cost is minutes-to-scene, penalised for a stale position and for
        # sending the wrong kind of unit.
        eta = distance / AVERAGE_RESPONSE_SPEED_KMH * 60.0
        cost = eta + type_rank * 2.5
        if idle_minutes > STALE_AFTER_MINUTES:
            cost += 15.0
        if online is not None and responder["responder_id"] not in online:
            cost += 6.0   # not currently connected: slower to see the alert

        ranked.append(
            {
                "responder_id": responder["responder_id"],
                "name": responder["name"],
                "type": responder_type,
                "phone": responder.get("phone"),
                "vehicle_number": responder.get("vehicle_number"),
                "latitude": responder["latitude"],
                "longitude": responder["longitude"],
                "distance_km": round(distance, 2),
                "eta_minutes": round(eta, 1),
                "idle_minutes": round(idle_minutes, 1),
                "connected": online is None or responder["responder_id"] in online,
                "score": round(cost, 2),
            }
        )

    ranked.sort(key=lambda item: item["score"])
    return ranked


def selection_size(severity: str) -> int:
    return DISPATCH_PLAN.get(severity.upper(), DISPATCH_PLAN["HIGH"])["count"]
