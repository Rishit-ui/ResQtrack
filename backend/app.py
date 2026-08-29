"""ResQTrack emergency-response backend.

The whole incident lifecycle lives here:

    detector confirms a crash
        -> POST /api/incidents
        -> nearest suitable responders are chosen and alerted over WebSocket
        -> a responder ACCEPTS; everyone else is told the case is taken
        -> the responder streams GPS; arrival is detected by geofence
        -> nearby hospitals are ranked by road time from the scene
        -> the responder picks one; a least-traffic route is returned
        -> the case is closed, and every step of it is in the database

Run it with:

    python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.database import (
    RESPONDER_STATES,
    SNAPSHOT_DIR,
    add_notification,
    can_transition,
    connect,
    initialise,
    log_event,
    next_id,
    now_utc,
    row_to_dict,
    rows_to_list,
)
from backend.realtime import hub
from backend.services.dispatch import rank_responders, selection_size
from backend.services.geo import haversine_km, valid_coordinate
from backend.services.hospital_service import HospitalSearchError, find_nearby_facilities
from backend.services.routing_service import RoutingError, get_route

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# A responder within this distance of the scene is treated as having arrived.
ARRIVAL_RADIUS_METRES = 75.0

app = FastAPI(
    title="ResQTrack Emergency Response API",
    description="Real-time accident detection, dispatch, navigation and handover.",
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialise()
STATIC_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")


# ============================================================
# REQUEST MODELS
# ============================================================

class IncidentCreate(BaseModel):
    camera_id: str
    latitude: float
    longitude: float
    confidence: float = 0.0
    kind: str = "unknown"
    label: str = "Incident"
    severity: str = "HIGH"
    reason: str = ""
    location_name: str | None = None
    frame: Optional[int] = None
    evidence_types: list[str] = Field(default_factory=list)
    signals: list[dict[str, Any]] = Field(default_factory=list)
    involved: list[dict[str, Any]] = Field(default_factory=list)
    vehicles: list[dict[str, Any]] = Field(default_factory=list)
    ml_probability: float = 0.0
    detected_at: str | None = None


class ResponderCreate(BaseModel):
    name: str
    responder_type: str = "AMBULANCE"
    latitude: float
    longitude: float
    phone: str | None = None
    vehicle_number: str | None = None


class LocationUpdate(BaseModel):
    latitude: float
    longitude: float
    accuracy: float | None = None
    speed_kmh: float | None = None
    heading: float | None = None


class StatusUpdate(BaseModel):
    status: str


class ResponderAction(BaseModel):
    responder_id: str


class HospitalChoice(BaseModel):
    responder_id: str
    name: str
    latitude: float
    longitude: float
    ref: str | None = None
    phone: str | None = None


class CloseIncident(BaseModel):
    responder_id: str | None = None
    outcome: str = "RESOLVED"
    notes: str | None = None


# ============================================================
# HELPERS
# ============================================================

def fetch_incident(connection, incident_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"Incident {incident_id} not found")
    return row_to_dict(row)  # type: ignore[return-value]


def fetch_responder(connection, responder_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM responders WHERE responder_id = ?", (responder_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"Responder {responder_id} not found")
    return row_to_dict(row)  # type: ignore[return-value]


def set_incident_status(connection, incident_id: str, current: str, target: str) -> None:
    if current == target:
        return
    if not can_transition(current, target):
        raise HTTPException(409, f"Cannot move incident from {current} to {target}")
    connection.execute(
        "UPDATE incidents SET status = ? WHERE incident_id = ?", (target, incident_id)
    )


# ============================================================
# SYSTEM
# ============================================================

@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse("/dashboard")


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/responder")
def responder_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "responder.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    connection = connect()
    try:
        counts = {
            "incidents": connection.execute("SELECT COUNT(*) c FROM incidents").fetchone()["c"],
            "open_incidents": connection.execute(
                "SELECT COUNT(*) c FROM incidents WHERE status NOT IN ('CLOSED','CANCELLED','FALSE_ALARM')"
            ).fetchone()["c"],
            "responders": connection.execute("SELECT COUNT(*) c FROM responders").fetchone()["c"],
            "available": connection.execute(
                "SELECT COUNT(*) c FROM responders WHERE status = 'AVAILABLE'"
            ).fetchone()["c"],
        }
    finally:
        connection.close()
    return {
        "service": "ResQTrack",
        "status": "ONLINE",
        "version": app.version,
        "server_time": now_utc(),
        "live_connections": hub.connections,
        **counts,
    }


@app.get("/api/stats")
def statistics() -> dict[str, Any]:
    connection = connect()
    try:
        by_kind = rows_to_list(
            connection.execute(
                "SELECT kind, COUNT(*) AS count FROM incidents GROUP BY kind ORDER BY count DESC"
            ).fetchall()
        )
        by_severity = rows_to_list(
            connection.execute(
                "SELECT severity, COUNT(*) AS count FROM incidents GROUP BY severity"
            ).fetchall()
        )
        response = connection.execute(
            """
            SELECT AVG((julianday(accepted_at) - julianday(created_at)) * 86400.0) AS seconds
            FROM incidents WHERE accepted_at IS NOT NULL
            """
        ).fetchone()["seconds"]
        on_scene = connection.execute(
            """
            SELECT AVG((julianday(arrived_at) - julianday(accepted_at)) * 86400.0) AS seconds
            FROM incidents WHERE arrived_at IS NOT NULL AND accepted_at IS NOT NULL
            """
        ).fetchone()["seconds"]
        totals = connection.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) AS closed,
              SUM(CASE WHEN status = 'FALSE_ALARM' THEN 1 ELSE 0 END) AS false_alarms
            FROM incidents
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "totals": dict(totals),
        "by_kind": by_kind,
        "by_severity": by_severity,
        "avg_acceptance_seconds": round(response or 0.0, 1),
        "avg_travel_seconds": round(on_scene or 0.0, 1),
    }


# ============================================================
# INCIDENTS
# ============================================================

@app.post("/api/incidents")
async def create_incident(data: IncidentCreate) -> dict[str, Any]:
    """Called by the detector the moment an accident is confirmed."""
    if not valid_coordinate(data.latitude, data.longitude):
        raise HTTPException(400, "Invalid camera coordinate")

    connection = connect()
    try:
        incident_id = next_id(connection, "incidents", "incident_id", "RQ-", 5)
        created_at = now_utc()
        connection.execute(
            """
            INSERT INTO incidents (
                incident_id, status, kind, label, severity, confidence, reason,
                evidence_types, signals, involved, vehicles, ml_probability,
                camera_id, location_name, latitude, longitude, frame,
                detected_at, created_at
            ) VALUES (?, 'DETECTED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                data.kind,
                data.label,
                data.severity.upper(),
                float(data.confidence),
                data.reason,
                json.dumps(data.evidence_types),
                json.dumps(data.signals),
                json.dumps(data.involved),
                json.dumps(data.vehicles),
                float(data.ml_probability),
                data.camera_id,
                data.location_name,
                data.latitude,
                data.longitude,
                data.frame,
                data.detected_at or created_at,
                created_at,
            ),
        )
        log_event(
            connection,
            incident_id,
            "detected",
            f"{data.label} confirmed on {data.camera_id}",
            actor=data.camera_id,
            payload={
                "confidence": data.confidence,
                "severity": data.severity,
                "evidence_types": data.evidence_types,
                "reason": data.reason,
            },
        )

        selected = _dispatch(connection, incident_id, data.severity, data.kind,
                             data.latitude, data.longitude)
        connection.commit()
        incident = fetch_incident(connection, incident_id)
    finally:
        connection.close()

    # The control room sees every detection; a responder only ever sees the
    # calls actually assigned to it, so nobody's phone screams for a crash on
    # the other side of the city.
    await hub.broadcast(
        "incident.created",
        {"incident": incident, "dispatched_to": selected},
        roles={"control"},
        incident_id=incident_id,
    )
    for responder in selected:
        await hub.broadcast(
            "incident.alert",
            {"incident": incident, "assignment": responder},
            roles=set(),
            responders={responder["responder_id"]},
        )

    return {"incident": incident, "dispatched_to": selected}


def _dispatch(
    connection,
    incident_id: str,
    severity: str,
    kind: str,
    latitude: float,
    longitude: float,
) -> list[dict[str, Any]]:
    """Pick and record the responders to alert.  Caller commits."""
    ranked = rank_responders(
        connection,
        latitude,
        longitude,
        severity=severity,
        kind=kind,
        online=hub.online_responders(),
    )
    selected = ranked[: selection_size(severity)]
    if not selected:
        log_event(
            connection,
            incident_id,
            "dispatch_failed",
            "No available responder within range - escalated to the control room",
        )
        add_notification(
            connection,
            "control",
            "No responder available",
            f"{incident_id} could not be assigned automatically.",
            incident_id,
            "critical",
        )
        return []

    timestamp = now_utc()
    for responder in selected:
        connection.execute(
            """
            INSERT INTO dispatches (incident_id, responder_id, status, distance_km,
                                    eta_minutes, sent_at)
            VALUES (?, ?, 'SENT', ?, ?, ?)
            ON CONFLICT(incident_id, responder_id) DO UPDATE SET
                status = 'SENT', sent_at = excluded.sent_at
            """,
            (
                incident_id,
                responder["responder_id"],
                responder["distance_km"],
                responder["eta_minutes"],
                timestamp,
            ),
        )
        connection.execute(
            "UPDATE responders SET status = 'ALERTED', last_seen = ? WHERE responder_id = ?",
            (timestamp, responder["responder_id"]),
        )
        add_notification(
            connection,
            responder["responder_id"],
            "Emergency dispatch",
            f"{incident_id}: {responder['distance_km']} km away",
            incident_id,
            "critical",
        )

    connection.execute(
        "UPDATE incidents SET status = 'DISPATCHED' WHERE incident_id = ?", (incident_id,)
    )
    log_event(
        connection,
        incident_id,
        "dispatched",
        f"Alert sent to {len(selected)} responder(s): "
        + ", ".join(item["responder_id"] for item in selected),
        payload={"responders": selected},
    )
    return selected


@app.post("/api/incidents/{incident_id}/snapshot")
async def upload_snapshot(incident_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Store the annotated frame that produced the alert."""
    connection = connect()
    try:
        fetch_incident(connection, incident_id)
        path = SNAPSHOT_DIR / f"{incident_id}.jpg"
        path.write_bytes(await file.read())
        relative = f"/snapshots/{path.name}"
        connection.execute(
            "UPDATE incidents SET snapshot_path = ? WHERE incident_id = ?", (relative, incident_id)
        )
        log_event(connection, incident_id, "snapshot", "Scene image stored")
        connection.commit()
    finally:
        connection.close()

    await hub.broadcast(
        "incident.snapshot", {"incident_id": incident_id, "snapshot_path": relative},
        incident_id=incident_id,
    )
    return {"incident_id": incident_id, "snapshot_path": relative}


@app.get("/api/incidents")
def list_incidents(
    status: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    active_only: bool = False,
) -> list[dict[str, Any]]:
    connection = connect()
    try:
        query = "SELECT * FROM incidents"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status.upper())
        elif active_only:
            query += " WHERE status NOT IN ('CLOSED','CANCELLED','FALSE_ALARM')"
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return rows_to_list(connection.execute(query, params).fetchall())
    finally:
        connection.close()


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str) -> dict[str, Any]:
    connection = connect()
    try:
        incident = fetch_incident(connection, incident_id)
        incident["dispatches"] = rows_to_list(
            connection.execute(
                """
                SELECT d.*, r.name, r.responder_type, r.phone, r.latitude, r.longitude, r.status AS responder_status
                FROM dispatches d JOIN responders r ON r.responder_id = d.responder_id
                WHERE d.incident_id = ? ORDER BY d.id
                """,
                (incident_id,),
            ).fetchall()
        )
        incident["timeline"] = rows_to_list(
            connection.execute(
                "SELECT * FROM incident_events WHERE incident_id = ? ORDER BY id",
                (incident_id,),
            ).fetchall()
        )
        return incident
    finally:
        connection.close()


@app.get("/api/incidents/{incident_id}/timeline")
def incident_timeline(incident_id: str) -> list[dict[str, Any]]:
    connection = connect()
    try:
        fetch_incident(connection, incident_id)
        return rows_to_list(
            connection.execute(
                "SELECT * FROM incident_events WHERE incident_id = ? ORDER BY id", (incident_id,)
            ).fetchall()
        )
    finally:
        connection.close()


@app.post("/api/incidents/{incident_id}/accept")
async def accept_incident(incident_id: str, action: ResponderAction) -> dict[str, Any]:
    """A responder takes the case.  First one wins; the rest are stood down."""
    connection = connect()
    try:
        incident = fetch_incident(connection, incident_id)
        responder = fetch_responder(connection, action.responder_id)

        if incident["assigned_responder"]:
            if incident["assigned_responder"] == action.responder_id:
                return {"incident": incident, "responder": responder, "already_yours": True}
            raise HTTPException(409, f"Already accepted by {incident['assigned_responder']}")
        if incident["status"] in ("CLOSED", "CANCELLED", "FALSE_ALARM"):
            raise HTTPException(409, f"Incident is {incident['status']}")
        if responder["status"] not in ("AVAILABLE", "ALERTED"):
            raise HTTPException(409, f"Responder is {responder['status']}")

        accepted_at = now_utc()
        set_incident_status(connection, incident_id, incident["status"], "ACCEPTED")
        connection.execute(
            "UPDATE incidents SET assigned_responder = ?, accepted_at = ? WHERE incident_id = ?",
            (action.responder_id, accepted_at, incident_id),
        )
        connection.execute(
            """
            UPDATE responders SET status = 'EN_ROUTE', active_incident = ?, last_seen = ?
            WHERE responder_id = ?
            """,
            (incident_id, accepted_at, action.responder_id),
        )
        connection.execute(
            """
            UPDATE dispatches SET status = 'ACCEPTED', responded_at = ?
            WHERE incident_id = ? AND responder_id = ?
            """,
            (accepted_at, incident_id, action.responder_id),
        )
        # Everyone else goes back in the pool.
        stood_down = [
            row["responder_id"]
            for row in connection.execute(
                "SELECT responder_id FROM dispatches WHERE incident_id = ? AND status = 'SENT'",
                (incident_id,),
            ).fetchall()
        ]
        connection.execute(
            "UPDATE dispatches SET status = 'SUPERSEDED', responded_at = ? WHERE incident_id = ? AND status = 'SENT'",
            (accepted_at, incident_id),
        )
        for other in stood_down:
            connection.execute(
                "UPDATE responders SET status = 'AVAILABLE' WHERE responder_id = ? AND active_incident IS NULL",
                (other,),
            )
        log_event(
            connection,
            incident_id,
            "accepted",
            f"{responder['name']} ({action.responder_id}) accepted the call",
            actor=action.responder_id,
        )
        connection.commit()

        incident = fetch_incident(connection, incident_id)
        responder = fetch_responder(connection, action.responder_id)
    finally:
        connection.close()

    route = None
    try:
        route = await get_route(
            responder["latitude"], responder["longitude"],
            incident["latitude"], incident["longitude"],
        )
    except RoutingError:
        route = None

    await hub.broadcast(
        "incident.accepted",
        {"incident": incident, "responder": responder, "stood_down": stood_down},
        incident_id=incident_id,
    )
    for other in stood_down:
        await hub.broadcast(
            "incident.stand_down",
            {"incident_id": incident_id, "reason": "another responder accepted"},
            roles=set(),
            responders={other},
            incident_id=incident_id,
        )

    return {"incident": incident, "responder": responder, "route": route}


@app.post("/api/incidents/{incident_id}/decline")
async def decline_incident(incident_id: str, action: ResponderAction) -> dict[str, Any]:
    connection = connect()
    try:
        fetch_incident(connection, incident_id)
        connection.execute(
            """
            UPDATE dispatches SET status = 'DECLINED', responded_at = ?
            WHERE incident_id = ? AND responder_id = ?
            """,
            (now_utc(), incident_id, action.responder_id),
        )
        connection.execute(
            "UPDATE responders SET status = 'AVAILABLE' WHERE responder_id = ? AND active_incident IS NULL",
            (action.responder_id,),
        )
        log_event(
            connection, incident_id, "declined",
            f"{action.responder_id} declined the call", actor=action.responder_id,
        )
        # Nobody left holding the case: widen the search.
        remaining = connection.execute(
            "SELECT COUNT(*) c FROM dispatches WHERE incident_id = ? AND status = 'SENT'",
            (incident_id,),
        ).fetchone()["c"]
        incident = fetch_incident(connection, incident_id)
        escalated: list[dict[str, Any]] = []
        if remaining == 0 and not incident["assigned_responder"]:
            escalated = _dispatch(
                connection, incident_id, incident["severity"], incident["kind"],
                incident["latitude"], incident["longitude"],
            )
        connection.commit()
    finally:
        connection.close()

    await hub.broadcast(
        "incident.declined",
        {"incident_id": incident_id, "responder_id": action.responder_id, "reassigned_to": escalated},
        incident_id=incident_id,
    )
    return {"incident_id": incident_id, "reassigned_to": escalated}


@app.post("/api/incidents/{incident_id}/arrive")
async def mark_arrived(incident_id: str, action: ResponderAction) -> dict[str, Any]:
    connection = connect()
    try:
        incident = _record_arrival(connection, incident_id, action.responder_id, manual=True)
        connection.commit()
    finally:
        connection.close()
    await hub.broadcast("incident.arrived", {"incident": incident}, incident_id=incident_id)
    return incident


def _record_arrival(connection, incident_id: str, responder_id: str, manual: bool) -> dict[str, Any]:
    incident = fetch_incident(connection, incident_id)
    if incident["assigned_responder"] != responder_id:
        raise HTTPException(403, "This incident is assigned to another responder")
    if incident["arrived_at"]:
        return incident
    arrived_at = now_utc()
    set_incident_status(connection, incident_id, incident["status"], "ON_SCENE")
    connection.execute(
        "UPDATE incidents SET arrived_at = ? WHERE incident_id = ?", (arrived_at, incident_id)
    )
    connection.execute(
        "UPDATE responders SET status = 'ON_SCENE', last_seen = ? WHERE responder_id = ?",
        (arrived_at, responder_id),
    )
    log_event(
        connection, incident_id, "on_scene",
        f"{responder_id} reached the scene" + ("" if manual else " (detected by geofence)"),
        actor=responder_id,
    )
    return fetch_incident(connection, incident_id)


@app.get("/api/incidents/{incident_id}/hospitals")
async def incident_hospitals(
    incident_id: str,
    radius_km: float = Query(12.0, ge=1.0, le=50.0),
    limit: int = Query(8, ge=1, le=25),
    kinds: str = "hospital,clinic",
) -> dict[str, Any]:
    """Facilities near the scene, ranked by road travel time from the scene."""
    connection = connect()
    try:
        incident = fetch_incident(connection, incident_id)
    finally:
        connection.close()

    try:
        result = await find_nearby_facilities(
            incident["latitude"],
            incident["longitude"],
            radius_km=radius_km,
            kinds=[item.strip() for item in kinds.split(",") if item.strip()],
            limit=limit,
        )
    except HospitalSearchError as error:
        raise HTTPException(503, f"Hospital lookup failed: {error}") from error

    await hub.broadcast(
        "incident.hospitals",
        {"incident_id": incident_id, "count": result["count"], "source": result["source"]},
        incident_id=incident_id,
    )
    return {"incident_id": incident_id, **result}


@app.post("/api/incidents/{incident_id}/hospital")
async def choose_hospital(incident_id: str, choice: HospitalChoice) -> dict[str, Any]:
    """Lock in the receiving hospital and return the least-traffic route to it."""
    if not valid_coordinate(choice.latitude, choice.longitude):
        raise HTTPException(400, "Invalid hospital coordinate")

    connection = connect()
    try:
        incident = fetch_incident(connection, incident_id)
        if incident["assigned_responder"] != choice.responder_id:
            raise HTTPException(403, "This incident is assigned to another responder")
        if not incident["arrived_at"]:
            raise HTTPException(409, "Mark arrival at the scene before selecting a hospital")

        started = now_utc()
        set_incident_status(connection, incident_id, incident["status"], "TRANSPORTING")
        connection.execute(
            """
            UPDATE incidents
            SET hospital_name = ?, hospital_latitude = ?, hospital_longitude = ?,
                transport_started_at = ?
            WHERE incident_id = ?
            """,
            (choice.name, choice.latitude, choice.longitude, started, incident_id),
        )
        connection.execute(
            "UPDATE responders SET status = 'TRANSPORTING', last_seen = ? WHERE responder_id = ?",
            (started, choice.responder_id),
        )
        log_event(
            connection, incident_id, "hospital_selected",
            f"Transporting to {choice.name}", actor=choice.responder_id,
            payload={"hospital": choice.model_dump()},
        )
        connection.commit()
        incident = fetch_incident(connection, incident_id)
    finally:
        connection.close()

    try:
        route = await get_route(
            incident["latitude"], incident["longitude"],
            choice.latitude, choice.longitude,
            prefer_least_traffic=True,
        )
    except RoutingError as error:
        raise HTTPException(503, f"Could not build a route: {error}") from error

    await hub.broadcast(
        "incident.transporting",
        {"incident": incident, "hospital": choice.model_dump(), "route": route},
        incident_id=incident_id,
    )
    return {"incident": incident, "hospital": choice.model_dump(), "route": route}


@app.post("/api/incidents/{incident_id}/close")
async def close_incident(incident_id: str, data: CloseIncident) -> dict[str, Any]:
    connection = connect()
    try:
        incident = fetch_incident(connection, incident_id)
        if incident["status"] in ("CLOSED", "CANCELLED", "FALSE_ALARM"):
            return incident

        outcome = data.outcome.upper()
        target = "FALSE_ALARM" if outcome == "FALSE_ALARM" else "CLOSED"
        if not can_transition(incident["status"], target):
            # A control-room operator may close a case from any live state.
            connection.execute(
                "UPDATE incidents SET status = ? WHERE incident_id = ?", (target, incident_id)
            )
        else:
            set_incident_status(connection, incident_id, incident["status"], target)

        closed_at = now_utc()
        connection.execute(
            "UPDATE incidents SET closed_at = ?, outcome = ? WHERE incident_id = ?",
            (closed_at, outcome, incident_id),
        )
        if incident["assigned_responder"]:
            connection.execute(
                """
                UPDATE responders SET status = 'AVAILABLE', active_incident = NULL, last_seen = ?
                WHERE responder_id = ?
                """,
                (closed_at, incident["assigned_responder"]),
            )
        log_event(
            connection, incident_id, "closed",
            data.notes or f"Case closed as {outcome}",
            actor=data.responder_id or "control",
            payload={"outcome": outcome},
        )
        connection.commit()
        incident = fetch_incident(connection, incident_id)
    finally:
        connection.close()

    await hub.broadcast("incident.closed", {"incident": incident}, incident_id=incident_id)
    return incident


# ============================================================
# RESPONDERS
# ============================================================

@app.post("/api/responders/register")
async def register_responder(data: ResponderCreate) -> dict[str, Any]:
    if not valid_coordinate(data.latitude, data.longitude):
        raise HTTPException(400, "Invalid responder coordinate")
    connection = connect()
    try:
        responder_id = next_id(connection, "responders", "responder_id", "RESP-", 4)
        timestamp = now_utc()
        connection.execute(
            """
            INSERT INTO responders (responder_id, name, responder_type, phone, vehicle_number,
                                    latitude, longitude, status, registered_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'AVAILABLE', ?, ?)
            """,
            (
                responder_id,
                data.name,
                data.responder_type.upper(),
                data.phone,
                data.vehicle_number,
                data.latitude,
                data.longitude,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()
        responder = fetch_responder(connection, responder_id)
    finally:
        connection.close()

    await hub.broadcast("responder.registered", {"responder": responder}, roles={"control"})
    return responder


@app.get("/api/responders")
def list_responders(status: str | None = None) -> list[dict[str, Any]]:
    connection = connect()
    try:
        if status:
            rows = connection.execute(
                "SELECT * FROM responders WHERE status = ? ORDER BY id", (status.upper(),)
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM responders ORDER BY id").fetchall()
        return rows_to_list(rows)
    finally:
        connection.close()


@app.get("/api/responders/{responder_id}")
def get_responder(responder_id: str) -> dict[str, Any]:
    connection = connect()
    try:
        responder = fetch_responder(connection, responder_id)
        responder["assignments"] = rows_to_list(
            connection.execute(
                """
                SELECT d.*, i.status AS incident_status, i.severity, i.label,
                       i.latitude, i.longitude, i.location_name
                FROM dispatches d JOIN incidents i ON i.incident_id = d.incident_id
                WHERE d.responder_id = ? ORDER BY d.id DESC LIMIT 25
                """,
                (responder_id,),
            ).fetchall()
        )
        return responder
    finally:
        connection.close()


@app.get("/api/responders/{responder_id}/alerts")
def responder_alerts(responder_id: str) -> list[dict[str, Any]]:
    """Open alerts for this responder - the poll-free client's safety net."""
    connection = connect()
    try:
        fetch_responder(connection, responder_id)
        return rows_to_list(
            connection.execute(
                """
                SELECT i.*, d.status AS dispatch_status, d.distance_km, d.eta_minutes, d.sent_at
                FROM dispatches d JOIN incidents i ON i.incident_id = d.incident_id
                WHERE d.responder_id = ?
                  AND d.status IN ('SENT', 'ACCEPTED')
                  AND i.status NOT IN ('CLOSED','CANCELLED','FALSE_ALARM')
                ORDER BY i.id DESC
                """,
                (responder_id,),
            ).fetchall()
        )
    finally:
        connection.close()


@app.post("/api/responders/{responder_id}/location")
async def update_location(responder_id: str, data: LocationUpdate) -> dict[str, Any]:
    """Stream a GPS fix.  Arrival at an active scene is detected here."""
    if not valid_coordinate(data.latitude, data.longitude):
        raise HTTPException(400, "Invalid coordinate")

    connection = connect()
    arrival: dict[str, Any] | None = None
    try:
        responder = fetch_responder(connection, responder_id)
        timestamp = now_utc()
        connection.execute(
            "UPDATE responders SET latitude = ?, longitude = ?, last_seen = ? WHERE responder_id = ?",
            (data.latitude, data.longitude, timestamp, responder_id),
        )
        connection.execute(
            """
            INSERT INTO location_history (responder_id, incident_id, latitude, longitude,
                                          accuracy, speed_kmh, heading, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                responder_id,
                responder.get("active_incident"),
                data.latitude,
                data.longitude,
                data.accuracy,
                data.speed_kmh,
                data.heading,
                timestamp,
            ),
        )

        incident_id = responder.get("active_incident")
        distance_m = None
        if incident_id:
            incident = fetch_incident(connection, incident_id)
            distance_m = haversine_km(
                data.latitude, data.longitude, incident["latitude"], incident["longitude"]
            ) * 1000.0
            if distance_m <= ARRIVAL_RADIUS_METRES and not incident["arrived_at"]:
                arrival = _record_arrival(connection, incident_id, responder_id, manual=False)
        connection.commit()
        responder = fetch_responder(connection, responder_id)
    finally:
        connection.close()

    await hub.broadcast(
        "responder.location",
        {
            "responder_id": responder_id,
            "latitude": data.latitude,
            "longitude": data.longitude,
            "speed_kmh": data.speed_kmh,
            "heading": data.heading,
            "incident_id": responder.get("active_incident"),
            "distance_to_scene_m": round(distance_m, 1) if distance_m is not None else None,
        },
        roles={"control"},
        incident_id=responder.get("active_incident"),
    )
    if arrival:
        await hub.broadcast(
            "incident.arrived", {"incident": arrival}, incident_id=arrival["incident_id"]
        )

    return {
        "responder": responder,
        "distance_to_scene_m": round(distance_m, 1) if distance_m is not None else None,
        "arrived": bool(arrival),
    }


@app.post("/api/responders/{responder_id}/status")
async def update_responder_status(responder_id: str, data: StatusUpdate) -> dict[str, Any]:
    status = data.status.upper()
    if status not in RESPONDER_STATES:
        raise HTTPException(400, f"Invalid status. Allowed: {sorted(RESPONDER_STATES)}")
    connection = connect()
    try:
        fetch_responder(connection, responder_id)
        connection.execute(
            "UPDATE responders SET status = ?, last_seen = ? WHERE responder_id = ?",
            (status, now_utc(), responder_id),
        )
        connection.commit()
        responder = fetch_responder(connection, responder_id)
    finally:
        connection.close()
    await hub.broadcast("responder.status", {"responder": responder}, roles={"control"})
    return responder


@app.get("/api/responders/{responder_id}/location-history")
def location_history(
    responder_id: str,
    incident_id: str | None = None,
    limit: int = Query(200, ge=1, le=2000),
) -> list[dict[str, Any]]:
    connection = connect()
    try:
        fetch_responder(connection, responder_id)
        if incident_id:
            rows = connection.execute(
                """
                SELECT * FROM location_history
                WHERE responder_id = ? AND incident_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (responder_id, incident_id, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM location_history WHERE responder_id = ? ORDER BY id DESC LIMIT ?",
                (responder_id, limit),
            ).fetchall()
        return rows_to_list(rows)
    finally:
        connection.close()


# ============================================================
# NAVIGATION
# ============================================================

@app.get("/api/route")
async def route(
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
    least_traffic: bool = True,
) -> dict[str, Any]:
    try:
        return await get_route(from_lat, from_lon, to_lat, to_lon, prefer_least_traffic=least_traffic)
    except RoutingError as error:
        raise HTTPException(503, str(error)) from error


@app.get("/api/incidents/{incident_id}/route")
async def route_to_incident(
    incident_id: str, from_lat: float, from_lon: float
) -> dict[str, Any]:
    connection = connect()
    try:
        incident = fetch_incident(connection, incident_id)
    finally:
        connection.close()
    try:
        return await get_route(from_lat, from_lon, incident["latitude"], incident["longitude"])
    except RoutingError as error:
        raise HTTPException(503, str(error)) from error


# ============================================================
# NOTIFICATIONS AND HISTORY
# ============================================================

@app.get("/api/notifications/{recipient}")
def notifications(recipient: str, limit: int = Query(30, ge=1, le=200)) -> list[dict[str, Any]]:
    connection = connect()
    try:
        return rows_to_list(
            connection.execute(
                "SELECT * FROM notifications WHERE recipient = ? ORDER BY id DESC LIMIT ?",
                (recipient, limit),
            ).fetchall()
        )
    finally:
        connection.close()


@app.get("/api/history")
def history(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Closed cases with their full response timings, for the reports view."""
    connection = connect()
    try:
        return rows_to_list(
            connection.execute(
                """
                SELECT i.*, r.name AS responder_name, r.responder_type,
                       (julianday(i.accepted_at) - julianday(i.created_at)) * 86400 AS accept_seconds,
                       (julianday(i.arrived_at)  - julianday(i.accepted_at)) * 86400 AS travel_seconds,
                       (julianday(i.closed_at)   - julianday(i.created_at))  * 86400 AS total_seconds
                FROM incidents i
                LEFT JOIN responders r ON r.responder_id = i.assigned_responder
                WHERE i.status IN ('CLOSED','CANCELLED','FALSE_ALARM')
                ORDER BY i.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        )
    finally:
        connection.close()


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    role: str = Query("control"),
    responder_id: str = Query(""),
) -> None:
    identity = responder_id or f"{role}-client"
    client = await hub.connect(websocket, role, identity)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            action = message.get("action")
            if action == "follow" and message.get("incident_id"):
                await hub.follow(client, message["incident_id"])
            elif action == "unfollow" and message.get("incident_id"):
                await hub.unfollow(client, message["incident_id"])
            elif action == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "sent_at": now_utc()}))
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - never let one socket take the server down
        pass
    finally:
        await hub.disconnect(client)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
