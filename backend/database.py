"""SQLite schema and helpers for the ResQTrack backend."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent

DB_FILE = Path(
    os.environ.get(
        "RESQTRACK_DB",
        BASE_DIR / "resqtrack.db"
    )
)

SNAPSHOT_DIR = Path(
    os.environ.get(
        "RESQTRACK_SNAPSHOTS",
        BASE_DIR / "snapshots"
    )
)


# ============================================================
# INCIDENT LIFECYCLE
# ============================================================

INCIDENT_FLOW = {
    "DETECTED": {
        "DISPATCHED",
        "CANCELLED",
        "FALSE_ALARM",
    },
    "DISPATCHED": {
        "ACCEPTED",
        "CANCELLED",
        "FALSE_ALARM",
        "DISPATCHED",
    },
    "ACCEPTED": {
        "ON_SCENE",
        "CANCELLED",
        "FALSE_ALARM",
    },
    "ON_SCENE": {
        "TRANSPORTING",
        "CLOSED",
        "CANCELLED",
    },
    "TRANSPORTING": {
        "CLOSED",
        "CANCELLED",
    },
    "CLOSED": set(),
    "CANCELLED": set(),
    "FALSE_ALARM": set(),
}


RESPONDER_STATES = {
    "AVAILABLE",
    "ALERTED",
    "EN_ROUTE",
    "ON_SCENE",
    "TRANSPORTING",
    "OFFLINE",
}


# ============================================================
# DATABASE SCHEMA
# ============================================================

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'DETECTED',
    kind TEXT NOT NULL DEFAULT 'unknown',
    label TEXT NOT NULL DEFAULT 'Incident',
    severity TEXT NOT NULL DEFAULT 'HIGH',
    confidence REAL NOT NULL DEFAULT 0.0,
    reason TEXT,
    evidence_types TEXT NOT NULL DEFAULT '[]',
    signals TEXT NOT NULL DEFAULT '[]',
    involved TEXT NOT NULL DEFAULT '[]',
    vehicles TEXT NOT NULL DEFAULT '[]',
    ml_probability REAL NOT NULL DEFAULT 0.0,
    camera_id TEXT NOT NULL,
    location_name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    frame INTEGER,
    snapshot_path TEXT,
    detected_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    assigned_responder TEXT,
    accepted_at TEXT,
    arrived_at TEXT,
    hospital_name TEXT,
    hospital_latitude REAL,
    hospital_longitude REAL,
    transport_started_at TEXT,
    closed_at TEXT,
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS responders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    responder_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    responder_type TEXT NOT NULL DEFAULT 'AMBULANCE',
    phone TEXT,
    vehicle_number TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'AVAILABLE',
    active_incident TEXT,
    registered_at TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dispatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT NOT NULL,
    responder_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'SENT',
    distance_km REAL,
    eta_minutes REAL,
    sent_at TEXT NOT NULL,
    responded_at TEXT,
    UNIQUE (incident_id, responder_id),
    FOREIGN KEY (incident_id)
        REFERENCES incidents (incident_id),
    FOREIGN KEY (responder_id)
        REFERENCES responders (responder_id)
);

CREATE TABLE IF NOT EXISTS incident_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    actor TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (incident_id)
        REFERENCES incidents (incident_id)
);

CREATE TABLE IF NOT EXISTS location_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    responder_id TEXT NOT NULL,
    incident_id TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    accuracy REAL,
    speed_kmh REAL,
    heading REAL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (responder_id)
        REFERENCES responders (responder_id)
);

CREATE TABLE IF NOT EXISTS facilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'hospital',
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    phone TEXT,
    address TEXT,
    emergency INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'builtin',
    cached_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient TEXT NOT NULL,
    incident_id TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'info',
    created_at TEXT NOT NULL,
    read_at TEXT
);

CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT UNIQUE NOT NULL,
    name TEXT,
    location_name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    source TEXT,
    status TEXT NOT NULL DEFAULT 'ONLINE',
    created_at TEXT NOT NULL,
    last_seen TEXT
);

CREATE TABLE IF NOT EXISTS vehicle_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT,
    camera_id TEXT NOT NULL,
    tracker_id INTEGER,
    vehicle_class TEXT,
    confidence REAL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    center_x REAL,
    center_y REAL,
    speed REAL,
    heading TEXT,
    observed_at TEXT NOT NULL,
    FOREIGN KEY (incident_id)
        REFERENCES incidents (incident_id),
    FOREIGN KEY (camera_id)
        REFERENCES cameras (camera_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    camera_id TEXT,
    source TEXT,
    model_name TEXT,
    model_version TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    frames_processed INTEGER NOT NULL DEFAULT 0,
    incidents_detected INTEGER NOT NULL DEFAULT 0,
    alerts_sent INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'RUNNING'
);

CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    incident_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (incident_id)
        REFERENCES incidents (incident_id)
);

CREATE INDEX IF NOT EXISTS idx_incidents_status
    ON incidents (status);

CREATE INDEX IF NOT EXISTS idx_incidents_created
    ON incidents (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_incident
    ON incident_events (incident_id, id);

CREATE INDEX IF NOT EXISTS idx_dispatch_incident
    ON dispatches (incident_id);

CREATE INDEX IF NOT EXISTS idx_dispatch_responder
    ON dispatches (responder_id, status);

CREATE INDEX IF NOT EXISTS idx_location_responder
    ON location_history (responder_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_notify_recipient
    ON notifications (recipient, id DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_incident
    ON vehicle_observations (incident_id);

CREATE INDEX IF NOT EXISTS idx_vehicle_camera
    ON vehicle_observations (camera_id);

CREATE INDEX IF NOT EXISTS idx_vehicle_time
    ON vehicle_observations (observed_at);

CREATE INDEX IF NOT EXISTS idx_model_runs_camera
    ON model_runs (camera_id);

CREATE INDEX IF NOT EXISTS idx_model_runs_started
    ON model_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_system_logs_incident
    ON system_logs (incident_id);

CREATE INDEX IF NOT EXISTS idx_system_logs_created
    ON system_logs (created_at DESC);
"""


# ============================================================
# BASIC HELPERS
# ============================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(
        DB_FILE,
        timeout=10.0,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialise() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    connection = connect()

    try:
        connection.executescript(SCHEMA)
        _migrate(connection)
        connection.commit()
    finally:
        connection.close()


# ============================================================
# MIGRATION
# ============================================================

def _migrate(connection: sqlite3.Connection) -> None:

    migrations = (
        (
            "incidents",
            {
                "kind": "TEXT NOT NULL DEFAULT 'unknown'",
                "label": "TEXT NOT NULL DEFAULT 'Incident'",
                "severity": "TEXT NOT NULL DEFAULT 'HIGH'",
                "reason": "TEXT",
                "evidence_types": "TEXT NOT NULL DEFAULT '[]'",
                "signals": "TEXT NOT NULL DEFAULT '[]'",
                "involved": "TEXT NOT NULL DEFAULT '[]'",
                "vehicles": "TEXT NOT NULL DEFAULT '[]'",
                "ml_probability": "REAL NOT NULL DEFAULT 0.0",
                "location_name": "TEXT",
                "snapshot_path": "TEXT",
                "detected_at": "TEXT",
                "accepted_at": "TEXT",
                "arrived_at": "TEXT",
                "hospital_name": "TEXT",
                "hospital_latitude": "REAL",
                "hospital_longitude": "REAL",
                "transport_started_at": "TEXT",
                "closed_at": "TEXT",
                "outcome": "TEXT",
            },
        ),
        (
            "responders",
            {
                "phone": "TEXT",
                "vehicle_number": "TEXT",
                "active_incident": "TEXT",
            },
        ),
    )

    for table, columns in migrations:

        existing = {
            row["name"]
            for row in connection.execute(
                f"PRAGMA table_info({table})"
            )
        }

        if not existing:
            continue

        for column, definition in columns.items():

            if column not in existing:

                connection.execute(
                    f"""
                    ALTER TABLE {table}
                    ADD COLUMN {column} {definition}
                    """
                )


# ============================================================
# ID GENERATION
# ============================================================

def next_id(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    prefix: str,
    width: int,
) -> str:

    row = connection.execute(
        f"SELECT MAX(id) AS top FROM {table}"
    ).fetchone()

    number = (row["top"] or 0) + 1

    while True:

        candidate = f"{prefix}{number:0{width}d}"

        exists = connection.execute(
            f"""
            SELECT 1
            FROM {table}
            WHERE {column} = ?
            """,
            (candidate,),
        ).fetchone()

        if exists is None:
            return candidate

        number += 1


# ============================================================
# ROW HELPERS
# ============================================================

JSON_COLUMNS = {
    "evidence_types",
    "signals",
    "involved",
    "vehicles",
    "payload",
}


def row_to_dict(
    row: sqlite3.Row | None,
) -> dict[str, Any] | None:

    if row is None:
        return None

    data = dict(row)

    for column in JSON_COLUMNS & data.keys():

        try:
            data[column] = json.loads(
                data[column] or "[]"
            )

        except (TypeError, ValueError):

            data[column] = []

    return data


def rows_to_list(
    rows: Iterable[sqlite3.Row],
) -> list[dict[str, Any]]:

    return [
        row_to_dict(row)
        for row in rows
    ]


# ============================================================
# INCIDENT EVENTS
# ============================================================

def log_event(
    connection: sqlite3.Connection,
    incident_id: str,
    event_type: str,
    message: str,
    actor: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:

    created_at = now_utc()

    cursor = connection.execute(
        """
        INSERT INTO incident_events (
            incident_id,
            event_type,
            message,
            actor,
            payload,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            incident_id,
            event_type,
            message,
            actor,
            json.dumps(payload or {}),
            created_at,
        ),
    )

    return {
        "id": cursor.lastrowid,
        "incident_id": incident_id,
        "event_type": event_type,
        "message": message,
        "actor": actor,
        "payload": payload or {},
        "created_at": created_at,
    }


# ============================================================
# NOTIFICATIONS
# ============================================================

def add_notification(
    connection: sqlite3.Connection,
    recipient: str,
    title: str,
    body: str,
    incident_id: str | None = None,
    level: str = "info",
) -> dict[str, Any]:

    created_at = now_utc()

    cursor = connection.execute(
        """
        INSERT INTO notifications (
            recipient,
            incident_id,
            title,
            body,
            level,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            recipient,
            incident_id,
            title,
            body,
            level,
            created_at,
        ),
    )

    return {
        "id": cursor.lastrowid,
        "recipient": recipient,
        "incident_id": incident_id,
        "title": title,
        "body": body,
        "level": level,
        "created_at": created_at,
    }


# ============================================================
# CAMERA LOGGING
# ============================================================

def register_camera(
    connection: sqlite3.Connection,
    camera_id: str,
    name: str,
    location_name: str,
    latitude: float,
    longitude: float,
    source: str,
    status: str = "ONLINE",
) -> None:

    timestamp = now_utc()

    connection.execute(
        """
        INSERT INTO cameras (
            camera_id,
            name,
            location_name,
            latitude,
            longitude,
            source,
            status,
            created_at,
            last_seen
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(camera_id)
        DO UPDATE SET
            name = excluded.name,
            location_name = excluded.location_name,
            latitude = excluded.latitude,
            longitude = excluded.longitude,
            source = excluded.source,
            status = excluded.status,
            last_seen = excluded.last_seen
        """,
        (
            camera_id,
            name,
            location_name,
            latitude,
            longitude,
            source,
            status,
            timestamp,
            timestamp,
        ),
    )


# ============================================================
# MODEL RUN LOGGING
# ============================================================

def start_model_run(
    connection: sqlite3.Connection,
    run_id: str,
    camera_id: str,
    source: str,
    model_name: str,
    model_version: str,
) -> None:

    connection.execute(
        """
        INSERT INTO model_runs (
            run_id,
            camera_id,
            source,
            model_name,
            model_version,
            started_at,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, 'RUNNING')
        """,
        (
            run_id,
            camera_id,
            source,
            model_name,
            model_version,
            now_utc(),
        ),
    )


def update_model_run(
    connection: sqlite3.Connection,
    run_id: str,
    frames_processed: int | None = None,
    incidents_detected: int | None = None,
    alerts_sent: int | None = None,
) -> None:

    connection.execute(
        """
        UPDATE model_runs
        SET
            frames_processed =
                COALESCE(?, frames_processed),

            incidents_detected =
                COALESCE(?, incidents_detected),

            alerts_sent =
                COALESCE(?, alerts_sent)

        WHERE run_id = ?
        """,
        (
            frames_processed,
            incidents_detected,
            alerts_sent,
            run_id,
        ),
    )


def finish_model_run(
    connection: sqlite3.Connection,
    run_id: str,
    frames_processed: int,
    incidents_detected: int,
    alerts_sent: int,
    status: str = "COMPLETED",
) -> None:

    connection.execute(
        """
        UPDATE model_runs
        SET
            ended_at = ?,
            frames_processed = ?,
            incidents_detected = ?,
            alerts_sent = ?,
            status = ?
        WHERE run_id = ?
        """,
        (
            now_utc(),
            frames_processed,
            incidents_detected,
            alerts_sent,
            status,
            run_id,
        ),
    )


# ============================================================
# VEHICLE OBSERVATION LOGGING
# ============================================================

def save_vehicle_observation(
    connection: sqlite3.Connection,
    camera_id: str,
    tracker_id: int,
    vehicle_class: str,
    confidence: float,
    box: tuple[float, float, float, float],
    center: tuple[float, float],
    speed: float | None = None,
    heading: str | None = None,
    incident_id: str | None = None,
) -> None:

    x1, y1, x2, y2 = box
    center_x, center_y = center

    connection.execute(
        """
        INSERT INTO vehicle_observations (
            incident_id,
            camera_id,
            tracker_id,
            vehicle_class,
            confidence,
            x1,
            y1,
            x2,
            y2,
            center_x,
            center_y,
            speed,
            heading,
            observed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            incident_id,
            camera_id,
            tracker_id,
            vehicle_class,
            confidence,
            x1,
            y1,
            x2,
            y2,
            center_x,
            center_y,
            speed,
            heading,
            now_utc(),
        ),
    )


# ============================================================
# SYSTEM LOGGING
# ============================================================

def log_system(
    connection: sqlite3.Connection,
    level: str,
    component: str,
    message: str,
    incident_id: str | None = None,
) -> None:

    connection.execute(
        """
        INSERT INTO system_logs (
            level,
            component,
            message,
            incident_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            level.upper(),
            component,
            message,
            incident_id,
            now_utc(),
        ),
    )


# ============================================================
# STATE TRANSITIONS
# ============================================================

def can_transition(
    current: str,
    target: str,
) -> bool:

    return target in INCIDENT_FLOW.get(
        current,
        set(),
    )