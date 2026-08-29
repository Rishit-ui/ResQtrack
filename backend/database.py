"""SQLite schema and helpers for the ResQTrack emergency-response backend.

Everything that happens to an incident is written here: the detection itself,
which responders were alerted, who accepted, every GPS ping on the way, the
hospital that was chosen, and when the case was closed.  ``incident_events`` is
the append-only audit trail that the dashboard replays as a timeline.

SQLite is used deliberately: the whole prototype runs from one file with no
server to install, and WAL mode keeps the detector's writes from blocking the
dashboard's reads.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent

# RESQTRACK_DB lets tests and alternate deployments point at another file
# without editing code.
DB_FILE = Path(os.environ.get("RESQTRACK_DB", BASE_DIR / "resqtrack.db"))
SNAPSHOT_DIR = Path(os.environ.get("RESQTRACK_SNAPSHOTS", BASE_DIR / "snapshots"))

# Incident lifecycle.  The API refuses transitions that are not in this map, so
# a responder cannot mark themselves "on scene" for a case they never accepted.
INCIDENT_FLOW = {
    "DETECTED": {"DISPATCHED", "CANCELLED", "FALSE_ALARM"},
    "DISPATCHED": {"ACCEPTED", "CANCELLED", "FALSE_ALARM", "DISPATCHED"},
    "ACCEPTED": {"ON_SCENE", "CANCELLED", "FALSE_ALARM"},
    "ON_SCENE": {"TRANSPORTING", "CLOSED", "CANCELLED"},
    "TRANSPORTING": {"CLOSED", "CANCELLED"},
    "CLOSED": set(),
    "CANCELLED": set(),
    "FALSE_ALARM": set(),
}

RESPONDER_STATES = {"AVAILABLE", "ALERTED", "EN_ROUTE", "ON_SCENE", "TRANSPORTING", "OFFLINE"}

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS incidents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id         TEXT    UNIQUE NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'DETECTED',
    kind                TEXT    NOT NULL DEFAULT 'unknown',
    label               TEXT    NOT NULL DEFAULT 'Incident',
    severity            TEXT    NOT NULL DEFAULT 'HIGH',
    confidence          REAL    NOT NULL DEFAULT 0.0,
    reason              TEXT,
    evidence_types      TEXT    NOT NULL DEFAULT '[]',
    signals             TEXT    NOT NULL DEFAULT '[]',
    involved            TEXT    NOT NULL DEFAULT '[]',
    vehicles            TEXT    NOT NULL DEFAULT '[]',
    ml_probability      REAL    NOT NULL DEFAULT 0.0,
    camera_id           TEXT    NOT NULL,
    location_name       TEXT,
    latitude            REAL    NOT NULL,
    longitude           REAL    NOT NULL,
    frame               INTEGER,
    snapshot_path       TEXT,
    detected_at         TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    assigned_responder  TEXT,
    accepted_at         TEXT,
    arrived_at          TEXT,
    hospital_name       TEXT,
    hospital_latitude   REAL,
    hospital_longitude  REAL,
    transport_started_at TEXT,
    closed_at           TEXT,
    outcome             TEXT
);

CREATE TABLE IF NOT EXISTS responders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    responder_id    TEXT    UNIQUE NOT NULL,
    name            TEXT    NOT NULL,
    responder_type  TEXT    NOT NULL DEFAULT 'AMBULANCE',
    phone           TEXT,
    vehicle_number  TEXT,
    latitude        REAL    NOT NULL,
    longitude       REAL    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'AVAILABLE',
    active_incident TEXT,
    registered_at   TEXT    NOT NULL,
    last_seen       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS dispatches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  TEXT NOT NULL,
    responder_id TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'SENT',
    distance_km  REAL,
    eta_minutes  REAL,
    sent_at      TEXT NOT NULL,
    responded_at TEXT,
    UNIQUE (incident_id, responder_id),
    FOREIGN KEY (incident_id)  REFERENCES incidents (incident_id),
    FOREIGN KEY (responder_id) REFERENCES responders (responder_id)
);

CREATE TABLE IF NOT EXISTS incident_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    message     TEXT NOT NULL,
    actor       TEXT,
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    FOREIGN KEY (incident_id) REFERENCES incidents (incident_id)
);

CREATE TABLE IF NOT EXISTS location_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    responder_id TEXT NOT NULL,
    incident_id  TEXT,
    latitude     REAL NOT NULL,
    longitude    REAL NOT NULL,
    accuracy     REAL,
    speed_kmh    REAL,
    heading      REAL,
    recorded_at  TEXT NOT NULL,
    FOREIGN KEY (responder_id) REFERENCES responders (responder_id)
);

CREATE TABLE IF NOT EXISTS facilities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ref         TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'hospital',
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    phone       TEXT,
    address     TEXT,
    emergency   INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT 'builtin',
    cached_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient    TEXT NOT NULL,
    incident_id  TEXT,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL,
    level        TEXT NOT NULL DEFAULT 'info',
    created_at   TEXT NOT NULL,
    read_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_status   ON incidents (status);
CREATE INDEX IF NOT EXISTS idx_incidents_created  ON incidents (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_incident    ON incident_events (incident_id, id);
CREATE INDEX IF NOT EXISTS idx_dispatch_incident  ON dispatches (incident_id);
CREATE INDEX IF NOT EXISTS idx_dispatch_responder ON dispatches (responder_id, status);
CREATE INDEX IF NOT EXISTS idx_location_responder ON location_history (responder_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_notify_recipient   ON notifications (recipient, id DESC);
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_FILE, timeout=10.0, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialise() -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    connection = connect()
    try:
        connection.executescript(SCHEMA)
        _migrate(connection)
        connection.commit()
    finally:
        connection.close()


def _migrate(connection: sqlite3.Connection) -> None:
    """Add columns that older prototype databases in this repo do not have."""
    for table, columns in (
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
    ):
        existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue
        for column, definition in columns.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# ============================================================
# ID ALLOCATION
# ============================================================

def next_id(connection: sqlite3.Connection, table: str, column: str, prefix: str, width: int) -> str:
    """Allocate the next sequential public id.

    Derived from the table's own AUTOINCREMENT rowid rather than ``COUNT(*)``,
    so deleting a row can never hand the same public id to two records.
    """
    row = connection.execute(f"SELECT MAX(id) AS top FROM {table}").fetchone()
    number = (row["top"] or 0) + 1
    while True:
        candidate = f"{prefix}{number:0{width}d}"
        clash = connection.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ?", (candidate,)
        ).fetchone()
        if clash is None:
            return candidate
        number += 1


# ============================================================
# ROW HELPERS
# ============================================================

JSON_COLUMNS = {"evidence_types", "signals", "involved", "vehicles", "payload"}


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for column in JSON_COLUMNS & data.keys():
        try:
            data[column] = json.loads(data[column] or "[]")
        except (TypeError, ValueError):
            data[column] = []
    return data


def rows_to_list(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [row_to_dict(row) for row in rows]  # type: ignore[misc]


def log_event(
    connection: sqlite3.Connection,
    incident_id: str,
    event_type: str,
    message: str,
    actor: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one entry to an incident's audit trail."""
    created_at = now_utc()
    cursor = connection.execute(
        """
        INSERT INTO incident_events (incident_id, event_type, message, actor, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (incident_id, event_type, message, actor, json.dumps(payload or {}), created_at),
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
        INSERT INTO notifications (recipient, incident_id, title, body, level, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (recipient, incident_id, title, body, level, created_at),
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


def can_transition(current: str, target: str) -> bool:
    return target in INCIDENT_FLOW.get(current, set())
