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

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'CITIZEN',
    phone TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    responder_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS manual_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'COLLISION',
    description TEXT NOT NULL,
    location_name TEXT,
    latitude REAL,
    longitude REAL,
    image_path TEXT,
    contact_phone TEXT,
    status TEXT NOT NULL DEFAULT 'SUBMITTED',
    admin_notes TEXT,
    reviewed_by TEXT,
    reviewed_at TEXT,
    linked_incident_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id TEXT UNIQUE NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    ip_address TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS password_resets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reset_token TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users (role);
CREATE INDEX IF NOT EXISTS idx_reports_user ON manual_reports (user_id);
CREATE INDEX IF NOT EXISTS idx_reports_status ON manual_reports (status);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs (action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs (created_at DESC);

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
        _bootstrap_defaults(connection)
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


# ============================================================
# PASSWORD HASHING AND AUTHENTICATION
# ============================================================

def hash_password(password: str) -> str:
    import bcrypt
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    import bcrypt
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


# ============================================================
# BOOTSTRAP DEFAULTS
# ============================================================

def _bootstrap_defaults(connection: sqlite3.Connection) -> None:
    now = now_utc()

    # 1. Seed base responders if table empty
    count = connection.execute("SELECT COUNT(*) c FROM responders").fetchone()["c"]
    if count == 0:
        responders = [
            ("RESP-0001", "Ambulance 108 - Rapid Unit A", "AMBULANCE", "+91-98450-11001", "KA-01-EA-1108", 12.9716, 77.5946),
            ("RESP-0002", "Ambulance 108 - City North B", "AMBULANCE", "+91-98450-11002", "KA-01-EA-2208", 12.9850, 77.6050),
            ("RESP-0003", "Paramedic Fast-Response Bike", "PARAMEDIC", "+91-98450-11003", "KA-01-PB-0091", 12.9650, 77.5850),
            ("RESP-0004", "Fire & Rescue Tender 1", "FIRE", "+91-98450-11005", "KA-01-FR-0003", 12.9550, 77.6100),
            ("RESP-0005", "Traffic Police Patrol Alpha", "POLICE", "+91-98450-11004", "KA-01-TP-4410", 12.9780, 77.5900),
        ]
        for r_id, name, kind, phone, veh, lat, lon in responders:
            connection.execute(
                """
                INSERT OR IGNORE INTO responders (
                    responder_id, name, responder_type, phone, vehicle_number,
                    latitude, longitude, status, registered_at, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'AVAILABLE', ?, ?)
                """,
                (r_id, name, kind, phone, veh, lat, lon, now, now),
            )

    # 2. Seed base facilities if table empty
    fac_count = connection.execute("SELECT COUNT(*) c FROM facilities").fetchone()["c"]
    if fac_count == 0:
        facilities = [
            ("seed:Bowring and Lady Curzon Hospital", "Bowring and Lady Curzon Hospital", "hospital", 12.9829, 77.6046, "080-25591325", "Shivaji Nagar", 1),
            ("seed:Victoria Hospital", "Victoria Hospital", "hospital", 12.9629, 77.5745, "080-26701150", "Fort", 1),
            ("seed:St. Johns Medical College", "St. John's Medical College Hospital", "hospital", 12.9279, 77.6212, "080-22065000", "Koramangala", 1),
            ("seed:Manipal Hospital", "Manipal Hospital, Old Airport Road", "hospital", 12.9591, 77.6494, "080-25023700", "HAL Airport Rd", 1),
            ("seed:Mallya Hospital", "Mallya Hospital", "hospital", 12.9689, 77.5966, "080-22277979", "Vittal Mallya Rd", 1),
            ("seed:KC General Hospital", "KC General Hospital, Malleswaram", "hospital", 12.9976, 77.5713, "080-23342778", "Malleswaram", 1),
            ("seed:Apollo Hospital", "Apollo Hospital, Sheshadripuram", "hospital", 12.9915, 77.5735, "080-46126666", "Sheshadripuram", 1),
        ]
        for ref, name, kind, lat, lon, phone, addr, emerg in facilities:
            connection.execute(
                """
                INSERT OR IGNORE INTO facilities (
                    ref, name, kind, latitude, longitude, phone, address, emergency, source, cached_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'seed', ?)
                """,
                (ref, name, kind, lat, lon, phone, addr, emerg, now),
            )

    # 3. Seed default accounts
    admin = connection.execute("SELECT * FROM users WHERE email = 'admin@resqtrack.gov'").fetchone()
    if not admin:
        connection.execute(
            """
            INSERT INTO users (user_id, email, password_hash, full_name, role, phone, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'ADMIN', ?, 'ACTIVE', ?, ?)
            """,
            ("USR-ADMIN-01", "admin@resqtrack.gov", hash_password("Admin@ResQTrack2026!"), "Emergency Command Director", "+91-80000-00001", now, now),
        )

    responder_user = connection.execute("SELECT * FROM users WHERE email = 'responder@resqtrack.gov'").fetchone()
    if not responder_user:
        connection.execute(
            """
            INSERT INTO users (user_id, email, password_hash, full_name, role, phone, status, responder_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'RESPONDER', ?, 'ACTIVE', 'RESP-0001', ?, ?)
            """,
            ("USR-RESP-01", "responder@resqtrack.gov", hash_password("Responder@2026!"), "Ambulance Crew 108", "+91-98450-11001", now, now),
        )

    citizen_user = connection.execute("SELECT * FROM users WHERE email = 'citizen@resqtrack.org'").fetchone()
    if not citizen_user:
        connection.execute(
            """
            INSERT INTO users (user_id, email, password_hash, full_name, role, phone, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'CITIZEN', ?, 'ACTIVE', ?, ?)
            """,
            ("USR-CITIZEN-01", "citizen@resqtrack.org", hash_password("Citizen@2026!"), "Rishit Citizen", "+91-98888-77777", now, now),
        )


# ============================================================
# USER OPERATIONS
# ============================================================

def create_user(
    connection: sqlite3.Connection,
    email: str,
    password: str,
    full_name: str,
    role: str = "CITIZEN",
    phone: str | None = None,
    responder_id: str | None = None,
) -> dict[str, Any]:
    email_clean = email.strip().lower()
    existing = connection.execute("SELECT user_id FROM users WHERE email = ?", (email_clean,)).fetchone()
    if existing:
        raise ValueError(f"User with email '{email_clean}' already exists")

    user_id = next_id(connection, "users", "user_id", "USR-", 5)
    now = now_utc()
    pwd_hash = hash_password(password)

    connection.execute(
        """
        INSERT INTO users (user_id, email, password_hash, full_name, role, phone, status, responder_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
        """,
        (user_id, email_clean, pwd_hash, full_name.strip(), role.upper(), phone, responder_id, now, now),
    )
    return get_user_by_id(connection, user_id)  # type: ignore[return-value]


def get_user_by_email(connection: sqlite3.Connection, email: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return row_to_dict(row) if row else None


def get_user_by_id(connection: sqlite3.Connection, user_id: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return row_to_dict(row) if row else None


def list_users(
    connection: sqlite3.Connection,
    role: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query = "SELECT user_id, email, full_name, role, phone, status, responder_id, created_at, updated_at FROM users"
    params: list[Any] = []
    clauses = []
    if role:
        clauses.append("role = ?")
        params.append(role.upper())
    if status:
        clauses.append("status = ?")
        params.append(status.upper())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return rows_to_list(connection.execute(query, params).fetchall())


def update_user_status(connection: sqlite3.Connection, user_id: str, status: str) -> dict[str, Any]:
    now = now_utc()
    connection.execute(
        "UPDATE users SET status = ?, updated_at = ? WHERE user_id = ?",
        (status.upper(), now, user_id),
    )
    return get_user_by_id(connection, user_id)  # type: ignore[return-value]


def update_user_profile(
    connection: sqlite3.Connection,
    user_id: str,
    full_name: str | None = None,
    phone: str | None = None,
) -> dict[str, Any]:
    now = now_utc()
    if full_name is not None and phone is not None:
        connection.execute(
            "UPDATE users SET full_name = ?, phone = ?, updated_at = ? WHERE user_id = ?",
            (full_name.strip(), phone.strip(), now, user_id),
        )
    elif full_name is not None:
        connection.execute(
            "UPDATE users SET full_name = ?, updated_at = ? WHERE user_id = ?",
            (full_name.strip(), now, user_id),
        )
    elif phone is not None:
        connection.execute(
            "UPDATE users SET phone = ?, updated_at = ? WHERE user_id = ?",
            (phone.strip(), now, user_id),
        )
    return get_user_by_id(connection, user_id)  # type: ignore[return-value]


def update_user_password(
    connection: sqlite3.Connection,
    user_id: str,
    new_password: str,
) -> None:
    now = now_utc()
    pwd_hash = hash_password(new_password)
    connection.execute(
        "UPDATE users SET password_hash = ?, updated_at = ? WHERE user_id = ?",
        (pwd_hash, now, user_id),
    )


# ============================================================
# AUDIT LOGGING
# ============================================================

def log_audit(
    connection: sqlite3.Connection,
    user_id: str | None,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> str:
    log_id = next_id(connection, "audit_logs", "log_id", "AUD-", 6)
    connection.execute(
        """
        INSERT INTO audit_logs (log_id, user_id, action, target_type, target_id, details, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            log_id,
            user_id,
            action,
            target_type,
            target_id,
            json.dumps(details or {}),
            ip_address,
            now_utc(),
        ),
    )
    return log_id


def list_audit_logs(
    connection: sqlite3.Connection,
    limit: int = 100,
    action: str | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    query = """
    SELECT a.*, u.email AS user_email, u.full_name AS user_name, u.role AS user_role
    FROM audit_logs a
    LEFT JOIN users u ON u.user_id = a.user_id
    """
    params: list[Any] = []
    clauses = []
    if action:
        clauses.append("a.action = ?")
        params.append(action)
    if user_id:
        clauses.append("a.user_id = ?")
        params.append(user_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY a.id DESC LIMIT ?"
    params.append(limit)
    return rows_to_list(connection.execute(query, params).fetchall())


# ============================================================
# CITIZEN MANUAL REPORTS
# ============================================================

def create_manual_report(
    connection: sqlite3.Connection,
    user_id: str,
    category: str,
    description: str,
    location_name: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    image_path: str | None = None,
    contact_phone: str | None = None,
) -> dict[str, Any]:
    report_id = next_id(connection, "manual_reports", "report_id", "REP-", 5)
    now = now_utc()
    connection.execute(
        """
        INSERT INTO manual_reports (
            report_id, user_id, category, description, location_name,
            latitude, longitude, image_path, contact_phone, status,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUBMITTED', ?, ?)
        """,
        (
            report_id,
            user_id,
            category.upper(),
            description.strip(),
            location_name,
            latitude,
            longitude,
            image_path,
            contact_phone,
            now,
            now,
        ),
    )
    return get_manual_report(connection, report_id)  # type: ignore[return-value]


def get_manual_report(connection: sqlite3.Connection, report_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT r.*, u.full_name AS reporter_name, u.email AS reporter_email
        FROM manual_reports r
        LEFT JOIN users u ON u.user_id = r.user_id
        WHERE r.report_id = ?
        """,
        (report_id,),
    ).fetchone()
    return row_to_dict(row) if row else None


def list_manual_reports(
    connection: sqlite3.Connection,
    user_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query = """
    SELECT r.*, u.full_name AS reporter_name, u.email AS reporter_email
    FROM manual_reports r
    LEFT JOIN users u ON u.user_id = r.user_id
    """
    params: list[Any] = []
    clauses = []
    if user_id:
        clauses.append("r.user_id = ?")
        params.append(user_id)
    if status:
        clauses.append("r.status = ?")
        params.append(status.upper())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY r.id DESC LIMIT ?"
    params.append(limit)
    return rows_to_list(connection.execute(query, params).fetchall())


def review_manual_report(
    connection: sqlite3.Connection,
    report_id: str,
    status: str,
    admin_notes: str | None,
    reviewed_by: str,
    linked_incident_id: str | None = None,
) -> dict[str, Any]:
    now = now_utc()
    connection.execute(
        """
        UPDATE manual_reports
        SET status = ?, admin_notes = ?, reviewed_by = ?, reviewed_at = ?,
            linked_incident_id = COALESCE(?, linked_incident_id), updated_at = ?
        WHERE report_id = ?
        """,
        (status.upper(), admin_notes, reviewed_by, now, linked_incident_id, now, report_id),
    )
    return get_manual_report(connection, report_id)  # type: ignore[return-value]
