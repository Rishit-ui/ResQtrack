import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_FILE = BASE_DIR / "resqtrack.db"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ResQTrack Emergency Response API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# DATABASE
# ============================================================

def get_db():

    connection = sqlite3.connect(
        DB_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db()

    cursor = connection.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS responders (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            responder_id TEXT UNIQUE NOT NULL,

            name TEXT NOT NULL,

            responder_type TEXT NOT NULL,

            latitude REAL NOT NULL,

            longitude REAL NOT NULL,

            status TEXT NOT NULL DEFAULT 'AVAILABLE',

            phone TEXT,

            registered_at TEXT NOT NULL,

            last_seen TEXT NOT NULL
        );


        CREATE TABLE IF NOT EXISTS incidents (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            incident_id TEXT UNIQUE NOT NULL,

            status TEXT NOT NULL,

            confidence REAL NOT NULL,

            latitude REAL NOT NULL,

            longitude REAL NOT NULL,

            camera_id TEXT NOT NULL,

            frame INTEGER,

            created_at TEXT NOT NULL,

            assigned_responder TEXT
        );


        CREATE TABLE IF NOT EXISTS location_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            responder_id TEXT NOT NULL,

            latitude REAL NOT NULL,

            longitude REAL NOT NULL,

            accuracy REAL,

            timestamp TEXT NOT NULL,

            FOREIGN KEY (
                responder_id
            )
            REFERENCES responders (
                responder_id
            )
        );


        CREATE TABLE IF NOT EXISTS dispatches (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            incident_id TEXT NOT NULL,

            responder_id TEXT NOT NULL,

            status TEXT NOT NULL,

            sent_at TEXT NOT NULL,

            accepted_at TEXT,

            FOREIGN KEY (
                incident_id
            )
            REFERENCES incidents (
                incident_id
            ),

            FOREIGN KEY (
                responder_id
            )
            REFERENCES responders (
                responder_id
            )
        );
    """)

    connection.commit()

    connection.close()


init_database()


# ============================================================
# DATA MODELS
# ============================================================

class IncidentCreate(BaseModel):

    confidence: float

    latitude: float

    longitude: float

    camera_id: str

    frame: Optional[int] = None


class ResponderCreate(BaseModel):

    name: str

    responder_type: str

    latitude: float

    longitude: float

    phone: Optional[str] = None


class LocationUpdate(BaseModel):

    latitude: float

    longitude: float

    accuracy: Optional[float] = None


class StatusUpdate(BaseModel):

    status: str


# ============================================================
# HELPERS
# ============================================================

def now_utc():

    return datetime.now(
        timezone.utc
    ).isoformat()


def next_incident_id(
    connection
):

    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM incidents
        """
    ).fetchone()

    number = (
        int(row["count"])
        + 1
    )

    return (
        f"RQ-{number:05d}"
    )


def next_responder_id(
    connection
):

    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM responders
        """
    ).fetchone()

    number = (
        int(row["count"])
        + 1
    )

    return (
        f"RESP-{number:04d}"
    )


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0

    dlat = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (

        math.sin(
            dlat / 2
        ) ** 2

        +

        math.cos(
            math.radians(lat1)
        )

        *

        math.cos(
            math.radians(lat2)
        )

        *

        math.sin(
            dlon / 2
        ) ** 2
    )

    c = (
        2
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )

    return (
        earth_radius
        *
        c
    )


def row_to_dict(row):

    if row is None:

        return None

    return dict(row)


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    connection = get_db()

    incidents_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM incidents
        """
    ).fetchone()["count"]

    responders_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM responders
        """
    ).fetchone()["count"]

    connection.close()

    return {

        "service":
            "ResQTrack",

        "status":
            "ONLINE",

        "incidents":
            incidents_count,

        "responders":
            responders_count
    }


# ============================================================
# RESPONDER PAGE
# ============================================================

@app.get("/responder")
def responder_page():

    return FileResponse(
        BASE_DIR / "responder.html"
    )


# ============================================================
# CREATE INCIDENT
# ============================================================

@app.post("/api/incidents")
def create_incident(
    data: IncidentCreate
):

    connection = get_db()

    incident_id = next_incident_id(
        connection
    )

    created_at = now_utc()

    connection.execute(
        """
        INSERT INTO incidents (
            incident_id,
            status,
            confidence,
            latitude,
            longitude,
            camera_id,
            frame,
            created_at,
            assigned_responder
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            incident_id,
            "ACCIDENT_CONFIRMED",
            data.confidence,
            data.latitude,
            data.longitude,
            data.camera_id,
            data.frame,
            created_at,
            None
        )
    )

    connection.commit()

    incident = connection.execute(
        """
        SELECT *
        FROM incidents
        WHERE incident_id = ?
        """,
        (incident_id,)
    ).fetchone()


    nearby_rows = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE status = 'AVAILABLE'
        """
    ).fetchall()


    nearby = []

    for responder in nearby_rows:

        distance = haversine_km(

            data.latitude,
            data.longitude,

            responder["latitude"],
            responder["longitude"]
        )

        nearby.append({

            "responder_id":
                responder[
                    "responder_id"
                ],

            "name":
                responder["name"],

            "type":
                responder["responder_type"],

            "distance_km":
                round(
                    distance,
                    2
                )
        })


    nearby.sort(
        key=lambda item:
        item["distance_km"]
    )

    connection.close()


    return {

        "incident":
            row_to_dict(
                incident
            ),

        "nearby_responders":
            nearby[:5]
    }


# ============================================================
# GET ALL INCIDENTS
# ============================================================

@app.get("/api/incidents")
def get_incidents():

    connection = get_db()

    rows = connection.execute(
        """
        SELECT *
        FROM incidents
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return [
        row_to_dict(row)
        for row in rows
    ]


# ============================================================
# GET INCIDENT
# ============================================================

@app.get(
    "/api/incidents/{incident_id}"
)
def get_incident(
    incident_id: str
):

    connection = get_db()

    row = connection.execute(
        """
        SELECT *
        FROM incidents
        WHERE incident_id = ?
        """,
        (incident_id,)
    ).fetchone()

    connection.close()

    if row is None:

        raise HTTPException(
            status_code=404,
            detail="Incident not found"
        )

    return row_to_dict(row)


# ============================================================
# REGISTER RESPONDER
# ============================================================

@app.post(
    "/api/responders/register"
)
def register_responder(
    data: ResponderCreate
):

    connection = get_db()

    responder_id = (
        next_responder_id(
            connection
        )
    )

    timestamp = now_utc()

    connection.execute(
        """
        INSERT INTO responders (
            responder_id,
            name,
            responder_type,
            latitude,
            longitude,
            status,
            phone,
            registered_at,
            last_seen
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            responder_id,
            data.name,
            data.responder_type,
            data.latitude,
            data.longitude,
            "AVAILABLE",
            data.phone,
            timestamp,
            timestamp
        )
    )

    connection.commit()

    row = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()

    connection.close()

    return row_to_dict(row)


# ============================================================
# GET RESPONDER
# ============================================================

@app.get(
    "/api/responders/{responder_id}"
)
def get_responder(
    responder_id: str
):

    connection = get_db()

    row = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()

    connection.close()

    if row is None:

        raise HTTPException(
            status_code=404,
            detail="Responder not found"
        )

    return row_to_dict(row)


# ============================================================
# GET ALL RESPONDERS
# ============================================================

@app.get("/api/responders")
def get_responders():

    connection = get_db()

    rows = connection.execute(
        """
        SELECT *
        FROM responders
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return [
        row_to_dict(row)
        for row in rows
    ]


# ============================================================
# UPDATE RESPONDER LOCATION
# ============================================================

@app.post(
    "/api/responders/{responder_id}/location"
)
def update_location(
    responder_id: str,
    data: LocationUpdate
):

    connection = get_db()

    existing = connection.execute(
        """
        SELECT responder_id
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()

    if existing is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Responder not found"
        )


    timestamp = now_utc()


    connection.execute(
        """
        UPDATE responders

        SET
            latitude = ?,
            longitude = ?,
            last_seen = ?

        WHERE responder_id = ?
        """,
        (
            data.latitude,
            data.longitude,
            timestamp,
            responder_id
        )
    )


    connection.execute(
        """
        INSERT INTO location_history (
            responder_id,
            latitude,
            longitude,
            accuracy,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            responder_id,
            data.latitude,
            data.longitude,
            data.accuracy,
            timestamp
        )
    )


    connection.commit()


    row = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()


    connection.close()

    return row_to_dict(row)


# ============================================================
# UPDATE RESPONDER STATUS
# ============================================================

@app.post(
    "/api/responders/{responder_id}/status"
)
def update_status(
    responder_id: str,
    data: StatusUpdate
):

    allowed = {

        "AVAILABLE",
        "BUSY",
        "ALERTED",
        "EN_ROUTE",
        "OFFLINE"
    }

    if data.status not in allowed:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid status. "
                f"Allowed: {sorted(allowed)}"
            )
        )


    connection = get_db()

    row = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()


    if row is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Responder not found"
        )


    connection.execute(
        """
        UPDATE responders

        SET
            status = ?,
            last_seen = ?

        WHERE responder_id = ?
        """,
        (
            data.status,
            now_utc(),
            responder_id
        )
    )


    connection.commit()


    updated = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()


    connection.close()

    return row_to_dict(updated)


# ============================================================
# DISPATCH
# ============================================================

@app.post(
    "/api/incidents/{incident_id}/dispatch"
)
def dispatch_incident(
    incident_id: str
):

    connection = get_db()


    incident = connection.execute(
        """
        SELECT *
        FROM incidents
        WHERE incident_id = ?
        """,
        (incident_id,)
    ).fetchone()


    if incident is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Incident not found"
        )


    responders_rows = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE status = 'AVAILABLE'
        """
    ).fetchall()


    nearby = []


    for responder in responders_rows:

        distance = haversine_km(

            incident["latitude"],
            incident["longitude"],

            responder["latitude"],
            responder["longitude"]
        )


        nearby.append({

            "responder_id":
                responder[
                    "responder_id"
                ],

            "name":
                responder["name"],

            "type":
                responder[
                    "responder_type"
                ],

            "distance_km":
                round(
                    distance,
                    2
                )
        })


    nearby.sort(
        key=lambda item:
        item["distance_km"]
    )


    selected = nearby[:3]

    timestamp = now_utc()


    for responder in selected:

        responder_id = (
            responder[
                "responder_id"
            ]
        )


        connection.execute(
            """
            UPDATE responders

            SET
                status = 'ALERTED',
                last_seen = ?

            WHERE responder_id = ?
            """,
            (
                timestamp,
                responder_id
            )
        )


        connection.execute(
            """
            INSERT INTO dispatches (
                incident_id,
                responder_id,
                status,
                sent_at,
                accepted_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                responder_id,
                "SENT",
                timestamp,
                None
            )
        )


    connection.commit()

    connection.close()


    return {

        "incident_id":
            incident_id,

        "status":
            "DISPATCH_SENT",

        "responders_notified":
            selected
    }


# ============================================================
# ACCEPT INCIDENT
# ============================================================

@app.post(
    "/api/incidents/{incident_id}/accept/{responder_id}"
)
def accept_incident(
    incident_id: str,
    responder_id: str
):

    connection = get_db()


    incident = connection.execute(
        """
        SELECT *
        FROM incidents
        WHERE incident_id = ?
        """,
        (incident_id,)
    ).fetchone()


    responder = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()


    if incident is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Incident not found"
        )


    if responder is None:

        connection.close()

        raise HTTPException(
            status_code=404,
            detail="Responder not found"
        )


    # --------------------------------------------------------
    # Prevent two responders from accepting simultaneously.
    # --------------------------------------------------------

    if incident[
        "assigned_responder"
    ] is not None:

        connection.close()

        raise HTTPException(
            status_code=409,
            detail=(
                "Incident already assigned."
            )
        )


    current_status = responder[
        "status"
    ]


    if current_status not in {
        "AVAILABLE",
        "ALERTED"
    }:

        connection.close()

        raise HTTPException(
            status_code=409,
            detail=(
                "Responder is unavailable."
            )
        )


    accepted_at = now_utc()


    # --------------------------------------------------------
    # ASSIGN INCIDENT
    # --------------------------------------------------------

    connection.execute(
        """
        UPDATE incidents

        SET
            assigned_responder = ?,
            status = 'RESPONDER_ASSIGNED'

        WHERE incident_id = ?
        """,
        (
            responder_id,
            incident_id
        )
    )


    connection.execute(
        """
        UPDATE responders

        SET
            status = 'EN_ROUTE',
            last_seen = ?

        WHERE responder_id = ?
        """,
        (
            accepted_at,
            responder_id
        )
    )


    connection.execute(
        """
        UPDATE dispatches

        SET
            status = 'ACCEPTED',
            accepted_at = ?

        WHERE incident_id = ?
        AND responder_id = ?
        AND status = 'SENT'
        """,
        (
            accepted_at,
            incident_id,
            responder_id
        )
    )


    connection.commit()


    updated_incident = connection.execute(
        """
        SELECT *
        FROM incidents
        WHERE incident_id = ?
        """,
        (incident_id,)
    ).fetchone()


    updated_responder = connection.execute(
        """
        SELECT *
        FROM responders
        WHERE responder_id = ?
        """,
        (responder_id,)
    ).fetchone()


    connection.close()


    return {

        "incident":
            row_to_dict(
                updated_incident
            ),

        "responder":
            row_to_dict(
                updated_responder
            )
    }


# ============================================================
# LOCATION HISTORY
# ============================================================

@app.get(
    "/api/responders/{responder_id}/location-history"
)
def get_location_history(
    responder_id: str,
    limit: int = 100
):

    limit = max(
        1,
        min(
            limit,
            1000
        )
    )


    connection = get_db()

    rows = connection.execute(
        """
        SELECT
            latitude,
            longitude,
            accuracy,
            timestamp

        FROM location_history

        WHERE responder_id = ?

        ORDER BY id DESC

        LIMIT ?
        """,
        (
            responder_id,
            limit
        )
    ).fetchall()


    connection.close()


    return [
        row_to_dict(row)
        for row in rows
    ]


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket(
    "/ws/responder/{responder_id}"
)
async def responder_socket(
    websocket: WebSocket,
    responder_id: str
):

    await websocket.accept()

    try:

        while True:

            await websocket.receive_text()

    except Exception:

        pass


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000
    )