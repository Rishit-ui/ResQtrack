from fastapi.responses import FileResponse
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ResQTrack Emergency Response API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# IN-MEMORY DATABASE
#
# We deliberately start simple.
# We will move this to SQLite next.
# ============================================================

incidents = {}

responders = {}

connections = {}


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


class StatusUpdate(BaseModel):

    status: str


# ============================================================
# HELPERS
# ============================================================

def generate_incident_id():

    return (
        f"RQ-{len(incidents) + 1:05d}"
    )


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0

    dlat = radians(
        lat2 - lat1
    )

    dlon = radians(
        lon2 - lon1
    )

    a = (

        sin(dlat / 2) ** 2

        +

        cos(radians(lat1))
        *
        cos(radians(lat2))
        *
        sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return (
        earth_radius * c
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "service": "ResQTrack",

        "status": "ONLINE",

        "incidents":
            len(incidents),

        "responders":
            len(responders)
    }


# ============================================================
# CREATE INCIDENT
# ============================================================

@app.post("/api/incidents")
async def create_incident(
    data: IncidentCreate
):

    incident_id = (
        generate_incident_id()
    )

    incident = {

        "incident_id":
            incident_id,

        "status":
            "ACCIDENT_CONFIRMED",

        "confidence":
            data.confidence,

        "latitude":
            data.latitude,

        "longitude":
            data.longitude,

        "camera_id":
            data.camera_id,

        "frame":
            data.frame,

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "assigned_responder":
            None
    }


    incidents[
        incident_id
    ] = incident


    # --------------------------------------------------------
    # FIND NEARBY RESPONDERS
    # --------------------------------------------------------

    nearby = []


    for responder in responders.values():

        if responder[
            "status"
        ] != "AVAILABLE":

            continue


        distance = haversine_km(

            data.latitude,

            data.longitude,

            responder[
                "latitude"
            ],

            responder[
                "longitude"
            ]
        )


        nearby.append({

            "responder_id":
                responder[
                    "responder_id"
                ],

            "name":
                responder[
                    "name"
                ],

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
        key=lambda x:
        x["distance_km"]
    )


    return {

        "incident":
            incident,

        "nearby_responders":
            nearby[:5]
    }


# ============================================================
# GET INCIDENTS
# ============================================================

@app.get("/api/incidents")
def get_incidents():

    return list(
        incidents.values()
    )


# ============================================================
# GET ONE INCIDENT
# ============================================================

@app.get(
    "/api/incidents/{incident_id}"
)
def get_incident(
    incident_id: str
):

    if incident_id not in incidents:

        raise HTTPException(
            status_code=404,
            detail="Incident not found"
        )

    return incidents[
        incident_id
    ]


# ============================================================
# REGISTER RESPONDER
# ============================================================

@app.post("/api/responders/register")
def register_responder(
    data: ResponderCreate
):

    responder_id = (
        f"RESP-{len(responders) + 1:04d}"
    )


    responder = {

        "responder_id":
            responder_id,

        "name":
            data.name,

        "responder_type":
            data.responder_type,

        "latitude":
            data.latitude,

        "longitude":
            data.longitude,

        "phone":
            data.phone,

        "status":
            "AVAILABLE",

        "registered_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }


    responders[
        responder_id
    ] = responder


    return responder


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

    if responder_id not in responders:

        raise HTTPException(
            status_code=404,
            detail="Responder not found"
        )


    responders[
        responder_id
    ]["latitude"] = data.latitude

    responders[
        responder_id
    ]["longitude"] = data.longitude


    return responders[
        responder_id
    ]


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

    if responder_id not in responders:

        raise HTTPException(
            status_code=404,
            detail="Responder not found"
        )


    allowed = {

        "AVAILABLE",
        "BUSY",
        "OFFLINE",
        "EN_ROUTE"
    }


    if data.status not in allowed:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid status. "
                f"Allowed: {allowed}"
            )
        )


    responders[
        responder_id
    ]["status"] = data.status


    return responders[
        responder_id
    ]


# ============================================================
# DISPATCH
# ============================================================

@app.post(
    "/api/incidents/{incident_id}/dispatch"
)
def dispatch_incident(
    incident_id: str
):

    if incident_id not in incidents:

        raise HTTPException(
            status_code=404,
            detail="Incident not found"
        )


    incident = incidents[
        incident_id
    ]


    nearby = []


    for responder in responders.values():

        if responder[
            "status"
        ] != "AVAILABLE":

            continue


        distance = haversine_km(

            incident[
                "latitude"
            ],

            incident[
                "longitude"
            ],

            responder[
                "latitude"
            ],

            responder[
                "longitude"
            ]
        )


        nearby.append({

            "responder_id":
                responder[
                    "responder_id"
                ],

            "name":
                responder[
                    "name"
                ],

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


    # --------------------------------------------------------
    # DEMO DISPATCH
    # --------------------------------------------------------

    selected = nearby[:3]


    for responder in selected:

        responder_id = (
            responder[
                "responder_id"
            ]
        )


        responders[
            responder_id
        ]["status"] = "ALERTED"


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

    if incident_id not in incidents:

        raise HTTPException(
            status_code=404,
            detail="Incident not found"
        )


    if responder_id not in responders:

        raise HTTPException(
            status_code=404,
            detail="Responder not found"
        )


    incident = incidents[
        incident_id
    ]

    responder = responders[
        responder_id
    ]


    if responder[
        "status"
    ] not in {
        "ALERTED",
        "AVAILABLE"
    }:

        raise HTTPException(
            status_code=409,
            detail="Responder unavailable"
        )


    incident[
        "assigned_responder"
    ] = responder_id

    incident[
        "status"
    ] = "RESPONDER_ASSIGNED"


    responder[
        "status"
    ] = "EN_ROUTE"


    return {

        "incident":
            incident,

        "responder":
            responder
    }


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

    connections[
        responder_id
    ] = websocket


    try:

        while True:

            await websocket.receive_text()

    except Exception:

        pass

    finally:

        connections.pop(
            responder_id,
            None
        )


# ============================================================
# RUN
# ============================================================
@app.get("/responder")
def responder_page():

    return FileResponse(
        "responder.html"
    )



if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )