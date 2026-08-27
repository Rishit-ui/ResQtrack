from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from routing_service import (
    get_route,
    RoutingError,
)


app = FastAPI(
    title="ResQTrack Routing API",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class RouteRequest(BaseModel):

    start_lat: float = Field(
        ...,
        ge=-90,
        le=90
    )

    start_lon: float = Field(
        ...,
        ge=-180,
        le=180
    )

    end_lat: float = Field(
        ...,
        ge=-90,
        le=90
    )

    end_lon: float = Field(
        ...,
        ge=-180,
        le=180
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "service": "ResQTrack Routing API",
        "status": "running",
        "routing_engine": "OSRM",
    }


# ============================================================
# ROUTING ENDPOINT
# ============================================================

@app.post("/route")
async def route(request: RouteRequest):

    try:

        result = await get_route(
            start_lat=request.start_lat,
            start_lon=request.start_lon,
            end_lat=request.end_lat,
            end_lon=request.end_lon,
        )

        return {
            "success": True,
            "routing_engine": "OSRM",
            "start": {
                "latitude": request.start_lat,
                "longitude": request.start_lon,
            },
            "destination": {
                "latitude": request.end_lat,
                "longitude": request.end_lon,
            },
            "route": result,
        }

    except RoutingError as error:

        raise HTTPException(
            status_code=502,
            detail=str(error)
        )