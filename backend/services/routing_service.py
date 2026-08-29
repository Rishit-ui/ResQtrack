"""Road routing with turn-by-turn guidance and least-traffic route selection.

Primary source is OSRM, asked for *alternative* routes rather than one answer.
Each alternative is then scored by the traffic provider and the one with the
lowest predicted travel time wins - that is the "route with least traffic" the
ambulance is guided along.

If no routing server is reachable (a common state at a demo venue), the module
degrades to a straight-line estimate rather than failing.  The response always
carries ``source`` and ``offline`` so the interface can say plainly whether it
is showing a real road route or a fallback estimate.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from backend.services.geo import bearing_degrees, compass_point, haversine_km, interpolate, valid_coordinate
from backend.services.traffic import assess_route

OSRM_BASE_URL = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")

# Average city ambulance speed used only by the offline estimator.
FALLBACK_SPEED_KMH = 32.0
ROAD_WINDING_FACTOR = 1.32   # roads are longer than the straight line


class RoutingError(Exception):
    pass


# ============================================================
# OSRM
# ============================================================

async def _osrm_routes(
    start: tuple[float, float], end: tuple[float, float], alternatives: bool
) -> list[dict[str, Any]]:
    coordinates = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    url = f"{OSRM_BASE_URL}/route/v1/driving/{coordinates}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
        "annotations": "duration,distance",
        "alternatives": "true" if alternatives else "false",
    }
    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.get(url, params=params)
    if response.status_code != 200:
        raise RoutingError(f"OSRM returned HTTP {response.status_code}")
    payload = response.json()
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise RoutingError(payload.get("message", "OSRM could not build a route"))
    return payload["routes"]


def _extract_steps(route: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten OSRM legs into plain turn-by-turn instructions."""
    steps: list[dict[str, Any]] = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            manoeuvre = step.get("maneuver", {})
            location = manoeuvre.get("location") or [None, None]
            steps.append(
                {
                    "instruction": _instruction_text(step, manoeuvre),
                    "type": manoeuvre.get("type", "continue"),
                    "modifier": manoeuvre.get("modifier"),
                    "road": step.get("name") or "unnamed road",
                    "distance_m": round(float(step.get("distance", 0.0)), 1),
                    "duration_s": round(float(step.get("duration", 0.0)), 1),
                    "latitude": location[1],
                    "longitude": location[0],
                }
            )
    return steps


def _instruction_text(step: dict[str, Any], manoeuvre: dict[str, Any]) -> str:
    road = step.get("name") or "the road"
    kind = manoeuvre.get("type", "continue")
    modifier = manoeuvre.get("modifier", "")
    if kind == "depart":
        return f"Head {modifier or 'out'} on {road}"
    if kind == "arrive":
        return "Arrive at the destination"
    if kind == "roundabout":
        exit_number = manoeuvre.get("exit")
        return f"At the roundabout take exit {exit_number} onto {road}" if exit_number else f"Take the roundabout onto {road}"
    if kind in ("turn", "end of road", "fork", "merge", "new name", "continue", "on ramp", "off ramp"):
        action = f"{kind.replace('_', ' ').title()}"
        if modifier:
            action = f"Turn {modifier}" if kind == "turn" else f"{action} {modifier}"
        return f"{action} onto {road}"
    return f"Continue on {road}"


def _road_classes(route: dict[str, Any]) -> dict[str, float]:
    """Distance travelled on each OSM road class, for the traffic model."""
    classes: dict[str, float] = {}
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            name = (
                step.get("intersections", [{}])[0].get("classes", ["unclassified"])[0]
                if step.get("intersections")
                else "unclassified"
            )
            if not isinstance(name, str):
                name = "unclassified"
            classes[name] = classes.get(name, 0.0) + float(step.get("distance", 0.0))
    return classes or {"unclassified": 1.0}


def _shape(route: dict[str, Any], label: str) -> dict[str, Any]:
    distance_km = float(route.get("distance", 0.0)) / 1000.0
    duration_min = float(route.get("duration", 0.0)) / 60.0
    steps = _extract_steps(route)
    return {
        "label": label,
        "distance_km": round(distance_km, 2),
        "duration_minutes": round(duration_min, 1),
        "base_duration_minutes": round(duration_min, 1),
        "geometry": route.get("geometry"),
        "steps": steps,
        "step_count": len(steps),
        "road_classes": _road_classes(route),
        "source": "osrm",
        "offline": False,
    }


# ============================================================
# OFFLINE ESTIMATE
# ============================================================

def _offline_route(
    start: tuple[float, float], end: tuple[float, float], reason: str
) -> dict[str, Any]:
    straight_km = haversine_km(start[0], start[1], end[0], end[1])
    distance_km = straight_km * ROAD_WINDING_FACTOR
    duration_min = (distance_km / FALLBACK_SPEED_KMH) * 60.0
    heading = compass_point(bearing_degrees(start[0], start[1], end[0], end[1]))
    return {
        "label": "direct estimate",
        "distance_km": round(distance_km, 2),
        "duration_minutes": round(duration_min, 1),
        "base_duration_minutes": round(duration_min, 1),
        "geometry": {"type": "LineString", "coordinates": interpolate(start, end, 24)},
        "steps": [
            {
                "instruction": f"Head {heading} towards the destination",
                "type": "depart",
                "modifier": heading,
                "road": "direct line",
                "distance_m": round(distance_km * 1000.0, 1),
                "duration_s": round(duration_min * 60.0, 1),
                "latitude": start[0],
                "longitude": start[1],
            },
            {
                "instruction": "Arrive at the destination",
                "type": "arrive",
                "modifier": None,
                "road": "destination",
                "distance_m": 0.0,
                "duration_s": 0.0,
                "latitude": end[0],
                "longitude": end[1],
            },
        ],
        "step_count": 2,
        "road_classes": {"unclassified": distance_km * 1000.0},
        "source": "offline-estimate",
        "offline": True,
        "note": f"No routing server reachable ({reason}); showing a straight-line estimate.",
    }


# ============================================================
# PUBLIC API
# ============================================================

async def get_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    *,
    prefer_least_traffic: bool = True,
) -> dict[str, Any]:
    """Best route from start to end, chosen for the shortest real travel time.

    Returns the winning route plus every alternative that was considered, each
    with its own traffic assessment, so the interface can explain *why* this
    route was chosen instead of the shortest one.
    """
    if not valid_coordinate(start_lat, start_lon):
        raise RoutingError("Invalid start coordinate")
    if not valid_coordinate(end_lat, end_lon):
        raise RoutingError("Invalid destination coordinate")

    start, end = (start_lat, start_lon), (end_lat, end_lon)

    try:
        raw_routes = await _osrm_routes(start, end, alternatives=prefer_least_traffic)
        candidates = [
            _shape(route, "fastest" if index == 0 else f"alternative {index}")
            for index, route in enumerate(raw_routes)
        ]
    except (RoutingError, httpx.HTTPError) as error:
        candidates = [_offline_route(start, end, type(error).__name__)]

    for candidate in candidates:
        assessment = await assess_route(candidate)
        candidate["traffic"] = assessment.as_dict()
        # The number an ambulance crew actually cares about.
        candidate["duration_minutes"] = round(
            candidate["base_duration_minutes"] * assessment.factor, 1
        )

    if prefer_least_traffic:
        candidates.sort(key=lambda item: item["duration_minutes"])
    best = candidates[0]
    best["chosen_because"] = (
        "only route available"
        if len(candidates) == 1
        else (
            f"{best['traffic']['level']} traffic - "
            f"{round(candidates[1]['duration_minutes'] - best['duration_minutes'], 1)} min "
            f"faster than the next option"
        )
    )

    return {
        "route": best,
        "alternatives": candidates,
        "traffic_source": best["traffic"]["source"],
        "offline": best["offline"],
        "distance_km": best["distance_km"],
        "duration_minutes": best["duration_minutes"],
        "geometry": best["geometry"],
        "steps": best["steps"],
    }


async def travel_matrix(
    origin: tuple[float, float], destinations: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    """Road distance and duration from one origin to many destinations.

    Uses one OSRM ``table`` call so ranking twenty hospitals costs one request.
    Falls back to great-circle estimates when the service is unreachable.
    """
    if not destinations:
        return []

    coordinates = ";".join(
        f"{longitude},{latitude}" for latitude, longitude in [origin, *destinations]
    )
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coordinates}"
    params = {
        "sources": "0",
        "destinations": ";".join(str(index + 1) for index in range(len(destinations))),
        "annotations": "duration,distance",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(url, params=params)
        payload = response.json()
        if response.status_code != 200 or payload.get("code") != "Ok":
            raise RoutingError("table service unavailable")
        durations = (payload.get("durations") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        results = []
        for index in range(len(destinations)):
            duration = durations[index] if index < len(durations) else None
            distance = distances[index] if index < len(distances) else None
            if duration is None or distance is None:
                raise RoutingError("incomplete table response")
            results.append(
                {
                    "distance_km": round(float(distance) / 1000.0, 2),
                    "duration_minutes": round(float(duration) / 60.0, 1),
                    "source": "osrm",
                }
            )
        return results
    except (RoutingError, httpx.HTTPError, ValueError, IndexError):
        return [
            {
                "distance_km": round(
                    haversine_km(origin[0], origin[1], lat, lon) * ROAD_WINDING_FACTOR, 2
                ),
                "duration_minutes": round(
                    haversine_km(origin[0], origin[1], lat, lon)
                    * ROAD_WINDING_FACTOR
                    / FALLBACK_SPEED_KMH
                    * 60.0,
                    1,
                ),
                "source": "offline-estimate",
            }
            for lat, lon in destinations
        ]
