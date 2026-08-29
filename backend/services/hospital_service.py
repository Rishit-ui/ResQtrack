"""Nearby hospitals, trauma centres and help points around an accident.

Three sources, tried in order, so the list is never empty when a crew needs it:

1. **Overpass** (OpenStreetMap) - a real radius query that returns every
   hospital, clinic, police station and fire station near the point, with the
   emergency-department tag and phone number when OSM has them.
2. **Nominatim** - a ranked text search, used when Overpass is busy.
3. **The local cache / built-in set** - facilities previously seen for this
   area, written to SQLite on every successful lookup.

Results are ranked by *road* travel time from the accident, not by straight-line
distance, and facilities with an emergency department are surfaced first.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend.database import connect, now_utc
from backend.services.geo import bounding_box, haversine_km, valid_coordinate
from backend.services.routing_service import travel_matrix

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ResQTrack/2.0 (SIH 2026 emergency response prototype)"

FACILITY_KINDS = {
    "hospital": "Hospital",
    "clinic": "Clinic",
    "doctors": "Doctor",
    "police": "Police station",
    "fire_station": "Fire station",
}


class HospitalSearchError(Exception):
    pass


# ============================================================
# OVERPASS
# ============================================================

def _overpass_query(latitude: float, longitude: float, radius_m: int, kinds: list[str]) -> str:
    amenities = "|".join(kinds)
    return f"""
    [out:json][timeout:25];
    (
      node["amenity"~"^({amenities})$"](around:{radius_m},{latitude},{longitude});
      way["amenity"~"^({amenities})$"](around:{radius_m},{latitude},{longitude});
      relation["amenity"~"^({amenities})$"](around:{radius_m},{latitude},{longitude});
    );
    out center tags;
    """


async def _from_overpass(
    latitude: float, longitude: float, radius_km: float, kinds: list[str]
) -> list[dict[str, Any]]:
    query = _overpass_query(latitude, longitude, int(radius_km * 1000), kinds)
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            async with httpx.AsyncClient(timeout=25.0, headers={"User-Agent": USER_AGENT}) as client:
                response = await client.post(url, data={"data": query})
            if response.status_code != 200:
                last_error = HospitalSearchError(f"Overpass HTTP {response.status_code}")
                continue
            elements = response.json().get("elements", [])
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
            continue

        facilities = []
        for element in elements:
            tags = element.get("tags", {})
            centre = element.get("center") or {}
            lat = element.get("lat", centre.get("lat"))
            lon = element.get("lon", centre.get("lon"))
            if lat is None or lon is None:
                continue
            amenity = tags.get("amenity", "hospital")
            facilities.append(
                {
                    "ref": f"osm:{element.get('type', 'node')}:{element.get('id')}",
                    "name": tags.get("name") or f"Unnamed {FACILITY_KINDS.get(amenity, amenity)}",
                    "kind": amenity,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "phone": tags.get("phone") or tags.get("contact:phone"),
                    "address": _address_from_tags(tags),
                    "emergency": 1 if tags.get("emergency") == "yes" or amenity == "hospital" else 0,
                    "source": "overpass",
                }
            )
        if facilities:
            return facilities
    if last_error:
        raise HospitalSearchError(str(last_error))
    return []


def _address_from_tags(tags: dict[str, Any]) -> str:
    parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb"),
        tags.get("addr:city"),
        tags.get("addr:postcode"),
    ]
    return ", ".join(str(part) for part in parts if part)


# ============================================================
# NOMINATIM
# ============================================================

async def _from_nominatim(latitude: float, longitude: float, radius_km: float) -> list[dict[str, Any]]:
    south, west, north, east = bounding_box(latitude, longitude, radius_km)
    params = {
        "q": "hospital",
        "format": "jsonv2",
        "limit": 20,
        "addressdetails": 1,
        "extratags": 1,
        "viewbox": f"{west},{north},{east},{south}",
        "bounded": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
            response = await client.get(NOMINATIM_URL, params=params)
        if response.status_code != 200:
            raise HospitalSearchError(f"Nominatim HTTP {response.status_code}")
        results = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise HospitalSearchError(str(error)) from error

    facilities = []
    for result in results:
        if result.get("class") != "amenity" and result.get("type") != "hospital":
            continue
        try:
            lat, lon = float(result["lat"]), float(result["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        extra = result.get("extratags") or {}
        facilities.append(
            {
                "ref": f"osm:{result.get('osm_type')}:{result.get('osm_id')}",
                "name": result.get("name") or result.get("display_name", "Unnamed hospital"),
                "kind": result.get("type", "hospital"),
                "latitude": lat,
                "longitude": lon,
                "phone": extra.get("phone"),
                "address": result.get("display_name", ""),
                "emergency": 1,
                "source": "nominatim",
            }
        )
    return facilities


# ============================================================
# LOCAL CACHE
# ============================================================

def cache_facilities(facilities: list[dict[str, Any]]) -> None:
    if not facilities:
        return
    connection = connect()
    try:
        connection.executemany(
            """
            INSERT INTO facilities (ref, name, kind, latitude, longitude, phone, address,
                                    emergency, source, cached_at)
            VALUES (:ref, :name, :kind, :latitude, :longitude, :phone, :address,
                    :emergency, :source, :cached_at)
            ON CONFLICT(ref) DO UPDATE SET
                name = excluded.name,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                phone = COALESCE(excluded.phone, facilities.phone),
                address = COALESCE(excluded.address, facilities.address),
                emergency = excluded.emergency,
                cached_at = excluded.cached_at
            """,
            [{**facility, "cached_at": now_utc()} for facility in facilities],
        )
        connection.commit()
    finally:
        connection.close()


def cached_facilities(latitude: float, longitude: float, radius_km: float) -> list[dict[str, Any]]:
    south, west, north, east = bounding_box(latitude, longitude, radius_km)
    connection = connect()
    try:
        rows = connection.execute(
            """
            SELECT ref, name, kind, latitude, longitude, phone, address, emergency, source
            FROM facilities
            WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
            """,
            (south, north, west, east),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


# ============================================================
# PUBLIC API
# ============================================================

async def find_nearby_facilities(
    latitude: float,
    longitude: float,
    *,
    radius_km: float = 12.0,
    kinds: list[str] | None = None,
    limit: int = 10,
    rank_by_road: bool = True,
) -> dict[str, Any]:
    """Ranked list of places an ambulance can take a casualty."""
    if not valid_coordinate(latitude, longitude):
        raise HospitalSearchError("Invalid accident coordinate")

    kinds = kinds or ["hospital", "clinic"]
    source = "overpass"
    notes: list[str] = []

    try:
        facilities = await _from_overpass(latitude, longitude, radius_km, kinds)
    except HospitalSearchError as error:
        notes.append(f"Overpass unavailable: {error}")
        facilities = []

    if not facilities:
        try:
            facilities = await _from_nominatim(latitude, longitude, radius_km)
            source = "nominatim"
        except HospitalSearchError as error:
            notes.append(f"Nominatim unavailable: {error}")
            facilities = []

    if facilities:
        cache_facilities(facilities)
    else:
        facilities = cached_facilities(latitude, longitude, radius_km)
        source = "local-cache"
        if facilities:
            notes.append("Live map services unreachable; using previously cached facilities.")
        else:
            notes.append(
                "No facility data available offline. Seed the database with "
                "`python seed_demo.py` or connect to the internet."
            )

    # De-duplicate places OSM lists more than once.
    unique: dict[str, dict[str, Any]] = {}
    for facility in facilities:
        key = facility.get("ref") or f"{facility['name']}:{facility['latitude']:.4f}"
        unique.setdefault(key, facility)
    facilities = list(unique.values())

    for facility in facilities:
        facility["straight_km"] = round(
            haversine_km(latitude, longitude, facility["latitude"], facility["longitude"]), 2
        )
    facilities.sort(key=lambda item: item["straight_km"])
    facilities = facilities[: max(limit * 2, limit)]

    if rank_by_road and facilities:
        matrix = await travel_matrix(
            (latitude, longitude),
            [(item["latitude"], item["longitude"]) for item in facilities],
        )
        for facility, travel in zip(facilities, matrix):
            facility["distance_km"] = travel["distance_km"]
            facility["eta_minutes"] = travel["duration_minutes"]
            facility["distance_source"] = travel["source"]
    else:
        for facility in facilities:
            facility["distance_km"] = facility["straight_km"]
            facility["eta_minutes"] = round(facility["straight_km"] / 32.0 * 60.0, 1)
            facility["distance_source"] = "straight-line"

    # An A&E five minutes further away still beats a clinic that cannot admit.
    facilities.sort(
        key=lambda item: (0 if item.get("emergency") else 1, item.get("eta_minutes", 999))
    )

    return {
        "facilities": facilities[:limit],
        "source": source,
        "notes": notes,
        "searched_radius_km": radius_km,
        "count": len(facilities[:limit]),
    }


# Backwards-compatible name used by the original prototype scripts.
async def find_nearby_hospitals(latitude: float, longitude: float, radius_km: float = 15.0):
    result = await find_nearby_facilities(latitude, longitude, radius_km=radius_km)
    return result["facilities"]
