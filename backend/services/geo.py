"""Geodesy helpers shared by dispatch, routing and the hospital search."""

from __future__ import annotations

from math import atan2, cos, degrees, radians, sin, sqrt

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from the first point to the second."""
    d_lon = radians(lon2 - lon1)
    y = sin(d_lon) * cos(radians(lat2))
    x = cos(radians(lat1)) * sin(radians(lat2)) - sin(radians(lat1)) * cos(radians(lat2)) * cos(d_lon)
    return (degrees(atan2(y, x)) + 360.0) % 360.0


def compass_point(bearing: float) -> str:
    points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return points[int((bearing + 22.5) % 360.0 // 45.0)]


def valid_coordinate(latitude: float, longitude: float) -> bool:
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def bounding_box(latitude: float, longitude: float, radius_km: float) -> tuple[float, float, float, float]:
    """(south, west, north, east) box around a point."""
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / max(1.0, 111.0 * max(0.1, abs(cos(radians(latitude)))))
    return (
        latitude - lat_delta,
        longitude - lon_delta,
        latitude + lat_delta,
        longitude + lon_delta,
    )


def interpolate(
    start: tuple[float, float], end: tuple[float, float], steps: int
) -> list[list[float]]:
    """Evenly spaced points between two coordinates, as [lon, lat] pairs.

    Used to synthesise a route line when no routing service is reachable, so
    the map still draws something meaningful instead of failing.
    """
    steps = max(2, steps)
    points: list[list[float]] = []
    for index in range(steps + 1):
        fraction = index / steps
        points.append(
            [
                start[1] + (end[1] - start[1]) * fraction,
                start[0] + (end[0] - start[0]) * fraction,
            ]
        )
    return points
