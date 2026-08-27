import httpx


OSRM_BASE_URL = "https://router.project-osrm.org"


class RoutingError(Exception):
    pass


async def get_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
):
    """
    Calculate a driving route using OSRM.

    Coordinates are supplied as:
        latitude, longitude

    OSRM internally requires:
        longitude, latitude
    """

    # ------------------------------------------------------------
    # Validate coordinates
    # ------------------------------------------------------------

    if not -90 <= start_lat <= 90:
        raise RoutingError("Invalid start latitude")

    if not -180 <= start_lon <= 180:
        raise RoutingError("Invalid start longitude")

    if not -90 <= end_lat <= 90:
        raise RoutingError("Invalid destination latitude")

    if not -180 <= end_lon <= 180:
        raise RoutingError("Invalid destination longitude")

    # ------------------------------------------------------------
    # OSRM requires longitude,latitude
    # ------------------------------------------------------------

    coordinates = (
        f"{start_lon},{start_lat};"
        f"{end_lon},{end_lat}"
    )

    url = (
        f"{OSRM_BASE_URL}/route/v1/driving/"
        f"{coordinates}"
    )

    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
    }

    # ------------------------------------------------------------
    # Request OSRM
    # ------------------------------------------------------------

    try:

        async with httpx.AsyncClient(
            timeout=15.0
        ) as client:

            response = await client.get(
                url,
                params=params,
            )

    except httpx.RequestError as error:

        raise RoutingError(
            f"Unable to contact routing service: {error}"
        )

    # ------------------------------------------------------------
    # HTTP validation
    # ------------------------------------------------------------

    if response.status_code != 200:

        raise RoutingError(
            f"OSRM HTTP error: {response.status_code}"
        )

    try:

        data = response.json()

    except ValueError:

        raise RoutingError(
            "OSRM returned invalid JSON"
        )

    # ------------------------------------------------------------
    # OSRM response validation
    # ------------------------------------------------------------

    if data.get("code") != "Ok":

        raise RoutingError(
            data.get(
                "message",
                "OSRM could not calculate the route"
            )
        )

    routes = data.get("routes")

    if not routes:

        raise RoutingError(
            "OSRM returned no routes"
        )

    route = routes[0]

    # ------------------------------------------------------------
    # Extract route information
    # ------------------------------------------------------------

    distance_meters = float(
        route.get("distance", 0)
    )

    duration_seconds = float(
        route.get("duration", 0)
    )

    distance_km = distance_meters / 1000.0

    duration_minutes = duration_seconds / 60.0

    return {
        "distance_km": round(
            distance_km,
            2
        ),

        "duration_minutes": round(
            duration_minutes,
            2
        ),

        "distance_meters": round(
            distance_meters,
            2
        ),

        "duration_seconds": round(
            duration_seconds,
            2
        ),

        "geometry": route.get(
            "geometry"
        ),

        "legs": route.get(
            "legs",
            []
        ),
    }