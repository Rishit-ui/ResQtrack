import os
from typing import Any

import requests


ROUTES_URL = (
    "https://routes.googleapis.com/"
    "directions/v2:computeRoutes"
)


class RoutingError(Exception):
    pass


def _duration_to_seconds(
    duration: str
) -> float:

    if not duration:
        return 0.0

    if not duration.endswith("s"):
        raise RoutingError(
            f"Unexpected duration format: {duration}"
        )

    return float(
        duration[:-1]
    )


def compute_best_route(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float
) -> dict[str, Any]:

    api_key = os.getenv(
        "GOOGLE_MAPS_API_KEY"
    )

    if not api_key:

        raise RoutingError(
            "GOOGLE_MAPS_API_KEY is not set."
        )


    payload = {

        "origin": {

            "location": {

                "latLng": {

                    "latitude":
                        origin_latitude,

                    "longitude":
                        origin_longitude
                }
            }
        },

        "destination": {

            "location": {

                "latLng": {

                    "latitude":
                        destination_latitude,

                    "longitude":
                        destination_longitude
                }
            }
        },

        "travelMode":
            "DRIVE",

        "routingPreference":
            "TRAFFIC_AWARE",

        "computeAlternativeRoutes":
            True,

        "departureTime":
            "now",

        "languageCode":
            "en-US",

        "units":
            "METRIC"
    }


    headers = {

        "Content-Type":
            "application/json",

        "X-Goog-Api-Key":
            api_key,

        "X-Goog-FieldMask":
            ",".join([

                "routes.duration",

                "routes.staticDuration",

                "routes.distanceMeters",

                "routes.polyline.encodedPolyline",

                "routes.description",

                "routes.routeLabels"

            ])
    }


    try:

        response = requests.post(

            ROUTES_URL,

            json=payload,

            headers=headers,

            timeout=15
        )

    except requests.RequestException as exc:

        raise RoutingError(
            f"Routing request failed: {exc}"
        ) from exc


    if not response.ok:

        raise RoutingError(

            "Google Routes API returned "
            f"{response.status_code}: "
            f"{response.text}"
        )


    data = response.json()


    routes = data.get(
        "routes",
        []
    )


    if not routes:

        raise RoutingError(
            "No routes returned."
        )


    formatted_routes = []


    for index, route in enumerate(
        routes
    ):

        duration_seconds = (
            _duration_to_seconds(
                route.get(
                    "duration"
                )
            )
        )


        static_duration_seconds = (
            _duration_to_seconds(
                route.get(
                    "staticDuration"
                )
            )
        )


        distance_meters = int(
            route.get(
                "distanceMeters",
                0
            )
        )


        traffic_delay_seconds = max(
            0.0,

            duration_seconds
            -
            static_duration_seconds
        )


        formatted_routes.append({

            "route_index":
                index,

            "duration_seconds":
                duration_seconds,

            "static_duration_seconds":
                static_duration_seconds,

            "traffic_delay_seconds":
                traffic_delay_seconds,

            "distance_meters":
                distance_meters,

            "distance_km":
                round(
                    distance_meters
                    /
                    1000.0,
                    2
                ),

            "description":
                route.get(
                    "description"
                ),

            "route_labels":
                route.get(
                    "routeLabels",
                    []
                ),

            "polyline":
                route
                .get(
                    "polyline",
                    {}
                )
                .get(
                    "encodedPolyline"
                )
        })


    # --------------------------------------------------------
    # BEST ROUTE
    #
    # Primary objective:
    # lowest traffic-aware ETA.
    # --------------------------------------------------------

    best_route = min(

        formatted_routes,

        key=lambda route:
        route["duration_seconds"]
    )


    return {

        "routing_provider":
            "Google Routes API",

        "routing_preference":
            "TRAFFIC_AWARE",

        "travel_mode":
            "DRIVE",

        "origin": {

            "latitude":
                origin_latitude,

            "longitude":
                origin_longitude
        },

        "destination": {

            "latitude":
                destination_latitude,

            "longitude":
                destination_longitude
        },

        "best_route":
            best_route,

        "alternatives":
            formatted_routes
    }