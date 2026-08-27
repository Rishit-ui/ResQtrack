import httpx


NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
)


class HospitalSearchError(Exception):
    pass


async def find_nearby_hospitals(
    latitude: float,
    longitude: float,
    radius_km: float = 15.0,
):
    """
    Find hospital candidates around an accident location
    using OpenStreetMap Nominatim.

    Nominatim returns ranked search results rather than
    guaranteeing every hospital in the area.
    """

    if not -90 <= latitude <= 90:
        raise HospitalSearchError(
            "Invalid latitude."
        )

    if not -180 <= longitude <= 180:
        raise HospitalSearchError(
            "Invalid longitude."
        )

    # --------------------------------------------------------
    # Approximate bounding box.
    #
    # 1 degree latitude ≈ 111 km.
    # --------------------------------------------------------

    lat_delta = radius_km / 111.0

    lon_delta = (
        radius_km
        /
        (
            111.0
            *
            max(
                0.1,
                abs(
                    __import__("math").cos(
                        __import__("math").radians(
                            latitude
                        )
                    )
                )
            )
        )
    )

    south = latitude - lat_delta
    north = latitude + lat_delta

    west = longitude - lon_delta
    east = longitude + lon_delta


    params = {

        "q": "hospital",

        "format": "jsonv2",

        "limit": 10,

        "addressdetails": 1,

        "extratags": 1,

        "viewbox": (
            f"{west},{north},"
            f"{east},{south}"
        ),

        "bounded": 1,

    }


    headers = {

        "User-Agent":
            "ResQTrack/1.0 "
            "(emergency-routing-hackathon-project)"

    }


    # --------------------------------------------------------
    # Nominatim request
    # --------------------------------------------------------

    try:

        async with httpx.AsyncClient(
            timeout=20.0,
            headers=headers
        ) as client:

            response = await client.get(

                NOMINATIM_URL,

                params=params

            )

    except httpx.RequestError as error:

        raise HospitalSearchError(

            f"Unable to contact Nominatim: {error}"

        )


    if response.status_code != 200:

        raise HospitalSearchError(

            f"Nominatim HTTP error: "
            f"{response.status_code}"

        )


    try:

        data = response.json()

    except ValueError:

        raise HospitalSearchError(

            "Nominatim returned invalid JSON."

        )


    hospitals = []


    # --------------------------------------------------------
    # Convert Nominatim results
    # --------------------------------------------------------

    for result in data:

        result_type = result.get(
            "type",
            ""
        )

        result_class = result.get(
            "class",
            ""
        )

        # Keep healthcare/hospital-like results.
        if (
            result_class != "amenity"
            and
            result_type != "hospital"
        ):
            continue


        name = result.get(
            "name"
        )


        if not name:

            name = result.get(
                "display_name",
                "Unnamed Hospital"
            )


        lat = result.get(
            "lat"
        )

        lon = result.get(
            "lon"
        )


        if lat is None or lon is None:

            continue


        hospitals.append({

            "name":
                name,

            "latitude":
                float(lat),

            "longitude":
                float(lon),

            "osm_id":
                result.get("osm_id"),

            "osm_type":
                result.get("osm_type"),

            "display_name":
                result.get(
                    "display_name"
                ),

            "address":
                result.get(
                    "address",
                    {}
                ),

            "extratags":
                result.get(
                    "extratags",
                    {}
                ),

        })


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = {}


    for hospital in hospitals:

        key = (

            hospital["name"].lower(),

            round(
                hospital["latitude"],
                5
            ),

            round(
                hospital["longitude"],
                5
            )

        )


        unique[key] = hospital


    return list(
        unique.values()
    )