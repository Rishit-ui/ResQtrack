import asyncio

from hospital_service import (
    find_nearby_hospitals,
    HospitalSearchError
)


async def main():

    # ========================================================
    # TEMPORARY TEST ACCIDENT LOCATION
    # ========================================================

    latitude = 12.653738
    longitude = 77.446470


    try:

        hospitals = await find_nearby_hospitals(
            latitude,
            longitude
        )


        print()

        print(
            "=============================================="
        )

        print(
            "NEARBY HOSPITALS"
        )

        print(
            "=============================================="
        )

        print(
            "Found:",
            len(hospitals)
        )

        print()


        # ====================================================
        # DISPLAY HOSPITALS
        # ====================================================

        for index, hospital in enumerate(
            hospitals,
            start=1
        ):

            name = hospital.get(
                "name",
                "Unnamed Hospital"
            )


            hospital_lat = hospital.get(
                "latitude"
            )


            hospital_lon = hospital.get(
                "longitude"
            )


            osm_id = hospital.get(
                "osm_id"
            )


            osm_type = hospital.get(
                "osm_type"
            )


            address = hospital.get(
                "address"
            )


            if not isinstance(
                address,
                dict
            ):

                address = {}


            extratags = hospital.get(
                "extratags"
            )


            if not isinstance(
                extratags,
                dict
            ):

                extratags = {}


            emergency = extratags.get(
                "emergency",
                "Not available"
            )


            phone = extratags.get(
                "phone",
                "Not available"
            )


            website = extratags.get(
                "website",
                "Not available"
            )


            print(
                f"{index}. {name}"
            )


            print(
                "   Coordinates:",
                hospital_lat,
                hospital_lon
            )


            print(
                "   OSM ID:",
                osm_id
            )


            print(
                "   OSM Type:",
                osm_type
            )


            print(
                "   Emergency:",
                emergency
            )


            print(
                "   Phone:",
                phone
            )


            print(
                "   Website:",
                website
            )


            print(
                "   Address:",
                address
            )


            print(
                "----------------------------------------------"
            )


    except HospitalSearchError as error:

        print()

        print(
            "HOSPITAL SEARCH ERROR:"
        )

        print(
            error
        )


if __name__ == "__main__":

    asyncio.run(
        main()
    )