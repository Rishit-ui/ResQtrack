"""Seed ResQTrack with demo responders and offline hospital data.

Run this once before a demonstration:

    python seed_demo.py                      # Bengaluru (default)
    python seed_demo.py --lat 28.61 --lon 77.21 --city Delhi
    python seed_demo.py --reset              # wipe and start clean

The hospitals it writes are real facilities near the chosen centre.  They live
in the ``facilities`` table, which the hospital service falls back to when the
venue's internet cannot reach OpenStreetMap - so the "nearby hospitals" step of
the demo works with the network unplugged.
"""

from __future__ import annotations

import argparse
import random
import sys

from backend.database import connect, initialise, next_id, now_utc

# Real hospitals around central Bengaluru, with approximate coordinates.
BENGALURU_HOSPITALS = [
    ("Bowring and Lady Curzon Hospital", 12.9829, 77.6046, "080-25591325", 1),
    ("Victoria Hospital", 12.9629, 77.5745, "080-26701150", 1),
    ("St. John's Medical College Hospital", 12.9279, 77.6212, "080-22065000", 1),
    ("Manipal Hospital, Old Airport Road", 12.9591, 77.6494, "080-25023700", 1),
    ("Mallya Hospital", 12.9689, 77.5966, "080-22277979", 1),
    ("Fortis Hospital, Bannerghatta Road", 12.8934, 77.5978, "080-66214444", 1),
    ("Apollo Hospital, Sheshadripuram", 12.9915, 77.5735, "080-46126666", 1),
    ("Vydehi Institute of Medical Sciences", 12.9908, 77.7069, "080-28413381", 1),
    ("KC General Hospital, Malleswaram", 12.9976, 77.5713, "080-23342778", 1),
    ("Sagar Hospital, Jayanagar", 12.9081, 77.5855, "080-42888100", 1),
    ("Cumballa Emergency Clinic, MG Road", 12.9748, 77.6096, None, 0),
    ("Cubbon Park Police Station", 12.9762, 77.5929, "080-22943600", 0),
]

DELHI_HOSPITALS = [
    ("All India Institute of Medical Sciences", 28.5672, 77.2100, "011-26588500", 1),
    ("Safdarjung Hospital", 28.5687, 77.2065, "011-26707444", 1),
    ("Lok Nayak Hospital", 28.6392, 77.2380, "011-23230838", 1),
    ("Ram Manohar Lohia Hospital", 28.6247, 77.2020, "011-23404446", 1),
    ("Sir Ganga Ram Hospital", 28.6390, 77.1895, "011-42251000", 1),
    ("Max Super Speciality, Saket", 28.5286, 77.2148, "011-26515050", 1),
]

RESPONDERS = [
    ("Ambulance 108 - Unit A", "AMBULANCE", "+91-98450-11001", "KA-01-EA-1108"),
    ("Ambulance 108 - Unit B", "AMBULANCE", "+91-98450-11002", "KA-01-EA-2208"),
    ("Rapid Paramedic Bike 1", "PARAMEDIC", "+91-98450-11003", "KA-01-PB-0091"),
    ("Traffic Patrol Alpha", "POLICE", "+91-98450-11004", "KA-01-TP-4410"),
    ("Fire and Rescue Tender 3", "FIRE", "+91-98450-11005", "KA-01-FR-0003"),
    ("City Hospital Ambulance", "AMBULANCE", "+91-98450-11006", "KA-01-CH-7712"),
]


def seed_facilities(connection, hospitals, city: str) -> int:
    timestamp = now_utc()
    written = 0
    for name, latitude, longitude, phone, emergency in hospitals:
        connection.execute(
            """
            INSERT INTO facilities (ref, name, kind, latitude, longitude, phone, address,
                                    emergency, source, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'seed', ?)
            ON CONFLICT(ref) DO UPDATE SET
                name = excluded.name, latitude = excluded.latitude,
                longitude = excluded.longitude, phone = excluded.phone,
                emergency = excluded.emergency, cached_at = excluded.cached_at
            """,
            (
                f"seed:{city}:{name}",
                name,
                "hospital" if emergency else "clinic",
                latitude,
                longitude,
                phone,
                city,
                emergency,
                timestamp,
            ),
        )
        written += 1
    return written


def seed_responders(connection, latitude: float, longitude: float, spread_km: float) -> list[str]:
    created = []
    generator = random.Random(20260101)   # stable positions between runs
    degrees = spread_km / 111.0
    timestamp = now_utc()
    for name, kind, phone, vehicle in RESPONDERS:
        existing = connection.execute(
            "SELECT responder_id FROM responders WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            created.append(existing["responder_id"])
            continue
        responder_id = next_id(connection, "responders", "responder_id", "RESP-", 4)
        connection.execute(
            """
            INSERT INTO responders (responder_id, name, responder_type, phone, vehicle_number,
                                    latitude, longitude, status, registered_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'AVAILABLE', ?, ?)
            """,
            (
                responder_id,
                name,
                kind,
                phone,
                vehicle,
                latitude + generator.uniform(-degrees, degrees),
                longitude + generator.uniform(-degrees, degrees),
                timestamp,
                timestamp,
            ),
        )
        created.append(responder_id)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the ResQTrack demo database")
    parser.add_argument("--lat", type=float, default=12.9719)
    parser.add_argument("--lon", type=float, default=77.5937)
    parser.add_argument("--city", default="Bengaluru", choices=("Bengaluru", "Delhi"))
    parser.add_argument("--spread-km", type=float, default=4.0,
                        help="how far the demo responders are scattered")
    parser.add_argument("--reset", action="store_true", help="delete existing rows first")
    args = parser.parse_args(argv)

    initialise()
    connection = connect()
    try:
        if args.reset:
            for table in ("incident_events", "dispatches", "location_history",
                          "notifications", "incidents", "responders", "facilities"):
                connection.execute(f"DELETE FROM {table}")
            print("Existing data cleared.")

        hospitals = DELHI_HOSPITALS if args.city == "Delhi" else BENGALURU_HOSPITALS
        facilities = seed_facilities(connection, hospitals, args.city)
        responders = seed_responders(connection, args.lat, args.lon, args.spread_km)
        connection.commit()
    finally:
        connection.close()

    print(f"\nResQTrack demo data ready ({args.city})")
    print(f"  facilities cached : {facilities}")
    print(f"  responders on duty: {len(responders)}  -> {', '.join(responders)}")
    print(f"  centre            : {args.lat:.5f}, {args.lon:.5f}")
    print("\nNext:")
    print("  1. python run_resqtrack.py          (backend + detector together)")
    print("  2. open http://localhost:8000/dashboard   (control room)")
    print("  3. open http://localhost:8000/responder   (ambulance app)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
