"""Compatibility shim - the hospital search now lives in the backend package.

Kept so the original standalone scripts (``test_hospitals.py``) keep working:

    from hospital_service import find_nearby_hospitals

New code should import :mod:`backend.services.hospital_service`, which adds the
Overpass source, the local offline cache and road-time ranking.
"""

from backend.services.hospital_service import (  # noqa: F401
    HospitalSearchError,
    find_nearby_facilities,
    find_nearby_hospitals,
)

__all__ = ["HospitalSearchError", "find_nearby_facilities", "find_nearby_hospitals"]
