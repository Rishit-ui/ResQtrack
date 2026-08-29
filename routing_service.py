"""Compatibility shim - routing now lives in the backend package.

Kept so ``routing_api.py`` and the original scripts keep working:

    from routing_service import get_route, RoutingError

New code should import :mod:`backend.services.routing_service`, which adds
alternative routes, traffic-aware selection, turn-by-turn steps and an offline
fallback.
"""

from backend.services.routing_service import (  # noqa: F401
    RoutingError,
    get_route,
    travel_matrix,
)

__all__ = ["RoutingError", "get_route", "travel_matrix"]
