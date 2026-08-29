"""Traffic weighting used to pick the least-congested route.

Public OSRM has no live traffic layer, so ResQTrack keeps traffic behind a
provider interface with two implementations:

``LiveTrafficProvider``
    Used when ``TOMTOM_API_KEY`` is set.  Queries TomTom's flow segment API at
    sample points along a candidate route and returns the real ratio of current
    speed to free-flow speed.

``ModelledTrafficProvider``
    The offline default.  It does *not* pretend to know live conditions: it
    applies a published-style congestion model from the time of day, the road
    classes the route actually uses (taken from the OSRM step metadata) and the
    number of junctions, which is enough to rank alternatives sensibly and to
    demonstrate the flow end to end.

Whichever provider is active is reported in the API response as
``traffic_source``, so nothing on screen claims to be live data when it is not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# Rush-hour multipliers by local hour.  1.0 = free flow.
HOUR_CONGESTION = {
    0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00, 4: 1.02, 5: 1.06,
    6: 1.14, 7: 1.32, 8: 1.52, 9: 1.46, 10: 1.24, 11: 1.18,
    12: 1.22, 13: 1.24, 14: 1.18, 15: 1.22, 16: 1.34,
    17: 1.56, 18: 1.62, 19: 1.48, 20: 1.28, 21: 1.14,
    22: 1.06, 23: 1.02,
}

# How much each road class slows down when it is busy.  Motorways degrade
# gracefully; narrow residential streets seize up.
ROAD_CLASS_SENSITIVITY = {
    "motorway": 0.35,
    "trunk": 0.55,
    "primary": 0.80,
    "secondary": 0.95,
    "tertiary": 1.05,
    "residential": 1.25,
    "living_street": 1.40,
    "unclassified": 1.10,
    "service": 1.30,
}


@dataclass
class TrafficAssessment:
    """How badly one candidate route is expected to be slowed."""

    factor: float                 # 1.0 = free flow, 1.6 = 60% slower
    level: str                    # free | light | moderate | heavy | severe
    source: str
    detail: str

    @property
    def score(self) -> float:
        return self.factor

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": round(self.factor, 3),
            "level": self.level,
            "source": self.source,
            "detail": self.detail,
        }


def _level_for(factor: float) -> str:
    if factor < 1.08:
        return "free"
    if factor < 1.22:
        return "light"
    if factor < 1.45:
        return "moderate"
    if factor < 1.75:
        return "heavy"
    return "severe"


class ModelledTrafficProvider:
    """Deterministic congestion model - no network, always available."""

    name = "modelled"

    def assess(self, route: dict[str, Any], when: datetime | None = None) -> TrafficAssessment:
        when = when or datetime.now()
        base = HOUR_CONGESTION.get(when.hour, 1.15)
        # Weekends are calmer.
        if when.weekday() >= 5:
            base = 1.0 + (base - 1.0) * 0.55

        classes = route.get("road_classes") or {}
        total = sum(classes.values()) or 1.0
        sensitivity = (
            sum(ROAD_CLASS_SENSITIVITY.get(name, 1.0) * share for name, share in classes.items())
            / total
        )

        # Every junction is somewhere to stop.
        junctions = route.get("step_count", 0)
        distance_km = max(0.3, route.get("distance_km", 1.0))
        junction_density = junctions / distance_km
        junction_penalty = 1.0 + min(0.30, 0.020 * junction_density)

        factor = 1.0 + (base - 1.0) * sensitivity * junction_penalty
        factor = max(1.0, min(2.2, factor))
        dominant = max(classes, key=classes.get) if classes else "mixed roads"
        return TrafficAssessment(
            factor=factor,
            level=_level_for(factor),
            source="modelled",
            detail=(
                f"{when.strftime('%H:%M')} on mostly {dominant} roads, "
                f"{junctions} turns over {distance_km:.1f} km"
            ),
        )


class LiveTrafficProvider:
    """TomTom flow segments sampled along the route geometry."""

    name = "tomtom"

    def __init__(self, api_key: str, samples: int = 4):
        self.api_key = api_key
        self.samples = samples
        self._fallback = ModelledTrafficProvider()

    async def assess_async(
        self, route: dict[str, Any], when: datetime | None = None
    ) -> TrafficAssessment:
        coordinates = (route.get("geometry") or {}).get("coordinates") or []
        if len(coordinates) < 2:
            return self._fallback.assess(route, when)

        step = max(1, len(coordinates) // (self.samples + 1))
        points = coordinates[step::step][: self.samples]
        ratios: list[float] = []
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                for longitude, latitude in points:
                    response = await client.get(
                        TOMTOM_FLOW_URL,
                        params={"key": self.api_key, "point": f"{latitude},{longitude}"},
                    )
                    if response.status_code != 200:
                        continue
                    segment = response.json().get("flowSegmentData") or {}
                    current = float(segment.get("currentSpeed") or 0)
                    free = float(segment.get("freeFlowSpeed") or 0)
                    if current > 0 and free > 0:
                        ratios.append(free / current)
        except httpx.HTTPError:
            return self._fallback.assess(route, when)

        if not ratios:
            return self._fallback.assess(route, when)

        factor = max(1.0, min(3.0, sum(ratios) / len(ratios)))
        return TrafficAssessment(
            factor=factor,
            level=_level_for(factor),
            source="tomtom-live",
            detail=f"live flow sampled at {len(ratios)} points along the route",
        )

    def assess(self, route: dict[str, Any], when: datetime | None = None) -> TrafficAssessment:
        return self._fallback.assess(route, when)


def get_provider() -> ModelledTrafficProvider | LiveTrafficProvider:
    api_key = os.environ.get("TOMTOM_API_KEY", "").strip()
    if api_key:
        return LiveTrafficProvider(api_key)
    return ModelledTrafficProvider()


async def assess_route(route: dict[str, Any]) -> TrafficAssessment:
    provider = get_provider()
    if isinstance(provider, LiveTrafficProvider):
        return await provider.assess_async(route)
    return provider.assess(route)
