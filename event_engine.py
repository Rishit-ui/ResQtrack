"""Compatibility shim for the original single-file event engine.

The incident policy now lives in :mod:`vision.incident_engine`, which covers
many more crash types and accumulates evidence across frames instead of judging
one frame at a time.  This module keeps the old import path working for the
existing validation and benchmark scripts:

    from event_engine import IncidentEventEngine, PERSON, VEHICLE
"""

from vision.incident_engine import (  # noqa: F401
    CONFIRMED,
    NORMAL,
    REVIEW,
    EnginePolicy,
    IncidentEngine,
    IncidentEventEngine,
    IncidentEvidence,
    IncidentSignal,
    KIND_LABELS,
    PERSON,
    SEVERITY_ORDER,
    VEHICLE,
)

__all__ = [
    "CONFIRMED",
    "NORMAL",
    "REVIEW",
    "EnginePolicy",
    "IncidentEngine",
    "IncidentEventEngine",
    "IncidentEvidence",
    "IncidentSignal",
    "KIND_LABELS",
    "PERSON",
    "SEVERITY_ORDER",
    "VEHICLE",
]
