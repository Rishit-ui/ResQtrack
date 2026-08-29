"""Physics-aware incident evidence for a single traffic camera.

The temporal ML model identifies unusual *traffic scenes*.  It must not, by
itself, label a scene as an accident: stationary vehicles close to each other
are a normal road-side pattern.  This module only promotes an incident when a
moving actor has a time-ordered interaction and an impact/disruption signal.

Coordinates are deliberately normalised by the detected actors' box sizes.
That keeps the policy usable across cameras with different resolutions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import acos, hypot
from typing import Any


VEHICLE = "vehicle"
PERSON = "person"


@dataclass(frozen=True)
class IncidentEvidence:
    """The strongest observation in the latest frame."""

    status: str = "NORMAL"
    kind: str = ""
    confidence: float = 0.0
    actor_ids: tuple[int, ...] = ()
    reason: str = "No dynamic incident evidence"

    @property
    def confirmed(self) -> bool:
        return self.status == "CONFIRMED"


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diagonal(box: tuple[float, float, float, float]) -> float:
    return max(30.0, hypot(box[2] - box[0], box[3] - box[1]))


def _box_gap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Euclidean gap between two boxes (zero means touching/overlapping)."""
    x_gap = max(first[0] - second[2], second[0] - first[2], 0.0)
    y_gap = max(first[1] - second[3], second[1] - first[3], 0.0)
    return hypot(x_gap, y_gap)


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_size, second_size = hypot(*first), hypot(*second)
    if first_size < 1e-6 or second_size < 1e-6:
        return 0.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_size * second_size)
    return acos(max(-1.0, min(1.0, cosine))) * 180.0 / 3.141592653589793


class IncidentEventEngine:
    """Keeps short actor trajectories and emits explainable incident evidence.

    It intentionally has no "proximity only" path to CONFIRMED.  A parked
    vehicle, queue of traffic, or detector-box overlap stays NORMAL.
    """

    def __init__(self, history_frames: int = 20, stale_after_frames: int = 30):
        self.history_frames = history_frames
        self.stale_after_frames = stale_after_frames
        self.frame_number = 0
        self.tracks: dict[int, deque[dict[str, Any]]] = {}
        self.last_seen: dict[int, int] = {}
        self.pair_history: dict[tuple[int, int], deque[float]] = {}
        self.contact_candidates: dict[tuple[int, int], dict[str, Any]] = {}

    def reset(self) -> None:
        self.frame_number = 0
        self.tracks.clear()
        self.last_seen.clear()
        self.pair_history.clear()
        self.contact_candidates.clear()

    def _motion(self, actor_id: int, actor: dict[str, Any]) -> dict[str, Any]:
        """Return current normalised motion before recording the actor."""
        box = tuple(map(float, actor["box"]))
        point = _center(box)
        history = self.tracks.get(actor_id)
        result = {
            "speed": 0.0,
            "previous_speed": 0.0,
            "speed_drop": 0.0,
            "speed_jump": 0.0,
            "turn": 0.0,
            "velocity": (0.0, 0.0),
        }
        if not history:
            return result

        previous = history[-1]
        scale = max(30.0, (_diagonal(box) + _diagonal(previous["box"])) / 2.0)
        velocity = (point[0] - previous["center"][0], point[1] - previous["center"][1])
        speed = hypot(*velocity) / scale
        result.update(speed=speed, velocity=velocity)

        if len(history) >= 2:
            before_previous = history[-2]
            previous_velocity = (
                previous["center"][0] - before_previous["center"][0],
                previous["center"][1] - before_previous["center"][1],
            )
            previous_scale = max(
                30.0,
                (_diagonal(previous["box"]) + _diagonal(before_previous["box"])) / 2.0,
            )
            previous_speed = hypot(*previous_velocity) / previous_scale
            result.update(
                previous_speed=previous_speed,
                speed_drop=max(0.0, previous_speed - speed),
                speed_jump=max(0.0, speed - previous_speed),
                turn=_angle(previous_velocity, velocity),
            )
        return result

    def _record(self, actor_id: int, actor: dict[str, Any], motion: dict[str, Any]) -> None:
        history = self.tracks.setdefault(actor_id, deque(maxlen=self.history_frames))
        box = tuple(map(float, actor["box"]))
        history.append(
            {
                "box": box,
                "center": _center(box),
                "kind": actor["kind"],
                "motion": motion,
            }
        )
        self.last_seen[actor_id] = self.frame_number

    @staticmethod
    def _motion_disrupted(motion: dict[str, Any]) -> bool:
        # Detection jitter should not look like an impact.  Both values are
        # ratios of movement to box diagonal, so they do not depend on pixels.
        return motion["speed_drop"] >= 0.075 or motion["speed_jump"] >= 0.20 or (
            motion["previous_speed"] >= 0.045 and motion["turn"] >= 65.0
        )

    def _candidate(
        self,
        status: str,
        kind: str,
        confidence: float,
        actor_ids: tuple[int, ...],
        reason: str,
    ) -> IncidentEvidence:
        return IncidentEvidence(status, kind, min(confidence, 0.99), actor_ids, reason)

    def _evaluate_pair(
        self,
        first_id: int,
        first: dict[str, Any],
        second_id: int,
        second: dict[str, Any],
        motions: dict[int, dict[str, Any]],
    ) -> IncidentEvidence | None:
        key = tuple(sorted((first_id, second_id)))
        first_box, second_box = first["box"], second["box"]
        scale = max(30.0, (_diagonal(first_box) + _diagonal(second_box)) / 2.0)
        gap_ratio = _box_gap(first_box, second_box) / scale
        overlap = _iou(first_box, second_box)
        history = self.pair_history.setdefault(key, deque(maxlen=5))
        previous_gap = history[-1] if history else None
        history.append(gap_ratio)

        near = gap_ratio <= 0.14 or overlap >= 0.025
        approaching = previous_gap is not None and previous_gap - gap_ratio >= 0.055
        first_motion, second_motion = motions[first_id], motions[second_id]
        first_is_vehicle = first["kind"] == VEHICLE
        second_is_vehicle = second["kind"] == VEHICLE

        if first_is_vehicle and second_is_vehicle:
            moving = max(first_motion["speed"], first_motion["previous_speed"], second_motion["speed"], second_motion["previous_speed"]) >= 0.045
            disrupted = self._motion_disrupted(first_motion) or self._motion_disrupted(second_motion)
            if moving and near and approaching and disrupted:
                return self._candidate(
                    "CONFIRMED",
                    "vehicle_vehicle_collision",
                    0.90 if overlap >= 0.025 else 0.84,
                    key,
                    "approach, contact proximity, and abrupt vehicle motion",
                )
            if moving and near and approaching:
                return self._candidate(
                    "REVIEW",
                    "vehicle_vehicle_risk",
                    0.48,
                    key,
                    "moving vehicles converged; no impact signature yet",
                )
            return None

        # Reorder to simplify vehicle-to-person policy.
        if first_is_vehicle and second["kind"] == PERSON:
            vehicle_id, vehicle, vehicle_motion = first_id, first, first_motion
            person_id, person, person_motion = second_id, second, second_motion
        elif second_is_vehicle and first["kind"] == PERSON:
            vehicle_id, vehicle, vehicle_motion = second_id, second, second_motion
            person_id, person, person_motion = first_id, first, first_motion
        else:
            return None

        vehicle_moving = max(vehicle_motion["speed"], vehicle_motion["previous_speed"]) >= 0.045
        person_disrupted = self._motion_disrupted(person_motion)
        contact_key = (vehicle_id, person_id)
        if vehicle_moving and near and approaching:
            self.contact_candidates[contact_key] = {
                "frame": self.frame_number,
                "person_center": _center(person["box"]),
                "gap_ratio": gap_ratio,
            }
            if person_disrupted:
                return self._candidate(
                    "CONFIRMED",
                    "vehicle_pedestrian_collision",
                    0.90 if overlap >= 0.025 else 0.84,
                    (vehicle_id, person_id),
                    "moving vehicle contacted a pedestrian with a trajectory disruption",
                )
            return self._candidate(
                "REVIEW",
                "vehicle_pedestrian_risk",
                0.58,
                (vehicle_id, person_id),
                "moving vehicle entered a pedestrian contact zone; awaiting impact evidence",
            )
        return None

    def _hit_and_run_evidence(
        self, actors: dict[int, dict[str, Any]], motions: dict[int, dict[str, Any]]
    ) -> IncidentEvidence | None:
        strongest: IncidentEvidence | None = None
        for (vehicle_id, person_id), contact in list(self.contact_candidates.items()):
            age = self.frame_number - contact["frame"]
            if age > 12:
                del self.contact_candidates[(vehicle_id, person_id)]
                continue
            missed = self.frame_number - self.last_seen.get(person_id, self.frame_number)
            vehicle = actors.get(vehicle_id)
            if not vehicle or not 2 <= missed <= 8:
                continue
            motion = motions[vehicle_id]
            vehicle_scale = _diagonal(vehicle["box"])
            departed = hypot(
                _center(vehicle["box"])[0] - contact["person_center"][0],
                _center(vehicle["box"])[1] - contact["person_center"][1],
            ) / vehicle_scale >= 0.35
            if max(motion["speed"], motion["previous_speed"]) >= 0.045 and departed:
                candidate = self._candidate(
                    "CONFIRMED",
                    "possible_hit_and_run",
                    0.82,
                    (vehicle_id, person_id),
                    "pedestrian disappeared after contact while the vehicle departed",
                )
                if strongest is None or candidate.confidence > strongest.confidence:
                    strongest = candidate
        return strongest

    def update(self, actors: dict[int, dict[str, Any]]) -> IncidentEvidence:
        """Process one tracked frame.

        ``actors`` maps ByteTrack IDs to ``kind`` (vehicle/person) and ``box``.
        Unsupported classes should be excluded by the caller.
        """
        self.frame_number += 1
        motions = {actor_id: self._motion(actor_id, actor) for actor_id, actor in actors.items()}
        candidates: list[IncidentEvidence] = []

        actor_ids = list(actors)
        for index, first_id in enumerate(actor_ids):
            for second_id in actor_ids[index + 1:]:
                candidate = self._evaluate_pair(
                    first_id, actors[first_id], second_id, actors[second_id], motions
                )
                if candidate:
                    candidates.append(candidate)

        for actor_id, actor in actors.items():
            self._record(actor_id, actor, motions[actor_id])

        hit_and_run = self._hit_and_run_evidence(actors, motions)
        if hit_and_run:
            candidates.append(hit_and_run)

        # Expire dead tracking state.  Its final observation remains available
        # long enough for the post-contact hit-and-run check above.
        stale_ids = [
            actor_id
            for actor_id, seen in self.last_seen.items()
            if self.frame_number - seen > self.stale_after_frames
        ]
        for actor_id in stale_ids:
            self.tracks.pop(actor_id, None)
            self.last_seen.pop(actor_id, None)

        if not candidates:
            return IncidentEvidence()
        return max(candidates, key=lambda item: item.confidence)
