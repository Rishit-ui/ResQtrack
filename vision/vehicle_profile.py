"""Rich per-vehicle information extracted from the YOLO11 detections.

The detector view is meant to be read by a control-room operator, so every
tracked road user is described the way a human would describe it: what it is,
how fast it is going in km/h, which way it is heading, what colour it is and
whether it is currently moving, braking, stopped or involved in an incident.

Speed calibration
-----------------
A single camera has no depth, so pixels cannot be converted to metres without a
reference.  Instead of inventing a number, this module estimates the scale from
the objects themselves: a passenger car is about 4.4 m long and 1.8 m wide, a
bus about 11 m.  Taking a robust median of ``real_size / observed_box_size``
across every confidently detected vehicle gives a metres-per-pixel estimate for
the part of the road the camera is looking at, which is refined every frame.

The result is displayed as "≈ 47 km/h" and is honest about being an estimate.
Set ``METERS_PER_PIXEL`` explicitly (from a surveyed reference length in the
scene) when a camera is properly calibrated.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from statistics import median
from typing import Any, Deque

import numpy as np

from vision.kinematics import aspect_ratio, compass_heading, height, width

# COCO ids that ResQTrack tracks.
PERSON_CLASS = 0
BICYCLE_CLASS = 1
CAR_CLASS = 2
MOTORCYCLE_CLASS = 3
BUS_CLASS = 5
TRUCK_CLASS = 7

VEHICLE_CLASSES = (BICYCLE_CLASS, CAR_CLASS, MOTORCYCLE_CLASS, BUS_CLASS, TRUCK_CLASS)
TRACKED_CLASSES = (PERSON_CLASS, *VEHICLE_CLASSES)

# Real-world footprint used for the scale estimate, in metres.
REAL_WIDTH_METRES = {
    CAR_CLASS: 1.80,
    BUS_CLASS: 2.55,
    TRUCK_CLASS: 2.45,
    MOTORCYCLE_CLASS: 0.80,
    BICYCLE_CLASS: 0.65,
    PERSON_CLASS: 0.50,
}

# Human-friendly names for the control room.
DISPLAY_NAMES = {
    PERSON_CLASS: "PERSON",
    BICYCLE_CLASS: "BICYCLE",
    CAR_CLASS: "CAR",
    MOTORCYCLE_CLASS: "MOTORCYCLE",
    BUS_CLASS: "BUS",
    TRUCK_CLASS: "TRUCK",
}

# Colour bands in HSV.  Hue is OpenCV's 0-179 range.
COLOUR_BANDS = (
    ("red", 0, 8),
    ("orange", 9, 20),
    ("yellow", 21, 33),
    ("green", 34, 85),
    ("blue", 86, 125),
    ("purple", 126, 155),
    ("red", 156, 179),
)


class ScaleEstimator:
    """Adaptive metres-per-pixel estimate from known vehicle dimensions."""

    def __init__(self, fixed_metres_per_pixel: float | None = None, samples: int = 240):
        self.fixed = fixed_metres_per_pixel
        self.samples: Deque[float] = deque(maxlen=samples)

    def observe(self, class_id: int, box: tuple[float, float, float, float], confidence: float) -> None:
        if self.fixed is not None or confidence < 0.55:
            return
        real_width = REAL_WIDTH_METRES.get(class_id)
        box_width = width(box)
        if real_width is None or box_width < 12.0:
            return
        # Only use boxes seen roughly side-on or head-on; extreme aspect ratios
        # mean the box is clipped by the frame edge and would skew the scale.
        ratio = aspect_ratio(box)
        if not 0.25 <= ratio <= 4.0:
            return
        self.samples.append(real_width / box_width)

    @property
    def metres_per_pixel(self) -> float:
        if self.fixed is not None:
            return self.fixed
        if len(self.samples) < 12:
            return 0.0
        return float(median(self.samples))

    @property
    def calibrated(self) -> bool:
        return self.metres_per_pixel > 0.0

    def to_kmh(self, pixels_per_second: float) -> float:
        scale = self.metres_per_pixel
        if scale <= 0.0:
            return 0.0
        return pixels_per_second * scale * 3.6


@dataclass
class VehicleProfile:
    """Everything the overlay and the alert payload need about one road user."""

    actor_id: int
    class_id: int
    class_name: str
    kind: str
    display_name: str
    box: tuple[float, float, float, float]
    confidence: float
    speed_pixels: float = 0.0
    speed_kmh: float = 0.0
    heading: str = "-"
    colour: str = "unknown"
    colour_bgr: tuple[int, int, int] = (200, 200, 200)
    state: str = "TRACKING"
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    involved: bool = False

    speed_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    colour_samples: Deque[str] = field(default_factory=lambda: deque(maxlen=12))

    @property
    def seconds_tracked_at(self) -> int:
        return self.last_seen_frame - self.first_seen_frame

    @property
    def size_pixels(self) -> tuple[int, int]:
        return int(width(self.box)), int(height(self.box))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.actor_id,
            "type": self.display_name,
            "class_name": self.class_name,
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "speed_kmh": round(self.speed_kmh, 1),
            "speed_pixels_per_second": round(self.speed_pixels, 1),
            "heading": self.heading,
            "colour": self.colour,
            "state": self.state,
            "box": [round(value, 1) for value in self.box],
            "involved": self.involved,
        }


def dominant_colour(frame: np.ndarray, box: tuple[float, float, float, float]) -> tuple[str, tuple[int, int, int]]:
    """Name the dominant colour of the middle of a detection box."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box
    # Sample the central half of the box: edges pick up road and background.
    pad_x, pad_y = width(box) * 0.25, height(box) * 0.25
    x1 = int(max(0, min(frame_width - 1, x1 + pad_x)))
    x2 = int(max(0, min(frame_width, x2 - pad_x)))
    y1 = int(max(0, min(frame_height - 1, y1 + pad_y)))
    y2 = int(max(0, min(frame_height, y2 - pad_y)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return "unknown", (200, 200, 200)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return "unknown", (200, 200, 200)

    import cv2  # local import keeps this module importable without a display

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[:, :, 0]))
    saturation = float(np.median(hsv[:, :, 1]))
    value = float(np.median(hsv[:, :, 2]))
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    swatch = (int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2]))

    if value < 55:
        return "black", swatch
    if saturation < 42:
        if value > 185:
            return "white", swatch
        return "silver" if value > 110 else "grey", swatch
    for name, low, high in COLOUR_BANDS:
        if low <= hue <= high:
            return name, swatch
    return "unknown", swatch


class VehicleRegistry:
    """Builds and keeps :class:`VehicleProfile` records across frames."""

    def __init__(self, scale: ScaleEstimator, fps: float, stale_frames: int = 45):
        self.scale = scale
        self.fps = fps if fps > 0 else 25.0
        self.stale_frames = stale_frames
        self.profiles: dict[int, VehicleProfile] = {}
        self._previous_centers: dict[int, tuple[float, float]] = {}

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        frame_number: int,
        involved_ids: set[int] | None = None,
        sample_colour: bool = True,
    ) -> dict[int, VehicleProfile]:
        involved_ids = involved_ids or set()
        seen: set[int] = set()

        for detection in detections:
            actor_id = detection["id"]
            box = detection["box"]
            class_id = detection["class_id"]
            seen.add(actor_id)

            self.scale.observe(class_id, box, detection["confidence"])

            profile = self.profiles.get(actor_id)
            if profile is None:
                profile = VehicleProfile(
                    actor_id=actor_id,
                    class_id=class_id,
                    class_name=detection["class_name"],
                    kind=detection["kind"],
                    display_name=DISPLAY_NAMES.get(class_id, detection["class_name"].upper()),
                    box=box,
                    confidence=detection["confidence"],
                    first_seen_frame=frame_number,
                )
                self.profiles[actor_id] = profile

            centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            previous = self._previous_centers.get(actor_id)
            velocity = (0.0, 0.0)
            pixels_per_second = 0.0
            if previous is not None:
                velocity = (centre[0] - previous[0], centre[1] - previous[1])
                pixels_per_second = hypot(*velocity) * self.fps
            self._previous_centers[actor_id] = centre

            profile.speed_samples.append(pixels_per_second)
            smoothed = sum(profile.speed_samples) / len(profile.speed_samples)

            # Re-classification happens: a distant car can flip to truck for a
            # frame.  Keep the highest-confidence label the track has produced.
            if detection["confidence"] >= profile.confidence or profile.class_id == class_id:
                profile.class_id = class_id
                profile.class_name = detection["class_name"]
                profile.display_name = DISPLAY_NAMES.get(class_id, detection["class_name"].upper())
            profile.confidence = detection["confidence"]
            profile.kind = detection["kind"]
            profile.box = box
            profile.speed_pixels = smoothed
            profile.speed_kmh = self.scale.to_kmh(smoothed)
            profile.heading = compass_heading(velocity)
            profile.last_seen_frame = frame_number
            profile.involved = actor_id in involved_ids

            # Colour sampling is the most expensive per-actor step, so it runs
            # on a subset of frames and is smoothed by majority vote.
            if sample_colour:
                name, swatch = dominant_colour(frame, box)
                profile.colour_samples.append(name)
                profile.colour_bgr = swatch
            if profile.colour_samples:
                profile.colour = max(
                    set(profile.colour_samples), key=profile.colour_samples.count
                )

            profile.state = self._state(profile)

        for actor_id in list(self.profiles):
            if frame_number - self.profiles[actor_id].last_seen_frame > self.stale_frames:
                del self.profiles[actor_id]
                self._previous_centers.pop(actor_id, None)

        return {actor_id: self.profiles[actor_id] for actor_id in seen if actor_id in self.profiles}

    def _state(self, profile: VehicleProfile) -> str:
        if profile.involved:
            return "INCIDENT"
        samples = list(profile.speed_samples)
        if len(samples) >= 4:
            recent = sum(samples[-2:]) / 2.0
            earlier = sum(samples[-5:-2]) / max(1, len(samples[-5:-2]))
            if earlier > 25.0 and recent < earlier * 0.45:
                return "BRAKING"
        if profile.speed_pixels < 4.0:
            return "STOPPED"
        if profile.speed_pixels < 18.0:
            return "SLOW"
        return "MOVING"

    def snapshot(self) -> list[dict[str, Any]]:
        return [profile.as_dict() for profile in self.profiles.values()]
