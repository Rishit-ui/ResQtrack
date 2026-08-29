"""Scale-invariant kinematics for tracked road users.

Every quantity in ResQTrack's incident policy is expressed as a ratio of the
actor's own bounding-box diagonal.  A car three metres from the camera and the
same car eighty metres away then produce comparable numbers, so one set of
thresholds works on any camera without per-site tuning.

Nothing in this module knows what an accident is.  It only answers "how is this
box moving, and how are these two boxes converging?".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import acos, atan2, degrees, hypot
from statistics import median
from typing import Any, Deque, Iterable

Box = tuple[float, float, float, float]
Point = tuple[float, float]

# A detector box never gets meaningfully smaller than this, and dividing by a
# tiny diagonal turns pixel jitter into fake "high speed".
MIN_DIAGONAL = 24.0


# ============================================================
# GEOMETRY
# ============================================================

def center(box: Box) -> Point:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def width(box: Box) -> float:
    return max(0.0, box[2] - box[0])


def height(box: Box) -> float:
    return max(0.0, box[3] - box[1])


def area(box: Box) -> float:
    return width(box) * height(box)


def diagonal(box: Box) -> float:
    return max(MIN_DIAGONAL, hypot(width(box), height(box)))


def aspect_ratio(box: Box) -> float:
    """Width / height.  A standing person is < 0.6, a fallen person is > 1.1."""
    return width(box) / max(1.0, height(box))


def box_gap(first: Box, second: Box) -> float:
    """Euclidean gap between two boxes.  Zero means touching or overlapping."""
    x_gap = max(first[0] - second[2], second[0] - first[2], 0.0)
    y_gap = max(first[1] - second[3], second[1] - first[3], 0.0)
    return hypot(x_gap, y_gap)


def iou(first: Box, second: Box) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area(first) + area(second) - intersection
    return intersection / union if union > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner`` that lies inside ``outer``.

    A pedestrian struck by a bus is swallowed by the bus box, so IoU stays low
    while containment goes to 1.  Both are needed to see a real contact.
    """
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / area(inner) if area(inner) > 0 else 0.0


def angle_between(first: Point, second: Point) -> float:
    """Angle in degrees between two velocity vectors."""
    first_size, second_size = hypot(*first), hypot(*second)
    if first_size < 1e-6 or second_size < 1e-6:
        return 0.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_size * second_size)
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def compass_heading(velocity: Point) -> str:
    """Screen-space heading label.  Image y grows downward, so north is -y."""
    if hypot(*velocity) < 1e-6:
        return "-"
    bearing = (degrees(atan2(velocity[0], -velocity[1])) + 360.0) % 360.0
    points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return points[int((bearing + 22.5) % 360.0 // 45.0)]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ramp(value: float, low: float, high: float) -> float:
    """Map ``value`` onto 0..1 across the ``low``..``high`` band."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return clamp((value - low) / (high - low))


# ============================================================
# PER-ACTOR STATE
# ============================================================

@dataclass
class Observation:
    frame: int
    timestamp: float
    box: Box
    center: Point
    velocity: Point
    speed: float
    aspect: float


@dataclass
class ActorState:
    """Rolling kinematic state for a single tracked road user."""

    actor_id: int
    kind: str
    class_name: str
    history: Deque[Observation] = field(default_factory=lambda: deque(maxlen=45))
    aspect_baseline: Deque[float] = field(default_factory=lambda: deque(maxlen=25))
    first_frame: int = 0
    last_frame: int = 0
    confidence: float = 0.0

    # Derived per-frame signals, refreshed by :meth:`observe`.
    speed: float = 0.0
    smooth_speed: float = 0.0
    previous_speed: float = 0.0
    peak_speed: float = 0.0
    speed_drop: float = 0.0
    speed_jump: float = 0.0
    turn: float = 0.0
    velocity: Point = (0.0, 0.0)
    aspect: float = 0.0
    aspect_shift: float = 0.0
    area_shift: float = 0.0
    stationary_frames: int = 0
    moving_frames: int = 0

    # ------------------------------------------------------------------
    def observe(self, frame: int, timestamp: float, box: Box, confidence: float) -> None:
        point = center(box)
        previous = self.history[-1] if self.history else None

        velocity: Point = (0.0, 0.0)
        speed = 0.0
        if previous is not None:
            # Normalise by the mean diagonal of the two frames so a box that
            # grows as it nears the camera does not read as acceleration.
            scale = (diagonal(box) + diagonal(previous.box)) / 2.0
            frame_gap = max(1, frame - previous.frame)
            velocity = (
                (point[0] - previous.center[0]) / frame_gap,
                (point[1] - previous.center[1]) / frame_gap,
            )
            speed = hypot(*velocity) / scale

        self.previous_speed = self.speed
        self.speed = speed
        self.velocity = velocity
        self.speed_drop = max(0.0, self.previous_speed - speed)
        self.speed_jump = max(0.0, speed - self.previous_speed)
        self.peak_speed = max(self.peak_speed, speed)
        self.turn = (
            angle_between(previous.velocity, velocity)
            if previous is not None and previous.speed > 0.006 and speed > 0.006
            else 0.0
        )

        # Exponential smoothing keeps single-frame detector jitter out of the
        # "was it actually moving?" question.
        self.smooth_speed = (
            speed if previous is None else 0.65 * self.smooth_speed + 0.35 * speed
        )

        self.aspect = aspect_ratio(box)
        baseline = median(self.aspect_baseline) if len(self.aspect_baseline) >= 5 else 0.0
        self.aspect_shift = (
            abs(self.aspect - baseline) / baseline if baseline > 0.05 else 0.0
        )
        self.area_shift = (
            abs(area(box) - area(previous.box)) / max(1.0, area(previous.box))
            if previous is not None
            else 0.0
        )

        if speed < 0.010:
            self.stationary_frames += 1
            self.moving_frames = 0
        else:
            self.stationary_frames = 0
            self.moving_frames += 1

        self.confidence = confidence
        self.last_frame = frame
        if not self.history:
            self.first_frame = frame

        self.history.append(
            Observation(frame, timestamp, box, point, velocity, speed, self.aspect)
        )
        # The baseline must describe the actor's *normal* pose, so it is only
        # fed while the actor is upright and not mid-disruption.
        if self.aspect_shift < 0.25:
            self.aspect_baseline.append(self.aspect)

    # ------------------------------------------------------------------
    @property
    def box(self) -> Box:
        return self.history[-1].box if self.history else (0.0, 0.0, 0.0, 0.0)

    @property
    def center(self) -> Point:
        return self.history[-1].center if self.history else (0.0, 0.0)

    @property
    def track_length(self) -> int:
        return len(self.history)

    @property
    def aspect_reference(self) -> float:
        """The shape this track normally presents, or 0 if not yet established."""
        if len(self.aspect_baseline) < 5:
            return 0.0
        return float(median(self.aspect_baseline))

    def speed_over(self, frames: int) -> float:
        """Mean speed across the last ``frames`` observations."""
        window = list(self.history)[-frames:]
        if not window:
            return 0.0
        return sum(item.speed for item in window) / len(window)

    def speed_before(self, frames_back: int, span: int = 4) -> float:
        """Mean speed in a window that ends ``frames_back`` observations ago."""
        history = list(self.history)
        end = max(0, len(history) - frames_back)
        window = history[max(0, end - span):end]
        if not window:
            return 0.0
        return sum(item.speed for item in window) / len(window)

    def displacement_since(self, frames_back: int) -> float:
        """Straight-line travel over the last ``frames_back`` frames, in diagonals."""
        history = list(self.history)
        if len(history) < 2:
            return 0.0
        past = history[max(0, len(history) - 1 - frames_back)]
        now = history[-1]
        return hypot(now.center[0] - past.center[0], now.center[1] - past.center[1]) / diagonal(now.box)

    def was_moving(self, lookback: int = 12, threshold: float = 0.020) -> bool:
        """True if the actor had real motion recently, not detector jitter."""
        window = list(self.history)[-lookback:]
        return any(item.speed >= threshold for item in window)

    def sustained_aspect_flip(self, frames: int = 4, shift: float = 0.42) -> bool:
        """A rollover or a fallen pedestrian holds its new shape; jitter does not."""
        if len(self.aspect_baseline) < 5:
            return False
        baseline = median(self.aspect_baseline)
        if baseline <= 0.05:
            return False
        window = list(self.history)[-frames:]
        if len(window) < frames:
            return False
        return all(abs(item.aspect - baseline) / baseline >= shift for item in window)

    def sustained_aspect_beyond(self, bound: float, frames: int, above: bool) -> bool:
        """Absolute pose test: the box shape leaves the range the class lives in.

        A car turning a corner changes its aspect ratio a lot, so a *relative*
        shift alone reports rollovers that never happened.  A car that is
        actually on its side produces a box taller than it is wide, and a person
        lying in the road produces a box wider than tall - shapes that upright
        traffic simply does not make.
        """
        window = list(self.history)[-frames:]
        if len(window) < frames:
            return False
        if above:
            return all(item.aspect >= bound for item in window)
        return all(item.aspect <= bound for item in window)


# ============================================================
# PAIR STATE
# ============================================================

@dataclass
class PairState:
    """Convergence history for one pair of actors."""

    key: tuple[int, int]
    gaps: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    overlaps: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    last_frame: int = 0

    gap: float = 999.0
    overlap: float = 0.0
    containment: float = 0.0
    closing_rate: float = 0.0
    approach_frames: int = 0
    contact_frame: int | None = None
    peak_closing_rate: float = 0.0

    def observe(self, frame: int, first: Box, second: Box) -> None:
        scale = (diagonal(first) + diagonal(second)) / 2.0
        self.gap = box_gap(first, second) / scale
        self.overlap = iou(first, second)
        smaller, larger = (first, second) if area(first) <= area(second) else (second, first)
        self.containment = containment(smaller, larger)

        previous_gap = self.gaps[-1] if self.gaps else None
        self.closing_rate = 0.0 if previous_gap is None else previous_gap - self.gap
        self.peak_closing_rate = max(self.peak_closing_rate, self.closing_rate)

        if self.closing_rate >= 0.012:
            self.approach_frames += 1
        elif self.closing_rate < -0.030:
            # Separating clearly: forget the old approach run.
            self.approach_frames = 0

        self.gaps.append(self.gap)
        self.overlaps.append(self.overlap)
        self.last_frame = frame

    @property
    def in_contact(self) -> bool:
        return self.gap <= 0.045 or self.overlap >= 0.02 or self.containment >= 0.35

    @property
    def near(self) -> bool:
        return self.gap <= 0.16 or self.overlap > 0.0

    def approached(self, minimum_frames: int = 2) -> bool:
        return self.approach_frames >= minimum_frames

    def gap_drop(self, frames: int = 6) -> float:
        """Total gap reduction over the recent window."""
        window = list(self.gaps)[-frames:]
        if len(window) < 2:
            return 0.0
        return max(0.0, max(window) - window[-1])


def pair_key(first_id: int, second_id: int) -> tuple[int, int]:
    return (first_id, second_id) if first_id <= second_id else (second_id, first_id)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def as_float_box(raw: Any) -> Box:
    x1, y1, x2, y2 = (float(value) for value in raw)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
