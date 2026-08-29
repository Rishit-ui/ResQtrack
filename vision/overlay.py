"""Control-room HUD drawn on top of the YOLO11 detections.

Three things are rendered:

* a **detail card** anchored to every tracked road user - type, track id,
  detection confidence, estimated km/h, heading, colour and motion state;
* a **roster panel** listing everything currently tracked, so an operator can
  read the scene without squinting at the boxes;
* an **incident banner** with the evidence that produced the verdict.

Cards are placed by a simple occupancy solver so two vehicles standing next to
each other never stack unreadable text on top of one another.
"""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from vision.incident_engine import CONFIRMED, REVIEW, IncidentEvidence
from vision.vehicle_profile import VehicleProfile

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Palette (BGR).
COLOUR_VEHICLE = (120, 220, 120)
COLOUR_PERSON = (0, 195, 255)
COLOUR_INCIDENT = (60, 60, 255)
COLOUR_REVIEW = (0, 170, 255)
COLOUR_NORMAL = (110, 235, 120)
COLOUR_PANEL = (22, 22, 26)
COLOUR_TEXT = (235, 238, 242)
COLOUR_MUTED = (150, 156, 166)

SEVERITY_COLOURS = {
    "CRITICAL": (60, 60, 255),
    "HIGH": (0, 110, 255),
    "MODERATE": (0, 190, 255),
    "LOW": (120, 200, 120),
}


def ui_scale(frame: np.ndarray) -> float:
    """Layout scale factor.

    The same overlay has to stay readable on a 640x360 CCTV clip and on a 4K
    camera, so every panel size and font is derived from the frame width rather
    than hard-coded in pixels.
    """
    return max(0.52, min(1.6, frame.shape[1] / 1280.0))


def _overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _panel(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    colour: tuple[int, int, int],
    alpha: float = 0.72,
    border: int = 1,
) -> None:
    """Translucent rounded-ish panel so the video stays visible underneath."""
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    backdrop = np.full(region.shape, COLOUR_PANEL, dtype=np.uint8)
    cv2.addWeighted(backdrop, alpha, region, 1.0 - alpha, 0, region)
    if border:
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), colour, border, cv2.LINE_AA)


def _text(
    frame: np.ndarray,
    label: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(frame, label, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_detection_boxes(
    frame: np.ndarray,
    profiles: Iterable[VehicleProfile],
) -> None:
    """Boxes only.  Labels are handled by the card layout below."""
    for profile in profiles:
        x1, y1, x2, y2 = (int(value) for value in profile.box)
        if profile.involved:
            colour, thickness = COLOUR_INCIDENT, 3
        elif profile.kind == "person":
            colour, thickness = COLOUR_PERSON, 2
        else:
            colour, thickness = COLOUR_VEHICLE, 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
        # Corner ticks make small boxes readable on a projector.
        tick = max(6, min(18, (x2 - x1) // 5))
        for cx, cy, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(frame, (cx, cy), (cx + dx * tick, cy), colour, thickness + 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * tick), colour, thickness + 1, cv2.LINE_AA)


def draw_vehicle_cards(
    frame: np.ndarray,
    profiles: Iterable[VehicleProfile],
    calibrated: bool,
    reserved: list[tuple[int, int, int, int]] | None = None,
    max_cards: int = 14,
) -> None:
    """Draw the per-vehicle detail card next to each detection."""
    frame_height, frame_width = frame.shape[:2]
    layout = ui_scale(frame)
    scale = 0.38 * layout
    line_height = int(15 * layout)
    occupied: list[tuple[int, int, int, int]] = list(reserved or [])
    # The bottom strip carries the key hints; cards must not sit on top of it.
    bottom_limit = frame_height - int(26 * layout)

    # Nearest / largest actors first: they matter most and get the best slots.
    ordered = sorted(
        profiles,
        key=lambda item: (not item.involved, -(item.box[3] - item.box[1])),
    )[:max_cards]

    for profile in ordered:
        x1, y1, x2, y2 = (int(value) for value in profile.box)
        speed_text = (
            f"{profile.speed_kmh:5.1f} km/h" if calibrated else f"{profile.speed_pixels:5.0f} px/s"
        )
        lines = [
            f"{profile.display_name} #{profile.actor_id}",
            f"{speed_text}  {profile.heading}",
            f"{profile.colour}  {profile.state}",
            f"conf {profile.confidence:.2f}",
        ]
        widths = [cv2.getTextSize(line, FONT, scale, 1)[0][0] for line in lines]
        card_width = max(widths) + int(30 * layout)
        card_height = line_height * len(lines) + int(11 * layout)
        if card_width > frame_width - 8:
            continue

        candidates = (
            (x1, y1 - card_height - 6),
            (x2 + 6, y1),
            (x1, y2 + 6),
            (x1 - card_width - 6, y1),
            (x1, y1 + 4),
        )
        placement = None
        for raw_x, raw_y in candidates:
            card_x = max(2, min(raw_x, frame_width - card_width - 2))
            card_y = max(2, min(raw_y, bottom_limit - card_height))
            rect = (card_x, card_y, card_x + card_width, card_y + card_height)
            if not any(_overlap(rect, item) for item in occupied):
                placement = rect
                break
        if placement is None:
            # Everything is crowded: slide down the frame until a gap appears.
            card_x = max(2, min(x1, frame_width - card_width - 2))
            card_y = 2
            while card_y + card_height < bottom_limit and any(
                _overlap((card_x, card_y, card_x + card_width, card_y + card_height), item)
                for item in occupied
            ):
                card_y += card_height // 2 + 4
            if card_y + card_height >= bottom_limit:
                continue   # genuinely no room left; better blank than illegible
            placement = (card_x, card_y, card_x + card_width, card_y + card_height)

        occupied.append(placement)
        accent = (
            COLOUR_INCIDENT
            if profile.involved
            else COLOUR_PERSON if profile.kind == "person" else COLOUR_VEHICLE
        )
        _panel(frame, placement, accent, alpha=0.78)

        card_x, card_y = placement[0], placement[1]
        # Colour swatch strip on the left edge of the card.
        cv2.rectangle(
            frame,
            (card_x + 4, card_y + 5),
            (card_x + int(9 * layout), placement[3] - 5),
            tuple(int(value) for value in profile.colour_bgr),
            -1,
        )
        cv2.rectangle(
            frame, (card_x + 4, card_y + 5),
            (card_x + int(9 * layout), placement[3] - 5), accent, 1
        )

        for index, line in enumerate(lines):
            colour = accent if index == 0 else COLOUR_TEXT if index < 3 else COLOUR_MUTED
            _text(
                frame,
                line,
                (card_x + int(15 * layout), card_y + line_height * (index + 1)),
                scale,
                colour,
                1,
            )

        # Leader line from card to box so the pairing is unambiguous.
        cv2.line(
            frame,
            (card_x + card_width // 2, card_y + card_height // 2),
            ((x1 + x2) // 2, (y1 + y2) // 2),
            accent,
            1,
            cv2.LINE_AA,
        )


def draw_roster(
    frame: np.ndarray,
    profiles: Iterable[VehicleProfile],
    calibrated: bool,
    metres_per_pixel: float,
    reserved: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int] | None:
    """Right-hand panel listing every tracked road user.

    Returns ``None`` when the frame is too narrow to carry it without covering
    the incident panel - the per-vehicle cards then do the work alone.
    """
    frame_height, frame_width = frame.shape[:2]
    scale = ui_scale(frame)
    panel_width = int(280 * scale)
    left_edge = max((item[2] for item in (reserved or [])), default=0)
    if frame_width - panel_width - 14 < left_edge + 12:
        return None

    profiles = sorted(profiles, key=lambda item: (item.kind != "vehicle", -item.speed_pixels))
    font_scale = 0.40 * scale
    row = int(19 * scale)
    capacity = max(3, (frame_height - int(120 * scale)) // row)
    shown = profiles[:min(12, capacity)]
    panel_height = int(62 * scale) + row * (len(shown) + 1)
    x1 = frame_width - panel_width - 14
    y1 = 14
    rect = (x1, y1, x1 + panel_width, y1 + panel_height)
    _panel(frame, rect, (70, 78, 92), alpha=0.80)

    _text(frame, "TRACKED ROAD USERS", (x1 + 11, y1 + int(22 * scale)), 0.44 * scale, COLOUR_TEXT, 1)
    calibration = (
        f"scale {metres_per_pixel:.3f} m/px" if calibrated else "scale: calibrating..."
    )
    _text(frame, calibration, (x1 + 11, y1 + int(39 * scale)), 0.36 * scale, COLOUR_MUTED, 1)
    header_y = y1 + int(58 * scale)
    cv2.line(frame, (x1 + 9, header_y - int(11 * scale)),
             (x1 + panel_width - 9, header_y - int(11 * scale)), (70, 78, 92), 1)
    unit = "km/h" if calibrated else "px/s"
    _text(frame, f"ID  TYPE       {unit:<6} STATE", (x1 + 11, header_y), 0.35 * scale, COLOUR_MUTED, 1)

    for index, profile in enumerate(shown):
        y = header_y + row * (index + 1)
        speed = f"{profile.speed_kmh:5.1f}" if calibrated else f"{profile.speed_pixels:5.0f}"
        line = f"{profile.actor_id:<3} {profile.display_name[:9]:<10} {speed:<6} {profile.state}"
        colour = (
            COLOUR_INCIDENT
            if profile.involved
            else COLOUR_PERSON if profile.kind == "person" else COLOUR_TEXT
        )
        _text(frame, line, (x1 + 11, y), 0.36 * scale, colour, 1)

    if len(profiles) > len(shown):
        _text(
            frame,
            f"+{len(profiles) - len(shown)} more",
            (x1 + 11, header_y + row * (len(shown) + 1)),
            0.35 * scale,
            COLOUR_MUTED,
            1,
        )
    return rect


def draw_status(
    frame: np.ndarray,
    evidence: IncidentEvidence,
    alert_active: bool,
    scene: dict[str, Any],
    camera_id: str,
    dispatch_state: str,
    fps_actual: float,
) -> tuple[int, int, int, int]:
    """Top-left incident panel: verdict, evidence chips and the ML context."""
    frame_width = frame.shape[1]
    scale = ui_scale(frame)
    panel_width = int(min(frame_width * 0.56, 540 * scale))
    line = int(20 * scale)

    if alert_active or evidence.status == CONFIRMED:
        accent = SEVERITY_COLOURS.get(evidence.severity, COLOUR_INCIDENT)
        headline = f"ACCIDENT CONFIRMED - {evidence.severity}"
    elif evidence.status == REVIEW:
        accent = COLOUR_REVIEW
        headline = "REVIEW - DYNAMIC INTERACTION"
    else:
        accent = COLOUR_NORMAL
        headline = "NORMAL - NO INCIDENT EVIDENCE"

    rows = 4 if evidence.status == "NORMAL" else 8
    panel_height = int(18 * scale) + line * rows
    rect = (12, 12, 12 + panel_width, 12 + panel_height)
    _panel(frame, rect, accent, alpha=0.80, border=2)

    x = 12 + int(14 * scale)
    y = 12 + line
    _text(frame, "ResQTrack  |  live incident policy", (x, y), 0.42 * scale, COLOUR_MUTED, 1)
    y += line + int(4 * scale)
    _text(frame, headline, (x, y), 0.62 * scale, accent, 2)

    if evidence.status != "NORMAL":
        y += line
        _text(frame, evidence.label[:46], (x, y), 0.46 * scale, COLOUR_TEXT, 1)
        y += line
        actors = ", ".join(f"#{item}" for item in evidence.actor_ids[:4])
        _text(
            frame,
            f"confidence {evidence.confidence:.2f}   actors {actors}",
            (x, y),
            0.40 * scale,
            COLOUR_MUTED,
            1,
        )
        # Evidence chips: the audit trail behind the verdict.
        y += int(10 * scale)
        chip_x = x
        chip_height = int(17 * scale)
        for signal in evidence.signals[:6]:
            label = signal.name.replace("_", " ")
            text_width = cv2.getTextSize(label, FONT, 0.34 * scale, 1)[0][0] + int(12 * scale)
            if chip_x + text_width > rect[2] - int(12 * scale):
                break
            cv2.rectangle(frame, (chip_x, y), (chip_x + text_width, y + chip_height),
                          accent, 1, cv2.LINE_AA)
            _text(frame, label, (chip_x + int(6 * scale), y + int(12 * scale)),
                  0.34 * scale, COLOUR_TEXT, 1)
            chip_x += text_width + int(5 * scale)
        y += chip_height + line - int(6 * scale)
        _text(frame, evidence.reason[:74], (x, y), 0.36 * scale, COLOUR_MUTED, 1)
    else:
        y += line
        _text(frame, "Parked, queued and overlapping vehicles are ignored.",
              (x, y), 0.38 * scale, COLOUR_MUTED, 1)

    y += line
    _text(
        frame,
        f"ML context {evidence.ml_probability:.3f} (corroborates only)   "
        f"veh {scene.get('tracked_vehicles', 0)}  ped {scene.get('tracked_people', 0)}",
        (x, y),
        0.36 * scale,
        COLOUR_MUTED,
        1,
    )
    y += line - int(4 * scale)
    _text(
        frame,
        f"{camera_id}   frame {scene.get('frame', 0)}   {fps_actual:.1f} fps   "
        f"dispatch: {dispatch_state}",
        (x, y),
        0.36 * scale,
        COLOUR_MUTED,
        1,
    )
    return (rect[0], rect[1], rect[2], max(rect[3], y + int(8 * scale)))


def draw_alert_border(frame: np.ndarray, evidence: IncidentEvidence, pulse: float) -> None:
    """Flashing frame border while an alert is live."""
    thickness = int(6 + 6 * pulse)
    colour = SEVERITY_COLOURS.get(evidence.severity, COLOUR_INCIDENT)
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1] - 1, frame.shape[0] - 1),
        colour,
        thickness,
        cv2.LINE_AA,
    )


def draw_help(frame: np.ndarray) -> None:
    scale = ui_scale(frame)
    _text(
        frame,
        "F full screen   I cards   R roster   T test alert   Q quit",
        (14, frame.shape[0] - int(9 * scale)),
        0.36 * scale,
        COLOUR_MUTED,
        1,
    )
