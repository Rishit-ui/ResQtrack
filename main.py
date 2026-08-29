import cv2
import time
import joblib
import numpy as np
import pandas as pd

from collections import deque, defaultdict
from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
    reset_vehicle_history,
)

from event_engine import (
    IncidentEvidence,
    IncidentEventEngine,
    PERSON,
    VEHICLE,
)

from temporal_features_engine import (
    extract_frame_features,
    aggregate_window,
    FEATURES,
    WINDOW_FRAMES,
    STRIDE_FRAMES,
)


# ============================================================
# RESQTRACK CONFIGURATION
# ============================================================

CAMERA_ID = "CAM-001"
CAMERA_LATITUDE = 12.9720
CAMERA_LONGITUDE = 77.5949

VIDEO_PATH = "data/accident.mp4"
YOLO_MODEL = "yolo11n.pt"
FINAL_MODEL = "resqtrack_final_model.pkl"

# COCO classes used by the current YOLO model:
# 0 = person, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
# People are intentionally included in tracking.  They are kept separate from
# the legacy vehicle-only ML features and handled by the event policy below.
PERSON_CLASS = 0
VEHICLE_CLASSES = [2, 3, 5, 7]
TRACKED_CLASSES = [PERSON_CLASS, *VEHICLE_CLASSES]
YOLO_CONFIDENCE = 0.40

# The trained temporal model's probability is context only.  It was trained on
# traffic windows, not a complete taxonomy of crashes, so it must never be the
# sole reason an emergency is confirmed.
ML_THRESHOLD = 0.50
ALERT_DURATION = 5.0

WINDOW_NAME = "ResQTrack - Accident Detection"

# Vehicle overlay settings. Speed is IMAGE-SPACE until camera
# calibration is introduced. Do not present it as km/h.
SHOW_VEHICLE_INFO = True
SPEED_HISTORY_LENGTH = 5
STALE_VEHICLE_FRAMES = 60

# Display controls: open full screen by default.  F toggles full screen, I
# toggles actor labels, and Q/Esc quits.
START_FULLSCREEN = True
WINDOWED_SIZE = (1280, 720)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_class_name(model, class_id: int) -> str:
    """Return a readable YOLO class name across Ultralytics versions."""
    names = getattr(model, "names", {})

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def pixel_speed(
    previous_center: tuple[float, float] | None,
    current_center: tuple[float, float],
    fps: float,
) -> float:
    """Calculate image-space speed in pixels/second."""
    if previous_center is None or fps <= 0:
        return 0.0

    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    distance_pixels = float(np.hypot(dx, dy))

    return distance_pixels * fps


def smoothed_speed(history: deque[float]) -> float:
    """Return a robust recent mean for display."""
    if not history:
        return 0.0
    return float(np.mean(history))


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def draw_actor_information(
    frame: np.ndarray,
    actors: dict,
    speed_histories: dict[int, deque[float]],
    show_labels: bool,
) -> None:
    """Draw compact, non-overlapping labels for tracked road users."""
    if not show_labels:
        return

    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.42, min(0.58, min(width, height) / 1450.0))
    # Keep actor labels out of the incident/status panel in the top left.
    occupied: list[tuple[int, int, int, int]] = [(0, 0, min(width, 690), 190)]

    # Sorting makes label allocation stable from frame to frame.
    for actor_id, actor in sorted(
        actors.items(), key=lambda item: (item[1]["box"][1], item[0])
    ):
        x1, y1, x2, y2 = map(int, actor["box"])
        kind = actor["kind"].upper()
        speed = smoothed_speed(speed_histories.get(actor_id, deque()))
        label = f"{kind} #{actor_id}  {speed:.1f}px/s"
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
        label_width, label_height = text_width + 10, text_height + baseline + 10

        # Try positions around the box before using the next free row.  This
        # avoids stacking vehicle details when two boxes are close together.
        candidates = [
            (x1, y1 - label_height - 4),
            (x1, y2 + 4),
            (x2 + 4, y1),
            (x1 - label_width - 4, y1),
        ]
        position: tuple[int, int] | None = None
        for raw_x, raw_y in candidates:
            label_x = max(0, min(raw_x, width - label_width))
            label_y = max(0, min(raw_y, height - label_height))
            rectangle = (label_x, label_y, label_x + label_width, label_y + label_height)
            if not any(_rectangles_overlap(rectangle, item) for item in occupied):
                position = (label_x, label_y)
                break

        if position is None:
            label_x = max(0, min(x1, width - label_width))
            label_y = 4
            while any(
                _rectangles_overlap(
                    (label_x, label_y, label_x + label_width, label_y + label_height), item
                )
                for item in occupied
            ) and label_y + label_height < height:
                label_y += label_height + 4
            position = (label_x, min(label_y, height - label_height))

        label_x, label_y = position
        rectangle = (label_x, label_y, label_x + label_width, label_y + label_height)
        occupied.append(rectangle)
        color = (46, 204, 113) if actor["kind"] == VEHICLE else (255, 191, 0)
        cv2.rectangle(frame, rectangle[:2], rectangle[2:], (18, 18, 18), -1)
        cv2.rectangle(frame, rectangle[:2], rectangle[2:], color, 1)
        cv2.putText(
            frame, label, (label_x + 5, label_y + text_height + 4), font, scale, color, 1, cv2.LINE_AA
        )


def draw_status_overlay(
    frame: np.ndarray,
    latest_probability: float,
    evidence: IncidentEvidence,
    accident_active: bool,
) -> None:
    """Draw incident policy state without presenting ML context as a verdict."""
    cv2.putText(
        frame,
        f"ML context: {latest_probability:.3f} (not a confirmation trigger)",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Evidence: {evidence.confidence:.2f}  {evidence.kind.replace('_', ' ') or 'none'}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "F: full screen   I: labels   Q/Esc: quit",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if accident_active:
        status = "!!! ACCIDENT CONFIRMED !!!"
        origin = (25, 140)
        scale = 0.85
        thickness = 3
        color = (0, 0, 255)
    elif evidence.status == "REVIEW":
        status = "REVIEW: DYNAMIC INTERACTION"
        origin = (25, 140)
        scale = 0.80
        thickness = 2
        color = (0, 165, 255)
    else:
        status = "STATUS: NORMAL (PARKED VEHICLES ARE IGNORED)"
        origin = (25, 140)
        scale = 0.80
        thickness = 3
        color = (0, 255, 0)

    cv2.putText(
        frame,
        status,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    if evidence.status != "NORMAL":
        cv2.putText(
            frame,
            evidence.reason[:92],
            (25, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )


# ============================================================
# STARTUP
# ============================================================

print()
print("============================================================")
print("             RESQTRACK FINAL LIVE SYSTEM")
print("============================================================")
print()

print("Camera:", CAMERA_ID)
print(
    "Camera location:",
    f"{CAMERA_LATITUDE:.6f}, {CAMERA_LONGITUDE:.6f}",
)
print()


# ============================================================
# LOAD FINAL TEMPORAL MODEL
# ============================================================

print("Loading final temporal model...")

try:
    package = joblib.load(FINAL_MODEL)
except Exception as exc:
    raise RuntimeError(
        f"Could not load final model '{FINAL_MODEL}': {exc}"
    ) from exc

try:
    final_model = package["model"]
    model_features = package["features"]
except (KeyError, TypeError) as exc:
    raise RuntimeError(
        "Final model package must contain 'model' and 'features'."
    ) from exc

if list(model_features) != list(FEATURES):
    raise ValueError(
        "Feature-order mismatch between trained model and "
        "canonical feature engine."
    )

print("Final model loaded successfully.")
print("Feature count:", len(FEATURES))
print()


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

try:
    yolo = YOLO(YOLO_MODEL)
except Exception as exc:
    raise RuntimeError(
        f"Could not load YOLO model '{YOLO_MODEL}': {exc}"
    ) from exc

print("YOLO loaded successfully.")
print()


# ============================================================
# OPEN VIDEO
# ============================================================

print("Opening video:", VIDEO_PATH)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    video.release()
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = float(video.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30.0

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened successfully.")
print("FPS:", round(fps, 3))
print("Frames:", frame_count if frame_count > 0 else "unknown")
print()


def set_fullscreen(enabled: bool) -> None:
    """Use an explicit resizable window instead of the backend's half-size default."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if enabled:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOWED_SIZE)


set_fullscreen(START_FULLSCREEN)


# ============================================================
# RESET TRACKING / STATE
# ============================================================

reset_vehicle_history()

previous_vehicles = None
previous_previous_vehicles = None
frame_features: list[dict] = []
frame_number = 0

# Per-actor display state.
speed_histories: dict[int, deque[float]] = defaultdict(
    lambda: deque(maxlen=SPEED_HISTORY_LENGTH)
)
vehicle_last_seen: dict[int, int] = {}
previous_actor_centers: dict[int, tuple[float, float]] = {}

# ML context state.
latest_probability = 0.0
max_probability = 0.0

# Event policy and alert state.
incident_engine = IncidentEventEngine()
latest_evidence = IncidentEvidence()
accident_active = False
accident_start_time = 0.0
confirmation_frame: int | None = None
show_actor_info = SHOW_VEHICLE_INFO
display_fullscreen = START_FULLSCREEN


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

try:
    while True:
        success, frame = video.read()

        if not success:
            print("Video finished.")
            break

        frame_number += 1

        # ====================================================
        # YOLO + BYTE TRACK
        # ====================================================

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONFIDENCE,
            classes=TRACKED_CLASSES,
            verbose=False,
        )

        if not results:
            continue

        result = results[0]

        # ====================================================
        # EXTRACT TRACKED VEHICLES
        # ====================================================

        vehicles: dict[int, dict] = {}
        actors: dict[int, dict] = {}

        has_boxes = (
            result.boxes is not None
            and result.boxes.id is not None
            and result.boxes.cls is not None
        )

        if has_boxes:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )
            class_ids = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, vehicle_id, class_id in zip(
                boxes,
                ids,
                class_ids,
            ):
                vehicle_id = int(vehicle_id)
                class_id = int(class_id)

                x1, y1, x2, y2 = map(float, box)
                center = (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                )

                class_name = safe_class_name(yolo, class_id)
                kind = PERSON if class_id == PERSON_CLASS else VEHICLE

                previous_center = previous_actor_centers.get(vehicle_id)

                speed_px_s = pixel_speed(
                    previous_center,
                    center,
                    fps,
                )

                speed_histories[vehicle_id].append(speed_px_s)
                vehicle_last_seen[vehicle_id] = frame_number

                actors[vehicle_id] = {
                    "center": center,
                    "box": (x1, y1, x2, y2),
                    "class_id": class_id,
                    "class_name": class_name,
                    "kind": kind,
                    "speed_pixels_per_second": speed_px_s,
                }

                # The shipped ML model was trained only on vehicles.  Do not
                # contaminate its feature vector with people; the event engine
                # below evaluates vehicle-to-person interactions separately.
                if kind == VEHICLE:
                    vehicles[vehicle_id] = actors[vehicle_id]
                    update_vehicle(vehicle_id, center)

        # Remove stale display state.
        stale_ids = [
            vehicle_id
            for vehicle_id, last_seen in vehicle_last_seen.items()
            if frame_number - last_seen > STALE_VEHICLE_FRAMES
        ]

        for vehicle_id in stale_ids:
            vehicle_last_seen.pop(vehicle_id, None)
            speed_histories.pop(vehicle_id, None)

        latest_evidence = incident_engine.update(actors)

        # ====================================================
        # CANONICAL FRAME FEATURES
        # ====================================================

        current_features = extract_frame_features(
            vehicles,
            previous_vehicles,
            previous_previous_vehicles,
        )

        frame_features.append(current_features)

        if len(frame_features) > WINDOW_FRAMES:
            frame_features = frame_features[-WINDOW_FRAMES:]

        # Save previous-frame references after feature extraction.
        previous_previous_vehicles = previous_vehicles
        previous_vehicles = vehicles
        previous_actor_centers = {
            actor_id: actor["center"] for actor_id, actor in actors.items()
        }

        # ====================================================
        # TEMPORAL ML WINDOW
        # Match training: WINDOW_FRAMES + STRIDE_FRAMES.
        # ====================================================

        window_ready = len(frame_features) >= WINDOW_FRAMES
        stride_ready = (
            frame_number >= WINDOW_FRAMES
            and (frame_number - WINDOW_FRAMES) % STRIDE_FRAMES == 0
        )

        if window_ready and stride_ready:
            aggregated = aggregate_window(frame_features)

            if aggregated is not None:
                model_input = pd.DataFrame(
                    [
                        [aggregated[feature] for feature in FEATURES]
                    ],
                    columns=FEATURES,
                )

                probability = float(
                    final_model.predict_proba(model_input)[0][1]
                )

                latest_probability = probability
                max_probability = max(max_probability, probability)

        # A model probability can support a human review, but it cannot
        # dispatch an emergency.  Confirmation needs dynamic, time-ordered
        # evidence from IncidentEventEngine (collision or pedestrian impact).
        if latest_evidence.confirmed and not accident_active:
            accident_active = True
            accident_start_time = time.time()
            confirmation_frame = frame_number

            print()
            print("================================================")
            print("🚨 ACCIDENT CONFIRMED")
            print("================================================")
            print("Camera:", CAMERA_ID)
            print("Frame:", frame_number)
            print("Event type:", latest_evidence.kind)
            print("Evidence:", latest_evidence.reason)
            print("Evidence confidence:", round(latest_evidence.confidence, 3))
            print("ML context (not trigger):", round(latest_probability, 4))
            print(
                "Location:",
                f"{CAMERA_LATITUDE:.6f}, {CAMERA_LONGITUDE:.6f}",
            )
            print("================================================")
            print()

        # ====================================================
        # ACCIDENT ALERT TIMER
        # ====================================================

        if accident_active:
            elapsed = time.time() - accident_start_time

            if elapsed >= ALERT_DURATION:
                accident_active = False
                confirmation_frame = None
                print("Accident alert cleared.")

        # ====================================================
        # RENDER FRAME
        # ====================================================

        # Ultralytics' default labels and our detailed labels used to render
        # on top of each other.  Keep its boxes, render one managed label per
        # actor below, and place those labels in free screen space.
        annotated_frame = result.plot(labels=False, conf=False)

        draw_actor_information(
            annotated_frame,
            actors,
            speed_histories,
            show_actor_info,
        )

        draw_status_overlay(
            annotated_frame,
            latest_probability,
            latest_evidence,
            accident_active,
        )

        # Show a compact camera identifier.
        cv2.putText(
            annotated_frame,
            f"Camera: {CAMERA_ID}",
            (25, max(170, annotated_frame.shape[0] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, annotated_frame)

        # Real-time-ish playback at the source video's FPS.
        delay = max(1, int(round(1000.0 / fps)))
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:
            print("Stopped by user.")
            break
        if key == ord("f"):
            display_fullscreen = not display_fullscreen
            set_fullscreen(display_fullscreen)
        if key == ord("i"):
            show_actor_info = not show_actor_info

finally:
    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()
    cv2.destroyAllWindows()
    reset_vehicle_history()
    incident_engine.reset()

    print()
    print("============================================================")
    print("ResQTrack stopped.")
    print("Frames processed:", frame_number)
    print("Maximum ML probability:", round(max_probability, 4))
    print("============================================================")
