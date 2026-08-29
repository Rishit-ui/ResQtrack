import cv2
import time
import joblib
import numpy as np
import pandas as pd

from collections import deque, defaultdict
from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
    collision_score,
    reset_vehicle_history,
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
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]
YOLO_CONFIDENCE = 0.40

# The trained temporal model's validated decision settings.
ML_THRESHOLD = 0.50
MIN_SUSPICIOUS_WINDOWS = 2
EVENT_GAP_SECONDS = 1.5
ALERT_DURATION = 5.0

WINDOW_NAME = "ResQTrack - Accident Detection"

# Vehicle overlay settings. Speed is IMAGE-SPACE until camera
# calibration is introduced. Do not present it as km/h.
SHOW_VEHICLE_INFO = True
SPEED_HISTORY_LENGTH = 5
STALE_VEHICLE_FRAMES = 60


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


def draw_vehicle_information(
    frame: np.ndarray,
    vehicles: dict,
    speed_histories: dict[int, deque[float]],
) -> None:
    """Draw tracking ID, vehicle class and image-space speed."""
    if not SHOW_VEHICLE_INFO:
        return

    for vehicle_id, vehicle in vehicles.items():
        x1, y1, _, _ = map(int, vehicle["box"])
        class_name = str(vehicle["class_name"]).upper()
        speed = smoothed_speed(speed_histories.get(vehicle_id, deque()))

        label = (
            f"ID {vehicle_id} | {class_name} | "
            f"{speed:.1f} px/s"
        )

        text_y = max(20, y1 - 8)

        # Black outline for readability.
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_status_overlay(
    frame: np.ndarray,
    physical_score: float,
    latest_probability: float,
    suspicious_events: deque[float],
    accident_active: bool,
) -> None:
    """Draw ResQTrack diagnostic/status information."""
    cv2.putText(
        frame,
        f"Physical Score: {physical_score}",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"ML Probability: {latest_probability:.3f}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Suspicious Windows: "
        f"{len(suspicious_events)}/{MIN_SUSPICIOUS_WINDOWS}",
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
    elif latest_probability >= ML_THRESHOLD:
        status = "SUSPICIOUS EVENT"
        origin = (25, 140)
        scale = 0.80
        thickness = 2
        color = (0, 165, 255)
    else:
        status = "STATUS: NORMAL"
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


# ============================================================
# RESET TRACKING / STATE
# ============================================================

reset_vehicle_history()

previous_vehicles = None
previous_previous_vehicles = None
frame_features: list[dict] = []
frame_number = 0

# Per-vehicle display state.
speed_histories: dict[int, deque[float]] = defaultdict(
    lambda: deque(maxlen=SPEED_HISTORY_LENGTH)
)
vehicle_last_seen: dict[int, int] = {}

# ML event state.
suspicious_events: deque[float] = deque()
latest_probability = 0.0
latest_prediction = 0
max_probability = 0.0
suspicious_window_count = 0

# Accident state.
accident_active = False
accident_start_time = 0.0
confirmation_frame: int | None = None


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
            classes=VEHICLE_CLASSES,
            verbose=False,
        )

        if not results:
            continue

        result = results[0]

        # ====================================================
        # EXTRACT TRACKED VEHICLES
        # ====================================================

        vehicles: dict[int, dict] = {}

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

                previous_center = None
                if (
                    previous_vehicles is not None
                    and vehicle_id in previous_vehicles
                ):
                    previous_center = previous_vehicles[vehicle_id]["center"]

                speed_px_s = pixel_speed(
                    previous_center,
                    center,
                    fps,
                )

                speed_histories[vehicle_id].append(speed_px_s)
                vehicle_last_seen[vehicle_id] = frame_number

                vehicles[vehicle_id] = {
                    "center": center,
                    "box": (x1, y1, x2, y2),
                    "class_id": class_id,
                    "class_name": class_name,
                    "speed_pixels_per_second": speed_px_s,
                }

                # Keep the existing trajectory/feature system intact.
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

        # ====================================================
        # PHYSICAL SCORE
        # Supporting information only.
        # It does NOT directly confirm accidents.
        # ====================================================

        physical_score, collision_pairs = collision_score(vehicles)

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

        # ====================================================
        # TEMPORAL ML WINDOW
        # Match training: WINDOW_FRAMES + STRIDE_FRAMES.
        # ====================================================

        window_ready = len(frame_features) >= WINDOW_FRAMES
        stride_ready = (
            frame_number >= WINDOW_FRAMES
            and (frame_number - WINDOW_FRAMES) % STRIDE_FRAMES == 0
        )

        if window_ready and stride_ready and not accident_active:
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
                latest_prediction = int(probability >= ML_THRESHOLD)

                if latest_prediction == 1:
                    current_time = frame_number / fps
                    suspicious_events.append(current_time)
                    suspicious_window_count += 1

                    while (
                        suspicious_events
                        and current_time - suspicious_events[0]
                        > EVENT_GAP_SECONDS
                    ):
                        suspicious_events.popleft()

                    if len(suspicious_events) >= MIN_SUSPICIOUS_WINDOWS:
                        # Guard against repeatedly confirming the same event.
                        accident_active = True
                        accident_start_time = time.time()
                        confirmation_frame = frame_number

                        print()
                        print("================================================")
                        print("🚨 ACCIDENT CONFIRMED")
                        print("================================================")
                        print("Camera:", CAMERA_ID)
                        print("Frame:", frame_number)
                        print("ML Probability:", round(probability, 4))
                        print("Physical Score:", physical_score)
                        print("Collision Pairs:", collision_pairs)
                        print(
                            "Suspicious Windows:",
                            len(suspicious_events),
                        )
                        print(
                            "Location:",
                            f"{CAMERA_LATITUDE:.6f}, "
                            f"{CAMERA_LONGITUDE:.6f}",
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
                suspicious_events.clear()
                suspicious_window_count = 0
                print("Accident alert cleared.")

        # ====================================================
        # RENDER FRAME
        # ====================================================

        annotated_frame = result.plot()

        draw_vehicle_information(
            annotated_frame,
            vehicles,
            speed_histories,
        )

        draw_status_overlay(
            annotated_frame,
            physical_score,
            latest_probability,
            suspicious_events,
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

finally:
    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()
    cv2.destroyAllWindows()
    reset_vehicle_history()

    print()
    print("============================================================")
    print("ResQTrack stopped.")
    print("Frames processed:", frame_number)
    print("Maximum ML probability:", round(max_probability, 4))
    print("============================================================")
