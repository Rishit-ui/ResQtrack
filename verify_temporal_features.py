import os
import cv2
import pandas as pd

from ultralytics import YOLO

from temporal_features_engine import (
    extract_frame_features,
    aggregate_window,
    FEATURES,
    WINDOW_FRAMES,
    STRIDE_FRAMES
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/accident.mp4"


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO(
    MODEL_PATH
)

print("YOLO loaded.")


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(
    VIDEO_PATH
)

if not video.isOpened():

    raise RuntimeError(
        f"Could not open {VIDEO_PATH}"
    )


# ============================================================
# FRAME HISTORY
# ============================================================

previous = None
previous_previous = None

frame_features = []

frame_count = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = video.read()

    if not success:
        break

    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    result = results[0]

    vehicles = {}


    # --------------------------------------------------------
    # VEHICLES
    # --------------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, vehicle_id in zip(
            boxes,
            ids
        ):

            x1, y1, x2, y2 = box

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            vehicles[vehicle_id] = {

                "center": (
                    center_x,
                    center_y
                ),

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }


    # --------------------------------------------------------
    # CANONICAL FRAME FEATURES
    # --------------------------------------------------------

    features = extract_frame_features(
        vehicles,
        previous,
        previous_previous
    )

    frame_features.append(
        features
    )


    # --------------------------------------------------------
    # UPDATE HISTORY
    # --------------------------------------------------------

    previous_previous = previous
    previous = vehicles


    # --------------------------------------------------------
    # STOP AFTER ENOUGH FRAMES FOR A TEST
    # --------------------------------------------------------

    if frame_count >= 60:
        break


video.release()


# ============================================================
# AGGREGATE
# ============================================================

window = aggregate_window(
    frame_features
)


# ============================================================
# CHECK
# ============================================================

print()
print("==============================================")
print("CANONICAL FEATURE ENGINE TEST")
print("==============================================")
print()

print(
    "Frames processed:",
    frame_count
)

print(
    "Feature count:",
    len(FEATURES)
)

print()

print("Features:")
for feature in FEATURES:
    print(
        f"  {feature}"
    )

print()

print(
    "Aggregated feature count:",
    len(window)
)

print()

missing = [
    feature
    for feature in FEATURES
    if feature not in window
]

extra = [
    feature
    for feature in window
    if feature not in FEATURES
]


print(
    "Missing features:",
    missing
)

print(
    "Extra features:",
    extra
)

print()

if (
    not missing
    and
    not extra
    and
    len(window) == len(FEATURES)
):

    print(
        "✅ CANONICAL FEATURE ENGINE PASSED"
    )

else:

    print(
        "❌ FEATURE ENGINE FAILED"
    )