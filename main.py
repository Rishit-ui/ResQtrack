import cv2
import time
import joblib
import numpy as np
import pandas as pd

from collections import deque

from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
    collision_score,
    reset_vehicle_history
)

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

VIDEO_PATH = "data/accident.mp4"

YOLO_MODEL = "yolo11n.pt"

FINAL_MODEL = "resqtrack_final_model.pkl"

YOLO_CONFIDENCE = 0.40

VEHICLE_CLASSES = [2, 3, 5, 7]


# ------------------------------------------------------------
# FINAL ML DECISION
# ------------------------------------------------------------

ML_THRESHOLD = 0.50

# Number of suspicious temporal windows required
# inside one event.
MIN_SUSPICIOUS_WINDOWS = 2

# Maximum time gap between suspicious windows
# before they are considered separate events.
EVENT_GAP_SECONDS = 1.5


# ------------------------------------------------------------
# ALERT
# ------------------------------------------------------------

ALERT_DURATION = 5


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

WINDOW_NAME = (
    "ResQTrack - Accident Detection"
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print()
print("============================================================")
print("          RESQTRACK FINAL LIVE SYSTEM")
print("============================================================")
print()

print(
    "Loading final temporal model..."
)

package = joblib.load(
    FINAL_MODEL
)

final_model = package["model"]

model_features = package["features"]


# ============================================================
# VERIFY MODEL FEATURES
# ============================================================

if model_features != FEATURES:

    raise ValueError(
        "Feature-order mismatch between "
        "trained model and canonical feature engine."
    )


print(
    "Final model loaded successfully."
)

print(
    "Feature count:",
    len(FEATURES)
)

print()


# ============================================================
# LOAD YOLO
# ============================================================

print(
    "Loading YOLO..."
)

yolo = YOLO(
    YOLO_MODEL
)

print(
    "YOLO loaded successfully."
)

print()


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(
    VIDEO_PATH
)

if not video.isOpened():

    print(
        "ERROR: Could not open video:"
    )

    print(
        VIDEO_PATH
    )

    raise SystemExit


fps = video.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30.0


print(
    "Video opened successfully."
)

print(
    "FPS:",
    fps
)

print()


# ============================================================
# RESET TRACKING
# ============================================================

reset_vehicle_history()


# ============================================================
# TEMPORAL STATE
# ============================================================

previous_vehicles = None

previous_previous_vehicles = None

frame_features = []

frame_number = 0


# ============================================================
# ML EVENT STATE
# ============================================================

suspicious_events = deque()

latest_probability = 0.0

latest_prediction = 0

max_probability = 0.0

suspicious_window_count = 0


# ============================================================
# ACCIDENT STATE
# ============================================================

accident_active = False

accident_start_time = 0

confirmation_frame = None


# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    success, frame = video.read()

    if not success:

        print(
            "Video finished."
        )

        break


    frame_number += 1


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = yolo.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=YOLO_CONFIDENCE,

        classes=VEHICLE_CLASSES,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # EXTRACT VEHICLES
    # ========================================================

    vehicles = {}


    if (
        result.boxes is not None

        and

        result.boxes.id is not None
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


            center_x = (
                float(x1 + x2)
                /
                2.0
            )


            center_y = (
                float(y1 + y2)
                /
                2.0
            )


            vehicles[
                int(vehicle_id)
            ] = {

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


            update_vehicle(

                int(vehicle_id),

                (
                    center_x,
                    center_y
                )
            )


    # ========================================================
    # PHYSICAL SCORE
    #
    # Supporting information only.
    # It DOES NOT directly confirm accidents.
    # ========================================================

    physical_score, collision_pairs = (
        collision_score(
            vehicles
        )
    )


    # ========================================================
    # CANONICAL FRAME FEATURES
    # ========================================================

    current_features = (
        extract_frame_features(

            vehicles,

            previous_vehicles,

            previous_previous_vehicles
        )
    )


    frame_features.append(
        current_features
    )


    # Keep only enough history for one temporal window.
    if len(frame_features) > WINDOW_FRAMES:

        frame_features = (
            frame_features[
                -WINDOW_FRAMES:
            ]
        )


    # ========================================================
    # UPDATE FRAME HISTORY
    # ========================================================

    previous_previous_vehicles = (
        previous_vehicles
    )

    previous_vehicles = vehicles


    # ========================================================
    # TEMPORAL ML WINDOW
    #
    # Match training:
    #
    # window = 30 frames
    # stride = 15 frames
    # ========================================================

    window_ready = (
        len(frame_features)
        >=
        WINDOW_FRAMES
    )


    stride_ready = (
        frame_number
        >=
        WINDOW_FRAMES
        and

        (
            (
                frame_number
                -
                WINDOW_FRAMES
            )
            %
            STRIDE_FRAMES
            ==
            0
        )
    )


    if (
        window_ready
        and
        stride_ready
        and
        not accident_active
    ):


        # ----------------------------------------------------
        # AGGREGATE
        # ----------------------------------------------------

        aggregated = (
            aggregate_window(
                frame_features
            )
        )


        if aggregated is not None:


            # ------------------------------------------------
            # CREATE MODEL INPUT
            # ------------------------------------------------

            model_input = pd.DataFrame(

                [[
                    aggregated[
                        feature
                    ]

                    for feature in FEATURES
                ]],

                columns=FEATURES
            )


            # ------------------------------------------------
            # FINAL MODEL PREDICTION
            # ------------------------------------------------

            probability = float(

                final_model.predict_proba(
                    model_input
                )[0][1]
            )


            latest_probability = (
                probability
            )


            max_probability = max(
                max_probability,
                probability
            )


            latest_prediction = int(

                probability
                >=
                ML_THRESHOLD
            )


            # ------------------------------------------------
            # SUSPICIOUS WINDOW
            # ------------------------------------------------

            if latest_prediction == 1:

                suspicious_window_count += 1

                current_time = (
                    frame_number
                    /
                    fps
                )


                suspicious_events.append(
                    current_time
                )


                # --------------------------------------------
                # REMOVE OLD EVENTS
                # --------------------------------------------

                while (

                    suspicious_events

                    and

                    (
                        current_time
                        -
                        suspicious_events[0]
                    )

                    >
                    EVENT_GAP_SECONDS
                ):

                    suspicious_events.popleft()


                # --------------------------------------------
                # ACCIDENT CONFIRMATION
                # --------------------------------------------

                if (

                    len(
                        suspicious_events
                    )
                    >=
                    MIN_SUSPICIOUS_WINDOWS
                ):

                    accident_active = True

                    accident_start_time = (
                        time.time()
                    )

                    confirmation_frame = (
                        frame_number
                    )


                    print()
                    print(
                        "================================================"
                    )

                    print(
                        "🚨 ACCIDENT CONFIRMED"
                    )

                    print(
                        "Frame:",
                        frame_number
                    )

                    print(
                        "ML Probability:",
                        round(
                            probability,
                            4
                        )
                    )

                    print(
                        "Physical Score:",
                        physical_score
                    )

                    print(
                        "Collision Pairs:",
                        collision_pairs
                    )

                    print(
                        "Suspicious Windows:",
                        len(
                            suspicious_events
                        )
                    )

                    print(
                        "================================================"
                    )

                    print()


    # ========================================================
    # ALERT TIMER
    # ========================================================

    if accident_active:

        elapsed = (
            time.time()
            -
            accident_start_time
        )


        if elapsed >= ALERT_DURATION:

            accident_active = False

            confirmation_frame = None

            suspicious_events.clear()

            suspicious_window_count = 0

            print(
                "Accident alert cleared."
            )


    # ========================================================
    # DRAW YOLO
    # ========================================================

    annotated_frame = (
        result.plot()
    )


    # ========================================================
    # SCORE DISPLAY
    # ========================================================

    cv2.putText(

        annotated_frame,

        f"Physical Score: {physical_score}",

        (25, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 0),

        2
    )


    # ========================================================
    # ML PROBABILITY
    # ========================================================

    cv2.putText(

        annotated_frame,

        f"ML Probability: {latest_probability:.3f}",

        (25, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 0),

        2
    )


    # ========================================================
    # SUSPICIOUS WINDOWS
    # ========================================================

    cv2.putText(

        annotated_frame,

        (
            f"Suspicious Windows: "
            f"{len(suspicious_events)}/"
            f"{MIN_SUSPICIOUS_WINDOWS}"
        ),

        (25, 95),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 0),

        2
    )


    # ========================================================
    # STATUS
    # ========================================================

    if accident_active:

        cv2.putText(

            annotated_frame,

            "!!! ACCIDENT CONFIRMED !!!",

            (25, 140),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            (0, 0, 255),

            3
        )


    elif latest_probability >= ML_THRESHOLD:

        cv2.putText(

            annotated_frame,

            "SUSPICIOUS EVENT",

            (25, 140),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.80,

            (0, 165, 255),

            2
        )


    else:

        cv2.putText(

            annotated_frame,

            "STATUS: NORMAL",

            (25, 140),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.80,

            (0, 255, 0),

            3
        )


    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(

        WINDOW_NAME,

        annotated_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

video.release()

cv2.destroyAllWindows()

reset_vehicle_history()


print()
print(
    "============================================================"
)

print(
    "ResQTrack stopped."
)

print(
    "Maximum ML probability:",
    round(
        max_probability,
        4
    )
)

print(
    "============================================================"
)