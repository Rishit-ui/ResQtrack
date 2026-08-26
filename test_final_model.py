import cv2
import joblib
import pandas as pd
import numpy as np

from ultralytics import YOLO

from temporal_features_engine import (
    extract_frame_features,
    aggregate_window,
    FEATURES,
    WINDOW_FRAMES
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "resqtrack_final_model.pkl"
VIDEO_PATH = "data/accident.mp4"


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print()
print("============================================================")
print("          RESQTRACK FINAL MODEL SANITY TEST")
print("============================================================")
print()

package = joblib.load(
    MODEL_FILE
)

model = package["model"]
model_features = package["features"]


# ============================================================
# FEATURE ORDER CHECK
# ============================================================

if model_features != FEATURES:

    raise ValueError(
        "Feature order mismatch between model and "
        "canonical feature engine."
    )


print(
    "Feature order verified."
)

print(
    "Feature count:",
    len(FEATURES)
)

print()


# ============================================================
# LOAD YOLO
# ============================================================

yolo = YOLO(
    "yolo11n.pt"
)


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(
    VIDEO_PATH
)

if not video.isOpened():

    raise RuntimeError(
        f"Could not open: {VIDEO_PATH}"
    )


# ============================================================
# TEMPORAL HISTORY
# ============================================================

previous = None
previous_previous = None

frame_features = []

frame_count = 0

predictions = []


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = (
        video.read()
    )

    if not success:
        break

    frame_count += 1


    results = yolo.track(
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
    # VEHICLE EXTRACTION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CANONICAL FRAME FEATURES
    # --------------------------------------------------------

    frame_feature = (
        extract_frame_features(

            vehicles,

            previous,

            previous_previous
        )
    )

    frame_features.append(
        frame_feature
    )


    # --------------------------------------------------------
    # UPDATE HISTORY
    # --------------------------------------------------------

    previous_previous = previous

    previous = vehicles


    # ========================================================
    # WINDOW READY?
    # ========================================================

    if len(frame_features) >= WINDOW_FRAMES:

        window = frame_features[
            -WINDOW_FRAMES:
        ]


        aggregated = (
            aggregate_window(
                window
            )
        )


        if aggregated is not None:

            model_input = pd.DataFrame(

                [[
                    aggregated[
                        feature
                    ]

                    for feature in FEATURES
                ]],

                columns=FEATURES
            )


            probability = float(

                model.predict_proba(
                    model_input
                )[0][1]
            )


            prediction = int(
                probability >= 0.50
            )


            predictions.append(
                (
                    frame_count,
                    probability,
                    prediction
                )
            )


    # --------------------------------------------------------
    # STOP AFTER FIRST 3000 FRAMES
    # --------------------------------------------------------

    if frame_count >= 3000:
        break


video.release()


# ============================================================
# RESULTS
# ============================================================

print()
print("============================================================")
print("                 SANITY TEST RESULTS")
print("============================================================")
print()

print(
    "Frames processed:",
    frame_count
)

print(
    "Predictions:",
    len(predictions)
)

print()


if not predictions:

    print(
        "No complete temporal window was produced."
    )

else:

    probabilities = np.array([
        item[1]
        for item in predictions
    ])


    print(
        "Maximum accident probability:",
        f"{probabilities.max():.4f}"
    )

    print(
        "Mean accident probability:",
        f"{probabilities.mean():.4f}"
    )

    print(
        "Minimum accident probability:",
        f"{probabilities.min():.4f}"
    )

    print()


    print(
        "Windows predicted ACCIDENT:",
        int(
            (
                probabilities
                >=
                0.50
            ).sum()
        )
    )

    print(
        "Total windows:",
        len(probabilities)
    )

    print()


    # --------------------------------------------------------
    # STRONGEST WINDOWS
    # --------------------------------------------------------

    strongest = sorted(
        predictions,
        key=lambda x: x[1],
        reverse=True
    )[:10]


    print(
        "Top accident-probability windows:"
    )

    print(
        "----------------------------------"
    )


    for (
        frame_number,
        probability,
        prediction
    ) in strongest:

        print(

            f"Frame {frame_number:<6}"

            f"Probability={probability:.4f}"

            f"  Prediction="

            + (
                "ACCIDENT"
                if prediction
                else
                "NORMAL"
            )
        )


print()

print(
    "Final model sanity test complete."
)

print()