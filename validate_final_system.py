import os
import cv2
import joblib
import numpy as np
import pandas as pd

from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
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

MODEL_FILE = "resqtrack_final_model.pkl"
YOLO_MODEL = "yolo11n.pt"

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

YOLO_CONFIDENCE = 0.40
VEHICLE_CLASSES = [2, 3, 5, 7]

ML_THRESHOLD = 0.50

MIN_SUSPICIOUS_WINDOWS = 2
EVENT_GAP_SECONDS = 1.5

OUTPUT_FILE = "final_system_validation.csv"


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print()
print("============================================================")
print("          RESQTRACK FINAL SYSTEM VALIDATION")
print("============================================================")
print()

print("Loading final model...")

package = joblib.load(
    MODEL_FILE
)

final_model = package["model"]
model_features = package["features"]

if model_features != FEATURES:

    raise ValueError(
        "Feature order mismatch between "
        "final model and canonical feature engine."
    )

print("Final model loaded.")
print("Feature count:", len(FEATURES))
print()


# ============================================================
# COLLECT VIDEOS
# ============================================================

videos = []

for directory, actual_label in [

    (ACCIDENT_DIR, 1),

    (NORMAL_DIR, 0)

]:

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            f"Missing directory: {directory}"
        )

    for filename in sorted(
        os.listdir(directory)
    ):

        if filename.lower().endswith(
            (
                ".mp4",
                ".avi",
                ".mov",
                ".mkv"
            )
        ):

            videos.append({

                "path":
                    os.path.join(
                        directory,
                        filename
                    ),

                "video":
                    filename,

                "actual":
                    actual_label
            })


print(
    "Videos found:",
    len(videos)
)

print()


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_info
):

    video_path = video_info["path"]
    video_name = video_info["video"]
    actual = video_info["actual"]

    print(
        "------------------------------------------------------------"
    )

    print(
        "Video:",
        video_name
    )

    print(
        "Actual:",
        "ACCIDENT"
        if actual
        else
        "NORMAL"
    )

    # --------------------------------------------------------
    # RESET TRACKER STATE
    # --------------------------------------------------------

    reset_vehicle_history()

    # --------------------------------------------------------
    # FRESH YOLO INSTANCE
    # --------------------------------------------------------

    yolo = YOLO(
        YOLO_MODEL
    )

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0


    # --------------------------------------------------------
    # TEMPORAL STATE
    # --------------------------------------------------------

    previous_vehicles = None
    previous_previous_vehicles = None

    frame_features = []

    frame_number = 0

    # --------------------------------------------------------
    # FINAL DECISION STATE
    # --------------------------------------------------------

    suspicious_events = []

    latest_probability = 0.0

    max_probability = 0.0

    confirmation_frame = None

    prediction = 0


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        success, frame = video.read()

        if not success:
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
            verbose=False
        )

        result = results[0]

        vehicles = {}


        # ====================================================
        # VEHICLE EXTRACTION
        # ====================================================

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


        # ====================================================
        # CANONICAL FRAME FEATURES
        # ====================================================

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


        if len(frame_features) > WINDOW_FRAMES:

            frame_features = (
                frame_features[
                    -WINDOW_FRAMES:
                ]
            )


        # ====================================================
        # UPDATE HISTORY
        # ====================================================

        previous_previous_vehicles = (
            previous_vehicles
        )

        previous_vehicles = vehicles


        # ====================================================
        # SAME WINDOW / STRIDE LOGIC AS MAIN.PY
        # ====================================================

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
        ):

            # ------------------------------------------------
            # AGGREGATE
            # ------------------------------------------------

            aggregated = (
                aggregate_window(
                    frame_features
                )
            )


            if aggregated is not None:

                # --------------------------------------------
                # MODEL INPUT
                # --------------------------------------------

                model_input = pd.DataFrame(

                    [[
                        aggregated[
                            feature
                        ]

                        for feature in FEATURES
                    ]],

                    columns=FEATURES
                )


                # --------------------------------------------
                # PREDICTION
                # --------------------------------------------

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


                # --------------------------------------------
                # SUSPICIOUS WINDOW
                # --------------------------------------------

                if probability >= ML_THRESHOLD:

                    current_time = (
                        frame_number
                        /
                        fps
                    )

                    suspicious_events.append(
                        current_time
                    )


                    # Remove events too far away
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

                        suspicious_events.pop(0)


                    # ----------------------------------------
                    # FINAL CONFIRMATION
                    # ----------------------------------------

                    if (

                        len(
                            suspicious_events
                        )

                        >=

                        MIN_SUSPICIOUS_WINDOWS
                    ):

                        prediction = 1

                        confirmation_frame = (
                            frame_number
                        )

                        # Same operational behaviour:
                        # once confirmed, classification is done.
                        break


    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()

    reset_vehicle_history()


    # ========================================================
    # RESULT
    # ========================================================

    print(
        "Predicted:",
        "ACCIDENT"
        if prediction
        else
        "NORMAL"
    )

    print(
        "Max probability:",
        f"{max_probability:.4f}"
    )

    print(
        "Confirmation frame:",
        confirmation_frame
    )

    print()


    return {

        "video":
            video_name,

        "actual":
            actual,

        "predicted":
            prediction,

        "max_probability":
            max_probability,

        "confirmation_frame":
            confirmation_frame
    }


# ============================================================
# RUN ALL VIDEOS
# ============================================================

results = []

for video_info in videos:

    result = process_video(
        video_info
    )

    results.append(
        result
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tp = int(
    (
        (results_df["actual"] == 1)
        &
        (results_df["predicted"] == 1)
    ).sum()
)

tn = int(
    (
        (results_df["actual"] == 0)
        &
        (results_df["predicted"] == 0)
    ).sum()
)

fp = int(
    (
        (results_df["actual"] == 0)
        &
        (results_df["predicted"] == 1)
    ).sum()
)

fn = int(
    (
        (results_df["actual"] == 1)
        &
        (results_df["predicted"] == 0)
    ).sum()
)


total = (
    tp
    +
    tn
    +
    fp
    +
    fn
)


accuracy = (
    (tp + tn)
    /
    total
    if total
    else
    0.0
)


precision = (
    tp
    /
    (tp + fp)
    if (tp + fp)
    else
    0.0
)


recall = (
    tp
    /
    (tp + fn)
    if (tp + fn)
    else
    0.0
)


f1 = (
    2
    *
    precision
    *
    recall
    /
    (precision + recall)
    if (precision + recall)
    else
    0.0
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("============================================================")
print("          RESQTRACK FINAL SYSTEM RESULT")
print("============================================================")
print()

print("CONFUSION MATRIX")
print("----------------")

print(
    "True Positives :",
    tp
)

print(
    "True Negatives :",
    tn
)

print(
    "False Positives:",
    fp
)

print(
    "False Negatives:",
    fn
)

print()

print("METRICS")
print("-------")

print(
    f"Accuracy : {accuracy:.3f}"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall   : {recall:.3f}"
)

print(
    f"F1 Score : {f1:.3f}"
)

print()

print("PERCENTAGE")
print("----------")

print(
    f"Accuracy : {accuracy * 100:.1f}%"
)

print(
    f"Precision: {precision * 100:.1f}%"
)

print(
    f"Recall   : {recall * 100:.1f}%"
)

print(
    f"F1 Score : {f1 * 100:.1f}%"
)

print()

print("PER-VIDEO RESULTS")
print("-----------------")

print(
    results_df.to_string(
        index=False
    )
)

print()

print(
    "Saved:",
    OUTPUT_FILE
)

print()

print("============================================================")
print("Final system validation complete.")
print("============================================================")
