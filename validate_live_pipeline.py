import os
import cv2
import pandas as pd

from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
    collision_score,
    reset_vehicle_history
)

from temporal_detector import TemporalAccidentDetector
from temporal_ml_inference import TemporalMLEngine


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

SUSPICIOUS_SCORE = 30
CONFIRMATION_FRAMES = 3
ALERT_DURATION = 5


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    print(
        f"Loading video: {os.path.basename(video_path)}"
    )

    # --------------------------------------------------------
    # RESET SOFTWARE STATE
    # --------------------------------------------------------

    reset_vehicle_history()

    ml_engine = TemporalMLEngine()
    ml_engine.reset()

    temporal_detector = (
        TemporalAccidentDetector(
            evidence_threshold=0.30,
            lookback=10,
            minimum_suspicious_observations=2,
            physical_event_threshold=0.40
        )
    )

    # --------------------------------------------------------
    # LOAD A FRESH YOLO MODEL FOR EACH VIDEO
    #
    # This prevents ByteTrack state from one video
    # leaking into the next video.
    # --------------------------------------------------------

    model = YOLO(MODEL_PATH)

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return {
            "predicted": 0,
            "max_physical_score": 0,
            "max_ml_probability": 0.0,
            "max_combined_evidence": 0.0,
            "confirmation_frame": None
        }

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    previous_vehicles = None
    previous_previous_vehicles = None

    max_physical_score = 0
    max_ml_probability = 0.0
    max_combined_evidence = 0.0

    confirmed = False
    confirmation_frame = None

    frame_number = 0

    # --------------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------------

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        # ====================================================
        # YOLO + BYTE TRACK
        # ====================================================

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

        # ====================================================
        # EXTRACT VEHICLES
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

                update_vehicle(
                    vehicle_id,
                    (
                        center_x,
                        center_y
                    )
                )

        # ====================================================
        # PHYSICAL EVIDENCE
        # ====================================================

        score, collision_pairs = (
            collision_score(
                vehicles
            )
        )

        max_physical_score = max(
            max_physical_score,
            score
        )

        physical_evidence = min(
            score / 100.0,
            1.0
        )

        collision_evidence = (
            len(collision_pairs) > 0
        )

        physical_suspicious = (
            score >= SUSPICIOUS_SCORE
        )

        # ====================================================
        # TEMPORAL ML
        # ====================================================

        ml_prediction = ml_engine.update(
            current=vehicles,
            previous=previous_vehicles,
            previous_previous=previous_previous_vehicles
        )

        ml_probability = None

        if ml_prediction is not None:

            ml_probability = float(
                ml_prediction
            )

            max_ml_probability = max(
                max_ml_probability,
                ml_probability
            )

        # ====================================================
        # COMBINED EVIDENCE
        # ====================================================

        if ml_probability is not None:

            combined_evidence = max(
                ml_probability,
                physical_evidence
            )

        else:

            combined_evidence = (
                physical_evidence
            )

        combined_evidence = min(
            max(
                combined_evidence,
                0.0
            ),
            1.0
        )

        max_combined_evidence = max(
            max_combined_evidence,
            combined_evidence
        )

        # ====================================================
        # TEMPORAL CONFIRMATION
        # ====================================================

        confirmation = (
            temporal_detector.update(
                evidence_score=combined_evidence,
                physical_score=physical_evidence,
                collision_evidence=collision_evidence
            )
        )

        # ====================================================
        # ACCIDENT CONFIRMED
        # ====================================================

        if (
            confirmation["confirmed"]
            and not confirmed
        ):

            confirmed = True

            confirmation_frame = (
                frame_number
            )

            # Once a video is confirmed,
            # we don't need to process the rest
            # for classification purposes.

            break

        # ====================================================
        # UPDATE FRAME HISTORY
        # ====================================================

        previous_previous_vehicles = (
            previous_vehicles
        )

        previous_vehicles = vehicles

    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()

    reset_vehicle_history()
    ml_engine.reset()

    return {

        "predicted":
            int(confirmed),

        "max_physical_score":
            max_physical_score,

        "max_ml_probability":
            max_ml_probability,

        "max_combined_evidence":
            max_combined_evidence,

        "confirmation_frame":
            confirmation_frame
    }


# ============================================================
# COLLECT VIDEOS
# ============================================================

videos = []

for directory, label in [

    (ACCIDENT_DIR, 1),

    (NORMAL_DIR, 0)

]:

    if not os.path.isdir(directory):

        print(
            f"WARNING: Directory missing: "
            f"{directory}"
        )

        continue

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

                "actual":
                    label
            })


# ============================================================
# START VALIDATION
# ============================================================

print()
print("==============================================")
print("      RESQTRACK LIVE PIPELINE VALIDATION")
print("==============================================")
print()

print(
    f"Videos found: {len(videos)}"
)

print()


results = []


# ============================================================
# PROCESS ALL VIDEOS
# ============================================================

for index, item in enumerate(
    videos,
    start=1
):

    video_path = item["path"]
    actual = item["actual"]

    video_name = os.path.basename(
        video_path
    )

    print(
        f"[{index}/{len(videos)}] "
        f"{video_name}"
    )

    result = process_video(
        video_path
    )

    predicted = result[
        "predicted"
    ]

    print(
        "    Actual   :",
        "ACCIDENT"
        if actual
        else
        "NORMAL"
    )

    print(
        "    Predicted:",
        "ACCIDENT"
        if predicted
        else
        "NORMAL"
    )

    print(
        "    Max Physical Score:",
        result[
            "max_physical_score"
        ]
    )

    print(
        "    Max ML Probability:",
        round(
            result[
                "max_ml_probability"
            ],
            3
        )
    )

    print(
        "    Max Combined Evidence:",
        round(
            result[
                "max_combined_evidence"
            ],
            3
        )
    )

    print(
        "    Confirmation Frame:",
        result[
            "confirmation_frame"
        ]
    )

    print()

    results.append({

        "video":
            video_name,

        "actual":
            actual,

        "predicted":
            predicted,

        "max_physical_score":
            result[
                "max_physical_score"
            ],

        "max_ml_probability":
            result[
                "max_ml_probability"
            ],

        "max_combined_evidence":
            result[
                "max_combined_evidence"
            ],

        "confirmation_frame":
            result[
                "confirmation_frame"
            ]
    })


# ============================================================
# CONFUSION MATRIX
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0


for result in results:

    actual = result["actual"]
    predicted = result["predicted"]

    if actual == 1 and predicted == 1:

        TP += 1

    elif actual == 0 and predicted == 0:

        TN += 1

    elif actual == 0 and predicted == 1:

        FP += 1

    elif actual == 1 and predicted == 0:

        FN += 1


# ============================================================
# METRICS
# ============================================================

total = (
    TP
    +
    TN
    +
    FP
    +
    FN
)

accuracy = (
    (TP + TN) / total
    if total > 0
    else 0.0
)

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0.0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    /
    (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

df = pd.DataFrame(
    results
)

df.to_csv(
    "live_pipeline_validation.csv",
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("==============================================")
print("       LIVE PIPELINE VALIDATION RESULT")
print("==============================================")
print()

print("CONFUSION MATRIX")
print("----------------")

print(
    f"True Positives : {TP}"
)

print(
    f"True Negatives : {TN}"
)

print(
    f"False Positives: {FP}"
)

print(
    f"False Negatives: {FN}"
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

print(
    "Detailed results saved as:"
)

print(
    "live_pipeline_validation.csv"
)

print()

print("==============================================")
print("Validation complete.")
print("==============================================")