import os
import cv2

from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
    collision_score,
    reset_vehicle_history
)

from temporal_detector import (
    TemporalAccidentDetector
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

SUSPICIOUS_SCORE = 30

CONFIRMATION_FRAMES = 3


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("==============================================")
print("       ResQTrack SIH Validation v2")
print("==============================================")
print()

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")
print()


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    # IMPORTANT:
    # Clear tracking state before every video.

    reset_vehicle_history()

    detector = TemporalAccidentDetector(

        evidence_threshold=0.30,

        confirmation_windows=
            CONFIRMATION_FRAMES,

        history_size=5,

        minimum_suspicious_ratio=0.67,

        strong_evidence_threshold=0.40,

        minimum_support_windows=2
    )

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        return (
            False,
            0,
            None
        )

    frame_count = 0

    max_score = 0

    confirmed = False

    confirmation_frame = None


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        success, frame = (
            video.read()
        )

        if not success:
            break

        frame_count += 1


        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=0.4,

            classes=[
                2,
                3,
                5,
                7
            ],

            verbose=False
        )

        result = results[0]

        vehicles = {}


        # ----------------------------------------------------
        # VEHICLES
        # ----------------------------------------------------

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

                vehicles[
                    vehicle_id
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

                    vehicle_id,

                    (
                        center_x,
                        center_y
                    )
                )


        # ----------------------------------------------------
        # ACCIDENT SCORE
        # ----------------------------------------------------

        score, collision_pairs = (
            collision_score(
                vehicles
            )
        )

        max_score = max(
            max_score,
            score
        )


        # ----------------------------------------------------
        # TEMPORAL EVIDENCE
        # ----------------------------------------------------

        evidence_score = min(
            score / 100.0,
            1.0
        )

        collision_evidence = (
            len(collision_pairs) > 0
        )

        approach_evidence = (
            score >= SUSPICIOUS_SCORE
        )


        # ----------------------------------------------------
        # TEMPORAL DETECTOR
        # ----------------------------------------------------

        confirmation = (
            detector.update(

                evidence_score=
                    evidence_score,

                collision_evidence=
                    collision_evidence,

                approach_evidence=
                    approach_evidence
            )
        )


        # ----------------------------------------------------
        # CONFIRMED
        # ----------------------------------------------------

        if confirmation["confirmed"]:

            confirmed = True

            confirmation_frame = (
                frame_count
            )

            break


    video.release()

    # IMPORTANT:
    # Clear state after every video too.

    reset_vehicle_history()

    return (
        confirmed,
        max_score,
        confirmation_frame
    )


# ============================================================
# FIND VIDEOS
# ============================================================

videos = []


for directory, label in [

    (
        ACCIDENT_DIR,
        1
    ),

    (
        NORMAL_DIR,
        0
    )

]:

    if not os.path.isdir(
        directory
    ):

        print(
            f"WARNING: Missing directory: {directory}"
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

            videos.append(

                (
                    os.path.join(
                        directory,
                        filename
                    ),

                    label
                )
            )


print(
    f"Videos found: {len(videos)}"
)

print()


# ============================================================
# VALIDATION
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0


for index, (
    video_path,
    actual
) in enumerate(
    videos,
    start=1
):

    name = os.path.basename(
        video_path
    )

    print(
        f"[{index}/{len(videos)}] {name}"
    )


    predicted, max_score, confirmation_frame = (

        process_video(
            video_path
        )
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    if (
        actual == 1
        and
        predicted
    ):

        TP += 1

    elif (
        actual == 0
        and
        not predicted
    ):

        TN += 1

    elif (
        actual == 0
        and
        predicted
    ):

        FP += 1

    else:

        FN += 1


    print(
        "  Actual   :",
        "ACCIDENT"
        if actual
        else
        "NORMAL"
    )

    print(
        "  Predicted:",
        "ACCIDENT"
        if predicted
        else
        "NORMAL"
    )

    print(
        "  Max score:",
        max_score
    )


    if (
        confirmation_frame
        is not None
    ):

        print(
            "  Confirmed frame:",
            confirmation_frame
        )

    print()


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

    (TP + TN)
    /
    total

    if total

    else
    0
)


precision = (

    TP
    /
    (TP + FP)

    if TP + FP

    else
    0
)


recall = (

    TP
    /
    (TP + FN)

    if TP + FN

    else
    0
)


f1 = (

    2
    *
    precision
    *
    recall
    /
    (precision + recall)

    if precision + recall

    else
    0
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("==============================================")
print("       RESQTRACK SIH VALIDATION RESULT")
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

print("==============================================")
print("Validation complete.")
print("==============================================")