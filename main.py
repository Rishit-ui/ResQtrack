import cv2
import time

from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
    collision_score
)

from temporal_detector import TemporalAccidentDetector


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "data/accident.mp4"

# Score at which the scene becomes suspicious
SUSPICIOUS_SCORE = 30

# Number of consecutive observations required
CONFIRMATION_FRAMES = 3

# How long the confirmed alert remains active
ALERT_DURATION = 5


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully.")


# ============================================================
# ACCIDENT STATE
# ============================================================

accident_active = False
accident_start_time = 0

temporal_detector = TemporalAccidentDetector(
    evidence_threshold=0.30,
    confirmation_windows=CONFIRMATION_FRAMES,
    history_size=5,
    minimum_suspicious_ratio=0.67
)

confirmation_status = "NORMAL"
confirmation_confidence = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        classes=[2, 3, 5, 7]
    )

    result = results[0]

    vehicles = {}

    # ========================================================
    # GET VEHICLE INFORMATION
    # ========================================================

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

        for box, vehicle_id in zip(boxes, ids):

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
                (center_x, center_y)
            )

    # ========================================================
    # CALCULATE ACCIDENT SCORE
    # ========================================================

    score, collision_pairs = collision_score(
        vehicles
    )

    # ========================================================
    # TEMPORAL CONFIRMATION
    # ========================================================

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

    confirmation = temporal_detector.update(
        evidence_score=evidence_score,
        collision_evidence=collision_evidence,
        approach_evidence=approach_evidence
    )

    confirmation_status = (
        confirmation["status"]
    )

    confirmation_confidence = (
        confirmation["confidence"]
    )

    # ========================================================
    # ACCIDENT CONFIRMED
    # ========================================================

    if (
        confirmation["confirmed"]
        and not accident_active
    ):

        accident_active = True

        accident_start_time = time.time()

        confirmation_status = "ACCIDENT CONFIRMED"

        print()
        print("================================")
        print("ACCIDENT CONFIRMED")
        print(
            "Evidence Score:",
            score
        )
        print(
            "Confirmation:",
            confirmation_confidence
        )
        print(
            "Vehicles:",
            collision_pairs
        )
        print("================================")
        print()

    # ========================================================
    # KEEP ALERT ACTIVE
    # ========================================================

    if accident_active:

        elapsed = (
            time.time()
            - accident_start_time
        )

        if elapsed >= ALERT_DURATION:

            accident_active = False

            temporal_detector.reset()

            confirmation_status = "NORMAL"

            confirmation_confidence = 0.0

            print(
                "Accident alert cleared."
            )

        else:

            # Keep confirmed state locked
            confirmation_status = "ACCIDENT CONFIRMED"

    # ========================================================
    # DRAW YOLO DETECTIONS
    # ========================================================

    annotated_frame = result.plot()

    # ========================================================
    # DISPLAY ACCIDENT SCORE
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Accident Score: {score}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # ========================================================
    # DISPLAY CONFIRMATION
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Confirmation: {confirmation_status}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    if accident_active:

        cv2.putText(
            annotated_frame,
            "!!! ACCIDENT CONFIRMED !!!",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    elif confirmation_status == "SUSPICIOUS":

        cv2.putText(
            annotated_frame,
            "SUSPICIOUS EVENT",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

    else:

        cv2.putText(
            annotated_frame,
            "STATUS: NORMAL",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            3
        )

    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "ResQTrack - Accident Detection",
        annotated_frame
    )

    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

video.release()

cv2.destroyAllWindows()

print("ResQTrack stopped.")