import cv2
import time

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

VIDEO_PATH = "data/accident.mp4"

SUSPICIOUS_SCORE = 30
CONFIRMATION_FRAMES = 3
ALERT_DURATION = 5


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("==============================================")
print("        RESQTRACK INITIALIZING")
print("==============================================")
print()

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

print("Loading Temporal ML model...")
ml_engine = TemporalMLEngine()

print("Models loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("ERROR: Could not open video:")
    print(VIDEO_PATH)
    raise SystemExit

print("Video opened successfully.")


# ============================================================
# RESET STATE
# ============================================================

reset_vehicle_history()
ml_engine.reset()

temporal_detector = TemporalAccidentDetector(
    evidence_threshold=0.30,
    confirmation_windows=CONFIRMATION_FRAMES,
    history_size=5,
    minimum_suspicious_ratio=0.67
)


# ============================================================
# ACCIDENT STATE
# ============================================================

accident_active = False
accident_start_time = 0

confirmation_status = "NORMAL"
confirmation_confidence = 0.0

ml_probability = None

previous_vehicles = None
previous_previous_vehicles = None

running = True


# ============================================================
# MAIN LOOP
# ============================================================

while running:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = video.read()

    if not success:
        print("Video finished.")
        running = False
        continue


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

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


    # ========================================================
    # EXTRACT VEHICLES
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
                (
                    center_x,
                    center_y
                )
            )


    # ========================================================
    # PHYSICAL ACCIDENT EVIDENCE
    # ========================================================

    score, collision_pairs = collision_score(
        vehicles
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


    # ========================================================
    # TEMPORAL ML
    # ========================================================

    ml_prediction = ml_engine.update(
        current=vehicles,
        previous=previous_vehicles,
        previous_previous=previous_previous_vehicles
    )

    if ml_prediction is not None:
        ml_probability = float(
            ml_prediction
        )


    # ========================================================
    # COMBINED EVIDENCE
    # ========================================================

    if ml_probability is not None:

        combined_evidence = max(
            ml_probability,
            physical_evidence
        )

    else:

        combined_evidence = physical_evidence


    combined_evidence = min(
        max(
            combined_evidence,
            0.0
        ),
        1.0
    )


    # ========================================================
    # PHYSICAL SUPPORT
    # ========================================================
    #
    # ML probability alone cannot create physical evidence.
    #

    approach_evidence = physical_suspicious


    # ========================================================
    # TEMPORAL CONFIRMATION
    # ========================================================

    confirmation = temporal_detector.update(
        evidence_score=combined_evidence,
        collision_evidence=collision_evidence,
        approach_evidence=approach_evidence
    )

    confirmation_status = confirmation["status"]

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

        confirmation_status = (
            "ACCIDENT CONFIRMED"
        )

        print()
        print("==========================================")
        print("🚨 ACCIDENT CONFIRMED")
        print("Physical Score:", score)

        if ml_probability is not None:
            print(
                "ML Probability:",
                round(
                    ml_probability,
                    3
                )
            )

        print(
            "Combined Evidence:",
            round(
                combined_evidence,
                3
            )
        )

        print(
            "Collision Pairs:",
            collision_pairs
        )

        print("==========================================")
        print()


    # ========================================================
    # ALERT TIMER
    # ========================================================

    if accident_active:

        elapsed = (
            time.time()
            - accident_start_time
        )

        if elapsed >= ALERT_DURATION:

            accident_active = False

            temporal_detector.reset()
            ml_engine.reset()

            confirmation_status = "NORMAL"
            confirmation_confidence = 0.0
            ml_probability = None

            print(
                "Accident alert cleared."
            )

        else:

            confirmation_status = (
                "ACCIDENT CONFIRMED"
            )


    # ========================================================
    # DRAW
    # ========================================================

    annotated_frame = result.plot()


    # ========================================================
    # DISPLAY SCORE
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Accident Score: {score}",
        (30, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY ML PROBABILITY
    # ========================================================

    if ml_probability is None:

        ml_text = "ML Probability: --"

    else:

        ml_text = (
            f"ML Probability: "
            f"{ml_probability:.2f}"
        )

    cv2.putText(
        annotated_frame,
        ml_text,
        (30, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY EVIDENCE
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Evidence: {combined_evidence:.2f}",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3
        )

    elif confirmation_status == "SUSPICIOUS":

        cv2.putText(
            annotated_frame,
            "SUSPICIOUS EVENT",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

    else:

        cv2.putText(
            annotated_frame,
            "STATUS: NORMAL",
            (30, 140),
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
    # UPDATE HISTORY
    # ========================================================

    previous_previous_vehicles = (
        previous_vehicles
    )

    previous_vehicles = vehicles


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        running = False


# ============================================================
# CLEANUP
# ============================================================

video.release()

cv2.destroyAllWindows()

reset_vehicle_history()

ml_engine.reset()

print("ResQTrack stopped.")