import cv2
import time

from ultralytics import YOLO

from accident_logic import (
    update_vehicle,
    collision_score
)

# -----------------------------
# SETTINGS
# -----------------------------

VIDEO_PATH = "data/accident.mp4"

ACCIDENT_THRESHOLD = 70
CONFIRMATION_FRAMES = 3
ALERT_DURATION = 5


# ------------------------

# LOAD YOLO
# --------------------------

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully.")


# -----------------------------
# ACCIDENT STATE
# -----------------------------

accident_active = False
accident_start_time = 0

suspicious_frames = 0


# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    # -----------------------------
    # YOLO + BYTE TRACK
    # -----------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        classes=[2, 3, 5, 7]
    )

    result = results[0]

    vehicles = {}


    # -----------------------------
    # GET VEHICLE INFORMATION
    # -----------------------------

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, vehicle_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            vehicles[vehicle_id] = {
                "center": (center_x, center_y),
                "box": (x1, y1, x2, y2)
            }

            update_vehicle(
                vehicle_id,
                (center_x, center_y)
            )


    # -----------------------------
    # CALCULATE ACCIDENT SCORE
    # -----------------------------

    score, collision_pairs = collision_score(
        vehicles
    )


    # -----------------------------
    # MULTI-FRAME CONFIRMATION
    # -----------------------------

    if not accident_active:

        if score >= ACCIDENT_THRESHOLD:

            suspicious_frames = min(
                suspicious_frames + 1,
                CONFIRMATION_FRAMES
            )

        else:

            suspicious_frames = 0


    # -----------------------------
    # CONFIRM ACCIDENT
    # -----------------------------

    if (
        suspicious_frames == CONFIRMATION_FRAMES
        and not accident_active
    ):

        accident_active = True
        accident_start_time = time.time()

        print()
        print("================================")
        print("ACCIDENT CONFIRMED")
        print("Score:", score)
        print("Vehicles:", collision_pairs)
        print("================================")
        print()


    # -----------------------------
    # KEEP ALERT ACTIVE
    # -----------------------------

    if accident_active:

        elapsed = time.time() - accident_start_time

        if elapsed >= ALERT_DURATION:

            accident_active = False
            suspicious_frames = 0

            print("Accident alert cleared.")


    # -----------------------------
    # DRAW DETECTIONS
    # -----------------------------

    annotated_frame = result.plot()


    # -----------------------------
    # DISPLAY SCORE
    # -----------------------------

    cv2.putText(
        annotated_frame,
        f"Accident Score: {score}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    # -----------------------------
    # DISPLAY CONFIRMATION
    # -----------------------------

    cv2.putText(
        annotated_frame,
        f"Confirmation: {suspicious_frames}/{CONFIRMATION_FRAMES}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # -----------------------------
    # DISPLAY STATUS
    # -----------------------------

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

    elif score >= 30:

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


    # -----------------------------
    # SHOW VIDEO
    # -----------------------------

    cv2.imshow(
        "ResQTrack - Accident Detection",
        annotated_frame
    )


    # -----------------------------
    # QUIT
    # -----------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

video.release()
cv2.destroyAllWindows()

print("ResQTrack stopped.")