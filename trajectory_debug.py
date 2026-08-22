import cv2
import math
from ultralytics import YOLO

VIDEO_PATH = "data/accident.mp4"

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("ERROR: Could not open video")
    exit()

history = {}
frame_number = 0


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def box_gap(box1, box2):

    if box1[2] < box2[0]:
        horizontal = box2[0] - box1[2]

    elif box2[2] < box1[0]:
        horizontal = box1[0] - box2[2]

    else:
        horizontal = 0

    if box1[3] < box2[1]:
        vertical = box2[1] - box1[3]

    elif box2[3] < box1[1]:
        vertical = box1[1] - box2[3]

    else:
        vertical = 0

    return math.sqrt(
        horizontal ** 2 +
        vertical ** 2
    )


while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        classes=[2, 3, 5, 7]
    )

    result = results[0]

    vehicles = {}

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, vehicle_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )

            vehicles[vehicle_id] = {
                "center": center,
                "box": box
            }

            if vehicle_id not in history:
                history[vehicle_id] = []

            history[vehicle_id].append(center)

            history[vehicle_id] = history[vehicle_id][-20:]

            # Draw trajectory
            points = history[vehicle_id]

            for k in range(1, len(points)):

                cv2.line(
                    frame,
                    points[k - 1],
                    points[k],
                    (255, 255, 0),
                    2
                )

            cv2.circle(
                frame,
                center,
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"ID {vehicle_id}",
                (center[0] + 5, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


    # ---------------------------------------
    # ANALYZE ALL VEHICLE PAIRS
    # ---------------------------------------

    pair_data = []

    ids = list(vehicles.keys())

    for i in range(len(ids)):

        for j in range(i + 1, len(ids)):

            id1 = ids[i]
            id2 = ids[j]

            current_distance = distance(
                vehicles[id1]["center"],
                vehicles[id2]["center"]
            )

            gap = box_gap(
                vehicles[id1]["box"],
                vehicles[id2]["box"]
            )

            approach = 0

            # Need at least 6 previous positions
            if (
                len(history[id1]) >= 6
                and len(history[id2]) >= 6
            ):

                old_distance = distance(
                    history[id1][-6],
                    history[id2][-6]
                )

                approach = old_distance - current_distance

            pair_data.append(
                (
                    approach,
                    id1,
                    id2,
                    current_distance,
                    gap
                )
            )


    # Sort by strongest approach
    pair_data.sort(
        reverse=True,
        key=lambda x: x[0]
    )


    # ---------------------------------------
    # DISPLAY TOP 3 APPROACHING PAIRS
    # ---------------------------------------

    y = 30

    cv2.putText(
        frame,
        f"FRAME: {frame_number}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y += 30

    for index, data in enumerate(pair_data[:3]):

        approach, id1, id2, current_distance, gap = data

        text = (
            f"{id1}-{id2} "
            f"Approach:{approach:.1f} "
            f"Dist:{current_distance:.1f} "
            f"Gap:{gap:.1f}"
        )

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        y += 25


    # ---------------------------------------
    # SHOW
    # ---------------------------------------

    cv2.imshow(
        "ResQTrack - Approach Debugger",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()