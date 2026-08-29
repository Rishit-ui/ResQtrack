import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/accident.mp4"

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

frame_no = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_no += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.40,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    result = results[0]

    count = 0
    ids = []

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):
        count = len(result.boxes.id)

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

    if frame_no <= 10 or frame_no % 30 == 0:
        print(
            f"Frame {frame_no}: "
            f"detections={count}, "
            f"IDs={ids}"
        )

cap.release()

print("Total frames processed:", frame_no)