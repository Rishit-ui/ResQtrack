import cv2
import math
import os
import numpy as np
import pandas as pd

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"

ACCIDENT_FOLDER = "dataset/accident"
NORMAL_FOLDER = "dataset/normal"

OUTPUT_FILE = "training_features.csv"


model = YOLO(MODEL_PATH)


def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    intersection = width * height

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


def extract_video_features(video_path):

    print()
    print("Processing:", video_path)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():

        print("Could not open:", video_path)

        return None

    speeds = []
    min_distances = []
    ious = []
    vehicle_counts = []

    previous_positions = {}

    frame_count = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_count += 1

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

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()

            ids = result.boxes.id.cpu().numpy().astype(int)

            for box, vehicle_id in zip(boxes, ids):

                x1, y1, x2, y2 = box

                center = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                )

                vehicles[vehicle_id] = {
                    "center": center,
                    "box": box
                }

        vehicle_counts.append(len(vehicles))

        # -----------------------------
        # SPEED FEATURES
        # -----------------------------

        current_positions = {}

        for vehicle_id, data in vehicles.items():

            current_positions[vehicle_id] = data["center"]

            if vehicle_id in previous_positions:

                speed = distance(
                    previous_positions[vehicle_id],
                    data["center"]
                )

                speeds.append(speed)

        previous_positions = current_positions

        # -----------------------------
        # PAIR FEATURES
        # -----------------------------

        ids = list(vehicles.keys())

        for i in range(len(ids)):

            for j in range(i + 1, len(ids)):

                v1 = vehicles[ids[i]]
                v2 = vehicles[ids[j]]

                d = distance(
                    v1["center"],
                    v2["center"]
                )

                iou = calculate_iou(
                    v1["box"],
                    v2["box"]
                )

                min_distances.append(d)
                ious.append(iou)

    video.release()

    # Avoid empty feature arrays

    if not speeds:
        speeds = [0]

    if not min_distances:
        min_distances = [999]

    if not ious:
        ious = [0]

    if not vehicle_counts:
        vehicle_counts = [0]

    # -----------------------------
    # AGGREGATE FEATURES
    # -----------------------------

    features = {

        "avg_speed":
            np.mean(speeds),

        "max_speed":
            np.max(speeds),

        "speed_std":
            np.std(speeds),

        "max_iou":
            np.max(ious),

        "avg_iou":
            np.mean(ious),

        "min_vehicle_distance":
            np.min(min_distances),

        "avg_vehicle_distance":
            np.mean(min_distances),

        "max_vehicle_count":
            np.max(vehicle_counts),

        "avg_vehicle_count":
            np.mean(vehicle_counts),

        "frames":
            frame_count
    }

    return features


# ==================================================
# BUILD DATASET
# ==================================================

all_features = []


# Accident videos

for filename in os.listdir(ACCIDENT_FOLDER):

    if filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):

        path = os.path.join(
            ACCIDENT_FOLDER,
            filename
        )

        features = extract_video_features(path)

        if features is not None:

            features["label"] = 1
            features["video"] = filename

            all_features.append(features)


# Normal videos

for filename in os.listdir(NORMAL_FOLDER):

    if filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):

        path = os.path.join(
            NORMAL_FOLDER,
            filename
        )

        features = extract_video_features(path)

        if features is not None:

            features["label"] = 0
            features["video"] = filename

            all_features.append(features)


# -----------------------------
# SAVE CSV
# -----------------------------

df = pd.DataFrame(all_features)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("================================")
print("FEATURE EXTRACTION COMPLETE")
print("================================")

print()
print(df)

print()
print("Saved as:")
print(OUTPUT_FILE)