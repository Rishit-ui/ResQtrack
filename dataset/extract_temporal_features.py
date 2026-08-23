import os
import math
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

ANNOTATIONS_FILE = "annotations.csv"

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

OUTPUT_FILE = "temporal_features.csv"

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15

CONFIDENCE = 0.40

VEHICLE_CLASSES = [2, 3, 5, 7]
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("     ResQTrack Temporal Feature Extractor")
print("==============================================\n")

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded successfully.")


# ============================================================
# BASIC GEOMETRY
# ============================================================

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

    area1 = max(
        0,
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = max(
        0,
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# VEHICLE FRAME EXTRACTION
# ============================================================

def extract_frame_data(frame):

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    result = results[0]

    vehicles = {}

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, vehicle_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            center = (
                float((x1 + x2) / 2),
                float((y1 + y2) / 2)
            )

            vehicles[int(vehicle_id)] = {
                "center": center,
                "box": box
            }

    return vehicles


# ============================================================
# FRAME-LEVEL BEHAVIOURAL FEATURES
# ============================================================

def calculate_frame_features(
    current,
    previous,
    previous_previous
):

    vehicle_count = len(current)

    speeds = []
    accelerations = []
    pair_distances = []
    pair_ious = []
    distance_changes = []

    # --------------------------------------------------------
    # VEHICLE MOTION
    # --------------------------------------------------------

    for vehicle_id, vehicle in current.items():

        center = vehicle["center"]

        if (
            previous is not None
            and vehicle_id in previous
        ):

            previous_center = previous[
                vehicle_id
            ]["center"]

            speed = distance(
                center,
                previous_center
            )

            speeds.append(speed)

            # --------------------------------------------
            # ACCELERATION
            # --------------------------------------------

            if (
                previous_previous is not None
                and vehicle_id in previous_previous
            ):

                previous_previous_center = (
                    previous_previous[
                        vehicle_id
                    ]["center"]
                )

                previous_speed = distance(
                    previous_center,
                    previous_previous_center
                )

                acceleration = (
                    speed - previous_speed
                )

                accelerations.append(
                    abs(acceleration)
                )

    # --------------------------------------------------------
    # VEHICLE PAIR INTERACTION
    # --------------------------------------------------------

    ids = list(current.keys())

    for i in range(len(ids)):

        for j in range(i + 1, len(ids)):

            id1 = ids[i]
            id2 = ids[j]

            vehicle1 = current[id1]
            vehicle2 = current[id2]

            current_distance = distance(
                vehicle1["center"],
                vehicle2["center"]
            )

            pair_distances.append(
                current_distance
            )

            iou = calculate_iou(
                vehicle1["box"],
                vehicle2["box"]
            )

            pair_ious.append(iou)

            # --------------------------------------------
            # RELATIVE DISTANCE CHANGE
            # --------------------------------------------

            if (
                previous is not None
                and id1 in previous
                and id2 in previous
            ):

                previous_distance = distance(
                    previous[id1]["center"],
                    previous[id2]["center"]
                )

                distance_change = (
                    previous_distance -
                    current_distance
                )

                distance_changes.append(
                    distance_change
                )

    # --------------------------------------------------------
    # SAFE DEFAULTS
    # --------------------------------------------------------

    if not speeds:
        speeds = [0.0]

    if not accelerations:
        accelerations = [0.0]

    if not pair_distances:
        pair_distances = [9999.0]

    if not pair_ious:
        pair_ious = [0.0]

    if not distance_changes:
        distance_changes = [0.0]

    # --------------------------------------------------------
    # RETURN FEATURES
    # --------------------------------------------------------

    return {
        "vehicle_count": vehicle_count,

        "mean_speed": np.mean(speeds),
        "max_speed": np.max(speeds),

        "mean_acceleration":
            np.mean(accelerations),

        "max_acceleration":
            np.max(accelerations),

        "min_pair_distance":
            np.min(pair_distances),

        "mean_pair_distance":
            np.mean(pair_distances),

        "max_iou":
            np.max(pair_ious),

        "mean_iou":
            np.mean(pair_ious),

        "max_approach_rate":
            np.max(distance_changes),

        "mean_approach_rate":
            np.mean(distance_changes)
    }


# ============================================================
# TEMPORAL WINDOW AGGREGATION
# ============================================================

def aggregate_window(
    frame_features
):

    if not frame_features:
        return None

    df = pd.DataFrame(
        frame_features
    )

    result = {}

    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    result["mean_speed"] = df[
        "mean_speed"
    ].mean()

    result["max_speed"] = df[
        "max_speed"
    ].max()

    result["speed_std"] = df[
        "mean_speed"
    ].std()

    result["mean_acceleration"] = df[
        "mean_acceleration"
    ].mean()

    result["max_acceleration"] = df[
        "max_acceleration"
    ].max()

    # --------------------------------------------------------
    # VEHICLE INTERACTION
    # --------------------------------------------------------

    result["min_pair_distance"] = df[
        "min_pair_distance"
    ].min()

    result["mean_pair_distance"] = df[
        "mean_pair_distance"
    ].mean()

    result["max_iou"] = df[
        "max_iou"
    ].max()

    result["mean_iou"] = df[
        "mean_iou"
    ].mean()

    # --------------------------------------------------------
    # APPROACH / COLLISION DYNAMICS
    # --------------------------------------------------------

    result["max_approach_rate"] = df[
        "max_approach_rate"
    ].max()

    result["mean_approach_rate"] = df[
        "mean_approach_rate"
    ].mean()

    # --------------------------------------------------------
    # TRAFFIC DENSITY
    # --------------------------------------------------------

    result["max_vehicle_count"] = df[
        "vehicle_count"
    ].max()

    result["mean_vehicle_count"] = df[
        "vehicle_count"
    ].mean()

    # --------------------------------------------------------
    # REMOVE NaN
    # --------------------------------------------------------

    for key in result:

        if pd.isna(result[key]):
            result[key] = 0.0

    return result


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    video_name,
    label,
    accident_start,
    accident_end
):

    print()
    print("----------------------------------------------")
    print("Processing:", video_name)
    print("----------------------------------------------")

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        print(
            "ERROR: Could not open:",
            video_path
        )

        return []

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    duration = total_frames / fps

    print(
        f"FPS: {fps:.2f} | "
        f"Frames: {total_frames} | "
        f"Duration: {duration:.2f}s"
    )

    # --------------------------------------------------------
    # EXTRACT FRAME FEATURES
    # --------------------------------------------------------

    all_frame_features = []

    previous = None
    previous_previous = None

    frame_number = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        vehicles = extract_frame_data(
            frame
        )

        features = calculate_frame_features(
            vehicles,
            previous,
            previous_previous
        )

        all_frame_features.append(
            features
        )

        previous_previous = previous
        previous = vehicles

        if frame_number % 100 == 0:

            print(
                f"  Frames processed: "
                f"{frame_number}/{total_frames}"
            )

    video.release()

    # --------------------------------------------------------
    # CREATE TEMPORAL WINDOWS
    # --------------------------------------------------------

    rows = []

    window_id = 0

    for start_frame in range(
        0,
        len(all_frame_features),
        STRIDE_FRAMES
    ):

        end_frame = (
            start_frame +
            WINDOW_FRAMES
        )

        if end_frame > len(
            all_frame_features
        ):
            break

        window = all_frame_features[
            start_frame:end_frame
        ]

        features = aggregate_window(
            window
        )

        if features is None:
            continue

        # ----------------------------------------------------
        # WINDOW TIMESTAMP
        # ----------------------------------------------------

        window_start_sec = (
            start_frame / fps
        )

        window_end_sec = (
            end_frame / fps
        )

        # ----------------------------------------------------
        # TEMPORAL LABEL
        # ----------------------------------------------------
        #
        # Accident window = overlap with annotated
        # accident interval.
        #
        # We require >= 50% overlap with the
        # accident interval to label it positive.
        #
        # ----------------------------------------------------

        window_label = 0

        if label == 1:

            overlap_start = max(
                window_start_sec,
                accident_start
            )

            overlap_end = min(
                window_end_sec,
                accident_end
            )

            overlap = max(
                0,
                overlap_end -
                overlap_start
            )

            window_duration = (
                window_end_sec -
                window_start_sec
            )

            if (
                window_duration > 0
                and
                overlap /
                window_duration >= 0.50
            ):

                window_label = 1

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        features["video"] = video_name

        features["window_id"] = window_id

        features["window_start_sec"] = (
            window_start_sec
        )

        features["window_end_sec"] = (
            window_end_sec
        )

        features["label"] = window_label

        rows.append(
            features
        )

        window_id += 1

    print(
        f"Generated {len(rows)} temporal windows."
    )

    return rows


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print("\nLoading annotations...")

annotations = pd.read_csv(
    ANNOTATIONS_FILE
)

required_columns = [
    "video",
    "label",
    "start_sec",
    "end_sec"
]

for column in required_columns:

    if column not in annotations.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# ============================================================
# PROCESS DATASET
# ============================================================

all_rows = []


for _, annotation in annotations.iterrows():

    video_name = str(
        annotation["video"]
    )

    label = int(
        annotation["label"]
    )

    accident_start = float(
        annotation["start_sec"]
    )

    accident_end = float(
        annotation["end_sec"]
    )

    if label == 1:

        video_path = os.path.join(
            ACCIDENT_DIR,
            video_name
        )

    else:

        video_path = os.path.join(
            NORMAL_DIR,
            video_name
        )

    if not os.path.exists(
        video_path
    ):

        print()
        print(
            "WARNING: File not found:",
            video_path
        )

        continue

    rows = process_video(
        video_path,
        video_name,
        label,
        accident_start,
        accident_end
    )

    all_rows.extend(
        rows
    )


# ============================================================
# SAVE TEMPORAL DATASET
# ============================================================

if not all_rows:

    raise RuntimeError(
        "No temporal features were generated."
    )


df = pd.DataFrame(
    all_rows
)


# Put important columns first

feature_columns = [
    "video",
    "window_id",
    "window_start_sec",
    "window_end_sec",
    "label"
]

other_columns = [
    column
    for column in df.columns
    if column not in feature_columns
]

df = df[
    feature_columns +
    other_columns
]


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("TEMPORAL FEATURE EXTRACTION COMPLETE")
print("==============================================")

print(
    f"\nTotal windows: {len(df)}"
)

print(
    "\nWindow distribution:"
)

print(
    df["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "NORMAL",
            1: "ACCIDENT"
        }
    )
)

print(
    "\nVideos represented:"
)

print(
    df["video"]
    .nunique()
)

print(
    "\nSaved as:"
)

print(
    OUTPUT_FILE
)

print(
    "\nNext stage:"
)

print(
    "Train the temporal classifier "
    "using VIDEO-LEVEL splitting."
)