import os
import cv2
import pandas as pd
import numpy as np

from ultralytics import YOLO

from accident_logic import reset_vehicle_history

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

MODEL_PATH = "yolo11n.pt"

ANNOTATIONS_FILE = "annotations.csv"

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

OUTPUT_FILE = "temporal_features_v4.csv"

CONFIDENCE = 0.40

VEHICLE_CLASSES = [2, 3, 5, 7]

# Temporal buffer around the manually annotated accident.
#
# Example:
# annotation = 7.0 -> 12.0
# buffered region = 6.5 -> 12.5
#
# This captures the approach / immediate aftermath that
# belongs to the same temporal accident event.
ACCIDENT_BUFFER_SEC = 0.5

# A window becomes ACCIDENT if at least this fraction
# of the window overlaps the buffered accident interval.
MIN_POSITIVE_OVERLAP = 0.30


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

annotations = pd.read_csv(
    ANNOTATIONS_FILE
)

required_annotation_columns = [
    "video",
    "label",
    "start_sec",
    "end_sec"
]

for column in required_annotation_columns:

    if column not in annotations.columns:

        raise ValueError(
            f"Missing annotation column: {column}"
        )


# ============================================================
# VALIDATE FEATURE ENGINE
# ============================================================

if len(FEATURES) != 19:

    raise ValueError(
        f"Expected 19 features, got {len(FEATURES)}"
    )


# ============================================================
# ACCIDENT INTERVAL
# ============================================================

def get_accident_interval(video_name):

    rows = annotations[
        annotations["video"]
        ==
        video_name
    ]

    if rows.empty:

        raise ValueError(
            f"No annotation found for {video_name}"
        )

    row = rows.iloc[0]

    label = int(
        row["label"]
    )

    if label == 0:

        return None

    start = float(
        row["start_sec"]
    )

    end = float(
        row["end_sec"]
    )

    if end <= start:

        raise ValueError(
            f"Invalid annotation for {video_name}: "
            f"{start} -> {end}"
        )

    # --------------------------------------------------------
    # TEMPORAL BUFFER
    # --------------------------------------------------------

    buffered_start = max(
        0.0,
        start - ACCIDENT_BUFFER_SEC
    )

    buffered_end = (
        end
        +
        ACCIDENT_BUFFER_SEC
    )

    return (
        buffered_start,
        buffered_end
    )


# ============================================================
# WINDOW LABEL
# ============================================================

def label_window(
    video_name,
    window_start_sec,
    window_end_sec
):

    interval = get_accident_interval(
        video_name
    )

    # --------------------------------------------------------
    # NORMAL VIDEO
    # --------------------------------------------------------

    if interval is None:

        return 0


    accident_start, accident_end = (
        interval
    )


    window_duration = (
        window_end_sec
        -
        window_start_sec
    )

    if window_duration <= 0:

        return 0


    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    overlap_start = max(
        window_start_sec,
        accident_start
    )

    overlap_end = min(
        window_end_sec,
        accident_end
    )

    overlap = max(
        0.0,
        overlap_end
        -
        overlap_start
    )


    overlap_ratio = (
        overlap
        /
        window_duration
    )


    return int(
        overlap_ratio
        >=
        MIN_POSITIVE_OVERLAP
    )


# ============================================================
# VIDEO LIST
# ============================================================

videos = []

for directory in [
    ACCIDENT_DIR,
    NORMAL_DIR
]:

    if not os.path.isdir(
        directory
    ):

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

            videos.append(
                os.path.join(
                    directory,
                    filename
                )
            )


print()
print("============================================================")
print("        RESQTRACK TEMPORAL FEATURE EXTRACTION V4")
print("============================================================")
print()

print(
    "Videos found:",
    len(videos)
)

print(
    "Window size:",
    WINDOW_FRAMES,
    "frames"
)

print(
    "Stride:",
    STRIDE_FRAMES,
    "frames"
)

print(
    "Accident buffer:",
    ACCIDENT_BUFFER_SEC,
    "seconds"
)

print(
    "Minimum positive overlap:",
    MIN_POSITIVE_OVERLAP
)

print(
    "Feature count:",
    len(FEATURES)
)

print()


# ============================================================
# PROCESS VIDEOS
# ============================================================

all_windows = []


for video_index, video_path in enumerate(
    videos,
    start=1
):

    video_name = os.path.basename(
        video_path
    )

    print(
        f"[{video_index}/{len(videos)}] "
        f"{video_name}"
    )


    # --------------------------------------------------------
    # RESET TRACKING
    # --------------------------------------------------------

    reset_vehicle_history()


    # --------------------------------------------------------
    # FRESH YOLO INSTANCE
    #
    # This prevents tracker state from one video
    # leaking into another.
    # --------------------------------------------------------

    model = YOLO(
        MODEL_PATH
    )


    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        print(
            "ERROR: Could not open:",
            video_path
        )

        continue


    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0


    frame_features = []

    previous = None
    previous_previous = None

    frame_count = 0


    # ========================================================
    # FRAME PROCESSING
    # ========================================================

    while True:

        success, frame = (
            video.read()
        )

        if not success:
            break

        frame_count += 1


        # ----------------------------------------------------
        # YOLO + BYTE TRACK
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # VEHICLE EXTRACTION
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

                center_x = (
                    (x1 + x2)
                    /
                    2.0
                )

                center_y = (
                    (y1 + y2)
                    /
                    2.0
                )


                vehicles[
                    int(vehicle_id)
                ] = {

                    "center": (
                        float(center_x),
                        float(center_y)
                    ),

                    "box": box
                }


        # ----------------------------------------------------
        # CANONICAL FEATURE ENGINE
        # ----------------------------------------------------

        current_features = (
            extract_frame_features(

                vehicles,

                previous,

                previous_previous
            )
        )


        frame_features.append(
            current_features
        )


        # ----------------------------------------------------
        # UPDATE TEMPORAL HISTORY
        # ----------------------------------------------------

        previous_previous = previous

        previous = vehicles


    video.release()

    reset_vehicle_history()


    # ========================================================
    # TEMPORAL WINDOW GENERATION
    # ========================================================

    window_count = 0

    positive_count = 0

    negative_count = 0


    maximum_start = (
        len(frame_features)
        -
        WINDOW_FRAMES
        +
        1
    )


    for start_frame in range(
        0,
        max(
            0,
            maximum_start
        ),
        STRIDE_FRAMES
    ):

        end_frame = (
            start_frame
            +
            WINDOW_FRAMES
        )


        window_features = (
            frame_features[
                start_frame:end_frame
            ]
        )


        if len(
            window_features
        ) < WINDOW_FRAMES:

            continue


        # ----------------------------------------------------
        # AGGREGATE USING CANONICAL ENGINE
        # ----------------------------------------------------

        aggregated = (
            aggregate_window(
                window_features
            )
        )


        if aggregated is None:

            continue


        # ----------------------------------------------------
        # TIME BOUNDARIES
        # ----------------------------------------------------

        start_sec = (
            start_frame
            /
            fps
        )

        end_sec = (
            end_frame
            /
            fps
        )


        # ----------------------------------------------------
        # OVERLAP-BASED LABEL
        # ----------------------------------------------------

        label = label_window(

            video_name,

            start_sec,

            end_sec
        )


        if label == 1:

            positive_count += 1

        else:

            negative_count += 1


        # ----------------------------------------------------
        # OUTPUT ROW
        # ----------------------------------------------------

        row = {

            "video":
                video_name,

            "window_id":
                window_count,

            "window_start_sec":
                round(
                    start_sec,
                    3
                ),

            "window_end_sec":
                round(
                    end_sec,
                    3
                )
        }


        for feature in FEATURES:

            value = aggregated[
                feature
            ]

            if pd.isna(value):

                value = 0.0

            if not np.isfinite(
                value
            ):

                value = 0.0

            row[feature] = float(
                value
            )


        row["label"] = label


        all_windows.append(
            row
        )


        window_count += 1


    print(
        f"    Frames: {frame_count}"
    )

    print(
        f"    Windows: {window_count}"
    )

    print(
        f"    Accident windows: "
        f"{positive_count}"
    )

    print(
        f"    Normal windows: "
        f"{negative_count}"
    )

    print()


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_windows
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

expected_columns = [

    "video",
    "window_id",
    "window_start_sec",
    "window_end_sec",

    *FEATURES,

    "label"
]


missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]


extra_columns = [
    column
    for column in df.columns
    if column not in expected_columns
]


if missing_columns:

    raise ValueError(
        "Missing columns: "
        +
        str(missing_columns)
    )


if extra_columns:

    raise ValueError(
        "Unexpected columns: "
        +
        str(extra_columns)
    )


# ============================================================
# CLEAN NUMERICAL FEATURES
# ============================================================

df[FEATURES] = (
    df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("============================================================")
print("       V4 TEMPORAL FEATURE EXTRACTION COMPLETE")
print("============================================================")
print()

print(
    "Total windows:",
    len(df)
)

print(
    "Videos represented:",
    df["video"].nunique()
)

print(
    "Feature count:",
    len(FEATURES)
)

print()

print(
    "Window distribution:"
)

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print()

print(
    "Positive windows by video:"
)

positive_by_video = (
    df.groupby("video")["label"]
    .sum()
)

print(
    positive_by_video
    .to_string()
)

print()


# ============================================================
# SAVE DATASET
# ============================================================

output_path = os.path.abspath(
    OUTPUT_FILE
)

df.to_csv(
    output_path,
    index=False
)

print()
print(
    "ACTUAL OUTPUT PATH:"
)

print(
    output_path
)

print(
    "FILE EXISTS:",
    os.path.exists(output_path)
)

print(
    "Saved as:",
    OUTPUT_FILE
)

print()

print(
    "Labeling method:"
)

print(
    "Buffered annotation + temporal overlap"
)

print()

print(
    "Buffer:",
    ACCIDENT_BUFFER_SEC,
    "seconds"
)

print(
    "Required overlap:",
    MIN_POSITIVE_OVERLAP * 100,
    "%"
)

print()