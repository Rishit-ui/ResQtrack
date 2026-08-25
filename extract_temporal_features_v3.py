import os
import cv2
import pandas as pd

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

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

ANNOTATIONS_FILE = "annotations.csv"

OUTPUT_FILE = "temporal_features_v3.csv"


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

annotations = pd.read_csv(
    ANNOTATIONS_FILE
)

print()
print("==============================================")
print("      RESQTRACK V3 FEATURE EXTRACTION")
print("==============================================")
print()

print(
    "Annotations loaded:",
    len(annotations)
)

print(
    "Features:",
    len(FEATURES)
)

print(
    "Window:",
    WINDOW_FRAMES,
    "frames"
)

print(
    "Stride:",
    STRIDE_FRAMES,
    "frames"
)

print()


# ============================================================
# HELPER
# ============================================================

def get_accident_interval(
    video_name
):

    rows = annotations[
        annotations["video"]
        ==
        video_name
    ]

    if rows.empty:

        return None

    row = rows.iloc[0]

    if int(row["label"]) != 1:

        return None

    return (
        float(row["start_sec"]),
        float(row["end_sec"])
    )


def window_label(
    video_name,
    start_sec,
    end_sec
):

    interval = get_accident_interval(
        video_name
    )

    # Normal video
    if interval is None:

        return 0

    accident_start, accident_end = (
        interval
    )

    # --------------------------------------------------------
    # Use the window midpoint for labeling.
    #
    # This prevents windows immediately before/after the
    # accident from being incorrectly labelled as accidents.
    # --------------------------------------------------------

    midpoint = (
        start_sec + end_sec
    ) / 2.0

    if (
        accident_start
        <= midpoint
        <= accident_end
    ):

        return 1

    return 0


# ============================================================
# VIDEO LIST
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

        raise FileNotFoundError(
            f"Missing dataset directory: "
            f"{directory}"
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
                (
                    os.path.join(
                        directory,
                        filename
                    ),
                    label
                )
            )


print(
    "Videos found:",
    len(videos)
)

print()


# ============================================================
# PROCESS VIDEOS
# ============================================================

all_windows = []

for video_index, (
    video_path,
    video_label
) in enumerate(
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
    # RESET TRACKING STATE
    # --------------------------------------------------------

    reset_vehicle_history()


    # --------------------------------------------------------
    # FRESH YOLO MODEL FOR EACH VIDEO
    #
    # Prevents tracker state leaking across videos.
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
    # FRAME LOOP
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
            classes=[2, 3, 5, 7],
            verbose=False
        )

        result = results[0]

        vehicles = {}


        # ----------------------------------------------------
        # EXTRACT VEHICLES
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


        # ----------------------------------------------------
        # CANONICAL FRAME FEATURES
        # ----------------------------------------------------

        features = (
            extract_frame_features(

                vehicles,

                previous,

                previous_previous
            )
        )

        frame_features.append(
            features
        )


        # ----------------------------------------------------
        # UPDATE HISTORY
        # ----------------------------------------------------

        previous_previous = previous
        previous = vehicles


    video.release()

    reset_vehicle_history()


    # ========================================================
    # TEMPORAL WINDOWS
    # ========================================================

    window_id = 0

    for start_frame in range(
        0,
        max(
            0,
            len(frame_features)
            -
            WINDOW_FRAMES
            +
            1
        ),
        STRIDE_FRAMES
    ):

        end_frame = (
            start_frame
            +
            WINDOW_FRAMES
        )

        window = frame_features[
            start_frame:end_frame
        ]


        if len(window) < WINDOW_FRAMES:

            continue


        # ----------------------------------------------------
        # AGGREGATE
        # ----------------------------------------------------

        aggregated = (
            aggregate_window(
                window
            )
        )


        if aggregated is None:

            continue


        start_sec = (
            start_frame / fps
        )

        end_sec = (
            end_frame / fps
        )


        label = window_label(

            video_name,

            start_sec,

            end_sec
        )


        row = {

            "video":
                video_name,

            "window_id":
                window_id,

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


        # ----------------------------------------------------
        # ADD EXACT MODEL FEATURES
        # ----------------------------------------------------

        for feature in FEATURES:

            row[feature] = (
                aggregated[feature]
            )


        row["label"] = label


        all_windows.append(
            row
        )

        window_id += 1


    print(
        "   Frames:",
        frame_count
    )

    print(
        "   Windows:",
        window_id
    )

    print()


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_windows
)


# ============================================================
# VALIDATION
# ============================================================

expected_columns = [

    "video",
    "window_id",
    "window_start_sec",
    "window_end_sec",

    *FEATURES,

    "label"
]


missing = [
    column
    for column in expected_columns
    if column not in df.columns
]


extra = [
    column
    for column in df.columns
    if column not in expected_columns
]


if missing:

    raise ValueError(
        "Missing columns: "
        +
        str(missing)
    )


if extra:

    raise ValueError(
        "Unexpected columns: "
        +
        str(extra)
    )


# ============================================================
# REMOVE NaN / INF
# ============================================================

feature_columns = list(
    FEATURES
)

df[feature_columns] = (
    df[feature_columns]
    .replace(
        [float("inf"), float("-inf")],
        0
    )
    .fillna(0)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("==============================================")
print("       V3 FEATURE EXTRACTION COMPLETE")
print("==============================================")
print()

print(
    "Total windows:",
    len(df)
)

print()

print(
    "Videos represented:",
    df["video"].nunique()
)

print()

print(
    "Window distribution:"
)

print(
    df["label"].value_counts()
    .sort_index()
)

print()

print(
    "Feature count:",
    len(FEATURES)
)

print()

print(
    "Saved as:",
    OUTPUT_FILE
)

print()

print(
    "Next stage:"
)

print(
    "Train V3 using VIDEO-LEVEL separation."
)

print()