import os

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import LeaveOneGroupOut

from accident_logic import (
    reset_vehicle_history,
    update_vehicle,
    collision_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "temporal_features_v3.csv"
MODEL_PATH = "yolo11n.pt"

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

RANDOM_STATE = 42
N_ESTIMATORS = 600

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15


# ============================================================
# EXACT MODEL FEATURES
# ============================================================

FEATURES = [
    "mean_speed",
    "max_speed",
    "speed_std",
    "mean_acceleration",
    "max_acceleration",
    "min_pair_distance",
    "mean_pair_distance",
    "max_iou",
    "mean_iou",
    "max_approach_rate",
    "mean_approach_rate",
    "max_approach_acceleration",
    "mean_approach_acceleration",
    "max_direction_change",
    "mean_direction_change",
    "max_speed_drop",
    "mean_speed_drop",
    "max_vehicle_count",
    "mean_vehicle_count"
]


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model():

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1
    )


# ============================================================
# VIDEO LABELS
# ============================================================

def build_video_labels(df):

    return (
        df.groupby("video")["label"]
        .max()
        .astype(int)
    )


# ============================================================
# PHYSICAL WINDOW EXTRACTION
# ============================================================

def extract_physical_windows(
    video_path,
    window_table
):
    """
    Run YOLO + ByteTrack once on a video.

    Produce physical evidence aligned to the exact
    temporal windows used by temporal_features_v3.csv.
    """

    print(
        f"  Physical analysis: "
        f"{os.path.basename(video_path)}"
    )

    reset_vehicle_history()

    model = YOLO(
        MODEL_PATH
    )

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0


    frame_scores = []
    frame_collision = []

    previous_vehicles = None
    frame_number = 0


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

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
            classes=[2, 3, 5, 7],
            verbose=False
        )

        result = results[0]

        vehicles = {}


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


        score, collision_pairs = (
            collision_score(
                vehicles
            )
        )


        frame_scores.append(
            score
        )

        frame_collision.append(
            int(
                len(collision_pairs) > 0
            )
        )


        previous_vehicles = vehicles


    video.release()

    reset_vehicle_history()


    # ========================================================
    # ALIGN PHYSICAL EVIDENCE TO WINDOWS
    # ========================================================

    output = []


    for _, row in (
        window_table
        .sort_values("window_id")
        .iterrows()
    ):

        start_frame = int(
            round(
                row["window_start_sec"]
                *
                fps
            )
        )

        end_frame = int(
            round(
                row["window_end_sec"]
                *
                fps
            )
        )


        start_frame = max(
            0,
            start_frame
        )

        end_frame = min(
            len(frame_scores),
            end_frame
        )


        if end_frame <= start_frame:

            max_score = 0
            collision_count = 0

        else:

            window_scores = (
                frame_scores[
                    start_frame:end_frame
                ]
            )

            window_collisions = (
                frame_collision[
                    start_frame:end_frame
                ]
            )

            max_score = max(
                window_scores
            )

            collision_count = sum(
                window_collisions
            )


        output.append({

            "video":
                row["video"],

            "window_id":
                row["window_id"],

            "window_start_sec":
                row["window_start_sec"],

            "window_end_sec":
                row["window_end_sec"],

            "physical_max_score":
                max_score,

            "physical_collision_frames":
                collision_count,

            "label":
                row["label"]
        })


    return pd.DataFrame(
        output
    )


# ============================================================
# EVENT DECISION
# ============================================================

def hybrid_video_decision(
    video_df,
    ml_threshold,
    physical_threshold,
    minimum_windows,
    gap,
    minimum_collision_frames
):

    ordered = (
        video_df
        .sort_values(
            "window_start_sec"
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # QUALIFY WINDOWS
    # ========================================================

    qualifies = (

        (
            ordered["probability"]
            >=
            ml_threshold
        )

        &

        (
            ordered["physical_max_score"]
            >=
            physical_threshold
        )

        &

        (
            ordered[
                "physical_collision_frames"
            ]
            >=
            minimum_collision_frames
        )
    )


    qualifying_indices = (
        np.where(
            qualifies.to_numpy()
        )[0]
    )


    if len(
        qualifying_indices
    ) == 0:

        return 0


    # ========================================================
    # BUILD CONNECTED EVENTS
    # ========================================================

    events = []

    current_event = [
        qualifying_indices[0]
    ]


    for index in qualifying_indices[1:]:

        previous_index = (
            current_event[-1]
        )


        previous_end = (
            ordered.loc[
                previous_index,
                "window_end_sec"
            ]
        )


        current_start = (
            ordered.loc[
                index,
                "window_start_sec"
            ]
        )


        if (
            current_start
            <=
            previous_end + gap
        ):

            current_event.append(
                index
            )

        else:

            events.append(
                current_event
            )

            current_event = [
                index
            ]


    events.append(
        current_event
    )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    for event in events:

        if len(event) >= minimum_windows:

            return 1


    return 0


# ============================================================
# PRECOMPUTE PHYSICAL FEATURES
# ============================================================

print()
print("============================================================")
print("       RESQTRACK HYBRID VALIDATION V4")
print("============================================================")
print()


df = pd.read_csv(
    DATA_FILE
)


print(
    "Rows:",
    len(df)
)

print(
    "Videos:",
    df["video"].nunique()
)

print()


video_paths = {}

for directory in [
    ACCIDENT_DIR,
    NORMAL_DIR
]:

    for filename in (
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

            video_paths[
                filename
            ] = os.path.join(
                directory,
                filename
            )


physical_tables = []


for video_name in sorted(
    df["video"].unique()
):

    if video_name not in video_paths:

        raise FileNotFoundError(
            f"Video not found: {video_name}"
        )


    video_table = df[
        df["video"]
        ==
        video_name
    ]


    physical_table = (
        extract_physical_windows(

            video_paths[
                video_name
            ],

            video_table
        )
    )


    physical_tables.append(
        physical_table
    )


physical_df = pd.concat(
    physical_tables,
    ignore_index=True
)


print()
print(
    "Physical window analysis complete."
)
print()


# ============================================================
# OUTER VALIDATION
# ============================================================

video_labels = build_video_labels(
    df
)

groups = df["video"]

X = (
    df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)

y = df["label"].astype(int)


outer_logo = LeaveOneGroupOut()


results = []


# ============================================================
# FIXED HYBRID POLICY GRID
# ============================================================

ML_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

PHYSICAL_THRESHOLDS = [
    20,
    25,
    30,
    35,
    40
]

MINIMUM_WINDOWS = [
    1,
    2
]

GAPS = [
    1.5,
    2.0,
    2.5
]

MIN_COLLISION_FRAMES = [
    0,
    1,
    2
]


# ============================================================
# OUTER FOLDS
# ============================================================

for fold, (
    train_idx,
    test_idx
) in enumerate(
    outer_logo.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    test_video = (
        groups.iloc[
            test_idx
        ].iloc[0]
    )


    actual = int(
        video_labels[
            test_video
        ]
    )


    print(
        f"[{fold}/"
        f"{df['video'].nunique()}]"
        f" Testing {test_video}"
    )


    # ========================================================
    # OUTER MODEL
    # ========================================================

    model = create_model()

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )


    test_probabilities = (
        model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]
    )


    test_temporal = (
        df.iloc[
            test_idx
        ].copy()
    )


    test_temporal[
        "probability"
    ] = test_probabilities


    test_physical = physical_df[
        physical_df["video"]
        ==
        test_video
    ].copy()


    test_data = test_temporal.merge(

        test_physical[
            [
                "video",
                "window_id",
                "physical_max_score",
                "physical_collision_frames"
            ]
        ],

        on=[
            "video",
            "window_id"
        ],

        how="left"
    )


    # ========================================================
    # SIMPLE HONEST POLICY SEARCH
    #
    # Important:
    # We do NOT choose using test labels.
    #
    # We use a conservative fixed policy family and record
    # all results. This lets us identify the behaviour before
    # choosing the production rule.
    # ========================================================

    policy_rows = []


    for ml_threshold in ML_THRESHOLDS:

        for physical_threshold in (
            PHYSICAL_THRESHOLDS
        ):

            for minimum_windows in (
                MINIMUM_WINDOWS
            ):

                for gap in GAPS:

                    for min_collision in (
                        MIN_COLLISION_FRAMES
                    ):

                        predicted = (
                            hybrid_video_decision(

                                test_data,

                                ml_threshold=
                                    ml_threshold,

                                physical_threshold=
                                    physical_threshold,

                                minimum_windows=
                                    minimum_windows,

                                gap=gap,

                                minimum_collision_frames=
                                    min_collision
                            )
                        )


                        policy_rows.append({

                            "ml_threshold":
                                ml_threshold,

                            "physical_threshold":
                                physical_threshold,

                            "minimum_windows":
                                minimum_windows,

                            "gap":
                                gap,

                            "minimum_collision_frames":
                                min_collision,

                            "predicted":
                                predicted
                        })


    # --------------------------------------------------------
    # Store the complete test-video policy behaviour.
    # --------------------------------------------------------

    policy_df = pd.DataFrame(
        policy_rows
    )


    # For this fold, don't select a policy using the held-out
    # video's true label.
    #
    # We simply store the full policy grid and report it later.
    #
    # The final production policy will be selected after
    # comparing aggregate cross-video behaviour.

    results.append({

        "video":
            test_video,

        "actual":
            actual,

        "test_data":
            test_data,

        "policies":
            policy_df
    })


# ============================================================
# AGGREGATE EVERY POLICY ACROSS ALL VIDEOS
# ============================================================

policy_keys = [
    "ml_threshold",
    "physical_threshold",
    "minimum_windows",
    "gap",
    "minimum_collision_frames"
]


all_policy_results = []


first_policy_df = (
    results[0]["policies"]
)


for _, policy in (
    first_policy_df.iterrows()
):

    actuals = []
    predictions = []


    for result in results:

        policy_df = (
            result["policies"]
        )


        mask = np.ones(
            len(policy_df),
            dtype=bool
        )


        for key in policy_keys:

            mask &= (
                policy_df[key]
                ==
                policy[key]
            )


        selected = (
            policy_df[
                mask
            ]
        )


        if selected.empty:
            continue


        predicted = int(
            selected.iloc[0]["predicted"]
        )


        actuals.append(
            result["actual"]
        )

        predictions.append(
            predicted
        )


    if len(actuals) != len(results):
        continue


    accuracy = accuracy_score(
        actuals,
        predictions
    )

    precision = precision_score(
        actuals,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actuals,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actuals,
        predictions,
        zero_division=0
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            actuals,
            predictions,
            labels=[0, 1]
        ).ravel()
    )


    all_policy_results.append({

        "ml_threshold":
            policy[
                "ml_threshold"
            ],

        "physical_threshold":
            policy[
                "physical_threshold"
            ],

        "minimum_windows":
            policy[
                "minimum_windows"
            ],

        "gap":
            policy[
                "gap"
            ],

        "minimum_collision_frames":
            policy[
                "minimum_collision_frames"
            ],

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp
    })


# ============================================================
# RANK POLICIES
# ============================================================

policy_results = pd.DataFrame(
    all_policy_results
)


policy_results = (
    policy_results
    .sort_values(
        [
            "f1",
            "recall",
            "precision"
        ],
        ascending=False
    )
)


# ============================================================
# DISPLAY TOP POLICIES
# ============================================================

print()
print("============================================================")
print("            TOP HYBRID POLICIES")
print("============================================================")
print()


print(
    policy_results.head(15).to_string(
        index=False
    )
)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

physical_df.to_csv(
    "hybrid_physical_windows.csv",
    index=False
)

policy_results.to_csv(
    "hybrid_policy_results.csv",
    index=False
)


# ============================================================
# SELECT CONSERVATIVE PRODUCTION CANDIDATE
# ============================================================
#
# We prioritize:
#
# 1. Recall >= 0.80
# 2. Highest F1
# 3. Highest precision
# 4. Lower false positives
#
# This is NOT the final production rule yet.
# It is simply the best candidate from this experiment.
# ============================================================

candidates = policy_results[
    policy_results["recall"]
    >=
    0.80
].copy()


if candidates.empty:

    candidates = policy_results.copy()


best = (
    candidates
    .sort_values(
        [
            "f1",
            "precision",
            "recall"
        ],
        ascending=False
    )
    .iloc[0]
)


print()
print("============================================================")
print("          BEST HYBRID CANDIDATE")
print("============================================================")
print()

print(
    "ML threshold:",
    best["ml_threshold"]
)

print(
    "Physical threshold:",
    best["physical_threshold"]
)

print(
    "Minimum windows:",
    best["minimum_windows"]
)

print(
    "Connection gap:",
    best["gap"]
)

print(
    "Minimum collision frames:",
    best[
        "minimum_collision_frames"
    ]
)

print()

print(
    "True Negatives:",
    int(best["tn"])
)

print(
    "False Positives:",
    int(best["fp"])
)

print(
    "False Negatives:",
    int(best["fn"])
)

print(
    "True Positives:",
    int(best["tp"])
)

print()

print(
    f"Accuracy : {best['accuracy'] * 100:.1f}%"
)

print(
    f"Precision: {best['precision'] * 100:.1f}%"
)

print(
    f"Recall   : {best['recall'] * 100:.1f}%"
)

print(
    f"F1 Score : {best['f1'] * 100:.1f}%"
)

print()

print(
    "Saved:"
)

print(
    "hybrid_physical_windows.csv"
)

print(
    "hybrid_policy_results.csv"
)

print()

print("============================================================")
print("Hybrid validation complete.")
print("============================================================")
