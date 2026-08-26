import os
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "temporal_features_v3.csv"

MODEL_PATH = "yolo11n.pt"

ACCIDENT_DIR = "dataset/accident"
NORMAL_DIR = "dataset/normal"

RANDOM_STATE = 42
N_ESTIMATORS = 600

ML_THRESHOLD = 0.30

WINDOW_GAP = 1.5


# ============================================================
# FEATURES
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
# LOAD TEMPORAL DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

X = (
    df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)

y = df["label"].astype(int)

groups = df["video"]


# ============================================================
# MODEL
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
# BUILD OOF PROBABILITIES
# ============================================================

print()
print("============================================================")
print("      RESQTRACK OOF EVENT GENERATION")
print("============================================================")
print()


logo = LeaveOneGroupOut()

oof_rows = []


for fold, (
    train_idx,
    test_idx
) in enumerate(
    logo.split(
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

    print(
        f"[{fold}/10] "
        f"Generating OOF probabilities: "
        f"{test_video}"
    )


    model = create_model()

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )


    probabilities = (
        model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]
    )


    fold_df = df.iloc[
        test_idx
    ].copy()

    fold_df["probability"] = (
        probabilities
    )

    oof_rows.append(
        fold_df
    )


# ============================================================
# COMBINE
# ============================================================

oof = pd.concat(
    oof_rows,
    ignore_index=True
)


oof = oof.sort_values(
    [
        "video",
        "window_start_sec"
    ]
).reset_index(
    drop=True
)


# ============================================================
# BUILD EVENTS
# ============================================================

events = []


for video_name, video_df in (
    oof.groupby("video")
):

    video_df = (
        video_df
        .sort_values(
            "window_start_sec"
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Suspicious windows
    # --------------------------------------------------------

    suspicious = (
        video_df["probability"]
        >=
        ML_THRESHOLD
    )


    indices = np.where(
        suspicious.to_numpy()
    )[0]


    if len(indices) == 0:

        continue


    # --------------------------------------------------------
    # Group connected windows
    # --------------------------------------------------------

    current = [
        indices[0]
    ]

    grouped = []


    for index in indices[1:]:

        previous = current[-1]

        previous_end = (
            video_df.loc[
                previous,
                "window_end_sec"
            ]
        )

        current_start = (
            video_df.loc[
                index,
                "window_start_sec"
            ]
        )


        if (
            current_start
            <=
            previous_end
            +
            WINDOW_GAP
        ):

            current.append(
                index
            )

        else:

            grouped.append(
                current
            )

            current = [
                index
            ]


    grouped.append(
        current
    )


    # ========================================================
    # EXTRACT EVENT FEATURES
    # ========================================================

    for event_id, event_indices in enumerate(
        grouped
    ):

        event_df = (
            video_df
            .iloc[event_indices]
            .copy()
        )


        probabilities = (
            event_df[
                "probability"
            ]
            .to_numpy()
        )


        start_time = float(
            event_df[
                "window_start_sec"
            ].min()
        )


        end_time = float(
            event_df[
                "window_end_sec"
            ].max()
        )


        duration = (
            end_time
            -
            start_time
        )


        peak_probability = float(
            probabilities.max()
        )


        mean_probability = float(
            probabilities.mean()
        )


        probability_std = float(
            probabilities.std()
        )


        window_count = len(
            event_df
        )


        # ----------------------------------------------------
        # PROBABILITY TREND
        # ----------------------------------------------------

        if len(probabilities) >= 2:

            probability_rise = float(
                probabilities.max()
                -
                probabilities[0]
            )

            probability_fall = float(
                probabilities.max()
                -
                probabilities[-1]
            )

        else:

            probability_rise = 0.0
            probability_fall = 0.0


        # ----------------------------------------------------
        # APPROXIMATE PROBABILITY AREA
        # ----------------------------------------------------

        probability_area = float(
            np.trapezoid(
                probabilities
            )
        )


        # ----------------------------------------------------
        # LABEL
        #
        # An event is ACCIDENT if any of its windows
        # overlap the annotated accident windows.
        # ----------------------------------------------------

        event_label = int(
            event_df["label"].max()
        )


        # ----------------------------------------------------
        # ORIGINAL TEMPORAL FEATURES
        #
        # Aggregate behaviour inside the event.
        # ----------------------------------------------------

        event_features = {

            "event_peak_probability":
                peak_probability,

            "event_mean_probability":
                mean_probability,

            "event_probability_std":
                probability_std,

            "event_window_count":
                window_count,

            "event_duration":
                duration,

            "event_probability_rise":
                probability_rise,

            "event_probability_fall":
                probability_fall,

            "event_probability_area":
                probability_area,


            "event_max_speed":
                event_df[
                    "max_speed"
                ].max(),

            "event_mean_speed":
                event_df[
                    "mean_speed"
                ].mean(),

            "event_speed_std":
                event_df[
                    "speed_std"
                ].mean(),

            "event_max_acceleration":
                event_df[
                    "max_acceleration"
                ].max(),

            "event_max_iou":
                event_df[
                    "max_iou"
                ].max(),

            "event_min_distance":
                event_df[
                    "min_pair_distance"
                ].min(),

            "event_max_approach_rate":
                event_df[
                    "max_approach_rate"
                ].max(),

            "event_max_approach_acceleration":
                event_df[
                    "max_approach_acceleration"
                ].max(),

            "event_max_direction_change":
                event_df[
                    "max_direction_change"
                ].max(),

            "event_max_speed_drop":
                event_df[
                    "max_speed_drop"
                ].max(),

            "event_max_vehicle_count":
                event_df[
                    "max_vehicle_count"
                ].max(),

            "event_mean_vehicle_count":
                event_df[
                    "mean_vehicle_count"
                ].mean()
        }


        events.append({

            "video":
                video_name,

            "event_id":
                event_id,

            "event_start_sec":
                start_time,

            "event_end_sec":
                end_time,

            "label":
                event_label,

            **event_features
        })


# ============================================================
# SAVE
# ============================================================

events_df = pd.DataFrame(
    events
)


events_df.to_csv(
    "event_oof_v4.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("============================================================")
print("           OOF EVENT DATASET COMPLETE")
print("============================================================")
print()

print(
    "Events generated:",
    len(events_df)
)

print(
    "Videos represented:",
    events_df["video"].nunique()
)

print()

print(
    "Event label distribution:"
)

print(
    events_df["label"]
    .value_counts()
    .sort_index()
)

print()

print(
    "Saved as:"
)

print(
    "event_oof_v4.csv"
)

print()

print(
    "Columns:"
)

for column in events_df.columns:

    print(
        " ",
        column
    )

print()