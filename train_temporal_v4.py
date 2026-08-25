import sys
import platform

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "temporal_features_v3.csv"
MODEL_FILE = "resqtrack_temporal_model_v4.pkl"

RANDOM_STATE = 42
N_ESTIMATORS = 600


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
# VIDEO LABELS
# ============================================================

def build_video_labels(df):
    """
    A video is ACCIDENT if it contains at least one
    accident-labelled temporal window.

    Therefore:
        video label = max(window labels)
    """

    return (
        df.groupby("video")["label"]
        .max()
        .astype(int)
    )


# ============================================================
# EVENT DECISION
# ============================================================

def video_decision(
    video_df,
    threshold,
    minimum_windows=2,
    connection_gap=1.5
):

    ordered = (
        video_df
        .sort_values("window_start_sec")
        .reset_index(drop=True)
    )

    if ordered.empty:
        return 0

    probabilities = (
        ordered["probability"]
        .to_numpy()
    )

    high_indices = np.where(
        probabilities >= threshold
    )[0]

    if len(high_indices) == 0:
        return 0

    # A very strong single window is enough.
    if probabilities.max() >= 0.75:
        return 1

    events = []

    current_event = [
        high_indices[0]
    ]

    for index in high_indices[1:]:

        previous_index = current_event[-1]

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
            previous_end + connection_gap
        ):

            current_event.append(index)

        else:

            events.append(current_event)

            current_event = [
                index
            ]

    events.append(current_event)

    for event in events:

        if len(event) >= minimum_windows:
            return 1

    return 0


# ============================================================
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("         RESQTRACK TEMPORAL MODEL V4")
print("============================================================")
print()

df = pd.read_csv(DATA_FILE)

print("Rows   :", len(df))
print("Videos :", df["video"].nunique())
print("Features:", len(FEATURES))
print()


# ============================================================
# PREPARE DATA
# ============================================================

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

video_labels = build_video_labels(df)


# ============================================================
# CALIBRATION GRID
# ============================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

minimum_windows_options = [
    1,
    2
]

connection_gaps = [
    1.5,
    2.0,
    2.5,
    3.0
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
# NESTED VIDEO-LEVEL VALIDATION
# ============================================================

outer_logo = LeaveOneGroupOut()

video_results = []


for outer_fold, (
    outer_train_idx,
    outer_test_idx
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
            outer_test_idx
        ].iloc[0]
    )

    actual_test_label = int(
        video_labels[test_video]
    )

    print(
        f"[{outer_fold}/"
        f"{df['video'].nunique()}]"
        f" Testing: {test_video}"
    )


    # ========================================================
    # OUTER TRAIN DATA
    # ========================================================

    outer_train_df = df.iloc[
        outer_train_idx
    ].copy()

    outer_train_X = X.iloc[
        outer_train_idx
    ]

    outer_train_y = y.iloc[
        outer_train_idx
    ]

    outer_train_groups = groups.iloc[
        outer_train_idx
    ]


    # ========================================================
    # INNER OOF PROBABILITIES
    #
    # Used ONLY for calibration.
    # ========================================================

    inner_logo = LeaveOneGroupOut()

    calibration_rows = []

    for (
        inner_train_idx,
        inner_validation_idx
    ) in inner_logo.split(
        outer_train_X,
        outer_train_y,
        groups=outer_train_groups
    ):

        inner_model = create_model()

        inner_model.fit(
            outer_train_X.iloc[
                inner_train_idx
            ],
            outer_train_y.iloc[
                inner_train_idx
            ]
        )

        probabilities = (
            inner_model.predict_proba(
                outer_train_X.iloc[
                    inner_validation_idx
                ]
            )[:, 1]
        )

        validation_rows = (
            outer_train_df.iloc[
                inner_validation_idx
            ].copy()
        )

        validation_rows[
            "probability"
        ] = probabilities

        calibration_rows.append(
            validation_rows
        )


    calibration_df = pd.concat(
        calibration_rows,
        ignore_index=True
    )


    # ========================================================
    # CORRECT VIDEO LABELS
    #
    # IMPORTANT: use MAX, not FIRST.
    # ========================================================

    calibration_labels = (
        calibration_df
        .groupby("video")["label"]
        .max()
        .astype(int)
    )


    # ========================================================
    # CALIBRATION SEARCH
    # ========================================================

    best_config = None

    best_key = None


    for threshold in thresholds:

        for minimum_windows in (
            minimum_windows_options
        ):

            for connection_gap in (
                connection_gaps
            ):

                actuals = []
                predictions = []


                for video_name in (
                    calibration_labels.index
                ):

                    video_data = (
                        calibration_df[
                            calibration_df["video"]
                            ==
                            video_name
                        ]
                    )

                    actual = int(
                        calibration_labels[
                            video_name
                        ]
                    )

                    predicted = video_decision(

                        video_data,

                        threshold=threshold,

                        minimum_windows=
                            minimum_windows,

                        connection_gap=
                            connection_gap
                    )

                    actuals.append(actual)
                    predictions.append(predicted)


                score_f1 = f1_score(
                    actuals,
                    predictions,
                    zero_division=0
                )

                score_recall = recall_score(
                    actuals,
                    predictions,
                    zero_division=0
                )

                score_precision = precision_score(
                    actuals,
                    predictions,
                    zero_division=0
                )

                # Prefer:
                # 1. F1
                # 2. Recall
                # 3. Precision

                key = (
                    score_f1,
                    score_recall,
                    score_precision
                )

                if (
                    best_key is None
                    or
                    key > best_key
                ):

                    best_key = key

                    best_config = {

                        "threshold":
                            threshold,

                        "minimum_windows":
                            minimum_windows,

                        "connection_gap":
                            connection_gap
                    }


    # ========================================================
    # TRAIN OUTER MODEL
    # ========================================================

    outer_model = create_model()

    outer_model.fit(
        outer_train_X,
        outer_train_y
    )


    # ========================================================
    # HELD-OUT VIDEO
    # ========================================================

    test_X = X.iloc[
        outer_test_idx
    ]

    test_probabilities = (
        outer_model
        .predict_proba(test_X)
        [:, 1]
    )

    test_df = df.iloc[
        outer_test_idx
    ].copy()

    test_df[
        "probability"
    ] = test_probabilities


    predicted_test_label = (
        video_decision(

            test_df,

            threshold=
                best_config["threshold"],

            minimum_windows=
                best_config["minimum_windows"],

            connection_gap=
                best_config["connection_gap"]
        )
    )


    max_probability = float(
        test_probabilities.max()
    )


    high_windows = int(
        (
            test_probabilities
            >=
            best_config["threshold"]
        ).sum()
    )


    print(
        "    Actual      :",
        "ACCIDENT"
        if actual_test_label
        else
        "NORMAL"
    )

    print(
        "    Predicted   :",
        "ACCIDENT"
        if predicted_test_label
        else
        "NORMAL"
    )

    print(
        "    Max prob.   :",
        f"{max_probability:.3f}"
    )

    print(
        "    Threshold   :",
        best_config[
            "threshold"
        ]
    )

    print(
        "    Min windows :",
        best_config[
            "minimum_windows"
        ]
    )

    print(
        "    Gap         :",
        best_config[
            "connection_gap"
        ]
    )

    print()


    video_results.append({

        "video":
            test_video,

        "actual":
            actual_test_label,

        "predicted":
            predicted_test_label,

        "max_probability":
            max_probability,

        "high_windows":
            high_windows,

        "threshold":
            best_config[
                "threshold"
            ],

        "minimum_windows":
            best_config[
                "minimum_windows"
            ],

        "connection_gap":
            best_config[
                "connection_gap"
            ]
    })


# ============================================================
# METRICS
# ============================================================

results_df = pd.DataFrame(
    video_results
)

actual = results_df["actual"]
predicted = results_df["predicted"]


accuracy = accuracy_score(
    actual,
    predicted
)

precision = precision_score(
    actual,
    predicted,
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)


cm = confusion_matrix(
    actual,
    predicted,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# RESULTS
# ============================================================

print()
print("============================================================")
print("            V4 VIDEO-LEVEL RESULTS")
print("============================================================")
print()

print("CONFUSION MATRIX")
print("----------------")

print(
    f"True Negatives : {tn}"
)

print(
    f"False Positives: {fp}"
)

print(
    f"False Negatives: {fn}"
)

print(
    f"True Positives : {tp}"
)

print()

print("METRICS")
print("-------")

print(
    f"Accuracy : {accuracy:.3f}"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall   : {recall:.3f}"
)

print(
    f"F1 Score : {f1:.3f}"
)

print()

print("PERCENTAGE")
print("----------")

print(
    f"Accuracy : {accuracy * 100:.1f}%"
)

print(
    f"Precision: {precision * 100:.1f}%"
)

print(
    f"Recall   : {recall * 100:.1f}%"
)

print(
    f"F1 Score : {f1 * 100:.1f}%"
)

print()

print("PER-VIDEO RESULTS")
print("-----------------")

print(
    results_df.to_string(
        index=False
    )
)

print()


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================

print(
    "Training final V4 model on complete dataset..."
)

final_model = create_model()

final_model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "model":
        final_model,

    "features":
        FEATURES,

    "window_frames":
        30,

    "stride_frames":
        15,

    "model_type":
        "RandomForest",

    "validation_method":
        "Nested Leave-One-Video-Out",

    "validation_accuracy":
        float(accuracy),

    "validation_precision":
        float(precision),

    "validation_recall":
        float(recall),

    "validation_f1":
        float(f1),

    "videos":
        sorted(
            df["video"]
            .unique()
            .tolist()
        ),

    "python_version":
        sys.version,

    "platform":
        platform.platform()
    }


joblib.dump(
    model_package,
    MODEL_FILE
)


print()
print("============================================================")
print("             V4 TRAINING COMPLETE")
print("============================================================")
print()

print(
    "Model saved as:"
)

print(
    MODEL_FILE
)

print()

print(
    "Validation method:"
)

print(
    "Nested Leave-One-Video-Out"
)

print()