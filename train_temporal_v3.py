import os
import sys
import platform
import sklearn

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "temporal_features_v3.csv"

MODEL_FILE = "resqtrack_temporal_model_v3.pkl"

RANDOM_STATE = 42

N_ESTIMATORS = 600


# ============================================================
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("          RESQTRACK TEMPORAL MODEL V3 TRAINING")
print("============================================================")
print()

print("Loading dataset...")

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


# ============================================================
# FEATURE LIST
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
# DATA VALIDATION
# ============================================================

missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing:

    raise ValueError(
        "Missing features: "
        +
        ", ".join(missing)
    )


required_columns = [
    "video",
    "label"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# CLEAN FEATURES
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


print("Feature count:", len(FEATURES))

print()

print("Window distribution:")

print(
    y.value_counts()
    .sort_index()
)

print()

print("Videos:")

print(
    df["video"]
    .nunique()
)

print()


# ============================================================
# VIDEO-LEVEL LEAVE-ONE-VIDEO-OUT VALIDATION
# ============================================================

logo = LeaveOneGroupOut()


all_window_predictions = []
all_window_probabilities = []

video_results = []


print("============================================================")
print("       LEAVE-ONE-VIDEO-OUT VALIDATION")
print("============================================================")
print()


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

    train_X = X.iloc[
        train_idx
    ]

    train_y = y.iloc[
        train_idx
    ]

    test_X = X.iloc[
        test_idx
    ]

    test_y = y.iloc[
        test_idx
    ]

    test_videos = (
        groups.iloc[
            test_idx
        ]
        .unique()
    )

    test_video = (
        test_videos[0]
    )


    print(
        f"[{fold}/{df['video'].nunique()}] "
        f"Testing: {test_video}"
    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        class_weight="balanced_subsample",

        min_samples_leaf=2,

        max_features="sqrt",

        n_jobs=-1
    )


    model.fit(
        train_X,
        train_y
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            test_X
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)


    # --------------------------------------------------------
    # WINDOW METRICS
    # --------------------------------------------------------

    window_accuracy = accuracy_score(
        test_y,
        predictions
    )


    # --------------------------------------------------------
    # VIDEO-LEVEL AGGREGATION
    # --------------------------------------------------------

    test_df = df.iloc[
        test_idx
    ].copy()

    test_df["probability"] = (
        probabilities
    )

    test_df["prediction"] = (
        predictions
    )


    actual_video_label = int(
        test_y.max()
    )


    max_probability = float(
        test_df["probability"].max()
    )


    # --------------------------------------------------------
    # HIGH-PROBABILITY WINDOWS
    # --------------------------------------------------------

    high_probability = (
        test_df["probability"]
        >= 0.60
    )


    high_probability_count = int(
        high_probability.sum()
    )


    # --------------------------------------------------------
    # EVENT RUNS
    #
    # We don't want one isolated high probability
    # window to define an accident.
    # --------------------------------------------------------

    ordered = test_df.sort_values(
        "window_start_sec"
    ).reset_index(
        drop=True
    )

    event_windows = []

    current_run = []


    for i in range(
        len(ordered)
    ):

        probability = (
            ordered.loc[
                i,
                "probability"
            ]
        )

        if probability >= 0.60:

            if current_run:

                previous_end = (
                    ordered.loc[
                        current_run[-1],
                        "window_end_sec"
                    ]
                )

                current_start = (
                    ordered.loc[
                        i,
                        "window_start_sec"
                    ]
                )

                # Windows are considered connected
                # when they overlap or are close in time.
                if (
                    current_start
                    <=
                    previous_end + 1.0
                ):

                    current_run.append(i)

                else:

                    event_windows.append(
                        current_run
                    )

                    current_run = [i]

            else:

                current_run = [i]

        else:

            if current_run:

                event_windows.append(
                    current_run
                )

                current_run = []


    if current_run:

        event_windows.append(
            current_run
        )


    # --------------------------------------------------------
    # VIDEO EVENT DECISION
    #
    # Require at least 2 connected high-probability
    # windows to reduce isolated false alarms.
    # --------------------------------------------------------

    predicted_video = int(
        any(
            len(run) >= 2
            for run in event_windows
        )
    )


    # --------------------------------------------------------
    # SAFETY OVERRIDE
    #
    # If the video has a large cluster of high-probability
    # windows, accept it even if only one run was created.
    # --------------------------------------------------------

    if (
        high_probability_count >= 3
    ):

        predicted_video = 1


    # --------------------------------------------------------
    # VIDEO RESULT
    # --------------------------------------------------------

    video_results.append({

        "video":
            test_video,

        "actual":
            actual_video_label,

        "predicted":
            predicted_video,

        "max_probability":
            max_probability,

        "high_probability_windows":
            high_probability_count,

        "event_count":
            len(event_windows),

        "window_accuracy":
            window_accuracy
    })


    all_window_predictions.extend(
        predictions.tolist()
    )

    all_window_probabilities.extend(
        probabilities.tolist()
    )


    print(
        "    Actual      :",
        "ACCIDENT"
        if actual_video_label
        else
        "NORMAL"
    )

    print(
        "    Predicted   :",
        "ACCIDENT"
        if predicted_video
        else
        "NORMAL"
    )

    print(
        "    Max prob.   :",
        f"{max_probability:.3f}"
    )

    print(
        "    High windows:",
        high_probability_count
    )

    print(
        "    Events      :",
        len(event_windows)
    )

    print()


# ============================================================
# VIDEO-LEVEL METRICS
# ============================================================

video_df = pd.DataFrame(
    video_results
)


video_actual = video_df[
    "actual"
].astype(int)

video_predicted = video_df[
    "predicted"
].astype(int)


video_accuracy = accuracy_score(
    video_actual,
    video_predicted
)

video_precision = precision_score(
    video_actual,
    video_predicted,
    zero_division=0
)

video_recall = recall_score(
    video_actual,
    video_predicted,
    zero_division=0
)

video_f1 = f1_score(
    video_actual,
    video_predicted,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    video_actual,
    video_predicted,
    labels=[0, 1]
)


tn, fp, fn, tp = (
    cm.ravel()
)


# ============================================================
# REPORT
# ============================================================

print()
print("============================================================")
print("             V3 VIDEO-LEVEL RESULTS")
print("============================================================")
print()

print("Confusion Matrix")
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

print("Metrics")
print("-------")

print(
    f"Accuracy : {video_accuracy:.3f}"
)

print(
    f"Precision: {video_precision:.3f}"
)

print(
    f"Recall   : {video_recall:.3f}"
)

print(
    f"F1 Score : {video_f1:.3f}"
)

print()

print("Percentage")
print("----------")

print(
    f"Accuracy : {video_accuracy * 100:.1f}%"
)

print(
    f"Precision: {video_precision * 100:.1f}%"
)

print(
    f"Recall   : {video_recall * 100:.1f}%"
)

print(
    f"F1 Score : {video_f1 * 100:.1f}%"
)

print()


# ============================================================
# PER-VIDEO RESULTS
# ============================================================

print("Per-video results")
print("-----------------")

print(
    video_df.to_string(
        index=False
    )
)

print()


# ============================================================
# TRAIN FINAL MODEL ON COMPLETE DATASET
# ============================================================

print("============================================================")
print("           TRAINING FINAL V3 MODEL")
print("============================================================")
print()


final_model = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    random_state=RANDOM_STATE,

    class_weight="balanced_subsample",

    min_samples_leaf=2,

    max_features="sqrt",

    n_jobs=-1
)


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

    "random_state":
        RANDOM_STATE,

    "training_videos":
        sorted(
            df["video"]
            .unique()
            .tolist()
        ),

    "sklearn_version":
        sklearn.__version__,

    "python_version":
        sys.version,

    "platform":
        platform.platform(),

    "validation_accuracy":
        float(
            video_accuracy
        ),

    "validation_precision":
        float(
            video_precision
        ),

    "validation_recall":
        float(
            video_recall
        ),

    "validation_f1":
        float(
            video_f1
        )
}


joblib.dump(
    model_package,
    MODEL_FILE
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = (
    final_model.feature_importances_
)

importance_df = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        importance
})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


print()
print("Feature Importance")
print("------------------")

print(
    importance_df.to_string(
        index=False
    )
)

print()


# ============================================================
# COMPLETE
# ============================================================

print("============================================================")
print("          V3 TRAINING COMPLETE")
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
    "Validation was performed using:"
)

print(
    "LEAVE-ONE-VIDEO-OUT SEPARATION"
)

print()

print(
    "The saved V3 model contains:"
)

print(
    "• Model"
)

print(
    "• 19 feature names"

)

print(
    "• Window configuration"
)

print(
    "• Training video list"
)

print(
    "• Validation metrics"
)

print(
    "• Software version metadata"
)

print()