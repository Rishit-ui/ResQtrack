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

DATA_FILE = "temporal_features_v4.csv"

MODEL_FILE = "resqtrack_temporal_model_v5.pkl"

RANDOM_STATE = 42

N_ESTIMATORS = 800


# ============================================================
# EXACT 19 FEATURES
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
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("        RESQTRACK TEMPORAL MODEL V5")
print("============================================================")
print()

df = pd.read_csv(
    DATA_FILE
)

print(
    "Rows    :",
    len(df)
)

print(
    "Videos  :",
    df["video"].nunique()
)

print(
    "Features:",
    len(FEATURES)
)

print()


# ============================================================
# VALIDATE
# ============================================================

required = [
    "video",
    "window_id",
    "window_start_sec",
    "window_end_sec",
    "label"
]

missing = [
    column
    for column in required + FEATURES
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns:\n"
        +
        "\n".join(missing)
    )


# ============================================================
# DATA
# ============================================================

X = (
    df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)

y = (
    df["label"]
    .astype(int)
)

groups = df["video"]


print(
    "Window distribution:"
)

print(
    y.value_counts()
    .sort_index()
)

print()


# ============================================================
# VIDEO LABEL
# ============================================================

video_labels = (
    df.groupby("video")["label"]
    .max()
    .astype(int)
)


# ============================================================
# LEAVE-ONE-VIDEO-OUT
# ============================================================

logo = LeaveOneGroupOut()

video_results = []

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

    actual_video = int(
        video_labels[
            test_video
        ]
    )


    print(
        f"[{fold}/10] "
        f"Testing: {test_video}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = create_model()

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )


    # --------------------------------------------------------
    # OOF PROBABILITY
    # --------------------------------------------------------

    probability = (
        model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]
    )


    test_df = df.iloc[
        test_idx
    ].copy()

    test_df[
        "probability"
    ] = probability


    oof_rows.append(
        test_df
    )


    # --------------------------------------------------------
    # WINDOW STATISTICS
    # --------------------------------------------------------

    max_probability = float(
        probability.max()
    )

    mean_probability = float(
        probability.mean()
    )


    # ========================================================
    # VIDEO DECISION
    # ========================================================
    #
    # IMPORTANT:
    # We are NOT using a fixed threshold yet.
    #
    # First we collect the honest OOF probability structure.
    #
    # ========================================================

    high_030 = int(
        (
            probability >= 0.30
        ).sum()
    )

    high_040 = int(
        (
            probability >= 0.40
        ).sum()
    )

    high_050 = int(
        (
            probability >= 0.50
        ).sum()
    )

    high_060 = int(
        (
            probability >= 0.60
        ).sum()
    )


    # --------------------------------------------------------
    # TEMPORAL CONCENTRATION
    # --------------------------------------------------------

    ordered = (
        test_df
        .sort_values(
            "window_start_sec"
        )
        .reset_index(
            drop=True
        )
    )


    # consecutive high-probability runs
    max_run_040 = 0
    max_run_050 = 0
    max_run_060 = 0

    run_040 = 0
    run_050 = 0
    run_060 = 0


    for p in (
        ordered[
            "probability"
        ]
    ):

        if p >= 0.40:
            run_040 += 1
        else:
            run_040 = 0


        if p >= 0.50:
            run_050 += 1
        else:
            run_050 = 0


        if p >= 0.60:
            run_060 += 1
        else:
            run_060 = 0


        max_run_040 = max(
            max_run_040,
            run_040
        )

        max_run_050 = max(
            max_run_050,
            run_050
        )

        max_run_060 = max(
            max_run_060,
            run_060
        )


    print(
        "    Actual label      :",
        "ACCIDENT"
        if actual_video
        else
        "NORMAL"
    )

    print(
        "    Max probability   :",
        f"{max_probability:.3f}"
    )

    print(
        "    Mean probability  :",
        f"{mean_probability:.3f}"
    )

    print(
        "    >= 0.30 windows  :",
        high_030
    )

    print(
        "    >= 0.40 windows  :",
        high_040
    )

    print(
        "    >= 0.50 windows  :",
        high_050
    )

    print(
        "    >= 0.60 windows  :",
        high_060
    )

    print(
        "    Max run >= 0.40  :",
        max_run_040
    )

    print(
        "    Max run >= 0.50  :",
        max_run_050
    )

    print(
        "    Max run >= 0.60  :",
        max_run_060
    )

    print()


    video_results.append({

        "video":
            test_video,

        "actual":
            actual_video,

        "max_probability":
            max_probability,

        "mean_probability":
            mean_probability,

        "high_030":
            high_030,

        "high_040":
            high_040,

        "high_050":
            high_050,

        "high_060":
            high_060,

        "max_run_040":
            max_run_040,

        "max_run_050":
            max_run_050,

        "max_run_060":
            max_run_060
    })


# ============================================================
# COMBINE OOF
# ============================================================

oof = pd.concat(
    oof_rows,
    ignore_index=True
)

oof.to_csv(
    "window_oof_v5_corrected.csv",
    index=False
)


# ============================================================
# VIDEO RESULTS
# ============================================================

video_df = pd.DataFrame(
    video_results
)


print()
print("============================================================")
print("         V5 OOF VIDEO SUMMARY")
print("============================================================")
print()

print(
    video_df.to_string(
        index=False
    )
)


# ============================================================
# EXPERIMENT WITH VIDEO RULES
# ============================================================
#
# We test a SMALL set of sensible rules.
# We will NOT optimize an enormous threshold grid because
# there are only 10 videos.
#
# ============================================================

rules = [

    {
        "name":
            "peak_050",

        "threshold":
            0.50,

        "minimum_windows":
            1
    },

    {
        "name":
            "peak_060",

        "threshold":
            0.60,

        "minimum_windows":
            1
    },

    {
        "name":
            "two_040",

        "threshold":
            0.40,

        "minimum_windows":
            2
    },

    {
        "name":
            "two_050",

        "threshold":
            0.50,

        "minimum_windows":
            2
    },

    {
        "name":
            "two_060",

        "threshold":
            0.60,

        "minimum_windows":
            2
    }
]


rule_results = []


for rule in rules:

    actuals = []
    predictions = []


    for _, row in video_df.iterrows():

        threshold = (
            rule["threshold"]
        )

        minimum_windows = (
            rule[
                "minimum_windows"
            ]
        )


        if threshold == 0.40:

            count = row[
                "high_040"
            ]

        elif threshold == 0.50:

            count = row[
                "high_050"
            ]

        elif threshold == 0.60:

            count = row[
                "high_060"
            ]

        else:

            count = 0


        predicted = int(
            count
            >=
            minimum_windows
        )


        actuals.append(
            int(
                row["actual"]
            )
        )

        predictions.append(
            predicted
        )


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


    rule_results.append({

        "rule":
            rule["name"],

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
# REPORT
# ============================================================

rules_df = pd.DataFrame(
    rule_results
)


print()
print("============================================================")
print("          V5 VIDEO DECISION COMPARISON")
print("============================================================")
print()

print(
    rules_df.to_string(
        index=False
    )
)

print()


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print(
    "Training final V5 model..."
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
        "Leave-One-Video-Out",

    "labeling_method":
        "Buffered annotation + temporal overlap",

    "random_state":
        RANDOM_STATE,

    "training_videos":
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
print("          V5 TRAINING COMPLETE")
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
    "OOF window data saved as:"
)

print(
    "window_oof_v5_corrected.csv"
)

print()