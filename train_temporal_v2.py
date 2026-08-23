import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "temporal_features_v2.csv"

MODEL_FILE = "resqtrack_temporal_model_v2.pkl"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("      ResQTrack Temporal ML Pipeline")
print("==============================================\n")

df = pd.read_csv(DATASET_FILE)

print(f"Total temporal windows: {len(df)}")
print(f"Videos represented: {df['video'].nunique()}")


# ============================================================
# FEATURE DEFINITIONS
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
# VALIDATION
# ============================================================

for feature in FEATURES:

    if feature not in df.columns:

        raise ValueError(
            f"Missing feature: {feature}"
        )


if "label" not in df.columns:
    raise ValueError("Missing label column.")

if "video" not in df.columns:
    raise ValueError("Missing video column.")


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()

y = df["label"].astype(int)

groups = df["video"]


X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n==============================================")
print("Dataset Distribution")
print("==============================================")

print(
    y.value_counts()
    .sort_index()
    .rename(
        index={
            0: "NORMAL",
            1: "ACCIDENT"
        }
    )
)


print("\nVideos:")

for video in sorted(
    df["video"].unique()
):

    video_data = df[
        df["video"] == video
    ]

    accident_windows = int(
        video_data["label"].sum()
    )

    print(
        f"{video:20s} "
        f"windows={len(video_data):3d} "
        f"accident={accident_windows:2d}"
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    return RandomForestClassifier(

        n_estimators=400,

        max_depth=8,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


# ============================================================
# VIDEO-LEVEL CROSS VALIDATION
# ============================================================

logo = LeaveOneGroupOut()

window_predictions = np.zeros(
    len(df),
    dtype=int
)

window_probabilities = np.zeros(
    len(df)
)

fold_number = 0


print("\n==============================================")
print("VIDEO-LEVEL CROSS VALIDATION")
print("==============================================\n")


for train_index, test_index in logo.split(
    X,
    y,
    groups
):

    fold_number += 1

    X_train = X.iloc[
        train_index
    ]

    X_test = X.iloc[
        test_index
    ]

    y_train = y.iloc[
        train_index
    ]

    y_test = y.iloc[
        test_index
    ]

    test_video = groups.iloc[
        test_index
    ].iloc[0]

    print(
        f"Fold {fold_number:02d} | "
        f"Test video: {test_video}"
    )

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    window_predictions[
        test_index
    ] = predictions

    window_probabilities[
        test_index
    ] = probabilities


# ============================================================
# WINDOW-LEVEL METRICS
# ============================================================

print("\n==============================================")
print("WINDOW-LEVEL RESULTS")
print("==============================================")

accuracy = accuracy_score(
    y,
    window_predictions
)

precision = precision_score(
    y,
    window_predictions,
    zero_division=0
)

recall = recall_score(
    y,
    window_predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    window_predictions,
    zero_division=0
)

print(
    f"\nAccuracy : {accuracy:.3f}"
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


# ============================================================
# ROC-AUC
# ============================================================

try:

    auc = roc_auc_score(
        y,
        window_probabilities
    )

    print(
        f"ROC-AUC  : {auc:.3f}"
    )

except ValueError:

    print(
        "ROC-AUC  : unavailable"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    window_predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix")
print("----------------")

print(
    "                 Predicted"
)

print(
    "              Normal  Accident"
)

print(
    f"Normal       {cm[0][0]:6d}  "
    f"{cm[0][1]:8d}"
)

print(
    f"Accident     {cm[1][0]:6d}  "
    f"{cm[1][1]:8d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y,
        window_predictions,
        target_names=[
            "NORMAL",
            "ACCIDENT"
        ],
        zero_division=0
    )
)


# ============================================================
# VIDEO-LEVEL RESULTS
# ============================================================

print("\n==============================================")
print("VIDEO-LEVEL EVENT RESULTS")
print("==============================================")

video_results = []


for video in sorted(
    df["video"].unique()
):

    mask = (
        df["video"] == video
    )

    actual = int(
        df.loc[
            mask,
            "label"
        ].max()
    )

    # Maximum accident probability
    # observed in this video.

    max_probability = float(
        np.max(
            window_probabilities[
                mask.values
            ]
        )
    )

    # Number of windows classified
    # as accident.

    predicted_windows = int(
        np.sum(
            window_predictions[
                mask.values
            ]
        )
    )

    total_windows = int(
        np.sum(mask)
    )

    # Event-level decision.
    #
    # A video is considered an accident
    # if at least one temporal window
    # reaches 0.50 probability.

    predicted_event = int(
        max_probability >= 0.50
    )

    video_results.append(
        {
            "video": video,
            "actual": actual,
            "max_probability":
                max_probability,
            "predicted_windows":
                predicted_windows,
            "total_windows":
                total_windows,
            "predicted_event":
                predicted_event
        }
    )


video_df = pd.DataFrame(
    video_results
)


print(
    video_df.to_string(
        index=False
    )
)


video_accuracy = accuracy_score(
    video_df["actual"],
    video_df["predicted_event"]
)

video_precision = precision_score(
    video_df["actual"],
    video_df["predicted_event"],
    zero_division=0
)

video_recall = recall_score(
    video_df["actual"],
    video_df["predicted_event"],
    zero_division=0
)

video_f1 = f1_score(
    video_df["actual"],
    video_df["predicted_event"],
    zero_division=0
)


print("\nVideo-level metrics:")

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


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n==============================================")
print("Training final temporal model...")
print("==============================================")

final_model = create_model()

final_model.fit(
    X,
    y
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    final_model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)


print("\nFeature Importance")
print("------------------")

for feature, value in importance.items():

    print(
        f"{feature:25s} "
        f"{value:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": final_model,
        "features": FEATURES
    },
    MODEL_FILE
)


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print("TEMPORAL MODEL TRAINING COMPLETE")
print("==============================================")

print(
    f"\nModel saved as:"
)

print(
    MODEL_FILE
)

print(
    "\nThis model was evaluated using "
    "video-level separation."
)