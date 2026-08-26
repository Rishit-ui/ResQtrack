import sys
import platform
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "temporal_features_v4.csv"

MODEL_FILE = "resqtrack_final_model.pkl"

RANDOM_STATE = 42


# ============================================================
# EXACT FEATURE ORDER
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
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("          RESQTRACK FINAL MODEL TRAINING")
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

print(
    "Features:",
    len(FEATURES)
)

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

y = (
    df["label"]
    .astype(int)
)


# ============================================================
# FINAL PIPELINE
# ============================================================

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",

        LogisticRegression(

            class_weight="balanced",

            max_iter=5000,

            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

print(
    "Training Logistic Regression..."
)

model.fit(
    X,
    y
)


# ============================================================
# SAVE COMPLETE MODEL PACKAGE
# ============================================================

package = {

    "model":
        model,

    "features":
        FEATURES,

    "window_frames":
        30,

    "stride_frames":
        15,

    "labeling_method":
        "Buffered annotation + temporal overlap",

    "accident_buffer_seconds":
        0.5,

    "minimum_positive_overlap":
        0.30,

    "model_type":
        "StandardScaler + LogisticRegression",

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
    package,
    MODEL_FILE
)


# ============================================================
# FEATURE COEFFICIENTS
# ============================================================

classifier = (
    model.named_steps[
        "classifier"
    ]
)

coefficients = (
    classifier.coef_[0]
)


importance = pd.DataFrame({

    "feature":
        FEATURES,

    "coefficient":
        coefficients,

    "absolute":
        np.abs(coefficients)

})


importance = (
    importance
    .sort_values(
        "absolute",
        ascending=False
    )
)


print()
print("============================================================")
print("           FEATURE COEFFICIENTS")
print("============================================================")
print()

print(
    importance[
        [
            "feature",
            "coefficient"
        ]
    ].to_string(
        index=False
    )
)

print()


# ============================================================
# COMPLETE
# ============================================================

print("============================================================")
print("             FINAL MODEL READY")
print("============================================================")
print()

print(
    "Saved as:"
)

print(
    MODEL_FILE
)

print()

print(
    "Model:"
)

print(
    "StandardScaler + LogisticRegression"
)

print()

print(
    "Feature count:",
    len(FEATURES)
)

print()