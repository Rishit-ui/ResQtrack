import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "temporal_features_v3.csv"
OUTPUT_FILE = "window_oof_v5.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 600


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
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("         RESQTRACK WINDOW OOF V5 GENERATION")
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
# VALIDATE COLUMNS
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
    for column in (
        required + FEATURES
    )
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


# ============================================================
# LEAVE-ONE-VIDEO-OUT
# ============================================================

logo = LeaveOneGroupOut()

oof_parts = []


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
        f"[{fold}/"
        f"{df['video'].nunique()}]"
        f" Held-out video: "
        f"{test_video}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = create_model()

    model.fit(

        X.iloc[
            train_idx
        ],

        y.iloc[
            train_idx
        ]
    )


    # --------------------------------------------------------
    # OOF PROBABILITY
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X.iloc[
                test_idx
            ]
        )[:, 1]
    )


    # --------------------------------------------------------
    # SAVE HELD-OUT WINDOWS
    # --------------------------------------------------------

    part = df.iloc[
        test_idx
    ].copy()


    part["oof_probability"] = (
        probabilities
    )


    part["oof_prediction_050"] = (
        probabilities >= 0.50
    ).astype(int)


    part["oof_prediction_035"] = (
        probabilities >= 0.35
    ).astype(int)


    part["oof_prediction_030"] = (
        probabilities >= 0.30
    ).astype(int)


    oof_parts.append(
        part
    )


# ============================================================
# COMBINE
# ============================================================

oof = pd.concat(
    oof_parts,
    ignore_index=True
)


# ============================================================
# SORT
# ============================================================

oof = (
    oof
    .sort_values(
        [
            "video",
            "window_start_sec"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(oof) != len(df):

    raise RuntimeError(
        "OOF row count does not match original dataset."
    )


if set(
    oof["video"]
) != set(
    df["video"]
):

    raise RuntimeError(
        "OOF video set does not match original dataset."
    )


# ============================================================
# SAVE
# ============================================================

oof.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("============================================================")
print("           WINDOW OOF V5 COMPLETE")
print("============================================================")
print()

print(
    "Rows:",
    len(oof)
)

print(
    "Videos:",
    oof["video"].nunique()
)

print()


# ------------------------------------------------------------
# Probability statistics
# ------------------------------------------------------------

print(
    "OOF probability statistics:"
)

print(
    oof[
        "oof_probability"
    ]
    .describe()
    .round(4)
    .to_string()
)

print()


# ------------------------------------------------------------
# Per-video summary
# ------------------------------------------------------------

summary = (
    oof
    .groupby(
        [
            "video",
            "label"
        ]
    )
    .agg(

        windows=(
            "window_id",
            "count"
        ),

        max_probability=(
            "oof_probability",
            "max"
        ),

        mean_probability=(
            "oof_probability",
            "mean"
        ),

        high_030=(
            "oof_probability",
            lambda x:
            int(
                (x >= 0.30).sum()
            )
        ),

        high_035=(
            "oof_probability",
            lambda x:
            int(
                (x >= 0.35).sum()
            )
        ),

        high_050=(
            "oof_probability",
            lambda x:
            int(
                (x >= 0.50).sum()
            )
        )
    )
    .reset_index()
)


print(
    summary.to_string(
        index=False
    )
)

print()


print(
    "Saved as:"
)

print(
    OUTPUT_FILE
)

print()