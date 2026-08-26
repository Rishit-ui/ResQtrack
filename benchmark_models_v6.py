import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import LeaveOneGroupOut

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "temporal_features_v4.csv"

RANDOM_STATE = 42


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
# MODELS
# ============================================================

def create_models():

    return {

        "RandomForest":

            RandomForestClassifier(

                n_estimators=800,

                class_weight="balanced_subsample",

                min_samples_leaf=2,

                max_features="sqrt",

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),


        "ExtraTrees":

            ExtraTreesClassifier(

                n_estimators=800,

                class_weight="balanced",

                min_samples_leaf=2,

                max_features="sqrt",

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),


        "LogisticRegression":

            Pipeline([

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",

                    LogisticRegression(

                        class_weight="balanced",

                        max_iter=5000,

                        random_state=RANDOM_STATE
                    )
                )
            ]),


        "SVM_RBF":

            Pipeline([

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",

                    SVC(

                        kernel="rbf",

                        probability=True,

                        class_weight="balanced",

                        C=1.0,

                        gamma="scale",

                        random_state=RANDOM_STATE
                    )
                )
            ]),


        "HistGradientBoosting":

            HistGradientBoostingClassifier(

                max_iter=300,

                learning_rate=0.05,

                max_leaf_nodes=15,

                l2_regularization=1.0,

                random_state=RANDOM_STATE
            )
    }


# ============================================================
# LOAD
# ============================================================

print()
print("============================================================")
print("       RESQTRACK MODEL BENCHMARK V6")
print("============================================================")
print()

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

y = (
    df["label"]
    .astype(int)
)

groups = df["video"]


# ============================================================
# VIDEO LABELS
# ============================================================

video_labels = (
    df.groupby("video")["label"]
    .max()
    .astype(int)
)


# ============================================================
# EVALUATION
# ============================================================

logo = LeaveOneGroupOut()

all_results = []


for model_name in create_models():

    print()
    print(
        "============================================================"
    )

    print(
        "MODEL:",
        model_name
    )

    print(
        "============================================================"
    )

    print()


    model_results = []


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
            ]

            .iloc[0]
        )


        actual = int(
            video_labels[
                test_video
            ]
        )


        model = (
            create_models()
            [model_name]
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(

            X.iloc[
                train_idx
            ],

            y.iloc[
                train_idx
            ]
        )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probabilities = (

            model.predict_proba(

                X.iloc[
                    test_idx
                ]

            )[:, 1]
        )


        # ----------------------------------------------------
        # VIDEO-LEVEL RULE
        #
        # Keep rule simple:
        #
        # >= 0.50 on at least two windows
        #
        # We are comparing classifiers, not optimizing
        # thousands of decision rules.
        # ----------------------------------------------------

        high_windows = int(

            (
                probabilities
                >=
                0.50
            ).sum()
        )


        predicted = int(

            high_windows
            >=
            2
        )


        max_probability = float(

            probabilities.max()
        )


        model_results.append({

            "video":
                test_video,

            "actual":
                actual,

            "predicted":
                predicted,

            "max_probability":
                max_probability,

            "high_windows":
                high_windows
        })


        print(

            f"{test_video:<20}"

            f"Actual={actual} "

            f"Predicted={predicted} "

            f"Max={max_probability:.3f} "

            f"High={high_windows}"
        )


    # ========================================================
    # METRICS
    # ========================================================

    result_df = pd.DataFrame(
        model_results
    )


    actuals = result_df[
        "actual"
    ]

    predictions = result_df[
        "predicted"
    ]


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

            labels=[
                0,
                1
            ]

        ).ravel()
    )


    print()

    print(
        "Accuracy :",
        f"{accuracy * 100:.1f}%"
    )

    print(
        "Precision:",
        f"{precision * 100:.1f}%"
    )

    print(
        "Recall   :",
        f"{recall * 100:.1f}%"
    )

    print(
        "F1 Score :",
        f"{f1 * 100:.1f}%"
    )

    print()


    all_results.append({

        "model":
            model_name,

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
# FINAL COMPARISON
# ============================================================

results = pd.DataFrame(
    all_results
)


results = (
    results
    .sort_values(
        "f1",
        ascending=False
    )
)


print()
print("============================================================")
print("             MODEL BENCHMARK V6")
print("============================================================")
print()

print(
    results.to_string(
        index=False
    )
)

print()

print(
    "Best model by F1:",
    results.iloc[0]["model"]
)

print()