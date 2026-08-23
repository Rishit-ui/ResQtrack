import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "training_features.csv"
MODEL_FILE = "resqtrack_accident_model.pkl"

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print("      ResQTrack ML Training Pipeline")
print("==========================================\n")

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded.")
print(f"Samples: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# VALIDATE DATASET
# ============================================================

if "label" not in df.columns:
    raise ValueError("ERROR: 'label' column not found.")

if "video" not in df.columns:
    raise ValueError("ERROR: 'video' column not found.")


print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "NORMAL",
            1: "ACCIDENT"
        }
    )
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df.drop(
    columns=["label", "video"]
)

y = df["label"]


# Make sure all feature columns are numeric

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Handle missing values

X = X.fillna(0)


print("\nFeatures used:")

for feature in X.columns:
    print(f"  • {feature}")


# ============================================================
# MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ============================================================
# LEAVE-ONE-OUT VALIDATION
# ============================================================

print("\n==========================================")
print("Leave-One-Video-Out Validation")
print("==========================================\n")

loo = LeaveOneOut()

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=loo,
    method="predict"
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)


print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix")
print("----------------")
print("                 Predicted")
print("              Normal  Accident")
print(f"Normal       {cm[0][0]:6d}  {cm[0][1]:8d}")
print(f"Accident     {cm[1][0]:6d}  {cm[1][1]:8d}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "NORMAL",
            "ACCIDENT"
        ],
        zero_division=0
    )
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining final model on complete dataset...")

model.fit(
    X,
    y
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
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
        "model": model,
        "features": list(X.columns)
    },
    MODEL_FILE
)


print("\n==========================================")
print("MODEL TRAINING COMPLETE")
print("==========================================")

print(f"\nModel saved as:")
print(MODEL_FILE)

print("\nResQTrack ML baseline is ready.")