import math
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "resqtrack_temporal_model_v2.pkl"

WINDOW_FRAMES = 30


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
# LOAD TRAINED MODEL
# ============================================================

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
trained_features = model_data["features"]

if trained_features != FEATURES:
    raise ValueError(
        "ERROR: Training/inference feature mismatch."
    )


# ============================================================
# GEOMETRY
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    intersection = (
        width * height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# FRAME FEATURES
# ============================================================

def frame_features(
    current,
    previous,
    previous_previous
):

    vehicle_count = len(current)

    speeds = []
    accelerations = []
    pair_distances = []
    pair_ious = []

    approach_rates = []
    approach_accelerations = []

    direction_changes = []
    speed_drops = []

    # --------------------------------------------------------
    # VEHICLE MOTION
    # --------------------------------------------------------

    for vehicle_id, vehicle in current.items():

        if (
            previous is None
            or vehicle_id not in previous
        ):
            continue

        center = vehicle["center"]

        previous_center = (
            previous[vehicle_id]["center"]
        )

        speed = distance(
            center,
            previous_center
        )

        speeds.append(speed)

        if (
            previous_previous is not None
            and vehicle_id in previous_previous
        ):

            previous_previous_center = (
                previous_previous[
                    vehicle_id
                ]["center"]
            )

            previous_speed = distance(
                previous_center,
                previous_previous_center
            )

            acceleration = (
                speed - previous_speed
            )

            accelerations.append(
                abs(acceleration)
            )

            speed_drops.append(
                max(
                    0.0,
                    previous_speed - speed
                )
            )

            v1 = (
                previous_center[0]
                -
                previous_previous_center[0],

                previous_center[1]
                -
                previous_previous_center[1]
            )

            v2 = (
                center[0]
                -
                previous_center[0],

                center[1]
                -
                previous_center[1]
            )

            mag1 = math.hypot(
                v1[0],
                v1[1]
            )

            mag2 = math.hypot(
                v2[0],
                v2[1]
            )

            if mag1 > 0 and mag2 > 0:

                cosine = (
                    v1[0] * v2[0]
                    +
                    v1[1] * v2[1]
                ) / (
                    mag1 * mag2
                )

                cosine = np.clip(
                    cosine,
                    -1.0,
                    1.0
                )

                direction_changes.append(
                    math.degrees(
                        math.acos(cosine)
                    )
                )


    # --------------------------------------------------------
    # VEHICLE PAIRS
    # --------------------------------------------------------

    ids = list(current.keys())

    for i in range(len(ids)):

        for j in range(i + 1, len(ids)):

            id1 = ids[i]
            id2 = ids[j]

            v1 = current[id1]
            v2 = current[id2]

            current_distance = distance(
                v1["center"],
                v2["center"]
            )

            pair_distances.append(
                current_distance
            )

            pair_ious.append(
                calculate_iou(
                    v1["box"],
                    v2["box"]
                )
            )

            if (
                previous is not None
                and
                id1 in previous
                and
                id2 in previous
            ):

                previous_distance = distance(
                    previous[id1]["center"],
                    previous[id2]["center"]
                )

                approach_rate = (
                    previous_distance
                    -
                    current_distance
                )

                approach_rates.append(
                    approach_rate
                )

                if (
                    previous_previous is not None
                    and
                    id1 in previous_previous
                    and
                    id2 in previous_previous
                ):

                    previous_previous_distance = distance(
                        previous_previous[id1]["center"],
                        previous_previous[id2]["center"]
                    )

                    previous_approach_rate = (
                        previous_previous_distance
                        -
                        previous_distance
                    )

                    approach_accelerations.append(
                        approach_rate
                        -
                        previous_approach_rate
                    )


    # --------------------------------------------------------
    # DEFAULTS
    # --------------------------------------------------------

    if not speeds:
        speeds = [0.0]

    if not accelerations:
        accelerations = [0.0]

    if not pair_distances:
        pair_distances = [9999.0]

    if not pair_ious:
        pair_ious = [0.0]

    if not approach_rates:
        approach_rates = [0.0]

    if not approach_accelerations:
        approach_accelerations = [0.0]

    if not direction_changes:
        direction_changes = [0.0]

    if not speed_drops:
        speed_drops = [0.0]


    return {

        "vehicle_count":
            vehicle_count,

        "mean_speed":
            np.mean(speeds),

        "max_speed":
            np.max(speeds),

        "speed_std":
            np.std(speeds),

        "mean_acceleration":
            np.mean(accelerations),

        "max_acceleration":
            np.max(accelerations),

        "min_pair_distance":
            np.min(pair_distances),

        "mean_pair_distance":
            np.mean(pair_distances),

        "max_iou":
            np.max(pair_ious),

        "mean_iou":
            np.mean(pair_ious),

        "max_approach_rate":
            np.max(approach_rates),

        "mean_approach_rate":
            np.mean(approach_rates),

        "max_approach_acceleration":
            np.max(
                approach_accelerations
            ),

        "mean_approach_acceleration":
            np.mean(
                approach_accelerations
            ),

        "max_direction_change":
            np.max(
                direction_changes
            ),

        "mean_direction_change":
            np.mean(
                direction_changes
            ),

        "max_speed_drop":
            np.max(speed_drops),

        "mean_speed_drop":
            np.mean(speed_drops)
    }


# ============================================================
# REAL-TIME ML ENGINE
# ============================================================

class TemporalMLEngine:

    def __init__(self):

        self.frames = []

    def update(
        self,
        current,
        previous,
        previous_previous
    ):

        features = frame_features(
            current,
            previous,
            previous_previous
        )

        self.frames.append(
            features
        )

        # Keep only required history.
        self.frames = self.frames[
            -WINDOW_FRAMES:
        ]

        if len(self.frames) < WINDOW_FRAMES:
            return None

        window = pd.DataFrame(
            self.frames
        )

        # EXACT SAME AGGREGATION
        # USED DURING TRAINING

        result = {

            "mean_speed":
                window["mean_speed"].mean(),

            "max_speed":
                window["max_speed"].max(),

            "speed_std":
                window["mean_speed"].std(),

            "mean_acceleration":
                window["mean_acceleration"].mean(),

            "max_acceleration":
                window["max_acceleration"].max(),

            "min_pair_distance":
                window["min_pair_distance"].min(),

            "mean_pair_distance":
                window["mean_pair_distance"].mean(),

            "max_iou":
                window["max_iou"].max(),

            "mean_iou":
                window["mean_iou"].mean(),

            "max_approach_rate":
                window["max_approach_rate"].max(),

            "mean_approach_rate":
                window["mean_approach_rate"].mean(),

            "max_approach_acceleration":
                window[
                    "max_approach_acceleration"
                ].max(),

            "mean_approach_acceleration":
                window[
                    "mean_approach_acceleration"
                ].mean(),

            "max_direction_change":
                window[
                    "max_direction_change"
                ].max(),

            "mean_direction_change":
                window[
                    "mean_direction_change"
                ].mean(),

            "max_speed_drop":
                window[
                    "max_speed_drop"
                ].max(),

            "mean_speed_drop":
                window[
                    "mean_speed_drop"
                ].mean(),

            "max_vehicle_count":
                window[
                    "vehicle_count"
                ].max(),

            "mean_vehicle_count":
                window[
                    "vehicle_count"
                ].mean()
        }

        input_data = pd.DataFrame(
            [[
                result[feature]
                for feature in FEATURES
            ]],
            columns=FEATURES
        )

        probability = model.predict_proba(
            input_data
        )[0][1]

        return float(
            probability
        )


    def reset(self):

        self.frames.clear()