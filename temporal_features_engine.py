"""
ResQTrack - Canonical Temporal Feature Engine

This file is the SINGLE source of truth for temporal features.

The same functions must be used for:

1. Training data generation
2. Validation
3. Live inference

This prevents training/inference feature mismatch.
"""

import math

import numpy as np
import pandas as pd


# ============================================================
# WINDOW CONFIGURATION
# ============================================================

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15


# ============================================================
# EXACT TRAINING FEATURE ORDER
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
# BASIC GEOMETRY
# ============================================================

def distance(p1, p2):

    return math.sqrt(

        (p1[0] - p2[0]) ** 2

        +

        (p1[1] - p2[1]) ** 2
    )


def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

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

        max(
            0,
            box1[2] - box1[0]
        )

        *

        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (

        max(
            0,
            box2[2] - box2[0]
        )

        *

        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
# FRAME-LEVEL FEATURES
# ============================================================

def extract_frame_features(
    current,
    previous,
    previous_previous
):
    """
    Extract exactly the frame-level behavioural features
    used by the temporal training pipeline.
    """

    vehicle_count = len(
        current
    )

    speeds = []
    accelerations = []

    pair_distances = []
    pair_ious = []

    approach_rates = []
    approach_accelerations = []

    direction_changes = []
    speed_drops = []


    # ========================================================
    # VEHICLE MOTION
    # ========================================================

    for vehicle_id, vehicle in current.items():

        center = vehicle["center"]

        if (
            previous is not None
            and
            vehicle_id in previous
        ):

            previous_center = (
                previous[
                    vehicle_id
                ]["center"]
            )

            speed = distance(
                center,
                previous_center
            )

            speeds.append(
                speed
            )


            # ------------------------------------------------
            # ACCELERATION
            # ------------------------------------------------

            if (
                previous_previous
                is not None

                and

                vehicle_id
                in previous_previous
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
                    speed
                    -
                    previous_speed
                )

                accelerations.append(
                    abs(acceleration)
                )


                # ------------------------------------------------
                # SPEED DROP
                # ------------------------------------------------

                speed_drop = max(
                    0.0,
                    previous_speed - speed
                )

                speed_drops.append(
                    speed_drop
                )


                # ------------------------------------------------
                # DIRECTION CHANGE
                # ------------------------------------------------

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

                magnitude_1 = math.sqrt(

                    v1[0] ** 2
                    +
                    v1[1] ** 2
                )

                magnitude_2 = math.sqrt(

                    v2[0] ** 2
                    +
                    v2[1] ** 2
                )

                if (
                    magnitude_1 > 0
                    and
                    magnitude_2 > 0
                ):

                    cosine = (

                        (
                            v1[0] * v2[0]
                        )

                        +

                        (
                            v1[1] * v2[1]
                        )
                    ) / (

                        magnitude_1
                        *
                        magnitude_2
                    )

                    cosine = np.clip(
                        cosine,
                        -1.0,
                        1.0
                    )

                    angle = math.degrees(
                        math.acos(
                            cosine
                        )
                    )

                    direction_changes.append(
                        angle
                    )


    # ========================================================
    # VEHICLE-TO-VEHICLE DYNAMICS
    # ========================================================

    vehicle_ids = list(
        current.keys()
    )

    for i in range(
        len(vehicle_ids)
    ):

        for j in range(
            i + 1,
            len(vehicle_ids)
        ):

            id1 = vehicle_ids[i]
            id2 = vehicle_ids[j]

            vehicle1 = current[id1]
            vehicle2 = current[id2]

            current_distance = distance(

                vehicle1["center"],
                vehicle2["center"]
            )

            pair_distances.append(
                current_distance
            )


            # ------------------------------------------------
            # IoU
            # ------------------------------------------------

            iou = calculate_iou(

                vehicle1["box"],
                vehicle2["box"]
            )

            pair_ious.append(
                iou
            )


            # ------------------------------------------------
            # APPROACH RATE
            # ------------------------------------------------

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


                # ------------------------------------------------
                # APPROACH ACCELERATION
                # ------------------------------------------------

                if (
                    previous_previous
                    is not None

                    and
                    id1
                    in previous_previous

                    and
                    id2
                    in previous_previous
                ):

                    previous_previous_distance = distance(

                        previous_previous[
                            id1
                        ]["center"],

                        previous_previous[
                            id2
                        ]["center"]
                    )

                    previous_approach_rate = (

                        previous_previous_distance
                        -
                        previous_distance
                    )

                    approach_acceleration = (

                        approach_rate
                        -
                        previous_approach_rate
                    )

                    approach_accelerations.append(
                        approach_acceleration
                    )


    # ========================================================
    # SAFE DEFAULTS
    #
    # These match the existing extractor behaviour.
    # ========================================================

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


    # ========================================================
    # RETURN FRAME FEATURES
    # ========================================================

    return {

        "vehicle_count":
            vehicle_count,

        "mean_speed":
            np.mean(
                speeds
            ),

        "max_speed":
            np.max(
                speeds
            ),

        "speed_std":
            np.std(
                speeds
            ),

        "mean_acceleration":
            np.mean(
                accelerations
            ),

        "max_acceleration":
            np.max(
                accelerations
            ),

        "min_pair_distance":
            np.min(
                pair_distances
            ),

        "mean_pair_distance":
            np.mean(
                pair_distances
            ),

        "max_iou":
            np.max(
                pair_ious
            ),

        "mean_iou":
            np.mean(
                pair_ious
            ),

        "max_approach_rate":
            np.max(
                approach_rates
            ),

        "mean_approach_rate":
            np.mean(
                approach_rates
            ),

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
            np.max(
                speed_drops
            ),

        "mean_speed_drop":
            np.mean(
                speed_drops
            )
    }


# ============================================================
# WINDOW AGGREGATION
# ============================================================

def aggregate_window(
    frame_features
):
    """
    Convert a list of frame-level feature dictionaries
    into exactly the same 20 features used for training.
    """

    if not frame_features:

        return None

    df = pd.DataFrame(
        frame_features
    )

    result = {}


    # ========================================================
    # MOTION
    # ========================================================

    result["mean_speed"] = (
        df["mean_speed"].mean()
    )

    result["max_speed"] = (
        df["max_speed"].max()
    )

    # IMPORTANT:
    # pandas std() uses ddof=1.
    # This matches the existing training extractor.

    result["speed_std"] = (
        df["mean_speed"].std()
    )

    result["mean_acceleration"] = (
        df[
            "mean_acceleration"
        ].mean()
    )

    result["max_acceleration"] = (
        df[
            "max_acceleration"
        ].max()
    )


    # ========================================================
    # VEHICLE INTERACTION
    # ========================================================

    result["min_pair_distance"] = (
        df[
            "min_pair_distance"
        ].min()
    )

    result["mean_pair_distance"] = (
        df[
            "mean_pair_distance"
        ].mean()
    )

    result["max_iou"] = (
        df["max_iou"].max()
    )

    result["mean_iou"] = (
        df["mean_iou"].mean()
    )


    # ========================================================
    # APPROACH DYNAMICS
    # ========================================================

    result["max_approach_rate"] = (
        df[
            "max_approach_rate"
        ].max()
    )

    result["mean_approach_rate"] = (
        df[
            "mean_approach_rate"
        ].mean()
    )

    result["max_approach_acceleration"] = (
        df[
            "max_approach_acceleration"
        ].max()
    )

    result["mean_approach_acceleration"] = (
        df[
            "mean_approach_acceleration"
        ].mean()
    )


    # ========================================================
    # TRAJECTORY CHANGE
    # ========================================================

    result["max_direction_change"] = (
        df[
            "max_direction_change"
        ].max()
    )

    result["mean_direction_change"] = (
        df[
            "mean_direction_change"
        ].mean()
    )

    result["max_speed_drop"] = (
        df[
            "max_speed_drop"
        ].max()
    )

    result["mean_speed_drop"] = (
        df[
            "mean_speed_drop"
        ].mean()
    )


    # ========================================================
    # TRAFFIC DENSITY
    # ========================================================

    result["max_vehicle_count"] = (
        df[
            "vehicle_count"
        ].max()
    )

    result["mean_vehicle_count"] = (
        df[
            "vehicle_count"
        ].mean()
    )


    # ========================================================
    # CLEAN NaN
    # ========================================================

    for key in result:

        if pd.isna(
            result[key]
        ):

            result[key] = 0.0


    # ========================================================
    # VALIDATE EXACT FEATURE SET
    # ========================================================

    missing = [
        feature
        for feature in FEATURES
        if feature not in result
    ]

    if missing:

        raise ValueError(
            "Missing temporal features: "
            +
            ", ".join(missing)
        )


    return result


# ============================================================
# MODEL INPUT
# ============================================================

def to_model_dataframe(
    aggregated_features
):
    """
    Creates a one-row DataFrame in the exact feature
    order expected by the trained model.
    """

    if aggregated_features is None:

        raise ValueError(
            "Cannot create model input "
            "from empty features."
        )

    return pd.DataFrame(
        [[
            aggregated_features[
                feature
            ]

            for feature in FEATURES
        ]],
        columns=FEATURES
    )


# ============================================================
# TEMPORAL WINDOW BUFFER
# ============================================================

class TemporalFeatureBuffer:

    def __init__(
        self,
        window_frames=WINDOW_FRAMES
    ):

        self.window_frames = (
            window_frames
        )

        self.frames = []


    def add(
        self,
        frame_feature_dict
    ):

        self.frames.append(
            frame_feature_dict
        )

        self.frames = self.frames[
            -self.window_frames:
        ]


    def ready(self):

        return (
            len(self.frames)
            >=
            self.window_frames
        )


    def aggregate(self):

        if not self.ready():

            return None

        return aggregate_window(
            self.frames[
                -self.window_frames:
            ]
        )


    def reset(self):

        self.frames.clear()