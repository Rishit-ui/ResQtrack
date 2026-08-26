import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

OOF_FILE = "temporal_features_v3.csv"
PROBABILITY_FILE = "event_oof_v4.csv"
ANNOTATIONS_FILE = "annotations.csv"

OUTPUT_FILE = "candidate_events_v5.csv"

ML_THRESHOLD = 0.30
EVENT_GAP = 1.5


# ============================================================
# LOAD DATA
# ============================================================

windows = pd.read_csv(
    OOF_FILE
)

oof = pd.read_csv(
    PROBABILITY_FILE
)

annotations = pd.read_csv(
    ANNOTATIONS_FILE
)


# ============================================================
# IMPORTANT
# ============================================================
#
# event_oof_v4.csv contains aggregated events, not the
# original per-window probabilities.
#
# Therefore we need the probability information to be
# reconstructed from the existing temporal windows.
#
# The event file already contains event clusters and their
# probability summaries, but for proper candidate construction
# we use the original temporal window table and its labels.
#
# ============================================================


# If temporal_features_v3.csv does not contain probabilities,
# merge them from event_oof_v4 through the event information
# is not possible window-by-window.
#
# Therefore this script will build candidates from:
#
# 1. annotation intervals
# 2. existing event_oof_v4 clusters
#
# and preserve their source.
# ============================================================


# ============================================================
# ANNOTATION LOOKUP
# ============================================================

annotation_lookup = {}

for _, row in annotations.iterrows():

    video = row["video"]

    label = int(
        row["label"]
    )

    if label == 1:

        annotation_lookup[video] = {

            "start":
                float(
                    row["start_sec"]
                ),

            "end":
                float(
                    row["end_sec"]
                )
        }

    else:

        annotation_lookup[video] = None


# ============================================================
# VIDEO LIST
# ============================================================

videos = sorted(
    windows["video"]
    .unique()
)


candidate_rows = []


# ============================================================
# BUILD CANDIDATES
# ============================================================

for video in videos:

    video_windows = (
        windows[
            windows["video"]
            ==
            video
        ]
        .sort_values(
            "window_start_sec"
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # ACCIDENT ANNOTATION CANDIDATE
    # --------------------------------------------------------

    annotation = (
        annotation_lookup.get(
            video
        )
    )


    if annotation is not None:

        start = annotation[
            "start"
        ]

        end = annotation[
            "end"
        ]


        overlapping = (
            video_windows[
                (
                    video_windows[
                        "window_end_sec"
                    ]
                    >=
                    start
                )

                &

                (
                    video_windows[
                        "window_start_sec"
                    ]
                    <=
                    end
                )
            ]
        )


        candidate_rows.append({

            "video":
                video,

            "candidate_id":
                len(candidate_rows),

            "source":
                "ANNOTATION",

            "candidate_start_sec":
                start,

            "candidate_end_sec":
                end,

            "label":
                1,

            "window_count":
                len(overlapping),

            "max_probability":
                np.nan
        })


    # --------------------------------------------------------
    # EXISTING ML EVENTS
    # --------------------------------------------------------

    video_events = (
        oof[
            oof["video"]
            ==
            video
        ]
        .sort_values(
            "event_start_sec"
        )
    )


    for _, event in (
        video_events.iterrows()
    ):

        event_start = float(
            event[
                "event_start_sec"
            ]
        )

        event_end = float(
            event[
                "event_end_sec"
            ]
        )


        # ----------------------------------------------------
        # Does this ML event overlap the real annotation?
        # ----------------------------------------------------

        overlaps_annotation = False


        if annotation is not None:

            overlap_start = max(
                event_start,
                annotation["start"]
            )

            overlap_end = min(
                event_end,
                annotation["end"]
            )

            overlap = max(
                0.0,
                overlap_end
                -
                overlap_start
            )


            annotation_duration = (
                annotation["end"]
                -
                annotation["start"]
            )


            if (
                annotation_duration > 0
                and
                overlap
                /
                annotation_duration
                >=
                0.25
            ):

                overlaps_annotation = True


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if overlaps_annotation:

            label = 1

        else:

            label = 0


        candidate_rows.append({

            "video":
                video,

            "candidate_id":
                len(candidate_rows),

            "source":
                "ML_EVENT",

            "candidate_start_sec":
                event_start,

            "candidate_end_sec":
                event_end,

            "label":
                label,

            "window_count":
                int(
                    event[
                        "event_window_count"
                    ]
                ),

            "max_probability":
                float(
                    event[
                        "event_peak_probability"
                    ]
                )
        })


    # --------------------------------------------------------
    # NORMAL BACKGROUND CANDIDATE
    #
    # If a normal video produced no suspicious event,
    # we still want one negative example.
    # --------------------------------------------------------

    if (
        annotation is None
        and
        len(video_events) == 0
    ):

        max_row = (
            video_windows.iloc[
                video_windows[
                    "window_start_sec"
                ].argmax()
            ]
        )


        candidate_rows.append({

            "video":
                video,

            "candidate_id":
                len(candidate_rows),

            "source":
                "NORMAL_BACKGROUND",

            "candidate_start_sec":
                float(
                    max_row[
                        "window_start_sec"
                    ]
                ),

            "candidate_end_sec":
                float(
                    max_row[
                        "window_end_sec"
                    ]
                ),

            "label":
                0,

            "window_count":
                1,

            "max_probability":
                np.nan
        })


# ============================================================
# DATAFRAME
# ============================================================

candidates = pd.DataFrame(
    candidate_rows
)


# ============================================================
# REPORT
# ============================================================

print()
print("============================================================")
print("        RESQTRACK V5 CANDIDATE EVENT DATASET")
print("============================================================")
print()

print(
    "Total candidates:",
    len(candidates)
)

print(
    "Videos:",
    candidates["video"].nunique()
)

print()

print(
    "Labels:"
)

print(
    candidates[
        "label"
    ]
    .value_counts()
    .sort_index()
)

print()

print(
    "Sources:"
)

print(
    candidates[
        "source"
    ]
    .value_counts()
)

print()

print(
    "Per-video candidates:"
)

print(
    candidates[
        [
            "video",
            "source",
            "candidate_start_sec",
            "candidate_end_sec",
            "label",
            "window_count",
            "max_probability"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)

print()

print(
    "Saved as:",
    OUTPUT_FILE
)

print()

print(
    "Next:"
)

print(
    "Inspect candidate events before training."
)

print()