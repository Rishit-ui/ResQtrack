"""
ResQTrack - Temporal Accident Confirmation Engine

Converts repeated suspicious observations
into a stable accident-confirmed decision.
"""

from collections import deque


class TemporalAccidentDetector:

    def __init__(
        self,
        evidence_threshold=0.30,
        confirmation_windows=3,
        history_size=5,
        minimum_suspicious_ratio=0.67,
        strong_evidence_threshold=0.40,
        minimum_support_windows=2
    ):

        self.evidence_threshold = (
            evidence_threshold
        )

        self.confirmation_windows = (
            confirmation_windows
        )

        self.minimum_suspicious_ratio = (
            minimum_suspicious_ratio
        )

        self.strong_evidence_threshold = (
            strong_evidence_threshold
        )

        self.minimum_support_windows = (
            minimum_support_windows
        )

        self.history = deque(
            maxlen=history_size
        )

        self.accident_confirmed = False


    # ==========================================================
    # UPDATE
    # ==========================================================

    def update(
        self,
        evidence_score,
        collision_evidence=False,
        approach_evidence=False
    ):

        suspicious = (
            evidence_score
            >= self.evidence_threshold
        )

        observation = {

            "evidence_score":
                float(evidence_score),

            "suspicious":
                bool(suspicious),

            "collision_evidence":
                bool(collision_evidence),

            "approach_evidence":
                bool(approach_evidence)
        }

        self.history.append(
            observation
        )

        return self._evaluate()


    # ==========================================================
    # EVALUATE
    # ==========================================================

    def _evaluate(self):

        if (
            len(self.history)
            <
            self.confirmation_windows
        ):

            return {
                "status": "NORMAL",
                "confirmed": False,
                "confidence": 0.0,
                "suspicious_ratio": 0.0
            }

        recent = list(
            self.history
        )[
            -self.confirmation_windows:
        ]

        # ------------------------------------------------------
        # SUSPICIOUS PERSISTENCE
        # ------------------------------------------------------

        suspicious_count = sum(
            item["suspicious"]
            for item in recent
        )

        suspicious_ratio = (
            suspicious_count
            /
            len(recent)
        )

        # ------------------------------------------------------
        # AVERAGE EVIDENCE
        # ------------------------------------------------------

        average_evidence = (
            sum(
                item["evidence_score"]
                for item in recent
            )
            /
            len(recent)
        )

        # ------------------------------------------------------
        # SUPPORTING EVIDENCE
        # ------------------------------------------------------

        collision_count = sum(
            item["collision_evidence"]
            for item in recent
        )

        approach_count = sum(
            item["approach_evidence"]
            for item in recent
        )

        strong_evidence_count = sum(
            item["evidence_score"]
            >= self.strong_evidence_threshold
            for item in recent
        )

        # ------------------------------------------------------
        # PERSISTENT SUSPICION
        # ------------------------------------------------------

        persistent_suspicion = (
            suspicious_ratio
            >=
            self.minimum_suspicious_ratio
        )

        # ------------------------------------------------------
        # PERSISTENT COLLISION
        # ------------------------------------------------------

        persistent_collision = (
            collision_count
            >=
            self.minimum_support_windows
        )

        # ------------------------------------------------------
        # PERSISTENT APPROACH
        # ------------------------------------------------------

        persistent_approach = (
            approach_count
            >=
            self.minimum_support_windows
        )

        # ------------------------------------------------------
        # PERSISTENT STRONG EVIDENCE
        # ------------------------------------------------------

        persistent_strong_evidence = (

            strong_evidence_count
            >=
            self.minimum_support_windows

            and

            average_evidence
            >=
            self.strong_evidence_threshold
        )

        # ------------------------------------------------------
        # SUPPORTING MOTION
        # ------------------------------------------------------

        supporting_motion = (

            persistent_collision

            or

            (
                persistent_approach
                and
                persistent_strong_evidence
            )
        )

        # ======================================================
        # ACCIDENT CONFIRMED
        # ======================================================

        if (
            persistent_suspicion
            and
            supporting_motion
        ):

            self.accident_confirmed = True

            return {

                "status":
                    "ACCIDENT CONFIRMED",

                "confirmed":
                    True,

                "confidence":
                    round(
                        average_evidence,
                        3
                    ),

                "suspicious_ratio":
                    round(
                        suspicious_ratio,
                        3
                    )
            }


        # ======================================================
        # SUSPICIOUS
        # ======================================================

        if suspicious_count > 0:

            return {

                "status":
                    "SUSPICIOUS",

                "confirmed":
                    False,

                "confidence":
                    round(
                        average_evidence,
                        3
                    ),

                "suspicious_ratio":
                    round(
                        suspicious_ratio,
                        3
                    )
            }


        # ======================================================
        # NORMAL
        # ======================================================

        return {

            "status":
                "NORMAL",

            "confirmed":
                False,

            "confidence":
                round(
                    average_evidence,
                    3
                ),

            "suspicious_ratio":
                round(
                    suspicious_ratio,
                    3
                )
        }


    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):

        self.history.clear()

        self.accident_confirmed = False