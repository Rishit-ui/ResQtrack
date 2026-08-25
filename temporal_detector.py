"""
ResQTrack - Temporal Accident Confirmation Engine

Designed for short-duration accident events.

Confirmation is based on:
1. Persistent suspicion across recent observations
2. At least one physical accident indication
3. At least one strong evidence observation

A brief collision does not need to remain visible
for every frame.
"""

from collections import deque


class TemporalAccidentDetector:

    def __init__(
        self,
        evidence_threshold=0.30,
        confirmation_windows=3,
        history_size=5,
        minimum_suspicious_ratio=0.67,
        strong_evidence_threshold=0.35
    ):

        self.evidence_threshold = (
            evidence_threshold
        )

        self.confirmation_windows = (
            confirmation_windows
        )

        self.history = deque(
            maxlen=history_size
        )

        self.minimum_suspicious_ratio = (
            minimum_suspicious_ratio
        )

        self.strong_evidence_threshold = (
            strong_evidence_threshold
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

        evidence_score = float(
            evidence_score
        )

        suspicious = (
            evidence_score >=
            self.evidence_threshold
        )

        physical_event = (
            bool(collision_evidence)
            or
            bool(approach_evidence)
        )

        self.history.append({

            "evidence_score":
                evidence_score,

            "suspicious":
                suspicious,

            "physical_event":
                physical_event
        })

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
        # HOW MANY RECENT OBSERVATIONS ARE SUSPICIOUS?
        # ------------------------------------------------------

        suspicious_count = sum(
            item["suspicious"]
            for item in recent
        )

        suspicious_ratio = (
            suspicious_count /
            len(recent)
        )


        # ------------------------------------------------------
        # WAS THERE ANY PHYSICAL EVENT?
        # ------------------------------------------------------

        physical_event_seen = any(
            item["physical_event"]
            for item in recent
        )


        # ------------------------------------------------------
        # STRONGEST EVIDENCE IN THE WINDOW
        #
        # We use MAX rather than average because an accident
        # can be extremely brief.
        # ------------------------------------------------------

        peak_evidence = max(
            item["evidence_score"]
            for item in recent
        )


        # ------------------------------------------------------
        # AVERAGE FOR DISPLAY ONLY
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
        # CONDITIONS
        # ------------------------------------------------------

        persistent_suspicion = (
            suspicious_ratio
            >=
            self.minimum_suspicious_ratio
        )

        strong_event = (
            peak_evidence
            >=
            self.strong_evidence_threshold
        )


        # ======================================================
        # FINAL ACCIDENT CONFIRMATION
        # ======================================================

        if (
            persistent_suspicion
            and
            physical_event_seen
            and
            strong_event
        ):

            self.accident_confirmed = True

            return {

                "status":
                    "ACCIDENT CONFIRMED",

                "confirmed":
                    True,

                "confidence":
                    round(
                        peak_evidence,
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
                        peak_evidence,
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