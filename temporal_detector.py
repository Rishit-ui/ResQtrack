from collections import deque


class TemporalAccidentDetector:

    def __init__(
        self,
        evidence_threshold=0.30,
        lookback=10,
        minimum_suspicious_observations=2,
        physical_event_threshold=0.40
    ):

        self.evidence_threshold = evidence_threshold
        self.lookback = lookback

        self.minimum_suspicious_observations = (
            minimum_suspicious_observations
        )

        self.physical_event_threshold = (
            physical_event_threshold
        )

        self.history = deque(
            maxlen=lookback
        )

        self.accident_confirmed = False


    # ==========================================================
    # UPDATE
    # ==========================================================

    def update(
        self,
        evidence_score,
        physical_score,
        collision_evidence=False
    ):

        evidence_score = float(
            evidence_score
        )

        physical_score = float(
            physical_score
        )

        suspicious = (
            evidence_score
            >=
            self.evidence_threshold
        )

        # A strong physical score itself is treated
        # as an accident-event observation.
        physical_event = (
            physical_score
            >=
            self.physical_event_threshold
            or
            bool(collision_evidence)
        )

        self.history.append({

            "evidence_score":
                evidence_score,

            "physical_score":
                physical_score,

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

        if not self.history:

            return {
                "status": "NORMAL",
                "confirmed": False,
                "confidence": 0.0
            }


        recent = list(self.history)


        # ------------------------------------------------------
        # ML / COMBINED SUSPICION
        # ------------------------------------------------------

        suspicious_count = sum(
            item["suspicious"]
            for item in recent
        )


        # ------------------------------------------------------
        # PHYSICAL EVENT
        # ------------------------------------------------------

        physical_event_seen = any(
            item["physical_event"]
            for item in recent
        )


        # ------------------------------------------------------
        # PEAK EVIDENCE
        # ------------------------------------------------------

        peak_evidence = max(
            item["evidence_score"]
            for item in recent
        )


        # ------------------------------------------------------
        # PEAK PHYSICAL SCORE
        # ------------------------------------------------------

        peak_physical = max(
            item["physical_score"]
            for item in recent
        )


        # ======================================================
        # CONFIRMATION
        # ======================================================
        #
        # We deliberately do NOT require the physical spike
        # to happen in the same frame as the ML peak.
        #
        # This is important because a real collision can cause:
        #
        # frame 1 → approach
        # frame 2 → collision
        # frame 3 → tracker separation
        #
        # ======================================================

        if (
            suspicious_count
            >=
            self.minimum_suspicious_observations

            and

            physical_event_seen

            and

            peak_evidence
            >=
            self.evidence_threshold
        ):

            self.accident_confirmed = True

            return {

                "status":
                    "ACCIDENT CONFIRMED",

                "confirmed":
                    True,

                "confidence":
                    round(
                        max(
                            peak_evidence,
                            peak_physical
                        ),
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
                        max(
                            peak_evidence,
                            peak_physical
                        ),
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
                0.0
        }


    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):

        self.history.clear()

        self.accident_confirmed = False