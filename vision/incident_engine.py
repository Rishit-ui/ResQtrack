"""ResQTrack incident reasoning engine.

The previous policy could only recognise three situations and it decided from a
single frame, which made it both blind (a rollover, a pile-up or a person
collapsing on the road were all "NORMAL") and jumpy (one noisy frame could
dispatch an ambulance).

This engine replaces that with a two-layer design:

**Layer 1 - hypothesis detectors.**  Every frame, a family of independent
detectors looks at the tracked actors and emits *findings*.  Each finding
carries a score, the actors involved and one or more `evidence types`:

    approach     the actors were converging before anything happened
    contact      their boxes actually met
    disruption   a trajectory broke: hard braking, a violent turn, a jolt
    pose_change  a box flipped shape - a rollover, or a person going down
    immobility   something that was moving fast is now stopped in the road
    departure    an actor left the contact point at speed
    multi_actor  several road users were disrupted together
    model_context the trained temporal model finds the scene anomalous

**Layer 2 - evidence accumulation.**  Findings are merged into hypotheses that
persist across frames.  A hypothesis is only CONFIRMED when it is supported on
several frames *and* by at least two independent evidence types.  That is what
separates a crash from a near-miss: a near-miss has approach and contact but
never acquires disruption or immobility.

The trained temporal model participates as `model_context` only.  It can be the
*second* corroborating type but never the first, so a model bias can never
dispatch an emergency on its own - the physical evidence has to be there.

Covered incident types
----------------------
* vehicle-to-vehicle collision (head-on, rear-end, side-swipe, T-bone)
* multi-vehicle pile-up
* vehicle-to-pedestrian collision
* pedestrian knocked down / person collapsed on the roadway
* vehicle rollover / overturn
* single-vehicle loss of control
* fixed-object / off-road impact
* possible hit-and-run (pedestrian or vehicle)
* post-crash immobilisation (a wreck blocking a live traffic lane)
* near-miss and vulnerable-road-user risk, reported as REVIEW only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable

from vision.kinematics import (
    ActorState,
    PairState,
    Point,
    as_float_box,
    clamp,
    compass_heading,
    containment as _containment,
    diagonal,
    iou as _iou,
    pair_key,
    ramp,
)

VEHICLE = "vehicle"
PERSON = "person"

NORMAL = "NORMAL"
REVIEW = "REVIEW"
CONFIRMED = "CONFIRMED"

SEVERITY_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

KIND_LABELS = {
    "vehicle_vehicle_collision": "Vehicle-to-vehicle collision",
    "multi_vehicle_pileup": "Multi-vehicle pile-up",
    "vehicle_pedestrian_collision": "Vehicle struck a pedestrian",
    "pedestrian_down": "Person down on the roadway",
    "vehicle_rollover": "Vehicle overturned",
    "single_vehicle_crash": "Single-vehicle loss of control",
    "fixed_object_impact": "Vehicle struck a fixed object",
    "possible_hit_and_run": "Possible hit-and-run",
    "vehicle_immobilised": "Wreck immobilised in a traffic lane",
    "vehicle_vehicle_risk": "Vehicles converging - near miss",
    "vehicle_pedestrian_risk": "Pedestrian in the path of a vehicle",
    "pedestrian_on_roadway": "Pedestrian exposed in live traffic",
}

# How dangerous each incident type is before speed and casualties are weighed.
KIND_BASE_SEVERITY = {
    "vehicle_pedestrian_collision": "CRITICAL",
    "possible_hit_and_run": "CRITICAL",
    "pedestrian_down": "HIGH",
    "multi_vehicle_pileup": "CRITICAL",
    "vehicle_rollover": "CRITICAL",
    "vehicle_vehicle_collision": "HIGH",
    "single_vehicle_crash": "HIGH",
    "fixed_object_impact": "HIGH",
    "vehicle_immobilised": "MODERATE",
}

# Ranked worst-first.  Used when one hypothesis collects several kinds.
KIND_PRIORITY = [
    "vehicle_pedestrian_collision",
    "possible_hit_and_run",
    "multi_vehicle_pileup",
    "vehicle_rollover",
    "vehicle_vehicle_collision",
    "pedestrian_down",
    "single_vehicle_crash",
    "fixed_object_impact",
    "vehicle_immobilised",
    "vehicle_pedestrian_risk",
    "vehicle_vehicle_risk",
    "pedestrian_on_roadway",
]

PHYSICAL_EVIDENCE = {
    "approach",
    "contact",
    "disruption",
    "pose_change",
    "immobility",
    "departure",
    "multi_actor",
    "vanished",
    "crowd",
}

# Something hit something, or a body/vehicle changed shape.
IMPACT_EVIDENCE = {"contact", "disruption", "pose_change", "multi_actor"}

# The road did not go back to normal afterwards.  This is the half that
# separates a crash from ordinary traffic: after a near miss everyone drives
# on, after a crash a vehicle is stopped in the lane, a person is down, or one
# party leaves the scene.
AFTERMATH_EVIDENCE = {"immobility", "pose_change", "departure", "crowd"}

# A track that dies right after a violent event is suggestive, but trackers
# lose identities to occlusion constantly, so it raises a hypothesis's score
# without ever being accepted as the corroboration that confirms one.
SUPPORTING_EVIDENCE = {"vanished"}


# ============================================================
# POLICY
# ============================================================

@dataclass
class EnginePolicy:
    """Every tunable threshold in one place.

    Motion values are box-diagonals per frame, so they are resolution and
    distance independent.  ``for_sensitivity`` gives the three presets exposed
    on the detector command line.
    """

    moving_speed: float = 0.020          # counts as "this actor is moving"
    fast_speed: float = 0.045            # counts as "moving with real energy"
    contact_gap: float = 0.040           # box gap that counts as contact
    near_gap: float = 0.16               # box gap that counts as a risk zone
    approach_frames: int = 2             # frames of closing before contact
    disruption_floor: float = 0.38       # minimum disruption worth reporting
    impact_disruption: float = 0.46      # disruption that reads as an impact
    impact_closing: float = 0.030        # closing rate that reads as a real hit

    confirm_score: float = 0.62          # peak score needed to confirm
    strong_score: float = 0.88           # violent impact that confirms alone
    review_score: float = 0.30           # peak score needed to show REVIEW
    support_frames: int = 3              # frames that must support a confirm
    hypothesis_ttl: int = 60             # frames a hypothesis survives unfed
    cooldown_frames: int = 150           # frames before the same key re-fires
    scene_cooldown_frames: int = 120     # one confirmed incident per scene window

    immobility_frames: int = 20          # stopped frames = wreck in the lane
    pileup_window: int = 25              # frames to gather a multi-vehicle event
    pileup_actors: int = 3               # disrupted vehicles that make a pile-up
    pileup_disruption: float = 0.58      # per-vehicle disruption inside a pile-up
    hit_and_run_min: int = 3             # frames a victim must be missing
    hit_and_run_max: int = 24
    min_track_frames: int = 12           # track age before a vanish means anything
    border_margin: float = 0.055         # frame fraction that counts as "the exit"
    ml_support: float = 0.55             # temporal-model probability that helps

    stale_frames: int = 45               # frames before a track is discarded

    @classmethod
    def for_sensitivity(cls, sensitivity: str) -> "EnginePolicy":
        sensitivity = (sensitivity or "balanced").lower()
        if sensitivity == "high":
            # Fewer misses, more REVIEW noise.  Useful on distant cameras.
            return cls(
                moving_speed=0.014,
                fast_speed=0.034,
                contact_gap=0.055,
                near_gap=0.20,
                approach_frames=1,
                disruption_floor=0.32,
                impact_disruption=0.38,
                impact_closing=0.022,
                confirm_score=0.56,
                strong_score=0.80,
                support_frames=2,
                immobility_frames=14,
                pileup_disruption=0.50,
            )
        if sensitivity == "strict":
            # For unattended dispatch, where a false ambulance is expensive.
            return cls(
                moving_speed=0.026,
                fast_speed=0.055,
                contact_gap=0.030,
                approach_frames=3,
                disruption_floor=0.45,
                impact_disruption=0.55,
                impact_closing=0.042,
                confirm_score=0.70,
                strong_score=0.93,
                support_frames=5,
                immobility_frames=26,
                pileup_disruption=0.65,
            )
        return cls()


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass(frozen=True)
class IncidentSignal:
    """One piece of supporting evidence, in words a human can audit."""

    name: str
    score: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 3), "detail": self.detail}


@dataclass(frozen=True)
class IncidentEvidence:
    """The strongest incident hypothesis in the current frame."""

    status: str = NORMAL
    kind: str = ""
    confidence: float = 0.0
    actor_ids: tuple[int, ...] = ()
    reason: str = "No dynamic incident evidence"
    severity: str = "LOW"
    signals: tuple[IncidentSignal, ...] = ()
    evidence_types: tuple[str, ...] = ()
    frame: int = 0
    ml_probability: float = 0.0
    involved: tuple[dict[str, Any], ...] = ()

    @property
    def confirmed(self) -> bool:
        return self.status == CONFIRMED

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind.replace("_", " ").title() or "Normal traffic")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "actor_ids": list(self.actor_ids),
            "reason": self.reason,
            "signals": [signal.as_dict() for signal in self.signals],
            "evidence_types": list(self.evidence_types),
            "frame": self.frame,
            "ml_probability": round(self.ml_probability, 4),
            "involved": [dict(item) for item in self.involved],
        }


@dataclass
class Finding:
    """A single detector's opinion about the current frame."""

    kind: str
    score: float
    actor_ids: tuple[int, ...]
    evidence_types: frozenset[str]
    detail: str
    reinforcement: bool = False  # attaches to existing hypotheses by actor


@dataclass
class Hypothesis:
    """Evidence accumulated for one candidate incident across frames."""

    key: str
    kind: str
    actor_ids: tuple[int, ...]
    first_frame: int
    last_frame: int
    peak: float = 0.0
    support_frames: int = 0
    evidence_types: set[str] = field(default_factory=set)
    signals: dict[str, IncidentSignal] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    confirmed_at: int | None = None

    def add(self, finding: Finding, frame: int) -> None:
        self.last_frame = frame
        self.peak = max(self.peak, finding.score)
        self.evidence_types |= set(finding.evidence_types)
        if finding.score >= 0.25:
            self.support_frames += 1
        for name in finding.evidence_types:
            existing = self.signals.get(name)
            if existing is None or finding.score > existing.score:
                self.signals[name] = IncidentSignal(name, finding.score, finding.detail)
        if finding.detail not in self.reasons:
            self.reasons.append(finding.detail)
            del self.reasons[:-6]
        # A hypothesis always reports its worst observed interpretation.
        if _kind_rank(finding.kind) < _kind_rank(self.kind):
            self.kind = finding.kind
        self.actor_ids = tuple(sorted(set(self.actor_ids) | set(finding.actor_ids)))

    @property
    def physical_types(self) -> set[str]:
        return self.evidence_types & PHYSICAL_EVIDENCE


def _kind_rank(kind: str) -> int:
    return KIND_PRIORITY.index(kind) if kind in KIND_PRIORITY else len(KIND_PRIORITY)


# ============================================================
# ENGINE
# ============================================================

class IncidentEngine:
    """Stateful, explainable incident detection for one camera."""

    def __init__(
        self,
        policy: EnginePolicy | None = None,
        fps: float = 25.0,
        frame_size: tuple[int, int] | None = None,
    ):
        self.policy = policy or EnginePolicy()
        self.fps = fps if fps > 0 else 25.0
        self.frame_size = frame_size
        self.frame_number = 0
        self.timestamp = 0.0
        self.ml_probability = 0.0

        self.actors: dict[int, ActorState] = {}
        self.pairs: dict[tuple[int, int], PairState] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.cooldown: dict[str, int] = {}
        self.contact_log: dict[tuple[int, int], dict[str, Any]] = {}
        self.impact_watch: dict[int, dict[str, Any]] = {}
        self.recent_disruptions: dict[int, int] = {}
        self.strong_disruptions: dict[int, int] = {}
        self.single_echo: dict[int, dict[str, Any]] = {}
        self.scene_cooldown_until: int = -1
        self.confirmed_history: list[IncidentEvidence] = []
        self.latest = IncidentEvidence()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.frame_number = 0
        self.timestamp = 0.0
        self.ml_probability = 0.0
        self.actors.clear()
        self.pairs.clear()
        self.hypotheses.clear()
        self.cooldown.clear()
        self.contact_log.clear()
        self.impact_watch.clear()
        self.recent_disruptions.clear()
        self.strong_disruptions.clear()
        self.single_echo.clear()
        self.scene_cooldown_until = -1
        self.confirmed_history.clear()
        self.latest = IncidentEvidence()

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def update(
        self,
        actors: dict[int, dict[str, Any]],
        *,
        frame_number: int | None = None,
        timestamp: float | None = None,
        ml_probability: float | None = None,
    ) -> IncidentEvidence:
        """Process one tracked frame and return the strongest evidence.

        ``actors`` maps tracker IDs to dicts with at least ``kind`` (``vehicle``
        or ``person``) and ``box`` as ``(x1, y1, x2, y2)``.  ``class_name`` and
        ``confidence`` are used for reporting when present.
        """
        self.frame_number = frame_number if frame_number is not None else self.frame_number + 1
        self.timestamp = (
            timestamp if timestamp is not None else self.frame_number / self.fps
        )
        if ml_probability is not None:
            # Smooth the model so a single spiky window cannot swing the policy.
            self.ml_probability = 0.7 * self.ml_probability + 0.3 * float(ml_probability)

        self._observe(actors)
        self._observe_pairs(actors)

        findings: list[Finding] = []
        findings.extend(self._pair_findings(actors))
        findings.extend(self._pose_findings(actors))
        findings.extend(self._single_vehicle_findings(actors))
        findings.extend(self._immobility_findings(actors))
        findings.extend(self._crowd_findings(actors))
        findings.extend(self._pileup_findings(actors))
        findings.extend(self._vanished_findings(actors))
        findings.extend(self._hit_and_run_findings(actors))
        findings.extend(self._exposure_findings(actors))
        findings.extend(self._model_context_findings())

        for finding in findings:
            self._ingest(finding)

        self._expire()
        self.latest = self._resolve()
        return self.latest

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _observe(self, actors: dict[int, dict[str, Any]]) -> None:
        for actor_id, actor in actors.items():
            box = as_float_box(actor["box"])
            state = self.actors.get(actor_id)
            if state is None:
                state = ActorState(
                    actor_id=actor_id,
                    kind=actor.get("kind", VEHICLE),
                    class_name=str(actor.get("class_name", actor.get("kind", "object"))),
                )
                self.actors[actor_id] = state
            state.kind = actor.get("kind", state.kind)
            state.class_name = str(actor.get("class_name", state.class_name))
            state.observe(
                self.frame_number,
                self.timestamp,
                box,
                float(actor.get("confidence", 0.0)),
            )
            disruption = self._disruption(state)
            if disruption >= self.policy.disruption_floor:
                self.recent_disruptions[actor_id] = self.frame_number
            if disruption >= self.policy.pileup_disruption:
                self.strong_disruptions[actor_id] = self.frame_number

    def _observe_pairs(self, actors: dict[int, dict[str, Any]]) -> None:
        ids = sorted(actors)
        for index, first_id in enumerate(ids):
            first = self.actors[first_id]
            for second_id in ids[index + 1:]:
                second = self.actors[second_id]
                key = pair_key(first_id, second_id)
                pair = self.pairs.get(key)
                if pair is None:
                    pair = PairState(key)
                    self.pairs[key] = pair
                pair.observe(self.frame_number, first.box, second.box)

    # ------------------------------------------------------------------
    # scoring helpers
    # ------------------------------------------------------------------
    def _disruption(self, actor: ActorState) -> float:
        """How violently did this actor's trajectory break, in 0..1?"""
        policy = self.policy
        # A track's first frames are meaningless: it starts at zero speed, so
        # frame two always looks like a violent jolt.  In dense traffic the
        # tracker creates new identities constantly, and treating each birth as
        # an impact was the main source of false alarms.
        if actor.track_length < 6:
            return 0.0
        # Every term is gated on the actor having had real speed beforehand, so
        # appearing, re-appearing or being re-identified cannot look like a crash.
        established = ramp(actor.previous_speed, 0.010, 0.028)
        drop = ramp(actor.speed_drop, 0.030, 0.110)
        jump = ramp(actor.speed_jump, 0.045, 0.180) * established
        turn = ramp(actor.turn, 45.0, 120.0) * ramp(actor.previous_speed, 0.018, 0.050)
        pose = ramp(actor.aspect_shift, 0.30, 0.70) * 0.9
        # A crash rarely stops a car within one frame, so multi-frame
        # decelerations have to count too.  The band is set well above firm
        # service braking: a driver stopping hard for a light loses speed over
        # several frames, an impact loses it almost all at once.
        sustained = ramp(
            actor.speed_before(2, 4) - actor.speed_over(3), 0.080, 0.250
        ) * ramp(actor.speed_before(2, 4), policy.moving_speed, policy.fast_speed)
        return clamp(max(drop, jump, turn, pose, sustained))

    def _energy(self, *actors: ActorState) -> float:
        """Impact energy proxy from the fastest actor just before the event."""
        speeds = [max(actor.speed_before(1, 5), actor.peak_speed * 0.6) for actor in actors]
        return ramp(max(speeds, default=0.0), self.policy.moving_speed, 0.11)

    def _moving(self, actor: ActorState) -> bool:
        if actor.track_length < 5:
            return False
        return (
            actor.smooth_speed >= self.policy.moving_speed
            or actor.speed_before(1, 5) >= self.policy.moving_speed
            or actor.was_moving(10, self.policy.moving_speed)
        )

    # ------------------------------------------------------------------
    # detectors
    # ------------------------------------------------------------------
    def _pair_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        policy = self.policy
        findings: list[Finding] = []
        ids = sorted(actors)

        for index, first_id in enumerate(ids):
            first = self.actors[first_id]
            for second_id in ids[index + 1:]:
                second = self.actors[second_id]
                pair = self.pairs[pair_key(first_id, second_id)]
                if not pair.near:
                    continue

                approached = pair.approached(policy.approach_frames) or pair.gap_drop(8) >= 0.09
                contact = pair.gap <= policy.contact_gap or pair.overlap >= 0.02 or pair.containment >= 0.35
                if contact:
                    self.contact_log[pair.key] = {
                        "frame": self.frame_number,
                        "kinds": (first.kind, second.kind),
                        "point": first.center,
                        "closing": pair.peak_closing_rate,
                    }

                if first.kind == VEHICLE and second.kind == VEHICLE:
                    findings.extend(
                        self._vehicle_pair(first, second, pair, approached, contact)
                    )
                elif {first.kind, second.kind} == {VEHICLE, PERSON}:
                    vehicle, person = (
                        (first, second) if first.kind == VEHICLE else (second, first)
                    )
                    findings.extend(
                        self._vehicle_person_pair(vehicle, person, pair, approached, contact)
                    )
        return findings

    def _vehicle_pair(
        self,
        first: ActorState,
        second: ActorState,
        pair: PairState,
        approached: bool,
        contact: bool,
    ) -> list[Finding]:
        policy = self.policy
        moving = self._moving(first) or self._moving(second)
        if not moving:
            # Parked cars, a stopped queue, or two boxes that merely overlap in
            # the image plane.  This is the false positive the old build had.
            return []

        disruption = max(self._disruption(first), self._disruption(second))
        energy = self._energy(first, second)
        ids = (first.actor_id, second.actor_id)

        if contact and approached:
            self._watch_for_immobility(first, second)
            violent_closing = pair.peak_closing_rate >= policy.impact_closing
            if disruption >= policy.impact_disruption and violent_closing:
                score = clamp(
                    0.40
                    + 0.22 * disruption
                    + 0.14 * energy
                    + 0.12 * ramp(pair.overlap, 0.02, 0.30)
                    + 0.12 * ramp(pair.peak_closing_rate, policy.impact_closing, 0.14)
                )
                return [
                    Finding(
                        "vehicle_vehicle_collision",
                        score,
                        ids,
                        frozenset({"approach", "contact", "disruption"}),
                        (
                            f"vehicles #{ids[0]} and #{ids[1]} converged, made contact "
                            f"and one trajectory broke (disruption {disruption:.2f})"
                        ),
                    )
                ]
            return [
                Finding(
                    "vehicle_vehicle_risk",
                    clamp(0.36 + 0.12 * energy + 0.10 * disruption),
                    ids,
                    frozenset({"approach", "contact"}),
                    f"vehicles #{ids[0]} and #{ids[1]} touched without an impact signature yet",
                )
            ]

        if approached and pair.gap <= policy.near_gap and energy >= 0.25:
            return [
                Finding(
                    "vehicle_vehicle_risk",
                    clamp(0.30 + 0.10 * energy),
                    ids,
                    frozenset({"approach"}),
                    f"vehicles #{ids[0]} and #{ids[1]} closing fast - near miss",
                )
            ]
        return []

    def _vehicle_person_pair(
        self,
        vehicle: ActorState,
        person: ActorState,
        pair: PairState,
        approached: bool,
        contact: bool,
    ) -> list[Finding]:
        policy = self.policy
        if not self._moving(vehicle):
            # A person walking past a parked car is not an incident.
            return []

        ids = (vehicle.actor_id, person.actor_id)
        energy = self._energy(vehicle)
        person_disruption = self._disruption(person)
        person_down = person.sustained_aspect_flip(3, 0.38)

        if contact and (approached or pair.peak_closing_rate >= 0.02):
            self._watch_for_immobility(vehicle)
            self.impact_watch.setdefault(
                person.actor_id,
                {"frame": self.frame_number, "partner": vehicle.actor_id, "point": person.center},
            )
            score = 0.50 + 0.16 * energy
            types = {"approach", "contact"}
            detail = f"moving vehicle #{ids[0]} made contact with pedestrian #{ids[1]}"
            if person_disruption >= policy.disruption_floor:
                score += 0.24 * person_disruption
                types.add("disruption")
                detail += f"; pedestrian trajectory broke ({person_disruption:.2f})"
            if person_down:
                score += 0.22
                types.add("pose_change")
                detail += "; pedestrian is down"
            return [
                Finding(
                    "vehicle_pedestrian_collision",
                    clamp(score),
                    ids,
                    frozenset(types),
                    detail,
                )
            ]

        if approached and pair.gap <= policy.near_gap and energy >= 0.20:
            return [
                Finding(
                    "vehicle_pedestrian_risk",
                    clamp(0.34 + 0.14 * energy),
                    ids,
                    frozenset({"approach"}),
                    f"vehicle #{ids[0]} is closing on pedestrian #{ids[1]}",
                )
            ]
        return []

    def _pose_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """Rollovers and people going down - shape changes, not speed changes."""
        findings: list[Finding] = []
        for actor_id in actors:
            actor = self.actors[actor_id]
            # A pose verdict compares the box against this track's own history,
            # so the track has to have a history worth comparing against.
            if actor.track_length < 25 or len(actor.aspect_baseline) < 12:
                continue

            if actor.kind == VEHICLE:
                # Both tests must agree: the box left the shape range a vehicle
                # can present from any camera angle *and* it changed sharply
                # from this track's own baseline.
                overturned = (
                    actor.aspect_reference >= 1.15
                    and actor.sustained_aspect_beyond(0.80, 8, above=False)
                    and actor.sustained_aspect_flip(8, 0.45)
                )
                if overturned and actor.was_moving(30, self.policy.fast_speed):
                    disruption = self._disruption(actor)
                    # An overturned vehicle does not drive on.  Requiring it to
                    # have stopped removes the partly-occluded cars that read
                    # as "tall" for a moment in dense traffic.
                    if actor.stationary_frames >= 6:
                        findings.append(
                            Finding(
                                "vehicle_rollover",
                                clamp(0.54 + 0.20 * self._energy(actor) + 0.16 * disruption),
                                (actor_id,),
                                frozenset({"pose_change", "disruption"}),
                                f"vehicle #{actor_id} is no longer upright - probable rollover",
                            )
                        )
                        self._watch_for_immobility(actor)

            if actor.kind == PERSON:
                # A standing person is much taller than wide.  A box that is
                # wider than tall for several frames means the person is down.
                down = actor.sustained_aspect_beyond(0.95, 5, above=True)
                if down and actor.smooth_speed <= self.policy.fast_speed:
                    struck = self.impact_watch.get(actor_id)
                    if struck and struck.get("partner") and self.frame_number - struck["frame"] <= 60:
                        kind, base = "vehicle_pedestrian_collision", 0.74
                        detail = f"pedestrian #{actor_id} is down after vehicle contact"
                        ids: tuple[int, ...] = (struck["partner"], actor_id)
                    else:
                        kind, base = "pedestrian_down", 0.56
                        detail = f"person #{actor_id} is down and not getting up"
                        ids = (actor_id,)
                    findings.append(
                        Finding(
                            kind,
                            clamp(base + 0.16 * ramp(actor.stationary_frames, 3, 25)),
                            ids,
                            frozenset({"pose_change"}),
                            detail,
                        )
                    )
        return findings

    def _single_vehicle_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """Loss of control and fixed-object impacts, with no second road user."""
        policy = self.policy
        findings: list[Finding] = []
        for actor_id in actors:
            actor = self.actors[actor_id]
            if actor.kind != VEHICLE or actor.track_length < 8:
                continue
            if self._has_live_contact(actor_id):
                continue  # a pair detector already owns this event

            approach_speed = actor.speed_before(1, 5)
            if approach_speed < policy.fast_speed:
                continue

            spin = ramp(actor.turn, 60.0, 140.0)
            braking = ramp(actor.speed_drop, 0.045, 0.130)
            sustained = ramp(approach_speed - actor.speed_over(3), 0.040, 0.120)
            severity_signal = max(spin, braking, sustained)
            if severity_signal < 0.45:
                continue

            kind = "single_vehicle_crash" if spin >= max(braking, sustained) else "fixed_object_impact"
            detail = (
                f"vehicle #{actor_id} swerved {actor.turn:.0f} deg at speed"
                if kind == "single_vehicle_crash"
                else f"vehicle #{actor_id} decelerated violently with no vehicle ahead"
            )
            score = clamp(0.42 + 0.22 * severity_signal + 0.16 * self._energy(actor))
            findings.append(
                Finding(kind, score, (actor_id,), frozenset({"disruption"}), detail)
            )
            # A loss of control is over in a frame or two, but the evidence for
            # it is not.  Echoing it keeps the hypothesis alive long enough for
            # the aftermath (the vehicle stopping, or its track breaking) to
            # arrive and corroborate it.
            self.single_echo[actor_id] = {
                "frame": self.frame_number,
                "kind": kind,
                "score": score,
                "detail": detail,
            }
            self._watch_for_immobility(actor)

        for actor_id, echo in list(self.single_echo.items()):
            age = self.frame_number - echo["frame"]
            if age <= 0:
                continue
            if age > 14 or actor_id not in self.actors:
                del self.single_echo[actor_id]
                continue
            findings.append(
                Finding(
                    echo["kind"],
                    clamp(echo["score"] * (1.0 - 0.04 * age)),
                    (actor_id,),
                    frozenset({"disruption"}),
                    echo["detail"],
                )
            )
        return findings

    def _traffic_is_flowing(self, actors: dict[int, dict[str, Any]]) -> tuple[bool, float]:
        """Is the rest of the traffic still moving, or is the whole scene stopped?

        This is the difference between a wreck and a traffic jam.  A vehicle
        standing still in a queue where nothing moves is congestion; a vehicle
        standing still while everything else flows past it is an obstruction.
        """
        vehicles = [
            self.actors[actor_id]
            for actor_id in actors
            if self.actors[actor_id].kind == VEHICLE and self.actors[actor_id].track_length >= 5
        ]
        if len(vehicles) < 3:
            return True, 1.0  # too few road users to call it a jam
        moving = sum(1 for actor in vehicles if actor.smooth_speed >= self.policy.moving_speed)
        fraction = moving / len(vehicles)
        return fraction >= 0.35, fraction

    def _vanished_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """A track that dies mid-scene moments after a violent disruption.

        Crashed vehicles change shape and orientation, which frequently breaks
        the tracker.  That is real aftermath - but only when the actor was
        violently disrupted first and did not simply leave through the edge of
        the frame.
        """
        findings: list[Finding] = []
        for actor_id, frame in list(self.strong_disruptions.items()):
            if actor_id in actors:
                continue
            actor = self.actors.get(actor_id)
            if actor is None or actor.track_length < self.policy.min_track_frames:
                continue
            gone = self.frame_number - actor.last_frame
            if not 4 <= gone <= 20:
                continue
            if self.frame_number - frame > 30:
                continue
            if self._left_through_border(actor):
                continue
            if self._disruption(actor) < 0.70 and actor.peak_speed < self.policy.fast_speed:
                continue
            if self._probably_occluded(actor, actors):
                continue
            findings.append(
                Finding(
                    "",
                    0.52,
                    (actor_id,),
                    frozenset({"vanished"}),
                    f"#{actor_id} stopped being trackable right after a violent disruption",
                    reinforcement=True,
                )
            )
        return findings

    def _crowd_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """People converging on a disrupted vehicle.

        On urban Indian roads the clearest sign that something went wrong is
        that bystanders walk towards it.  A crash draws a crowd; a car merely
        driving past does not.  The test is the *change* in how many people are
        beside the vehicle, so a busy footpath does not register - only a
        gathering that was not there a second ago.
        """
        findings: list[Finding] = []
        people = [
            self.actors[actor_id]
            for actor_id in actors
            if self.actors[actor_id].kind == PERSON and self.actors[actor_id].track_length >= 6
        ]
        if len(people) < 3:
            return findings

        for actor_id, watch in self.impact_watch.items():
            vehicle = self.actors.get(actor_id)
            if vehicle is None or actor_id not in actors or vehicle.kind != VEHICLE:
                continue
            if self.frame_number - watch["frame"] > 200:
                continue

            radius = 2.6 * diagonal(vehicle.box)
            approaching = 0
            for person in people:
                distance_now = hypot(
                    person.center[0] - vehicle.center[0],
                    person.center[1] - vehicle.center[1],
                )
                if distance_now > radius:
                    continue
                history = list(person.history)
                if len(history) < 14:
                    continue
                past = history[-14]
                distance_before = hypot(
                    past.center[0] - vehicle.center[0],
                    past.center[1] - vehicle.center[1],
                )
                # Closed at least a fifth of the way in half a second.
                if distance_before - distance_now >= 0.20 * radius:
                    approaching += 1

            baseline = watch.get("crowd_baseline", 0)
            watch["crowd_baseline"] = max(baseline, approaching)
            if approaching >= 3:
                findings.append(
                    Finding(
                        "",
                        clamp(0.45 + 0.10 * (approaching - 3)),
                        (actor_id,),
                        frozenset({"crowd"}),
                        f"{approaching} bystanders converged on vehicle #{actor_id}",
                        reinforcement=True,
                    )
                )
        return findings

    def _immobility_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """A vehicle that was fast and is now parked in the road is a wreck.

        This is the signal that separates a real crash from a near miss: after a
        near miss everyone drives on, after a crash the vehicle stays put - while
        the rest of the traffic keeps moving around it.
        """
        policy = self.policy
        findings: list[Finding] = []
        flowing, flow_fraction = self._traffic_is_flowing(actors)
        if not flowing:
            # Everything is stopped: this is congestion, not a wreck.
            return findings
        for actor_id, watch in list(self.impact_watch.items()):
            actor = self.actors.get(actor_id)
            if actor is None or self.frame_number - watch["frame"] > 240:
                self.impact_watch.pop(actor_id, None)
                continue
            if actor_id not in actors:
                continue
            if actor.stationary_frames < policy.immobility_frames:
                continue
            if watch.get("reported_at") and self.frame_number - watch["reported_at"] < 5:
                continue
            if actor.peak_speed < policy.fast_speed:
                continue
            watch["reported_at"] = self.frame_number
            held = actor.stationary_frames / self.fps
            findings.append(
                Finding(
                    "vehicle_immobilised",
                    clamp(0.46 + 0.24 * ramp(held, 0.6, 4.0) + 0.10 * ramp(flow_fraction, 0.35, 0.9)),
                    (actor_id,),
                    frozenset({"immobility"}),
                    (
                        f"vehicle #{actor_id} has not moved for {held:.1f}s while "
                        f"{flow_fraction * 100:.0f}% of the traffic keeps flowing"
                    ),
                    reinforcement=True,
                )
            )
        return findings

    def _pileup_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """Several vehicles violently disrupted together, with a real contact.

        Heavy traffic makes vehicles brake together all the time, so braking
        alone is worthless here.  A pile-up needs hard disruption *and* at least
        one pair inside the cluster that actually touched.
        """
        policy = self.policy
        window = policy.pileup_window
        disrupted = [
            actor_id
            for actor_id, record in self.strong_disruptions.items()
            if self.frame_number - record <= window
            and actor_id in actors
            and self.actors[actor_id].kind == VEHICLE
        ]
        if len(disrupted) < policy.pileup_actors:
            return []

        # Only count vehicles that are actually in one another's neighbourhood;
        # two unrelated hard-braking events at opposite ends of a junction are
        # not a pile-up.
        cluster = self._largest_cluster(disrupted)
        if len(cluster) < policy.pileup_actors:
            return []

        contacts = [
            key
            for key, contact in self.contact_log.items()
            if key[0] in cluster
            and key[1] in cluster
            and self.frame_number - contact["frame"] <= window
            and contact.get("closing", 0.0) >= policy.impact_closing
        ]
        if not contacts:
            return []

        # A pile-up comes to rest.  Queueing traffic keeps rolling, so without
        # vehicles that have actually stopped this is just congestion.
        stopped = [
            actor_id
            for actor_id in cluster
            if self.actors[actor_id].stationary_frames >= policy.immobility_frames // 2
        ]
        if len(stopped) < 2:
            return []

        energy = self._energy(*[self.actors[actor_id] for actor_id in cluster])
        for actor_id in cluster:
            self._watch_for_immobility(self.actors[actor_id])
        return [
            Finding(
                "multi_vehicle_pileup",
                clamp(0.56 + 0.06 * (len(cluster) - policy.pileup_actors) + 0.18 * energy),
                tuple(sorted(cluster)),
                frozenset({"multi_actor", "contact", "disruption", "immobility"}),
                (
                    f"{len(cluster)} vehicles collided and {len(stopped)} of them are "
                    "stopped in the carriageway - probable pile-up"
                ),
            )
        ]

    def _largest_cluster(self, candidates: list[int]) -> list[int]:
        """Group disrupted vehicles that are within a few box-widths of each other."""
        best: list[int] = []
        for anchor in candidates:
            anchor_state = self.actors[anchor]
            scale = diagonal(anchor_state.box)
            group = [
                other
                for other in candidates
                if hypot(
                    self.actors[other].center[0] - anchor_state.center[0],
                    self.actors[other].center[1] - anchor_state.center[1],
                )
                <= 3.5 * scale
            ]
            if len(group) > len(best):
                best = group
        return best

    def _hit_and_run_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """A contact, then the victim's track dies while the other party leaves.

        A vanished track is weak evidence on its own - trackers lose identities
        behind occlusions all the time, and every road user eventually leaves
        through the edge of the frame.  So the victim must have been tracked
        long enough to be real, must have disappeared away from the frame
        border, and must have been physically disrupted at the contact.
        """
        policy = self.policy
        findings: list[Finding] = []
        for key, contact in list(self.contact_log.items()):
            age = self.frame_number - contact["frame"]
            if age > policy.hit_and_run_max + 10:
                del self.contact_log[key]
                continue
            if age < policy.hit_and_run_min:
                continue
            if contact.get("closing", 0.0) < 0.020:
                continue  # boxes drifted together; nothing actually ran into anything

            for victim_id, runner_id in ((key[0], key[1]), (key[1], key[0])):
                victim = self.actors.get(victim_id)
                runner = self.actors.get(runner_id)
                if victim is None or runner is None:
                    continue
                if victim_id in actors or runner_id not in actors:
                    continue
                if victim.track_length < policy.min_track_frames:
                    continue  # a track this young vanishing is a tracker artefact
                missing = self.frame_number - victim.last_frame
                if not policy.hit_and_run_min <= missing <= policy.hit_and_run_max:
                    continue
                if runner.kind != VEHICLE or not self._moving(runner):
                    continue
                if self._left_through_border(victim):
                    continue  # the victim simply drove/walked out of shot
                disrupted_at_contact = (
                    self.frame_number - self.strong_disruptions.get(victim_id, -999)
                    <= age + 8
                )
                if not disrupted_at_contact:
                    continue  # nothing actually happened before the track ended
                if victim.kind == VEHICLE and victim.stationary_frames < 6:
                    # A moving car box that vanishes is occlusion or a lost
                    # identity.  Only a vehicle that was already stopped -
                    # a wreck - counts as a victim left behind.
                    continue
                departed = hypot(
                    runner.center[0] - contact["point"][0],
                    runner.center[1] - contact["point"][1],
                ) / diagonal(runner.box)
                if departed < 0.40:
                    continue
                victim_label = "pedestrian" if victim.kind == PERSON else "vehicle"
                base = 0.70 if victim.kind == PERSON else 0.62
                findings.append(
                    Finding(
                        "possible_hit_and_run",
                        clamp(base + 0.16 * ramp(departed, 0.4, 2.0) + 0.10 * self._energy(runner)),
                        (runner_id, victim_id),
                        frozenset({"contact", "departure"}),
                        (
                            f"{victim_label} #{victim_id} disappeared after contact while "
                            f"vehicle #{runner_id} drove away"
                        ),
                    )
                )
        return findings

    def _probably_occluded(
        self, actor: ActorState, actors: dict[int, dict[str, Any]]
    ) -> bool:
        """Did something drive in front of the place this track disappeared?

        In dense traffic a lost identity almost always means one road user
        passed behind another, so a disappearance is only evidence of a crash
        when nothing is standing where the track went dark.
        """
        last_box = actor.box
        for other_id in actors:
            other = self.actors.get(other_id)
            if other is None or other_id == actor.actor_id:
                continue
            if _iou(last_box, other.box) >= 0.08 or _containment(last_box, other.box) >= 0.30:
                return True
        return False

    def _left_through_border(self, actor: ActorState) -> bool:
        """True if the actor's last box was touching the edge of the frame."""
        if not self.frame_size:
            return False
        frame_width, frame_height = self.frame_size
        margin_x = frame_width * self.policy.border_margin
        margin_y = frame_height * self.policy.border_margin
        x1, y1, x2, y2 = actor.box
        return (
            x1 <= margin_x
            or y1 <= margin_y
            or x2 >= frame_width - margin_x
            or y2 >= frame_height - margin_y
        )

    def _exposure_findings(self, actors: dict[int, dict[str, Any]]) -> list[Finding]:
        """A pedestrian standing in live, fast traffic - advisory REVIEW only."""
        findings: list[Finding] = []
        for actor_id in actors:
            person = self.actors[actor_id]
            if person.kind != PERSON or person.track_length < 10:
                continue
            fast_neighbours = [
                other_id
                for other_id in actors
                if self.actors[other_id].kind == VEHICLE
                and self.pairs.get(pair_key(actor_id, other_id))
                and self.pairs[pair_key(actor_id, other_id)].gap <= 0.45
                and self.actors[other_id].smooth_speed >= self.policy.fast_speed
            ]
            if len(fast_neighbours) >= 2:
                findings.append(
                    Finding(
                        "pedestrian_on_roadway",
                        0.32,
                        (actor_id, *fast_neighbours[:2]),
                        frozenset({"approach"}),
                        f"pedestrian #{actor_id} is exposed to {len(fast_neighbours)} moving vehicles",
                    )
                )
        return findings

    def _model_context_findings(self) -> list[Finding]:
        """Let the trained temporal model corroborate, never initiate."""
        if self.ml_probability < self.policy.ml_support or not self.hypotheses:
            return []
        boost = 0.20 * ramp(self.ml_probability, self.policy.ml_support, 0.95)
        return [
            Finding(
                "",
                boost,
                tuple(sorted({actor for item in self.hypotheses.values() for actor in item.actor_ids})),
                frozenset({"model_context"}),
                f"temporal model rates this window anomalous ({self.ml_probability:.2f})",
                reinforcement=True,
            )
        ]

    # ------------------------------------------------------------------
    # hypothesis bookkeeping
    # ------------------------------------------------------------------
    def _watch_for_immobility(self, *actors: ActorState) -> None:
        for actor in actors:
            self.impact_watch.setdefault(
                actor.actor_id,
                {"frame": self.frame_number, "partner": None, "point": actor.center},
            )

    def _has_live_contact(self, actor_id: int) -> bool:
        return any(
            actor_id in key and self.frame_number - contact["frame"] <= 8
            for key, contact in self.contact_log.items()
        )

    def _ingest(self, finding: Finding) -> None:
        if finding.reinforcement:
            targets = [
                item
                for item in self.hypotheses.values()
                if set(item.actor_ids) & set(finding.actor_ids)
            ]
            if targets:
                for hypothesis in targets:
                    if finding.kind:
                        score = finding.score
                        actor_ids = finding.actor_ids
                    else:
                        # A supporting signal nudges the hypothesis; it must
                        # never be able to push one over the "violent impact
                        # confirms on its own" line, or model context and lost
                        # tracks would start dispatching ambulances.
                        score = min(
                            hypothesis.peak + 0.25 * finding.score,
                            self.policy.strong_score - 0.02,
                        )
                        score = max(score, hypothesis.peak)
                        actor_ids = hypothesis.actor_ids
                    hypothesis.add(
                        Finding(
                            finding.kind or hypothesis.kind,
                            score,
                            actor_ids,
                            finding.evidence_types,
                            finding.detail,
                        ),
                        self.frame_number,
                    )
                return
            if not finding.kind:
                return  # pure model context with nothing to corroborate

        key = self._hypothesis_key(finding)
        hypothesis = self.hypotheses.get(key)
        if hypothesis is None:
            hypothesis = Hypothesis(
                key=key,
                kind=finding.kind,
                actor_ids=finding.actor_ids,
                first_frame=self.frame_number,
                last_frame=self.frame_number,
            )
            self.hypotheses[key] = hypothesis
        hypothesis.add(finding, self.frame_number)

    @staticmethod
    def _hypothesis_key(finding: Finding) -> str:
        return "actors:" + "-".join(str(actor) for actor in sorted(finding.actor_ids))

    def _expire(self) -> None:
        policy = self.policy
        for key, hypothesis in list(self.hypotheses.items()):
            if self.frame_number - hypothesis.last_frame > policy.hypothesis_ttl:
                del self.hypotheses[key]
        for actor_id, state in list(self.actors.items()):
            if self.frame_number - state.last_frame > policy.stale_frames:
                del self.actors[actor_id]
                self.recent_disruptions.pop(actor_id, None)
                self.strong_disruptions.pop(actor_id, None)
        for key in list(self.pairs):
            if self.frame_number - self.pairs[key].last_frame > policy.stale_frames:
                del self.pairs[key]
        for key, frame in list(self.cooldown.items()):
            if self.frame_number - frame > policy.cooldown_frames:
                del self.cooldown[key]

    # ------------------------------------------------------------------
    # verdict
    # ------------------------------------------------------------------
    def _resolve(self) -> IncidentEvidence:
        policy = self.policy
        best: tuple[float, IncidentEvidence] | None = None

        for hypothesis in self.hypotheses.values():
            if not hypothesis.physical_types:
                continue

            types = hypothesis.evidence_types
            # Something has to have actually happened ...
            has_impact = (
                ("contact" in types and "disruption" in types)
                or "pose_change" in types
                or ("multi_actor" in types and "contact" in types)
                # A single-vehicle crash has no second party to touch.  Its
                # impact evidence is a violent disruption that ends with the
                # vehicle stopped where it should not be.
                or ("disruption" in types and "immobility" in types)
            )
            # ... and the road has to still be wrong afterwards.  Only a
            # genuinely violent, unambiguous impact skips the aftermath test.
            has_aftermath = bool(types & AFTERMATH_EVIDENCE)
            corroborated = has_aftermath or hypothesis.peak >= policy.strong_score

            confirmable = (
                hypothesis.peak >= policy.confirm_score
                and hypothesis.support_frames >= policy.support_frames
                and has_impact
                and corroborated
            )
            recently_fired = (
                hypothesis.key in self.cooldown
                or self.frame_number <= self.scene_cooldown_until
            )

            if confirmable and not recently_fired:
                status = CONFIRMED
            elif confirmable and recently_fired:
                # Same wreck, already dispatched.  Keep it visible, do not
                # raise a second emergency for it.
                status = REVIEW
            elif hypothesis.peak >= policy.review_score:
                status = REVIEW
            else:
                continue

            evidence = self._to_evidence(hypothesis, status)
            rank = (
                {CONFIRMED: 2.0, REVIEW: 1.0}[status] * 10.0
                + SEVERITY_ORDER[evidence.severity]
                + evidence.confidence
            )
            if best is None or rank > best[0]:
                best = (rank, evidence)

        if best is None:
            return IncidentEvidence(ml_probability=self.ml_probability, frame=self.frame_number)

        evidence = best[1]
        if evidence.confirmed:
            key = "actors:" + "-".join(str(actor) for actor in evidence.actor_ids)
            self.cooldown[key] = self.frame_number
            # One crash produces many overlapping hypotheses (the pair, each
            # vehicle's own immobility, the pile-up cluster).  A scene-wide
            # cooldown makes that one incident instead of a burst of alerts.
            self.scene_cooldown_until = self.frame_number + self.policy.scene_cooldown_frames
            hypothesis = self.hypotheses.get(key)
            if hypothesis is not None:
                hypothesis.confirmed_at = self.frame_number
            self.confirmed_history.append(evidence)
        return evidence

    def _to_evidence(self, hypothesis: Hypothesis, status: str) -> IncidentEvidence:
        involved = tuple(self.describe_actor(actor_id) for actor_id in hypothesis.actor_ids)
        involved = tuple(item for item in involved if item)
        return IncidentEvidence(
            status=status,
            kind=hypothesis.kind,
            confidence=round(min(hypothesis.peak, 0.99), 3),
            actor_ids=tuple(hypothesis.actor_ids),
            reason="; ".join(hypothesis.reasons[-2:]) or KIND_LABELS.get(hypothesis.kind, ""),
            severity=self._severity(hypothesis, involved),
            signals=tuple(
                sorted(hypothesis.signals.values(), key=lambda item: -item.score)
            ),
            evidence_types=tuple(sorted(hypothesis.evidence_types)),
            frame=self.frame_number,
            ml_probability=round(self.ml_probability, 4),
            involved=involved,
        )

    def _severity(self, hypothesis: Hypothesis, involved: Iterable[dict[str, Any]]) -> str:
        if hypothesis.kind not in KIND_BASE_SEVERITY:
            return "LOW"
        level = SEVERITY_ORDER[KIND_BASE_SEVERITY[hypothesis.kind]]
        people = sum(1 for item in involved if item.get("kind") == PERSON)
        vehicles = sum(1 for item in involved if item.get("kind") == VEHICLE)
        energy = self._energy(
            *[self.actors[actor_id] for actor_id in hypothesis.actor_ids if actor_id in self.actors]
        )
        if people >= 1:
            level += 1
        if vehicles >= 3:
            level += 1
        if energy >= 0.70:
            level += 1
        if energy <= 0.25 and people == 0:
            level -= 1
        level = max(0, min(3, level))
        return next(name for name, value in SEVERITY_ORDER.items() if value == level)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def describe_actor(self, actor_id: int) -> dict[str, Any]:
        actor = self.actors.get(actor_id)
        if actor is None:
            return {}
        return {
            "id": actor_id,
            "kind": actor.kind,
            "class_name": actor.class_name,
            "confidence": round(actor.confidence, 3),
            "speed_units": round(actor.smooth_speed, 4),
            "peak_speed_units": round(actor.peak_speed, 4),
            "heading": compass_heading(actor.velocity),
            "stationary_frames": actor.stationary_frames,
            "box": [round(value, 1) for value in actor.box],
        }

    def scene_summary(self) -> dict[str, Any]:
        vehicles = [a for a in self.actors.values() if a.kind == VEHICLE]
        people = [a for a in self.actors.values() if a.kind == PERSON]
        return {
            "frame": self.frame_number,
            "tracked_vehicles": len(vehicles),
            "tracked_people": len(people),
            "open_hypotheses": len(self.hypotheses),
            "ml_probability": round(self.ml_probability, 4),
        }


# ------------------------------------------------------------------
# Backwards-compatible alias for the original class name.
# ------------------------------------------------------------------
IncidentEventEngine = IncidentEngine
