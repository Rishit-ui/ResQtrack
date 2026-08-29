"""Behavioural tests for the ResQTrack incident policy.

These are the guarantees the detector is expected to hold.  The negative tests
matter as much as the positive ones: an emergency system that cries wolf on
parked cars is worse than no system at all.

    python -m unittest -v tests.test_incident_policy
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accident_logic import collision_score, reset_vehicle_history, update_vehicle  # noqa: E402
from vision.incident_engine import (  # noqa: E402
    CONFIRMED,
    NORMAL,
    PERSON,
    REVIEW,
    VEHICLE,
    EnginePolicy,
    IncidentEngine,
)

FRAME_SIZE = (1280, 720)
SIZE = 60.0


def actor(kind: str, x: float, y: float = 300.0, width: float = SIZE, height: float = SIZE) -> dict:
    return {"kind": kind, "box": (x, y, x + width, y + height), "class_name": kind, "confidence": 0.9}


def person(x: float, y: float = 300.0, standing: bool = True) -> dict:
    # A standing person is much taller than wide; a fallen one is the reverse.
    return actor(PERSON, x, y, 26.0, 62.0) if standing else actor(PERSON, x, y, 62.0, 26.0)


def new_engine(**overrides) -> IncidentEngine:
    policy = EnginePolicy(**overrides) if overrides else EnginePolicy()
    return IncidentEngine(policy=policy, fps=25.0, frame_size=FRAME_SIZE)


def run(engine: IncidentEngine, frames: list[dict[int, dict]]) -> list:
    return [engine.update(scene) for scene in frames]


def best(results: list) -> str:
    """The strongest verdict seen anywhere in the sequence."""
    if any(item.status == CONFIRMED for item in results):
        return CONFIRMED
    if any(item.status == REVIEW for item in results):
        return REVIEW
    return NORMAL


def confirmed_events(results: list) -> list:
    return [item for item in results if item.status == CONFIRMED]


# ============================================================
# NEGATIVE CASES - these must never dispatch an ambulance
# ============================================================

class NoFalseAlarms(unittest.TestCase):
    def test_parked_vehicles_touching_stay_normal(self):
        """Kerbside parking puts boxes side by side for minutes on end."""
        engine = new_engine()
        scene = {1: actor(VEHICLE, 100), 2: actor(VEHICLE, 158)}
        results = run(engine, [scene] * 60)
        self.assertEqual(best(results), NORMAL)

    def test_overlapping_boxes_from_perspective_stay_normal(self):
        """A distant car behind a near one overlaps in the image, not in reality."""
        engine = new_engine()
        scene = {1: actor(VEHICLE, 200, 300, 90, 90), 2: actor(VEHICLE, 230, 320, 40, 40)}
        results = run(engine, [scene] * 50)
        self.assertEqual(best(results), NORMAL)

    def test_traffic_queue_stopping_together_is_not_a_pileup(self):
        """Everyone braking at a red light is congestion, not a crash."""
        engine = new_engine()
        frames = []
        for step in range(14):                      # rolling up to the lights
            offset = 14 * step
            frames.append({i: actor(VEHICLE, 100 + i * 90 + offset) for i in range(1, 6)})
        for _ in range(45):                         # all stopped together
            offset = 14 * 13
            frames.append({i: actor(VEHICLE, 100 + i * 90 + offset) for i in range(1, 6)})
        results = run(engine, frames)
        self.assertNotIn(CONFIRMED, [item.status for item in results])

    def test_a_new_track_appearing_is_not_an_impact(self):
        """A track's first frames start at zero speed and must not read as a jolt."""
        engine = new_engine()
        frames = [{1: actor(VEHICLE, 100)}]
        for step in range(1, 30):                   # a second vehicle pops in mid-scene
            frames.append({1: actor(VEHICLE, 100), 2: actor(VEHICLE, 140 + step * 12)})
        results = run(engine, frames)
        self.assertEqual(best(results), NORMAL)

    def test_a_car_leaving_through_the_frame_edge_is_not_a_hit_and_run(self):
        engine = new_engine()
        frames = []
        for step in range(20):
            frames.append({1: actor(VEHICLE, 900 + step * 18), 2: actor(VEHICLE, 940 + step * 18)})
        for _ in range(15):                         # vehicle 1 has driven out of shot
            frames.append({2: actor(VEHICLE, 1200)})
        results = run(engine, frames)
        self.assertNotIn("possible_hit_and_run", [item.kind for item in confirmed_events(results)])

    def test_model_probability_alone_never_confirms(self):
        """The trained model may corroborate physical evidence, never replace it."""
        engine = new_engine()
        scene = {1: actor(VEHICLE, 100), 2: actor(VEHICLE, 400)}
        results = [engine.update(scene, ml_probability=0.99) for _ in range(60)]
        self.assertEqual(best(results), NORMAL)

    def test_near_miss_is_review_not_confirmed(self):
        """A fast car closes on a slower one, brakes in time, and both drive on."""
        engine = new_engine()
        frames = []
        follower, leader = 100.0, 400.0
        for _ in range(11):                          # closing hard: gap 240 -> ~9 px
            follower += 33
            leader += 12
            frames.append({1: actor(VEHICLE, follower), 2: actor(VEHICLE, leader)})
        speed = 33.0
        for _ in range(8):                           # smooth braking to match speed
            speed = max(12.0, speed - 4.0)
            follower += speed
            leader += 12
            frames.append({1: actor(VEHICLE, follower), 2: actor(VEHICLE, leader)})
        for _ in range(25):                          # both cruise on, nobody stops
            follower += 12
            leader += 12
            frames.append({1: actor(VEHICLE, follower), 2: actor(VEHICLE, leader)})
        results = run(engine, frames)
        self.assertNotEqual(best(results), CONFIRMED)


# ============================================================
# POSITIVE CASES - these must reach CONFIRMED
# ============================================================

class DetectsRealCrashes(unittest.TestCase):
    def test_vehicle_to_vehicle_collision_with_aftermath(self):
        """Approach, contact, violent stop, then the wreck stays put."""
        engine = new_engine()
        frames = []
        position = 100.0
        for _ in range(13):                          # vehicle 1 runs into vehicle 2
            position = min(308.0, position + 16)     # 308 + 60 wide = 8 px into the leader
            frames.append({1: actor(VEHICLE, position), 2: actor(VEHICLE, 360)})
        for _ in range(40):                          # both immobile afterwards
            frames.append({1: actor(VEHICLE, position), 2: actor(VEHICLE, 360)})
        results = run(engine, frames)
        events = confirmed_events(results)
        self.assertTrue(events, "a rear-end collision followed by immobility must confirm")
        self.assertEqual(events[0].kind, "vehicle_vehicle_collision")
        self.assertIn("contact", events[0].evidence_types)
        self.assertIn("disruption", events[0].evidence_types)

    def test_vehicle_strikes_a_pedestrian(self):
        engine = new_engine()
        frames = []
        position = 100.0
        for _ in range(12):
            position += 18
            frames.append({1: actor(VEHICLE, position), 9: person(360)})
        for _ in range(30):                          # the pedestrian is down and still
            frames.append({1: actor(VEHICLE, position), 9: person(356, standing=False)})
        results = run(engine, frames)
        events = confirmed_events(results)
        self.assertTrue(events, "a struck pedestrian must confirm")
        self.assertEqual(events[0].kind, "vehicle_pedestrian_collision")
        self.assertEqual(events[0].severity, "CRITICAL")

    def test_person_collapsing_alone_is_detected(self):
        """A medical emergency on the carriageway, with no vehicle involved."""
        engine = new_engine()
        frames = [{9: person(300 + step * 3)} for step in range(20)]   # walking
        frames += [{9: person(360, standing=False)} for _ in range(30)]  # down
        results = run(engine, frames)
        self.assertEqual(best(results), CONFIRMED)
        self.assertEqual(confirmed_events(results)[0].kind, "pedestrian_down")

    def test_one_incident_produces_one_alert_not_a_burst(self):
        """The scene cooldown is what keeps a single crash from paging six crews."""
        engine = new_engine()
        frames = []
        position = 100.0
        for _ in range(13):
            position = min(308.0, position + 16)
            frames.append({1: actor(VEHICLE, position), 2: actor(VEHICLE, 360)})
        for _ in range(90):
            frames.append({1: actor(VEHICLE, position), 2: actor(VEHICLE, 360)})
        results = run(engine, frames)
        self.assertEqual(len(confirmed_events(results)), 1)


# ============================================================
# POLICY MECHANICS
# ============================================================

class PolicyMechanics(unittest.TestCase):
    def test_sensitivity_presets_are_ordered(self):
        strict = EnginePolicy.for_sensitivity("strict")
        balanced = EnginePolicy.for_sensitivity("balanced")
        high = EnginePolicy.for_sensitivity("high")
        self.assertGreater(strict.confirm_score, balanced.confirm_score)
        self.assertGreater(balanced.confirm_score, high.confirm_score)
        self.assertGreater(strict.support_frames, high.support_frames)

    def test_evidence_is_explainable(self):
        """Every confirmation must carry the reasoning that produced it."""
        engine = new_engine()
        frames = []
        position = 100.0
        for _ in range(13):
            position = min(308.0, position + 16)
            frames.append({1: actor(VEHICLE, position), 2: actor(VEHICLE, 360)})
        frames += [{1: actor(VEHICLE, position), 2: actor(VEHICLE, 360)}] * 40
        events = confirmed_events(run(engine, frames))
        self.assertTrue(events)
        evidence = events[0]
        self.assertTrue(evidence.reason)
        self.assertTrue(evidence.signals)
        self.assertTrue(evidence.evidence_types)
        payload = evidence.as_dict()
        for key in ("status", "kind", "label", "confidence", "severity", "signals"):
            self.assertIn(key, payload)

    def test_reset_clears_all_state(self):
        engine = new_engine()
        engine.update({1: actor(VEHICLE, 100)})
        engine.reset()
        self.assertEqual(engine.frame_number, 0)
        self.assertFalse(engine.actors)
        self.assertFalse(engine.hypotheses)

    def test_engine_tolerates_an_empty_scene(self):
        engine = new_engine()
        for _ in range(5):
            evidence = engine.update({})
        self.assertEqual(evidence.status, NORMAL)


# ============================================================
# LEGACY HELPER
# ============================================================

class LegacyCollisionHelper(unittest.TestCase):
    def test_static_proximity_scores_zero(self):
        reset_vehicle_history()
        vehicles = {
            1: {"center": (10.0, 10.0), "box": (0.0, 0.0, 20.0, 20.0)},
            2: {"center": (20.0, 10.0), "box": (10.0, 0.0, 30.0, 20.0)},
        }
        for _ in range(12):
            update_vehicle(1, vehicles[1]["center"])
            update_vehicle(2, vehicles[2]["center"])
            score, pairs = collision_score(vehicles)
            self.assertEqual(score, 0)
            self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
