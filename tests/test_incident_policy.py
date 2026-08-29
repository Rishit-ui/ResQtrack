import unittest

from accident_logic import collision_score, reset_vehicle_history, update_vehicle
from event_engine import IncidentEventEngine, PERSON, VEHICLE


def actor(kind, x, y=0.0, width=30.0, height=30.0):
    return {"kind": kind, "box": (x, y, x + width, y + height)}


class IncidentPolicyTests(unittest.TestCase):
    def test_parked_or_overlapping_vehicles_never_confirm(self):
        engine = IncidentEventEngine()
        actors = {1: actor(VEHICLE, 0), 2: actor(VEHICLE, 20)}
        for _ in range(12):
            evidence = engine.update(actors)
            self.assertEqual(evidence.status, "NORMAL")

    def test_vehicle_collision_requires_approach_and_disruption(self):
        engine = IncidentEventEngine()
        engine.update({1: actor(VEHICLE, 0), 2: actor(VEHICLE, 100)})
        engine.update({1: actor(VEHICLE, 25), 2: actor(VEHICLE, 75)})
        evidence = engine.update({1: actor(VEHICLE, 40), 2: actor(VEHICLE, 60)})
        self.assertTrue(evidence.confirmed)
        self.assertEqual(evidence.kind, "vehicle_vehicle_collision")

    def test_vehicle_pedestrian_collision_is_supported(self):
        engine = IncidentEventEngine()
        engine.update({1: actor(VEHICLE, 0), 9: actor(PERSON, 100)})
        engine.update({1: actor(VEHICLE, 30), 9: actor(PERSON, 100)})
        evidence = engine.update({1: actor(VEHICLE, 65), 9: actor(PERSON, 85)})
        self.assertTrue(evidence.confirmed)
        self.assertEqual(evidence.kind, "vehicle_pedestrian_collision")

    def test_possible_hit_and_run_is_supported(self):
        engine = IncidentEventEngine()
        engine.update({1: actor(VEHICLE, 0), 9: actor(PERSON, 100)})
        engine.update({1: actor(VEHICLE, 30), 9: actor(PERSON, 100)})
        engine.update({1: actor(VEHICLE, 65), 9: actor(PERSON, 100)})
        engine.update({1: actor(VEHICLE, 105)})
        evidence = engine.update({1: actor(VEHICLE, 145)})
        self.assertTrue(evidence.confirmed)
        self.assertEqual(evidence.kind, "possible_hit_and_run")

    def test_legacy_score_does_not_score_static_proximity(self):
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
