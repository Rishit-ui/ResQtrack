"""End-to-end tests for the ResQTrack emergency-response backend.

They exercise the real API against a throwaway database: detection, dispatch,
acceptance, arrival by geofence, hospital handover and closure - plus the
guards that stop two crews taking the same case or a crew skipping a step.

    python -m unittest -v tests.test_backend
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the backend at a scratch database *before* it is imported, because the
# app initialises its schema at import time.
_TEMP = tempfile.mkdtemp(prefix="resqtrack-test-")
os.environ["RESQTRACK_DB"] = str(Path(_TEMP) / "test.db")
os.environ["RESQTRACK_SNAPSHOTS"] = str(Path(_TEMP) / "snapshots")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.database import can_transition, connect, next_id  # noqa: E402
from backend.services.dispatch import rank_responders, selection_size  # noqa: E402
from backend.services.geo import bounding_box, compass_point, haversine_km  # noqa: E402

SCENE = {"latitude": 12.9719, "longitude": 77.5937}


def make_incident(client: TestClient, severity: str = "CRITICAL", **extra) -> dict:
    payload = {
        "camera_id": "CAM-TEST",
        "latitude": SCENE["latitude"],
        "longitude": SCENE["longitude"],
        "confidence": 0.88,
        "kind": "vehicle_pedestrian_collision",
        "label": "Vehicle struck a pedestrian",
        "severity": severity,
        "reason": "contact then the pedestrian went down",
        "location_name": "Test Junction",
        "evidence_types": ["approach", "contact", "disruption", "pose_change"],
        **extra,
    }
    response = client.post("/api/incidents", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def make_responder(client: TestClient, name: str, lat: float, lon: float,
                   kind: str = "AMBULANCE") -> dict:
    response = client.post("/api/responders/register", json={
        "name": name, "responder_type": kind, "latitude": lat, "longitude": lon,
        "phone": "+91-99999-00000", "vehicle_number": "KA-01-TEST",
    })
    assert response.status_code == 200, response.text
    return response.json()


class IncidentLifecycle(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        connection = connect()
        for table in ("incident_events", "dispatches", "location_history",
                      "notifications", "incidents", "responders"):
            connection.execute(f"DELETE FROM {table}")
        connection.commit()
        connection.close()

    # ------------------------------------------------------------------
    def test_detection_dispatches_to_the_nearest_units(self):
        make_responder(self.client, "Near unit", 12.975, 77.596)
        make_responder(self.client, "Far unit", 13.10, 77.75)
        body = make_incident(self.client)

        self.assertEqual(body["incident"]["status"], "DISPATCHED")
        dispatched = body["dispatched_to"]
        self.assertTrue(dispatched)
        # Closest first.
        self.assertEqual(dispatched[0]["name"], "Near unit")
        self.assertLess(dispatched[0]["distance_km"], 2.0)

    def test_full_response_flow(self):
        responder = make_responder(self.client, "Ambulance 1", 12.975, 77.596)
        incident = make_incident(self.client)["incident"]
        incident_id = incident["incident_id"]
        responder_id = responder["responder_id"]

        accepted = self.client.post(f"/api/incidents/{incident_id}/accept",
                                    json={"responder_id": responder_id})
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json()["incident"]["status"], "ACCEPTED")
        self.assertEqual(accepted.json()["responder"]["status"], "EN_ROUTE")

        # A fix far from the scene must not count as arrival.
        away = self.client.post(f"/api/responders/{responder_id}/location",
                                json={"latitude": 12.980, "longitude": 77.600})
        self.assertFalse(away.json()["arrived"])

        # A fix on top of the scene triggers the geofence.
        near = self.client.post(f"/api/responders/{responder_id}/location",
                                json={"latitude": SCENE["latitude"], "longitude": SCENE["longitude"]})
        self.assertTrue(near.json()["arrived"])
        self.assertEqual(self.client.get(f"/api/incidents/{incident_id}").json()["status"], "ON_SCENE")

        hospitals = self.client.get(f"/api/incidents/{incident_id}/hospitals")
        self.assertEqual(hospitals.status_code, 200, hospitals.text)

        chosen = self.client.post(f"/api/incidents/{incident_id}/hospital", json={
            "responder_id": responder_id, "name": "Test General Hospital",
            "latitude": 12.9829, "longitude": 77.6046,
        })
        self.assertEqual(chosen.status_code, 200, chosen.text)
        body = chosen.json()
        self.assertEqual(body["incident"]["status"], "TRANSPORTING")
        # A route always comes back, even with no routing server reachable.
        self.assertIn("route", body)
        self.assertGreater(body["route"]["distance_km"], 0)
        self.assertIn("traffic", body["route"]["route"])

        closed = self.client.post(f"/api/incidents/{incident_id}/close",
                                  json={"responder_id": responder_id, "outcome": "RESOLVED"})
        self.assertEqual(closed.json()["status"], "CLOSED")
        # The crew is released for the next call.
        self.assertEqual(
            self.client.get(f"/api/responders/{responder_id}").json()["status"], "AVAILABLE"
        )

        timeline = self.client.get(f"/api/incidents/{incident_id}/timeline").json()
        recorded = [event["event_type"] for event in timeline]
        for expected in ("detected", "dispatched", "accepted", "on_scene",
                         "hospital_selected", "closed"):
            self.assertIn(expected, recorded)

    def test_second_responder_cannot_take_an_accepted_case(self):
        first = make_responder(self.client, "Unit A", 12.975, 77.596)
        second = make_responder(self.client, "Unit B", 12.974, 77.595)
        incident_id = make_incident(self.client)["incident"]["incident_id"]

        self.client.post(f"/api/incidents/{incident_id}/accept",
                         json={"responder_id": first["responder_id"]})
        clash = self.client.post(f"/api/incidents/{incident_id}/accept",
                                 json={"responder_id": second["responder_id"]})
        self.assertEqual(clash.status_code, 409)

    def test_hospital_cannot_be_chosen_before_arrival(self):
        responder = make_responder(self.client, "Unit A", 12.975, 77.596)
        incident_id = make_incident(self.client)["incident"]["incident_id"]
        self.client.post(f"/api/incidents/{incident_id}/accept",
                         json={"responder_id": responder["responder_id"]})
        early = self.client.post(f"/api/incidents/{incident_id}/hospital", json={
            "responder_id": responder["responder_id"], "name": "Somewhere",
            "latitude": 12.98, "longitude": 77.60,
        })
        self.assertEqual(early.status_code, 409)

    def test_declining_reassigns_to_another_unit(self):
        first = make_responder(self.client, "Unit A", 12.9720, 77.5938)
        make_responder(self.client, "Unit B", 12.9730, 77.5945)
        incident_id = make_incident(self.client, severity="LOW")["incident"]["incident_id"]

        declined = self.client.post(f"/api/incidents/{incident_id}/decline",
                                    json={"responder_id": first["responder_id"]})
        self.assertEqual(declined.status_code, 200, declined.text)
        self.assertTrue(declined.json()["reassigned_to"])

    def test_incident_with_no_responders_is_escalated_not_lost(self):
        body = make_incident(self.client)
        self.assertEqual(body["dispatched_to"], [])
        self.assertEqual(body["incident"]["status"], "DETECTED")
        timeline = self.client.get(
            f"/api/incidents/{body['incident']['incident_id']}/timeline"
        ).json()
        self.assertIn("dispatch_failed", [event["event_type"] for event in timeline])

    def test_snapshot_is_stored_and_served(self):
        incident_id = make_incident(self.client)["incident"]["incident_id"]
        # A one-pixel JPEG is enough to prove the path end to end.
        import cv2
        import numpy as np
        ok, buffer = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(ok)
        response = self.client.post(
            f"/api/incidents/{incident_id}/snapshot",
            files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        path = response.json()["snapshot_path"]
        self.assertEqual(self.client.get(path).status_code, 200)

    def test_unknown_records_return_404(self):
        self.assertEqual(self.client.get("/api/incidents/RQ-99999").status_code, 404)
        self.assertEqual(self.client.get("/api/responders/RESP-9999").status_code, 404)

    def test_invalid_coordinates_are_rejected(self):
        response = self.client.post("/api/incidents", json={
            "camera_id": "CAM-BAD", "latitude": 999.0, "longitude": 0.0,
        })
        self.assertEqual(response.status_code, 400)

    def test_health_and_stats_are_serialisable(self):
        make_responder(self.client, "Unit A", 12.975, 77.596)
        make_incident(self.client)
        health = self.client.get("/api/health").json()
        self.assertEqual(health["status"], "ONLINE")
        self.assertGreaterEqual(health["incidents"], 1)
        stats = self.client.get("/api/stats").json()
        self.assertIn("by_kind", stats)
        self.assertIn("totals", stats)


class Helpers(unittest.TestCase):
    def test_state_machine_blocks_illegal_transitions(self):
        self.assertTrue(can_transition("DISPATCHED", "ACCEPTED"))
        self.assertTrue(can_transition("ACCEPTED", "ON_SCENE"))
        self.assertFalse(can_transition("DETECTED", "ON_SCENE"))
        self.assertFalse(can_transition("CLOSED", "ACCEPTED"))

    def test_public_ids_never_collide_after_a_delete(self):
        connection = connect()
        try:
            connection.execute("DELETE FROM incidents")
            connection.execute(
                """INSERT INTO incidents (incident_id, camera_id, latitude, longitude,
                   detected_at, created_at) VALUES ('RQ-00001','C',1.0,1.0,'t','t')"""
            )
            connection.commit()
            # COUNT(*) based numbering would hand out RQ-00002 here after a delete;
            # deriving from MAX(id) cannot.
            self.assertNotEqual(next_id(connection, "incidents", "incident_id", "RQ-", 5), "RQ-00001")
            connection.execute("DELETE FROM incidents")
            connection.commit()
        finally:
            connection.close()

    def test_dispatch_plan_scales_with_severity(self):
        self.assertGreater(selection_size("CRITICAL"), selection_size("LOW"))

    def test_busy_responders_are_not_dispatched(self):
        connection = connect()
        try:
            connection.execute("DELETE FROM responders")
            connection.execute(
                """INSERT INTO responders (responder_id, name, responder_type, latitude,
                   longitude, status, active_incident, registered_at, last_seen)
                   VALUES ('RESP-9001','Busy','AMBULANCE',12.9720,77.5938,'AVAILABLE',
                           'RQ-00001','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"""
            )
            connection.commit()
            ranked = rank_responders(connection, SCENE["latitude"], SCENE["longitude"])
            self.assertEqual(ranked, [])
        finally:
            connection.execute("DELETE FROM responders")
            connection.commit()
            connection.close()

    def test_geo_helpers(self):
        # Bengaluru to Delhi is roughly 1740 km.
        distance = haversine_km(12.9719, 77.5937, 28.6139, 77.2090)
        self.assertGreater(distance, 1700)
        self.assertLess(distance, 1800)
        self.assertEqual(compass_point(0.0), "N")
        self.assertEqual(compass_point(90.0), "E")
        south, west, north, east = bounding_box(12.9719, 77.5937, 10.0)
        self.assertLess(south, 12.9719)
        self.assertGreater(north, 12.9719)
        self.assertLess(west, 77.5937)
        self.assertGreater(east, 77.5937)


if __name__ == "__main__":
    unittest.main()
