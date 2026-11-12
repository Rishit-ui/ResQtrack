"""Tests for Authentication, Authorization, Citizen Reports, and Admin Operations in ResQTrack."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TEMP = tempfile.mkdtemp(prefix="resqtrack-auth-test-")
os.environ["RESQTRACK_DB"] = str(Path(_TEMP) / "test_auth.db")
os.environ["RESQTRACK_SNAPSHOTS"] = str(Path(_TEMP) / "snapshots")
os.environ["RESQTRACK_HTTP_TIMEOUT"] = "1.0"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.database import connect, hash_password, verify_password, initialise  # noqa: E402


class AuthAndFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Ensure schema and seed responders/facilities/users are present
        initialise()
        connection = connect()
        try:
            count = connection.execute("SELECT COUNT(*) AS c FROM responders").fetchone()["c"]
            if count == 0:
                from backend.database import seed_responders
                seed_responders(connection)
                connection.commit()
        finally:
            connection.close()

    def test_01_password_hashing(self):
        pwd = "SecureSecretPassword123!"
        hashed = hash_password(pwd)
        self.assertNotEqual(pwd, hashed)
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))

    def test_02_default_users_exist(self):
        # Admin, Responder, and Citizen should be seeded on initialise
        admin_login = self.client.post("/api/auth/login", json={
            "email": "admin@resqtrack.gov",
            "password": "Admin@ResQTrack2026!",
        })
        self.assertEqual(admin_login.status_code, 200, admin_login.text)
        admin_data = admin_login.json()
        self.assertEqual(admin_data["user"]["role"], "ADMIN")
        self.assertIn("access_token", admin_data)

        # Responder login
        resp_login = self.client.post("/api/auth/login", json={
            "email": "responder@resqtrack.gov",
            "password": "Responder@2026!",
        })
        self.assertEqual(resp_login.status_code, 200)
        self.assertEqual(resp_login.json()["user"]["role"], "RESPONDER")
        self.assertEqual(resp_login.json()["user"]["responder_id"], "RESP-0001")

    def test_03_citizen_registration_and_login(self):
        reg = self.client.post("/api/auth/register", json={
            "email": "test_citizen@example.com",
            "password": "Password123!",
            "full_name": "Test Citizen User",
            "phone": "+91-98765-43210",
        })
        self.assertEqual(reg.status_code, 200, reg.text)
        data = reg.json()
        self.assertEqual(data["user"]["role"], "CITIZEN")
        self.assertEqual(data["user"]["email"], "test_citizen@example.com")
        self.assertIn("access_token", data)

        # Duplicate email should fail
        dup = self.client.post("/api/auth/register", json={
            "email": "test_citizen@example.com",
            "password": "Password123!",
            "full_name": "Another Citizen",
        })
        self.assertEqual(dup.status_code, 400)

        # Login with correct password
        login_res = self.client.post("/api/auth/login", json={
            "email": "test_citizen@example.com",
            "password": "Password123!",
        })
        self.assertEqual(login_res.status_code, 200)
        token = login_res.json()["access_token"]

        # Current user endpoint /api/auth/me
        me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["email"], "test_citizen@example.com")

        # Wrong password fails
        bad_login = self.client.post("/api/auth/login", json={
            "email": "test_citizen@example.com",
            "password": "WrongPassword999!",
        })
        self.assertEqual(bad_login.status_code, 401)

    def test_04_role_authorization_guards(self):
        # Register citizen
        reg = self.client.post("/api/auth/register", json={
            "email": "regular_guy@example.com",
            "password": "Password123!",
            "full_name": "Regular Guy",
        })
        citizen_token = reg.json()["access_token"]
        citizen_headers = {"Authorization": f"Bearer {citizen_token}"}

        # Citizen attempts to list admin users -> 403 Forbidden
        admin_users_res = self.client.get("/api/admin/users", headers=citizen_headers)
        self.assertEqual(admin_users_res.status_code, 403)

        # Citizen attempts to review a report -> 403 Forbidden
        review_res = self.client.post("/api/admin/reports/REP-00001/review", json={"status": "APPROVED"}, headers=citizen_headers)
        self.assertEqual(review_res.status_code, 403)

        # Citizen attempts to get admin audit logs -> 403 Forbidden
        audit_res = self.client.get("/api/admin/audit-logs", headers=citizen_headers)
        self.assertEqual(audit_res.status_code, 403)

        # Admin login and access -> 200 OK
        admin_login = self.client.post("/api/auth/login", json={
            "email": "admin@resqtrack.gov",
            "password": "Admin@ResQTrack2026!",
        })
        admin_token = admin_login.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        admin_users_res = self.client.get("/api/admin/users", headers=admin_headers)
        self.assertEqual(admin_users_res.status_code, 200)
        self.assertGreaterEqual(len(admin_users_res.json()), 1)

        admin_audit = self.client.get("/api/admin/audit-logs", headers=admin_headers)
        self.assertEqual(admin_audit.status_code, 200)

    def test_05_manual_reports_and_ownership(self):
        # Create User 1
        u1_res = self.client.post("/api/auth/register", json={
            "email": "user1@example.com",
            "password": "Password123!",
            "full_name": "User One",
        })
        u1_headers = {"Authorization": f"Bearer {u1_res.json()['access_token']}"}

        # Create User 2
        u2_res = self.client.post("/api/auth/register", json={
            "email": "user2@example.com",
            "password": "Password123!",
            "full_name": "User Two",
        })
        u2_headers = {"Authorization": f"Bearer {u2_res.json()['access_token']}"}

        # User 1 submits report
        report_payload = {
            "category": "COLLISION",
            "description": "Two cars collided at MG Road junction. Fluid leaking.",
            "location_name": "MG Road / Brigade Road Junction",
            "latitude": 12.9750,
            "longitude": 77.6050,
            "contact_phone": "+91-98765-11111",
        }
        create_res = self.client.post("/api/reports", json=report_payload, headers=u1_headers)
        self.assertEqual(create_res.status_code, 200, create_res.text)
        report = create_res.json()
        report_id = report["report_id"]
        self.assertEqual(report["status"], "SUBMITTED")
        self.assertTrue(report_id.startswith("REP-"))

        # User 1 can view their report
        view_u1 = self.client.get(f"/api/reports/{report_id}", headers=u1_headers)
        self.assertEqual(view_u1.status_code, 200)
        self.assertEqual(view_u1.json()["report_id"], report_id)

        # User 2 CANNOT view User 1's private report -> 403 Forbidden
        view_u2 = self.client.get(f"/api/reports/{report_id}", headers=u2_headers)
        self.assertEqual(view_u2.status_code, 403)

        # User 1 list reports
        my_reports = self.client.get("/api/reports/my", headers=u1_headers)
        self.assertEqual(my_reports.status_code, 200)
        self.assertEqual(len(my_reports.json()), 1)

        # Admin logs in and reviews report
        admin_login = self.client.post("/api/auth/login", json={
            "email": "admin@resqtrack.gov",
            "password": "Admin@ResQTrack2026!",
        })
        admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}

        # Admin can view any report
        admin_view = self.client.get(f"/api/reports/{report_id}", headers=admin_headers)
        self.assertEqual(admin_view.status_code, 200)

        # Admin reviews with ESCALATE -> dispatches responders and links incident
        escalate_res = self.client.post(
            f"/api/admin/reports/{report_id}/review",
            json={"status": "ESCALATED", "admin_notes": "Verified high impact, dispatching emergency crew", "severity": "HIGH"},
            headers=admin_headers,
        )
        self.assertEqual(escalate_res.status_code, 200, escalate_res.text)
        escalate_data = escalate_res.json()
        self.assertEqual(escalate_data["report"]["status"], "ESCALATED")
        self.assertIsNotNone(escalate_data["report"]["linked_incident_id"])
        self.assertIsNotNone(escalate_data["incident"])
        self.assertEqual(escalate_data["incident"]["status"], "DISPATCHED")

    def test_06_profile_update_and_password_lifecycle(self):
        reg = self.client.post("/api/auth/register", json={
            "email": "profile_user@example.com",
            "password": "OldPassword123!",
            "full_name": "Old Name",
        })
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Update profile
        up_res = self.client.put("/api/auth/profile", json={"full_name": "Updated Name", "phone": "+91-91234-56789"}, headers=headers)
        self.assertEqual(up_res.status_code, 200)
        self.assertEqual(up_res.json()["full_name"], "Updated Name")

        # Change password
        ch_res = self.client.post("/api/auth/change-password", json={
            "current_password": "OldPassword123!",
            "new_password": "NewSecretPassword123!",
        }, headers=headers)
        self.assertEqual(ch_res.status_code, 200)

        # Login with old password fails
        bad_login = self.client.post("/api/auth/login", json={
            "email": "profile_user@example.com",
            "password": "OldPassword123!",
        })
        self.assertEqual(bad_login.status_code, 401)

        # Login with new password succeeds
        good_login = self.client.post("/api/auth/login", json={
            "email": "profile_user@example.com",
            "password": "NewSecretPassword123!",
        })
        self.assertEqual(good_login.status_code, 200)

        # Forgot password flow
        forgot_res = self.client.post("/api/auth/forgot-password", json={"email": "profile_user@example.com"})
        self.assertEqual(forgot_res.status_code, 200)
        reset_token = forgot_res.json()["reset_token"]

        # Reset password
        reset_res = self.client.post("/api/auth/reset-password", json={
            "token": reset_token,
            "new_password": "BrandNewPassword123!",
        })
        self.assertEqual(reset_res.status_code, 200)

        # Login with brand new password succeeds
        latest_login = self.client.post("/api/auth/login", json={
            "email": "profile_user@example.com",
            "password": "BrandNewPassword123!",
        })
        self.assertEqual(latest_login.status_code, 200)

    def test_07_system_health(self):
        res = self.client.get("/api/system/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertIn("database", data)
        self.assertIn("realtime", data)


if __name__ == "__main__":
    unittest.main()
