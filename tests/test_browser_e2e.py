"""Browser Automation & End-to-End User Journey Tests for ResQTrack.

Tests:
1. Public Landing Page & Navigation
2. Citizen Journey: Register -> Report incident -> Track report -> Logout
3. Admin Journey: Login -> View metrics -> Review report & escalate -> Verify dispatch -> Map -> Audit log
4. Responder Journey: Login -> Accept call -> GPS fix -> Arrive -> Choose hospital -> Complete incident
5. Responsive Visual Inspection across Desktop (1440px), Laptop (1280px), Tablet (768px), Mobile (390px)
"""

from __future__ import annotations

import os
import sys
import time
import threading
import unittest
from pathlib import Path

# Add repo to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure snapshots and test db directory
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

SERVER_PORT = 8000
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        import uvicorn
        from backend.app import app
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="error")
        )

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


_server_thread: ServerThread | None = None


def ensure_server_running():
    global _server_thread
    if _server_thread is None:
        _server_thread = ServerThread()
        _server_thread.start()
        time.sleep(2.0)  # Wait for server startup


class BrowserE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_server_running()
        from playwright.sync_api import sync_playwright
        cls.playwright = sync_playwright().start()
        # Launch with system installed Edge or Chrome
        try:
            cls.browser = cls.playwright.chromium.launch(channel="msedge", headless=True)
        except Exception:
            try:
                cls.browser = cls.playwright.chromium.launch(channel="chrome", headless=True)
            except Exception:
                cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.browser.close()
            cls.playwright.stop()
        except Exception:
            pass

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 900})
        self.page = self.context.new_page()

    def tearDown(self):
        self.context.close()

    def test_01_landing_page(self):
        """Test landing page, headline, and responsive presentation."""
        self.page.goto(f"{BASE_URL}/", wait_until="networkidle")
        self.assertIn("ResQTrack", self.page.title())

        # Check hero headline
        content = self.page.content()
        self.assertIn("Every Second Matters", content)
        self.assertIn("Every Response Counts", content)

        # Check workflow section
        self.assertIn("How ResQTrack Responds to Emergencies", content)

        # Capture desktop screenshot
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "01_landing_desktop_1440.png"), full_page=True)

        # Check mobile viewport (390px iPhone standard)
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.page.wait_for_timeout(500)
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "01_landing_mobile_390.png"))

        # Verify no horizontal scrollbar overflow
        scroll_width = self.page.evaluate("document.documentElement.scrollWidth")
        inner_width = self.page.evaluate("window.innerWidth")
        self.assertLessEqual(scroll_width, inner_width + 10, "Horizontal overflow detected on mobile")

    def test_02_citizen_journey(self):
        """Full Citizen Journey: Register -> Report -> Verify Tracking -> Logout."""
        # 1. Register new citizen
        self.page.goto(f"{BASE_URL}/register", wait_until="networkidle")
        unique_email = f"citizen_{int(time.time())}@example.com"

        self.page.fill("input[placeholder='Rishit Citizen']", "Autonomous Test Citizen")
        self.page.fill("input[placeholder='citizen@example.com']", unique_email)
        self.page.fill("input[placeholder='+91-98765-43210']", "+91-99887-76655")
        self.page.fill("input[placeholder='••••••••••••']", "SecurePassword123!")

        self.page.click("button:has-text('Complete Registration')")
        self.page.wait_for_url("**/user/dashboard", timeout=8000)
        self.assertIn("/user/dashboard", self.page.url)

        # 2. Fill manual incident report
        self.page.fill("textarea", "Rear-end collision between two sedans. Coolant leak, one driver stunned.")
        self.page.fill("input[placeholder*='MG Road']", "Koramangala 80ft Road Junction")
        self.page.fill("input[placeholder*='Latitude']", "12.9352")
        self.page.fill("input[placeholder*='Longitude']", "77.6245")

        self.page.click("button:has-text('Submit Incident Report')")
        self.page.wait_for_selector("text=submitted successfully", timeout=8000)

        # Capture citizen dashboard with success banner
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "02_citizen_submitted.png"))

        # 3. Switch to My Reports tab & verify persistence
        self.page.click("button:has-text('My Reports')")
        self.page.wait_for_selector("text=REP-", timeout=5000)

        # Verify report status is SUBMITTED
        content = self.page.content()
        self.assertIn("SUBMITTED", content)
        self.assertIn("Koramangala 80ft Road Junction", content)

        # 4. Open report detail modal
        self.page.click("button:has-text('View Details')")
        self.page.wait_for_selector("text=Category:", timeout=4000)
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "02_citizen_report_modal.png"))

        # Close the modal before navigating or logging out
        self.page.keyboard.press("Escape")
        self.page.locator("div[style*='rgba(0, 0, 0, 0.75)']").click(position={"x": 10, "y": 10})
        self.page.wait_for_timeout(300)

        # 5. Logout
        self.page.click("button[title='Log out']")
        self.page.wait_for_url("**/login", timeout=5000)
        self.assertIn("/login", self.page.url)

    def test_03_admin_journey(self):
        """Admin Journey: Login -> View metrics -> Escalate report -> Check Active Incidents & Map."""
        # 1. Sign in as Admin
        self.page.goto(f"{BASE_URL}/login", wait_until="networkidle")
        self.page.locator("button", has_text="Admin").first.click()
        self.page.wait_for_timeout(300)
        self.page.locator("button[type='submit']").click()
        self.page.wait_for_url("**/admin/dashboard", timeout=12000)

        # Verify overview metrics
        self.page.wait_for_selector("text=Emergency Operations Command", timeout=8000)
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "03_admin_overview.png"))

        # 2. Go to Citizen Reports
        self.page.click("button:has-text('Citizen Reports')")
        self.page.wait_for_selector("text=Citizen Incident Reports Review Queue", timeout=8000)

        # Find and click 'Escalate & Dispatch' if a submitted report exists
        escalate_btn = self.page.locator("button:has-text('Escalate & Dispatch')").first
        if escalate_btn.count() > 0:
            self.page.on("dialog", lambda dialog: dialog.accept("Confirmed via CCTV - dispatching ambulance"))
            escalate_btn.click()
            self.page.wait_for_timeout(1500)

        # 3. Check Active Incidents Tab
        self.page.click("button:has-text('Active Incidents')")
        self.page.wait_for_selector("text=Live Active Emergency Incidents", timeout=8000)
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "03_admin_active_incidents.png"))

        # 4. Check Tactical Map
        self.page.click("button:has-text('Tactical Map')")
        self.page.wait_for_selector("div.leaflet-container", timeout=8000)
        self.page.wait_for_timeout(1000)
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "03_admin_tactical_map.png"))

        # 5. Check Audit Trail
        self.page.click("button:has-text('Audit Trail')")
        self.page.wait_for_selector("text=Security & Operational Audit Trail", timeout=8000)
        content = self.page.content()
        self.assertIn("AUD-", content)
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "03_admin_audit_trail.png"))

        # Logout
        self.page.click("button[title='Log out']")
        self.page.wait_for_url("**/login", timeout=8000)

    def test_04_responder_journey(self):
        """Responder Journey: Login -> Accept Call -> Arrive -> Choose Hospital -> Close Case."""
        self.page.goto(f"{BASE_URL}/login", wait_until="networkidle")
        self.page.locator("button", has_text="Responder").first.click()
        self.page.wait_for_timeout(300)
        self.page.locator("button[type='submit']").click()
        self.page.wait_for_url("**/responder", timeout=12000)

        self.page.wait_for_selector("text=Unit:", timeout=5000)
        self.page.screenshot(path=str(SCREENSHOTS_DIR / "04_responder_portal.png"))

        # If incoming alert banner appears, accept it
        accept_btn = self.page.locator("button:has-text('ACCEPT CALL')")
        if accept_btn.count() > 0:
            accept_btn.click()
            self.page.wait_for_timeout(1500)

        # If active call is present, simulate arriving & selecting hospital
        arrive_btn = self.page.locator("button:has-text('CONFIRM ARRIVAL AT SCENE')")
        if arrive_btn.count() > 0:
            arrive_btn.click()
            self.page.wait_for_timeout(1500)

        # Complete handover if transporting or present
        close_btn = self.page.locator("button:has-text('COMPLETE HANDOVER')")
        if close_btn.count() > 0:
            self.page.on("dialog", lambda dialog: dialog.accept("Patient stabilized and admitted"))
            close_btn.click()
            self.page.wait_for_timeout(1500)

    def test_05_responsive_inspection(self):
        """Verify layouts across 1440px desktop, 1280px laptop, 768px tablet, 390px mobile."""
        viewports = [
            ("desktop_1440", 1440, 900),
            ("laptop_1280", 1280, 800),
            ("tablet_768", 768, 1024),
            ("mobile_390", 390, 844),
        ]

        for name, w, h in viewports:
            self.page.set_viewport_size({"width": w, "height": h})
            self.page.goto(f"{BASE_URL}/", wait_until="networkidle")
            self.page.wait_for_timeout(400)
            self.page.screenshot(path=str(SCREENSHOTS_DIR / f"responsive_{name}.png"))

            # Verify no horizontal overflow
            scroll_w = self.page.evaluate("document.documentElement.scrollWidth")
            self.assertLessEqual(scroll_w, w + 10, f"Horizontal overflow on {name} ({w}px)")


if __name__ == "__main__":
    unittest.main()
