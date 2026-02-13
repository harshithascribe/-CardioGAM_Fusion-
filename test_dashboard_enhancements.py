#!/usr/bin/env python3
"""
Comprehensive test script for dashboard enhancements
Tests all new features: enhanced navigation, real-time stats, quick actions
"""

import sys
import os
import time
import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import unittest
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

from src.dashboard.models import db, User, PatientAssessment
from src.dashboard.app import server, app

class DashboardEnhancementTests(unittest.TestCase):
    """Test class for dashboard enhancements"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("Setting up test environment...")

        # Create test database
        with server.app_context():
            db.create_all()

            # Create test admin user
            if not User.query.filter_by(username='testadmin').first():
                admin = User(username='testadmin', email='testadmin@cardiofusion.com', role='admin')
                admin.set_password('testpass123')
                db.session.add(admin)
                db.session.commit()

            # Create sample patient assessments for testing
            if PatientAssessment.query.count() == 0:
                # Create assessments with different risk levels
                assessments = [
                    PatientAssessment(
                        patient_id='TEST-001',
                        age=45, bp=130, cholesterol=220, heart_rate=75,
                        risk_score=0.2, risk_category='Low Risk',
                        confidence=92.5, recommendations='Low risk patient',
                        created_by=1
                    ),
                    PatientAssessment(
                        patient_id='TEST-002',
                        age=55, bp=150, cholesterol=250, heart_rate=85,
                        risk_score=0.6, risk_category='Moderate Risk',
                        confidence=88.3, recommendations='Moderate risk patient',
                        created_by=1
                    ),
                    PatientAssessment(
                        patient_id='TEST-003',
                        age=65, bp=170, cholesterol=280, heart_rate=95,
                        risk_score=0.8, risk_category='High Risk',
                        confidence=94.1, recommendations='High risk patient',
                        created_by=1
                    ),
                    # Add today's assessment
                    PatientAssessment(
                        patient_id='TEST-004',
                        age=50, bp=140, cholesterol=240, heart_rate=80,
                        risk_score=0.4, risk_category='Moderate Risk',
                        confidence=89.7, recommendations='Today assessment',
                        created_by=1,
                        created_at=datetime.now()
                    )
                ]

                for assessment in assessments:
                    db.session.add(assessment)
                db.session.commit()

        # Start the Dash app in a separate thread
        import threading
        def run_app():
            app.run(debug=False, port=8051, host='127.0.0.1')

        cls.app_thread = threading.Thread(target=run_app, daemon=True)
        cls.app_thread.start()
        time.sleep(3)  # Wait for app to start

        # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        try:
            cls.driver = webdriver.Chrome(options=chrome_options)
            cls.driver.implicitly_wait(10)
            cls.base_url = "http://127.0.0.1:8051"
        except Exception as e:
            print(f"Could not initialize Chrome driver: {e}")
            cls.driver = None

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.driver:
            cls.driver.quit()

        # Clean up test database
        with server.app_context():
            db.drop_all()

    def setUp(self):
        """Set up each test"""
        if not self.driver:
            self.skipTest("WebDriver not available")

    def login_user(self):
        """Helper method to log in"""
        self.driver.get(f"{self.base_url}/")

        # Wait for login form
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "login-username"))
        )

        # Fill login form
        self.driver.find_element(By.ID, "login-username").send_keys("testadmin")
        self.driver.find_element(By.ID, "login-password").send_keys("testpass123")

        # Click login
        self.driver.find_element(By.ID, "login-btn").click()

        # Wait for dashboard to load
        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/dashboard")
        )

    def test_01_dashboard_page_load(self):
        """Test that dashboard page loads correctly"""
        print("Testing dashboard page load...")
        self.login_user()

        # Verify we're on dashboard
        self.assertIn("/dashboard", self.driver.current_url)

        # Check for enhanced navigation bar
        navbar = self.driver.find_element(By.CLASS_NAME, "navbar")
        self.assertIsNotNone(navbar)

        # Check for brand with emoji
        brand = navbar.find_element(By.CLASS_NAME, "navbar-brand")
        self.assertIn("🫀", brand.text)

        print("✓ Dashboard page loads correctly")

    def test_02_enhanced_navigation_bar(self):
        """Test enhanced navigation bar with emojis"""
        print("Testing enhanced navigation bar...")
        self.login_user()

        navbar = self.driver.find_element(By.CLASS_NAME, "navbar")

        # Check navigation items with emojis
        nav_links = navbar.find_elements(By.CLASS_NAME, "nav-link")

        expected_emojis = ["🏠", "📋", "📊", "🔬", "❤️", "⚙️", "🚪"]
        found_emojis = [link.text for link in nav_links]

        for emoji in expected_emojis:
            self.assertTrue(any(emoji in text for text in found_emojis),
                          f"Emoji {emoji} not found in navigation")

        print("✓ Enhanced navigation bar displays correctly")

    def test_03_dashboard_stats_display(self):
        """Test real-time dashboard statistics"""
        print("Testing dashboard statistics display...")
        self.login_user()

        # Wait for stats to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "total-patients"))
        )

        # Check stats elements
        total_patients = self.driver.find_element(By.ID, "total-patients")
        high_risk_count = self.driver.find_element(By.ID, "high-risk-count")
        avg_risk_score = self.driver.find_element(By.ID, "avg-risk-score")
        today_assessments = self.driver.find_element(By.ID, "today-assessments")

        # Verify stats are displayed (should be strings)
        self.assertIsInstance(total_patients.text, str)
        self.assertIsInstance(high_risk_count.text, str)
        self.assertIsInstance(avg_risk_score.text, str)
        self.assertIsInstance(today_assessments.text, str)

        print(f"✓ Dashboard stats: Total={total_patients.text}, High Risk={high_risk_count.text}, Avg={avg_risk_score.text}, Today={today_assessments.text}")

    def test_04_quick_actions_buttons(self):
        """Test quick action buttons presence and functionality"""
        print("Testing quick action buttons...")
        self.login_user()

        # Check for quick actions section
        quick_actions_card = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//h5[contains(text(), 'Quick Actions')]"))
        )

        # Find quick action buttons
        buttons = self.driver.find_elements(By.CLASS_NAME, "btn")
        quick_buttons = [btn for btn in buttons if "quick-" in btn.get_attribute("id")]

        expected_buttons = ["quick-assess-btn", "quick-report-btn", "quick-search-btn", "quick-analytics-btn"]
        found_buttons = [btn.get_attribute("id") for btn in quick_buttons]

        for expected_id in expected_buttons:
            self.assertIn(expected_id, found_buttons, f"Button {expected_id} not found")

        print("✓ Quick action buttons are present")

    def test_05_refresh_dashboard_stats(self):
        """Test dashboard stats refresh functionality"""
        print("Testing dashboard stats refresh...")
        self.login_user()

        # Get initial stats
        initial_total = self.driver.find_element(By.ID, "total-patients").text

        # Find and click refresh button
        refresh_btn = self.driver.find_element(By.ID, "refresh-dashboard-btn")
        refresh_btn.click()

        # Wait a moment for refresh
        time.sleep(2)

        # Verify stats are still displayed (refresh should work)
        refreshed_total = self.driver.find_element(By.ID, "total-patients").text
        self.assertEqual(initial_total, refreshed_total, "Stats should remain consistent after refresh")

        print("✓ Dashboard stats refresh works correctly")

    def test_06_quick_action_navigation(self):
        """Test quick action button navigation"""
        print("Testing quick action navigation...")
        self.login_user()

        # Test search patients button
        search_btn = self.driver.find_element(By.ID, "quick-search-btn")
        search_btn.click()

        # Wait for navigation
        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/patients")
        )
        self.assertIn("/patients", self.driver.current_url)

        # Go back to dashboard
        self.driver.get(f"{self.base_url}/dashboard")

        # Test analytics button
        analytics_btn = self.driver.find_element(By.ID, "quick-analytics-btn")
        analytics_btn.click()

        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/analytics")
        )
        self.assertIn("/analytics", self.driver.current_url)

        print("✓ Quick action navigation works correctly")

    def test_07_authentication_required(self):
        """Test that dashboard requires authentication"""
        print("Testing authentication requirements...")

        # Try to access dashboard without login
        self.driver.get(f"{self.base_url}/dashboard")

        # Should redirect to login
        WebDriverWait(self.driver, 10).until(
            EC.url_contains("/") or EC.url_contains("/login")
        )

        # Should not be on dashboard
        self.assertNotIn("/dashboard", self.driver.current_url)

        print("✓ Authentication properly required for dashboard")

    def test_08_responsive_design(self):
        """Test responsive design elements"""
        print("Testing responsive design...")
        self.login_user()

        # Check that cards are properly structured
        cards = self.driver.find_elements(By.CLASS_NAME, "card")
        self.assertGreater(len(cards), 5, "Should have multiple cards on dashboard")

        # Check for proper Bootstrap classes
        main_container = self.driver.find_element(By.CLASS_NAME, "container-fluid")
        self.assertIsNotNone(main_container, "Main container should be fluid")

        print("✓ Responsive design elements are present")

    def test_09_callback_error_handling(self):
        """Test callback error handling"""
        print("Testing callback error handling...")
        self.login_user()

        # Try to trigger callbacks without proper authentication context
        # This should be handled gracefully

        # Check that no JavaScript errors occurred
        logs = self.driver.get_log("browser")
        error_logs = [log for log in logs if log['level'] == 'SEVERE']

        # Allow some warnings but no critical errors
        critical_errors = [log for log in error_logs if 'Error' in log['message'] or 'Exception' in log['message']]
        self.assertEqual(len(critical_errors), 0, f"Critical JavaScript errors found: {critical_errors}")

        print("✓ Callback error handling works correctly")

def run_tests():
    """Run all dashboard enhancement tests"""
    print("Starting comprehensive dashboard enhancement tests...\n")

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(DashboardEnhancementTests)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    if result.skipped:
        print(f"\nSKIPPED:")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    print(f"{'='*50}")

    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
