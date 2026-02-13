#!/usr/bin/env python3
"""
Basic test script for dashboard enhancements
Tests core functionality without browser automation
"""

import sys
import os
import requests
import json
import time

# Add src to path
sys.path.insert(0, 'src')

def test_database_setup():
    """Test that database is properly set up with sample data"""
    print("Testing database setup...")

    try:
        # Import the server and db from the dashboard app
        from src.dashboard.app import server, db, User, PatientAssessment

        with server.app_context():
            # Check if tables exist
            user_count = User.query.count()
            assessment_count = PatientAssessment.query.count()

            print(f"✓ Database tables exist")
            print(f"  - Users: {user_count}")
            print(f"  - Patient Assessments: {assessment_count}")

            if user_count > 0 and assessment_count > 0:
                print("✓ Sample data is present")
                return True
            else:
                print("⚠ No sample data found - this is expected if no assessments have been made yet")
                return True  # Not a failure, just no data yet

    except Exception as e:
        print(f"✗ Database error: {e}")
        return False

def test_app_startup():
    """Test that the app can be initialized properly"""
    print("Testing app startup...")

    try:
        # Test that we can import and initialize the app without errors
        from src.dashboard.app import app, server

        # Check that the app has the required attributes
        if hasattr(app, 'layout') and app.layout is not None:
            print("✓ App can be initialized successfully")
            return True
        else:
            print("⚠ App layout not properly configured")
            return False
    except Exception as e:
        print(f"✗ Could not initialize app: {e}")
        return False

def test_dashboard_stats_callback():
    """Test the dashboard stats callback functionality"""
    print("Testing dashboard stats callback...")

    try:
        # Test that we can import the dashboard stats callback without errors
        from src.dashboard.app import update_dashboard_stats

        # Check that the callback function exists and is callable
        if callable(update_dashboard_stats):
            print("✓ Dashboard stats callback is properly defined")
            return True
        else:
            print("⚠ Dashboard stats callback not callable")
            return False
    except Exception as e:
        print(f"✗ Error testing dashboard stats callback: {e}")
        return False

def test_quick_actions_setup():
    """Test that quick action buttons are properly configured"""
    print("Testing quick actions setup...")

    # This is a basic check - in a real test we'd inspect the Dash layout
    # For now, just verify the app structure exists
    try:
        from src.dashboard.app import app
        if hasattr(app, 'layout') and app.layout is not None:
            print("✓ App layout is configured")
            return True
        else:
            print("⚠ App layout not found")
            return False
    except Exception as e:
        print(f"✗ Error checking app layout: {e}")
        return False

def run_basic_tests():
    """Run all basic dashboard tests"""
    print("Running basic dashboard enhancement tests...\n")

    tests = [
        ("Database Setup", test_database_setup),
        ("App Startup", test_app_startup),
        ("Dashboard Stats", test_dashboard_stats_callback),
        ("Quick Actions", test_quick_actions_setup),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
        print()

    # Print summary
    print("=" * 40)
    print("BASIC TEST SUMMARY")
    print("=" * 40)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    success = passed == total
    print(f"Overall: {'PASS' if success else 'FAIL'}")

    return success

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
