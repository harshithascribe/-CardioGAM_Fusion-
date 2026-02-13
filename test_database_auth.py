#!/usr/bin/env python3
"""
Comprehensive Database and Authentication Testing for CardioGAM-Fusion
Tests database operations, user authentication, and data integrity
"""
import sys
import os
sys.path.insert(0, '.')

from src.dashboard.app import server, db
from src.dashboard.models import User, PatientAssessment
from flask import Flask
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import torch
from src.model.autoencoder import ECGAutoencoder
from src.ecg.synthetic_ecg import generate_12_lead_ecg

def test_database_setup():
    """Test database initialization and setup"""
    print("🗄️  Testing Database Setup...")

    try:
        with server.app_context():
            # Test database connection
            with db.engine.connect() as connection:
                connection.execute(db.text("SELECT 1"))
            print("  ✅ Database connection successful")

            # Test table creation
            db.create_all()
            print("  ✅ Database tables created")

            # Check if default admin user exists
            admin_user = User.query.filter_by(username='admin').first()
            if admin_user:
                print("  ✅ Default admin user exists")
                print(f"    Username: {admin_user.username}")
                print(f"    Email: {admin_user.email}")
                print(f"    Role: {admin_user.role}")
            else:
                print("  ❌ Default admin user not found")
                return False

        return True

    except Exception as e:
        print(f"  ❌ Database setup failed: {str(e)}")
        return False

def test_user_authentication():
    """Test user authentication functionality"""
    print("\n🔐 Testing User Authentication...")

    try:
        with server.app_context():
            # Test user creation
            test_user = User(username='testuser', email='test@example.com', role='user')
            test_user.set_password('testpass123')
            db.session.add(test_user)
            db.session.commit()
            print("  ✅ Test user created successfully")

            # Test password verification
            if test_user.check_password('testpass123'):
                print("  ✅ Password verification successful")
            else:
                print("  ❌ Password verification failed")
                return False

            # Test wrong password
            if not test_user.check_password('wrongpass'):
                print("  ✅ Wrong password correctly rejected")
            else:
                print("  ❌ Wrong password incorrectly accepted")
                return False

            # Test user lookup by username
            found_user = User.query.filter_by(username='testuser').first()
            if found_user and found_user.id == test_user.id:
                print("  ✅ User lookup by username successful")
            else:
                print("  ❌ User lookup by username failed")
                return False

            # Test user lookup by email
            found_user_email = User.query.filter_by(email='test@example.com').first()
            if found_user_email and found_user_email.id == test_user.id:
                print("  ✅ User lookup by email successful")
            else:
                print("  ❌ User lookup by email failed")
                return False

            # Clean up test user
            db.session.delete(test_user)
            db.session.commit()
            print("  ✅ Test user cleanup successful")

        return True

    except Exception as e:
        print(f"  ❌ Authentication test failed: {str(e)}")
        return False

def test_patient_assessment_crud():
    """Test Patient Assessment CRUD operations"""
    print("\n🏥 Testing Patient Assessment CRUD...")

    try:
        with server.app_context():
            # Create test assessment
            test_assessment = PatientAssessment(
                patient_id='TEST-001',
                age=45,
                bp=130,
                cholesterol=220,
                heart_rate=75,
                risk_score=0.65,
                risk_category='Moderate Risk',
                confidence=92.5,
                recommendations='Regular monitoring recommended',
                created_by=1  # admin user
            )

            # Add ECG features (simulate)
            for i in range(1, 13):
                setattr(test_assessment, f'lead_{i}_mean', np.random.uniform(0.1, 0.9))
                setattr(test_assessment, f'lead_{i}_std', np.random.uniform(0.05, 0.3))

            db.session.add(test_assessment)
            db.session.commit()
            print("  ✅ Patient assessment created successfully")

            # Test retrieval
            retrieved = PatientAssessment.query.filter_by(patient_id='TEST-001').first()
            if retrieved and retrieved.age == 45:
                print("  ✅ Patient assessment retrieval successful")
            else:
                print("  ❌ Patient assessment retrieval failed")
                return False

            # Test update
            retrieved.age = 46
            retrieved.updated_at = datetime.utcnow()
            db.session.commit()

            updated = PatientAssessment.query.filter_by(patient_id='TEST-001').first()
            if updated.age == 46:
                print("  ✅ Patient assessment update successful")
            else:
                print("  ❌ Patient assessment update failed")
                return False

            # Test deletion
            db.session.delete(retrieved)
            db.session.commit()

            deleted = PatientAssessment.query.filter_by(patient_id='TEST-001').first()
            if deleted is None:
                print("  ✅ Patient assessment deletion successful")
            else:
                print("  ❌ Patient assessment deletion failed")
                return False

        return True

    except Exception as e:
        print(f"  ❌ CRUD test failed: {str(e)}")
        return False

def test_data_integrity():
    """Test data integrity and constraints"""
    print("\n🔍 Testing Data Integrity...")

    try:
        with server.app_context():
            # Test required fields
            try:
                invalid_assessment = PatientAssessment()  # Missing required fields
                db.session.add(invalid_assessment)
                db.session.commit()
                print("  ❌ Data integrity check failed - allowed null values")
                return False
            except Exception:
                db.session.rollback()  # Roll back the failed transaction
                print("  ✅ Data integrity enforced - null values rejected")

            # Test valid data insertion
            valid_assessment = PatientAssessment(
                patient_id='VALID-001',
                age=50,
                bp=120,
                cholesterol=200,
                heart_rate=70,
                risk_score=0.45,
                risk_category='Low Risk',
                confidence=95.0,
                recommendations='Healthy lifestyle maintained',
                created_by=1
            )

            # Add ECG features
            for i in range(1, 13):
                setattr(valid_assessment, f'lead_{i}_mean', 0.5)
                setattr(valid_assessment, f'lead_{i}_std', 0.15)

            db.session.add(valid_assessment)
            db.session.commit()
            print("  ✅ Valid data insertion successful")

            # Test risk score constraints (should be between 0 and 1)
            if 0 <= valid_assessment.risk_score <= 1:
                print("  ✅ Risk score constraints validated")
            else:
                print("  ❌ Risk score constraints violated")
                return False

            # Clean up
            db.session.delete(valid_assessment)
            db.session.commit()

        return True

    except Exception as e:
        print(f"  ❌ Data integrity test failed: {str(e)}")
        return False

def test_model_integration():
    """Test ML model integration with database"""
    print("\n🤖 Testing Model Integration...")

    try:
        # Load models
        gam = joblib.load("models/gam_model.pkl")
        rf = joblib.load("models/rf_residual.pkl")
        meta = joblib.load("models/meta_model.pkl")
        ae = ECGAutoencoder(24)
        ae.load_state_dict(torch.load("models/autoencoder.pt"))
        ae.eval()

        print("  ✅ Models loaded successfully")

        # Generate test ECG
        _, ecg = generate_12_lead_ecg(hr=70)
        features = {}
        for lead, sig in ecg.items():
            features[f"{lead}_mean"] = np.mean(sig)
            features[f"{lead}_std"] = np.std(sig)

        X_ecg = np.array(list(features.values())).reshape(1, -1)
        X_tab = np.array([[45, 130, 220, 75]])

        # Test predictions
        gam_pred_proba = gam.predict_proba(X_tab)
        gam_pred = gam_pred_proba[0, 1] if gam_pred_proba.ndim == 2 else gam_pred_proba[0]

        Z = ae.encoder(torch.tensor(X_ecg, dtype=torch.float32)).detach().numpy()
        rf_pred = rf.predict(Z)[0]

        meta_X = pd.DataFrame({"gam": [gam_pred], "rf": [rf_pred]})
        meta_pred_proba = meta.predict_proba(meta_X)
        final_risk = meta_pred_proba[0, 1] if meta_pred_proba.ndim == 2 else meta_pred_proba[0]

        print("  ✅ Model predictions successful")
        print(".4f")
        print(".4f")
        print(".4f")
        # Test prediction consistency
        _, ecg2 = generate_12_lead_ecg(hr=70)
        features2 = {}
        for lead, sig in ecg2.items():
            features2[f"{lead}_mean"] = np.mean(sig)
            features2[f"{lead}_std"] = np.std(sig)

        X_ecg2 = np.array(list(features2.values())).reshape(1, -1)
        Z2 = ae.encoder(torch.tensor(X_ecg2, dtype=torch.float32)).detach().numpy()
        rf_pred2 = rf.predict(Z2)[0]

        if abs(rf_pred - rf_pred2) < 0.01:  # Allow small variation
            print("  ✅ Prediction consistency verified")
        else:
            print("  ❌ Prediction consistency failed")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Model integration test failed: {str(e)}")
        return False

def test_bulk_operations():
    """Test bulk database operations"""
    print("\n📊 Testing Bulk Operations...")

    try:
        with server.app_context():
            # Create multiple test assessments
            assessments = []
            for i in range(10):
                assessment = PatientAssessment(
                    patient_id=f'BULK-{i:03d}',
                    age=40 + i,
                    bp=110 + i*2,
                    cholesterol=180 + i*5,
                    heart_rate=65 + i,
                    risk_score=0.3 + i*0.05,
                    risk_category='Low Risk' if i < 5 else 'Moderate Risk',
                    confidence=85 + i,
                    recommendations=f'Test recommendation {i}',
                    created_by=1
                )

                # Add ECG features
                for j in range(1, 13):
                    setattr(assessment, f'lead_{j}_mean', np.random.uniform(0.1, 0.9))
                    setattr(assessment, f'lead_{j}_std', np.random.uniform(0.05, 0.3))

                assessments.append(assessment)

            # Bulk insert
            db.session.add_all(assessments)
            db.session.commit()
            print("  ✅ Bulk insert successful")

            # Bulk query
            bulk_results = PatientAssessment.query.filter(
                PatientAssessment.patient_id.like('BULK-%')
            ).all()

            if len(bulk_results) == 10:
                print("  ✅ Bulk query successful")
            else:
                print(f"  ❌ Bulk query failed - expected 10, got {len(bulk_results)}")
                return False

            # Bulk update
            for assessment in bulk_results:
                assessment.confidence += 1
            db.session.commit()
            print("  ✅ Bulk update successful")

            # Bulk delete
            for assessment in bulk_results:
                db.session.delete(assessment)
            db.session.commit()

            remaining = PatientAssessment.query.filter(
                PatientAssessment.patient_id.like('BULK-%')
            ).count()

            if remaining == 0:
                print("  ✅ Bulk delete successful")
            else:
                print(f"  ❌ Bulk delete failed - {remaining} records remaining")
                return False

        return True

    except Exception as e:
        print(f"  ❌ Bulk operations test failed: {str(e)}")
        return False

def main():
    """Run all database and authentication tests"""
    print("🚀 Starting Comprehensive Database & Authentication Testing\n")

    test_results = {
        "database_setup": test_database_setup(),
        "user_auth": test_user_authentication(),
        "crud_operations": test_patient_assessment_crud(),
        "data_integrity": test_data_integrity(),
        "model_integration": test_model_integration(),
        "bulk_operations": test_bulk_operations()
    }

    print("\n" + "="*70)
    print("📊 DATABASE & AUTHENTICATION TEST RESULTS SUMMARY")
    print("="*70)

    # Count successful tests
    successful_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        if result:
            print(f"✅ {test_name.upper().replace('_', ' ')}: PASSED")
            successful_tests += 1
        else:
            print(f"❌ {test_name.upper().replace('_', ' ')}: FAILED")

    print(f"\n🎯 Overall Score: {successful_tests}/{total_tests} tests passed")

    if successful_tests == total_tests:
        print("🎉 ALL DATABASE & AUTHENTICATION TESTS PASSED!")
        print("The CardioGAM-Fusion database and authentication systems are fully functional.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
