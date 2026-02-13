#!/usr/bin/env python3
"""
Debug script to test the dashboard callback logic and identify authentication/database issues
"""
import sys
import os
sys.path.insert(0, '.')

import joblib
import pickle
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from src.model.autoencoder import ECGAutoencoder
from src.ecg.synthetic_ecg import generate_12_lead_ecg
from src.dashboard.models import db, PatientAssessment
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

def test_callback_logic():
    print("=== Testing Dashboard Callback Logic ===")

    # Test data
    age = 50
    bp = 120
    cholesterol = 200
    hr = 70

    print(f"Test inputs: age={age}, bp={bp}, cholesterol={cholesterol}, heart_rate={hr}")

    # Create Flask app context for database operations
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cardio_fusion.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    try:
        with app.app_context():
            # Load models (same as in app.py)
            print("Loading models...")
            gam = joblib.load("models/gam_model.pkl")
            rf = joblib.load("models/rf_residual.pkl")
            meta = joblib.load("models/meta_model.pkl")
            ae = ECGAutoencoder(24)
            ae.load_state_dict(torch.load("models/autoencoder.pt"))
            ae.eval()
            print("✓ Models loaded successfully")

            # Simulate the prediction logic from the callback
            print("Running prediction logic...")

            # Generate synthetic ECG
            seed = hash((age, bp, cholesterol, hr)) % (2**32)
            np.random.seed(seed)
            _, ecg = generate_12_lead_ecg(hr=hr/60)

            if not ecg:
                raise ValueError("Failed to generate ECG data")

            features = {}
            for lead, sig in ecg.items():
                if len(sig) == 0:
                    raise ValueError(f"Empty ECG signal for lead {lead}")
                features[f"{lead}_mean"] = np.mean(sig)
                features[f"{lead}_std"] = np.std(sig)

            X_ecg = np.array(list(features.values())).reshape(1, -1)
            X_tab = pd.DataFrame([[age, bp, cholesterol, hr]], columns=["age","bp","cholesterol","heart_rate"])

            # GAM prediction
            gam_pred_proba = gam.predict_proba(X_tab)
            if gam_pred_proba.ndim == 2:
                gam_pred = gam_pred_proba[0, 1]
            else:
                gam_pred = gam_pred_proba[0]

            # Autoencoder encoding
            X_tensor = torch.tensor(X_ecg, dtype=torch.float32)
            Z = ae.encoder(X_tensor).detach().numpy()

            # RF prediction
            rf_pred = rf.predict(Z)[0]

            # Meta model
            meta_X = pd.DataFrame({"gam": [gam_pred], "rf": [rf_pred]})
            meta_pred_proba = meta.predict_proba(meta_X)
            if meta_pred_proba.ndim == 2:
                final_risk = meta_pred_proba[0, 1]
            else:
                final_risk = meta_pred_proba[0]

            print(f"✓ Prediction successful: risk_score={final_risk}")

            # Determine risk category
            if final_risk < 0.3:
                category = "Low Risk"
                color = "success"
                recommendations = "Patient shows low cardiovascular risk. Recommend annual check-ups and healthy lifestyle maintenance."
            elif final_risk < 0.7:
                category = "Moderate Risk"
                color = "warning"
                recommendations = "Patient shows moderate cardiovascular risk. Recommend lifestyle modifications, medication review, and closer monitoring."
            else:
                category = "High Risk"
                color = "danger"
                recommendations = "Patient shows high cardiovascular risk. Immediate intervention recommended: cardiology consultation, aggressive risk factor management, and close monitoring."

            # Generate confidence
            confidence_seed = hash((age, bp, cholesterol, hr, 'confidence')) % (2**32)
            np.random.seed(confidence_seed)
            confidence = f"{np.random.uniform(85, 98):.1f}%"

            print(f"✓ Category: {category}, Confidence: {confidence}")

            # Test database save (without authentication)
            print("Testing database save...")
            patient_id = f"PAT-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

            assessment = PatientAssessment(
                patient_id=patient_id,
                age=age,
                bp=bp,
                cholesterol=cholesterol,
                heart_rate=hr,
                risk_score=final_risk,
                risk_category=category,
                confidence=float(confidence.strip('%')),
                recommendations=recommendations,
                created_by=1  # Default admin user ID
            )

            # Add ECG features
            for i, lead in enumerate(['lead_1', 'lead_2', 'lead_3', 'lead_4', 'lead_5', 'lead_6',
                                     'lead_7', 'lead_8', 'lead_9', 'lead_10', 'lead_11', 'lead_12']):
                setattr(assessment, f"{lead}_mean", features[f"{lead}_mean"])
                setattr(assessment, f"{lead}_std", features[f"{lead}_std"])

            db.session.add(assessment)
            db.session.commit()

            print("✓ Database save successful")
            print("✓ Callback logic works correctly!")

            return True

    except Exception as e:
        print(f"✗ Error in callback logic: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_callback_logic()
    if success:
        print("\n🎉 Callback logic works correctly!")
    else:
        print("\n❌ Callback logic has issues - check the error above")
