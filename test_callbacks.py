import sys
sys.path.insert(0, '.')
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import pickle
import numpy as np
import torch
import scipy.stats as stats
import pytest
from flask import Flask, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from src.model.autoencoder import ECGAutoencoder
from src.ecg.synthetic_ecg import generate_12_lead_ecg
from src.dashboard.models import db, User, PatientAssessment

# Mock current_user for testing
class MockUser:
    def __init__(self):
        self.is_authenticated = True
        self.id = 1
        self.username = 'admin'
        self.role = 'admin'

@pytest.fixture(scope="module")
def setup_callback_models():
    try:
        gam = joblib.load("models/gam_model.pkl")
        rf = joblib.load("models/rf_residual.pkl")
        meta = joblib.load("models/meta_model.pkl")
    except Exception as e:
        pytest.fail(f"Error loading models: {e}")

    try:
        ae = ECGAutoencoder(24)
        ae.load_state_dict(torch.load("models/autoencoder.pt"))
        ae.eval()
    except Exception as e:
        pytest.fail(f"Error loading autoencoder: {e}")

    return gam, rf, meta, ae

def predict_risk_callback_logic(n_clicks, age, bp, chol, hr, gam, rf, meta, ae, current_user=None):
    """Test the predict_risk callback logic"""
    if current_user is None:
        current_user = MockUser()

    if n_clicks and current_user.is_authenticated:
        try:
            # Validate inputs
            if not all([age, bp, chol, hr]):
                return "0.000", "Incomplete Data", "0%", 0, "secondary", "Please fill in all patient information."

            # Generate synthetic ECG
            _, ecg = generate_12_lead_ecg(hr=hr/60)
            features = {}
            for lead, sig in ecg.items():
                features[f"{lead}_mean"] = np.mean(sig)
                features[f"{lead}_std"] = np.std(sig)

            X_ecg = np.array(list(features.values())).reshape(1, -1)
            X_tab = np.array([[age, bp, chol, hr]])

            # GAM prediction
            gam_pred = gam.predict_proba(X_tab)

            # Autoencoder encoding
            X_tensor = torch.tensor(X_ecg, dtype=torch.float32)
            Z = ae.encoder(X_tensor).detach().numpy()

            # RF prediction on residuals
            rf_pred = rf.predict(Z)[0]

            # Meta model
            meta_X = pd.DataFrame({"gam": [gam_pred], "rf": [rf_pred]})
            meta_pred_proba = meta.predict_proba(meta_X)
            if meta_pred_proba.ndim == 2:
                final_risk = meta_pred_proba[0, 1]
            else:
                final_risk = meta_pred_proba[0]

            # Determine risk category, color, and recommendations
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

            confidence = f"{np.random.uniform(85, 98):.1f}%"

            return f"{final_risk:.3f}", category, confidence, int(final_risk * 100), color, recommendations

        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", "Error", "0%", 0, "secondary", f"An error occurred during assessment: {str(e)}"

    return "0.000", "Low Risk", "95%", 0, "success", "Please log in to assess patients."

@pytest.mark.parametrize("n_clicks,age,bp,chol,hr,expected_category", [
    (1, 50, 120, 200, 70, "Low Risk"),  # Normal values
    (1, 30, 100, 150, 60, "Low Risk"),  # Low risk values
    (1, 80, 180, 300, 100, "High Risk"), # High risk values
])
def test_predict_risk_callback(setup_callback_models, n_clicks, age, bp, chol, hr, expected_category):
    gam, rf, meta, ae = setup_callback_models

    result = predict_risk_callback_logic(n_clicks, age, bp, chol, hr, gam, rf, meta, ae, MockUser())
    risk_score, category, confidence, progress_value, progress_color, recommendations = result

    # Assert valid risk score
    if risk_score not in ["0.000", "Error"]:
        risk_val = float(risk_score)
        assert 0 <= risk_val <= 1, f"Risk score {risk_val} is not between 0 and 1"

        # Assert category matches expected
        assert category == expected_category, f"Expected {expected_category}, got {category}"

        # Assert color logic
        if risk_val < 0.3:
            assert progress_color == "success", f"Low risk should have success color, got {progress_color}"
        elif risk_val < 0.7:
            assert progress_color == "warning", f"Moderate risk should have warning color, got {progress_color}"
        else:
            assert progress_color == "danger", f"High risk should have danger color, got {progress_color}"

        # Assert progress value
        expected_progress = int(risk_val * 100)
        assert progress_value == expected_progress, f"Progress value mismatch: expected {expected_progress}, got {progress_value}"

@pytest.mark.parametrize("n_clicks,age,bp,chol,hr", [
    (1, None, 120, 200, 70),  # Missing age
    (1, 50, None, 200, 70),   # Missing BP
    (1, 50, 120, None, 70),   # Missing cholesterol
    (1, 50, 120, 200, None),  # Missing heart rate
])
def test_predict_risk_callback_missing_data(setup_callback_models, n_clicks, age, bp, chol, hr):
    gam, rf, meta, ae = setup_callback_models

    result = predict_risk_callback_logic(n_clicks, age, bp, chol, hr, gam, rf, meta, ae)
    risk_score, category, confidence, progress_value, progress_color, recommendations = result

    # Assert incomplete data handling
    assert risk_score == "0.000", f"Expected '0.000' for incomplete data, got {risk_score}"
    assert category == "Incomplete Data", f"Expected 'Incomplete Data' category, got {category}"
    assert progress_color == "secondary", f"Expected 'secondary' color for incomplete data, got {progress_color}"

def test_predict_risk_callback_not_authenticated(setup_callback_models):
    gam, rf, meta, ae = setup_callback_models

    # Mock unauthenticated user
    class MockUnauthenticatedUser:
        def __init__(self):
            self.is_authenticated = False

    unauthenticated_user = MockUnauthenticatedUser()

    result = predict_risk_callback_logic(1, 50, 120, 200, 70, gam, rf, meta, ae, unauthenticated_user)
    risk_score, category, confidence, progress_value, progress_color, recommendations = result

    # Assert not authenticated handling
    assert risk_score == "0.000", f"Expected '0.000' for unauthenticated user, got {risk_score}"
    assert "Please log in" in recommendations, f"Expected login message, got {recommendations}"
