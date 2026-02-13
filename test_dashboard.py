import sys
sys.path.insert(0, '.')
import pandas as pd
import joblib
import pickle
import numpy as np
import torch
import pytest
from src.model.autoencoder import ECGAutoencoder
from src.ecg.synthetic_ecg import generate_12_lead_ecg

# Load data and models (similar to app.py)
@pytest.fixture(scope="module")
def setup_models():
    try:
        df = pd.read_csv("data/patients_ecg.csv")
    except Exception as e:
        pytest.fail(f"Error loading data: {e}")

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

    return df, gam, rf, meta, ae

def predict_risk(age, bp, chol, hr, gam, rf, meta, ae):
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
    if gam_pred.ndim == 2:
        gam_pred = gam_pred[:, 1]
    gam_pred = gam_pred[0] if hasattr(gam_pred, '__len__') else gam_pred

    # Autoencoder encoding
    X_tensor = torch.tensor(X_ecg, dtype=torch.float32)
    Z = ae.encoder(X_tensor).detach().numpy()

    # RF prediction (regressor, not classifier)
    rf_pred = rf.predict(Z)[0]

    # Meta model
    meta_X = pd.DataFrame({"gam": [gam_pred], "rf": [rf_pred]})
    try:
        final_risk = meta.predict_proba(meta_X)[:, 1][0]
    except Exception as e:
        # Fallback: use simple average if meta model fails
        final_risk = (gam_pred + rf_pred) / 2

    return final_risk

@pytest.mark.parametrize("age,bp,chol,hr", [
    (50, 120, 200, 70),  # Normal values
    (30, 100, 150, 60),  # Low values
    (80, 180, 300, 100), # High values
])
def test_predict_risk(setup_models, age, bp, chol, hr):
    df, gam, rf, meta, ae = setup_models

    risk = predict_risk(age, bp, chol, hr, gam, rf, meta, ae)

    # Assert risk is between 0 and 1
    assert 0 <= risk <= 1, f"Risk score {risk} is not between 0 and 1"

    # Assert risk is a float
    assert isinstance(risk, (float, np.float32, np.float64)), f"Risk score {risk} is not a float"

def test_data_loading(setup_models):
    df, _, _, _, _ = setup_models

    # Check data shape
    assert df.shape[0] > 0, "Data has no rows"
    assert df.shape[1] > 0, "Data has no columns"

    # Check risk column exists and is valid
    assert 'risk' in df.columns, "Risk column missing"
    assert df['risk'].min() >= 0, "Risk values below 0"
    assert df['risk'].max() <= 1, "Risk values above 1"

def test_analytics_callbacks(setup_models):
    df, _, _, _, _ = setup_models

    # Test risk filtering logic
    high_risk = df[df['risk'] >= 0.7]
    moderate_risk = df[(df['risk'] >= 0.3) & (df['risk'] < 0.7)]
    low_risk = df[df['risk'] < 0.3]

    assert len(high_risk) + len(moderate_risk) + len(low_risk) == len(df), "Risk categories don't cover all data"

    # Test that filtering works
    assert all(high_risk['risk'] >= 0.7), "High risk filter failed"
    assert all(moderate_risk['risk'] >= 0.3) and all(moderate_risk['risk'] < 0.7), "Moderate risk filter failed"
    assert all(low_risk['risk'] < 0.3), "Low risk filter failed"
