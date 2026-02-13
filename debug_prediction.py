#!/usr/bin/env python3
"""
Debug script to test the risk prediction logic and identify the source of System Error
"""
import sys
import os
sys.path.insert(0, '.')

import joblib
import pickle
import numpy as np
import torch
import pandas as pd
from src.model.autoencoder import ECGAutoencoder
from src.ecg.synthetic_ecg import generate_12_lead_ecg

def test_prediction():
    print("=== Testing Risk Prediction Logic ===")

    # Test data
    age = 50
    bp = 120
    cholesterol = 200
    hr = 70

    print(f"Test inputs: age={age}, bp={bp}, cholesterol={cholesterol}, heart_rate={hr}")

    try:
        # Load models
        print("Loading models...")
        gam = joblib.load("models/gam_model.pkl")
        rf = joblib.load("models/rf_residual.pkl")
        meta = joblib.load("models/meta_model.pkl")
        ae = ECGAutoencoder(24)
        ae.load_state_dict(torch.load("models/autoencoder.pt"))
        ae.eval()
        print("✓ Models loaded successfully")

        # Generate synthetic ECG
        print("Generating synthetic ECG...")
        seed = hash((age, bp, cholesterol, hr)) % (2**32)
        np.random.seed(seed)
        _, ecg = generate_12_lead_ecg(hr=hr/60)
        print(f"✓ ECG generated with {len(ecg)} leads")

        # Extract features
        print("Extracting ECG features...")
        features = {}
        for lead, sig in ecg.items():
            if len(sig) == 0:
                raise ValueError(f"Empty ECG signal for lead {lead}")
            features[f"{lead}_mean"] = np.mean(sig)
            features[f"{lead}_std"] = np.std(sig)
        print(f"✓ Features extracted: {len(features)} features")

        X_ecg = np.array(list(features.values())).reshape(1, -1)
        X_tab = pd.DataFrame([[age, bp, cholesterol, hr]], columns=["age","bp","cholesterol","heart_rate"])
        print(f"✓ X_ecg shape: {X_ecg.shape}, X_tab shape: {X_tab.shape}")

        # GAM prediction
        print("Running GAM prediction...")
        gam_pred_proba = gam.predict_proba(X_tab)
        if gam_pred_proba.ndim == 2:
            gam_pred = gam_pred_proba[0, 1]
        else:
            gam_pred = gam_pred_proba[0]
        print(f"✓ GAM prediction: {gam_pred}")

        # Autoencoder encoding
        print("Running autoencoder encoding...")
        X_tensor = torch.tensor(X_ecg, dtype=torch.float32)
        Z = ae.encoder(X_tensor).detach().numpy()
        print(f"✓ Encoded shape: {Z.shape}")

        # RF prediction
        print("Running RF prediction...")
        rf_pred = rf.predict(Z)[0]
        print(f"✓ RF prediction: {rf_pred}")

        # Meta model
        print("Running meta model prediction...")
        meta_X = pd.DataFrame({"gam": [gam_pred], "rf": [rf_pred]})
        meta_pred_proba = meta.predict_proba(meta_X)
        if meta_pred_proba.ndim == 2:
            final_risk = meta_pred_proba[0, 1]
        else:
            final_risk = meta_pred_proba[0]
        print(f"✓ Final risk score: {final_risk}")

        # Determine category
        if final_risk < 0.3:
            category = "Low Risk"
        elif final_risk < 0.7:
            category = "Moderate Risk"
        else:
            category = "High Risk"

        print(f"✓ Risk category: {category}")
        print("✓ Prediction completed successfully!")

        return True

    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n🎉 Prediction logic works correctly!")
    else:
        print("\n❌ Prediction logic has issues - check the error above")
