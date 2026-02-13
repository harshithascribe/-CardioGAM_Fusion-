import sys
sys.path.insert(0, '.')
import pandas as pd
import joblib
import numpy as np
import torch
from src.model.autoencoder import ECGAutoencoder
from src.ecg.synthetic_ecg import generate_12_lead_ecg

def test_prediction():
    try:
        print("Loading data...")
        df = pd.read_csv("data/patients_ecg.csv")
        print(f"Data loaded: {df.shape}")

        print("Loading models...")
        gam = joblib.load("models/gam_model.pkl")
        rf = joblib.load("models/rf_residual.pkl")
        meta = joblib.load("models/meta_model.pkl")
        print("Models loaded successfully")

        print("Loading autoencoder...")
        ae = ECGAutoencoder(24)
        ae.load_state_dict(torch.load("models/autoencoder.pt"))
        ae.eval()
        print("Autoencoder loaded successfully")

        # Test prediction with sample data
        age, bp, chol, hr = 50, 120, 200, 70
        print(f"Testing prediction with: age={age}, bp={bp}, chol={chol}, hr={hr}")

        # Generate synthetic ECG
        print("Generating ECG...")
        _, ecg = generate_12_lead_ecg(hr=hr/60)
        print(f"ECG generated with {len(ecg)} leads")

        features = {}
        for lead, sig in ecg.items():
            features[f"{lead}_mean"] = np.mean(sig)
            features[f"{lead}_std"] = np.std(sig)

        X_ecg = np.array(list(features.values())).reshape(1, -1)
        X_tab = np.array([[age, bp, chol, hr]])
        print(f"ECG features shape: {X_ecg.shape}")
        print(f"Tabular data shape: {X_tab.shape}")

        # GAM prediction
        print("Running GAM prediction...")
        gam_pred_proba = gam.predict_proba(X_tab)
        if gam_pred_proba.ndim == 2:
            gam_pred = gam_pred_proba[0, 1]
        else:
            gam_pred = gam_pred_proba[0]
        print(f"GAM prediction: {gam_pred}")

        # Autoencoder encoding
        print("Running autoencoder encoding...")
        X_tensor = torch.tensor(X_ecg, dtype=torch.float32)
        Z = ae.encoder(X_tensor).detach().numpy()
        print(f"Encoded features shape: {Z.shape}")

        # RF prediction
        print("Running RF prediction...")
        rf_pred = rf.predict(Z)[0]
        print(f"RF prediction: {rf_pred}")

        # Meta model
        print("Running meta model prediction...")
        meta_X = pd.DataFrame({"gam": [gam_pred], "rf": [rf_pred]})
        meta_pred_proba = meta.predict_proba(meta_X)
        if meta_pred_proba.ndim == 2:
            final_risk = meta_pred_proba[0, 1]
        else:
            final_risk = meta_pred_proba[0]
        print(f"Final risk score: {final_risk}")

        print("Prediction completed successfully!")
        return final_risk

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_prediction()
