import joblib
import pickle
import torch
from src.model.autoencoder import ECGAutoencoder

try:
    gam = joblib.load("models/gam_model.pkl")
    print("GAM loaded successfully")
except Exception as e:
    print(f"Error loading GAM: {e}")

try:
    rf = joblib.load("models/rf_residual.pkl")
    print("RF loaded successfully")
except Exception as e:
    print(f"Error loading RF: {e}")

try:
    meta = joblib.load("models/meta_model.pkl")
    print("Meta loaded successfully")
except Exception as e:
    print(f"Error loading Meta: {e}")

try:
    ae = ECGAutoencoder(24)
    ae.load_state_dict(torch.load("models/autoencoder.pt"))
    ae.eval()
    print("Autoencoder loaded successfully")
except Exception as e:
    print(f"Error loading Autoencoder: {e}")
