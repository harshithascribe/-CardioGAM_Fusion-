import sys
sys.path.insert(0, '.')
import joblib
import torch
from src.model.autoencoder import ECGAutoencoder

def test_model_loading():
    try:
        print("Testing model loading...")

        # Test GAM model
        print("Loading GAM model...")
        gam = joblib.load("models/gam_model.pkl")
        print("GAM model loaded successfully")

        # Test RF model
        print("Loading RF model...")
        rf = joblib.load("models/rf_residual.pkl")
        print("RF model loaded successfully")

        # Test Meta model
        print("Loading Meta model...")
        meta = joblib.load("models/meta_model.pkl")
        print("Meta model loaded successfully")

        # Test Autoencoder
        print("Loading Autoencoder...")
        ae = ECGAutoencoder(24)
        ae.load_state_dict(torch.load("models/autoencoder.pt"))
        ae.eval()
        print("Autoencoder loaded successfully")

        print("All models loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()
