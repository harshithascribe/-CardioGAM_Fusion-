import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from src.model.autoencoder import ECGAutoencoder
from src.ecg.synthetic_ecg import generate_12_lead_ecg

# Load data
print("🔄 Loading data and models...")
df = pd.read_csv("data/patients_ecg.csv")
print(f"📊 Dataset shape: {df.shape}")

# Load trained models
gam = joblib.load("models/gam_model.pkl")
rf = joblib.load("models/rf_residual.pkl")
meta = joblib.load("models/meta_model.pkl")

ae = ECGAutoencoder(24)
ae.load_state_dict(torch.load("models/autoencoder.pt"))
ae.eval()

print("✅ All models loaded successfully")

# Prepare data
ecg_cols = [c for c in df.columns if "lead" in c]
X_ecg = df[ecg_cols].values
X_tab = df[["age","bp","cholesterol","heart_rate"]]
y = df["risk"]

# Split data for evaluation
X_tab_train, X_tab_test, y_train, y_test = train_test_split(X_tab, y, test_size=0.2, random_state=42)
X_ecg_train, X_ecg_test = train_test_split(X_ecg, test_size=0.2, random_state=42)

print(f"🔀 Train/Test split: {len(X_tab_train)}/{len(X_tab_test)} samples")

# Evaluate Autoencoder
print("\n🧠 AUTOENCODER EVALUATION")
X_tensor_test = torch.tensor(X_ecg_test, dtype=torch.float32)
with torch.no_grad():
    recon_test, z_test = ae(X_tensor_test)
    ae_mse = torch.nn.functional.mse_loss(recon_test, X_tensor_test).item()
    ae_rmse = np.sqrt(ae_mse)

print(f"   MSE: {ae_mse:.6f}")
print(f"   RMSE: {ae_rmse:.6f}")

# Evaluate GAM
print("\n🎯 GAM MODEL EVALUATION")
gam_pred_proba = gam.predict_proba(X_tab_test)
if gam_pred_proba.ndim == 2:
    gam_pred_binary = (gam_pred_proba[:, 1] > 0.5).astype(int)
    gam_auc_input = gam_pred_proba[:, 1]
else:
    gam_pred_binary = (gam_pred_proba > 0.5).astype(int)
    gam_auc_input = gam_pred_proba

gam_accuracy = accuracy_score(y_test, gam_pred_binary)
gam_precision = precision_score(y_test, gam_pred_binary)
gam_recall = recall_score(y_test, gam_pred_binary)
gam_f1 = f1_score(y_test, gam_pred_binary)
gam_auc = roc_auc_score(y_test, gam_auc_input)

print(f"   Accuracy: {gam_accuracy:.4f}")
print(f"   Precision: {gam_precision:.4f}")
print(f"   Recall: {gam_recall:.4f}")
print(f"   F1-Score: {gam_f1:.4f}")
print(f"   AUC-ROC: {gam_auc:.4f}")

# Evaluate RF on encoded features
print("\n🌲 RANDOM FOREST EVALUATION")
X_tensor_train = torch.tensor(X_ecg_train, dtype=torch.float32)
with torch.no_grad():
    z_train = ae.encoder(X_tensor_train).detach().numpy()

rf_pred = rf.predict(z_test)

# Convert to binary for classification metrics
rf_pred_binary = (rf_pred > 0.5).astype(int)
rf_accuracy = accuracy_score(y_test, rf_pred_binary)
rf_precision = precision_score(y_test, rf_pred_binary)
rf_recall = recall_score(y_test, rf_pred_binary)
rf_f1 = f1_score(y_test, rf_pred_binary)
rf_auc = roc_auc_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"   Accuracy: {rf_accuracy:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall: {rf_recall:.4f}")
print(f"   F1-Score: {rf_f1:.4f}")
print(f"   AUC-ROC: {rf_auc:.4f}")
print(f"   MSE: {rf_mse:.6f}")
print(f"   R²: {rf_r2:.4f}")

# Evaluate Meta Model (Ensemble)
print("\n🤖 META MODEL (ENSEMBLE) EVALUATION")

# Get predictions for meta model
gam_pred_all = gam.predict_proba(X_tab)
if gam_pred_all.ndim == 2:
    gam_pred_all = gam_pred_all[:, 1]
rf_pred_all = rf.predict(ae.encoder(torch.tensor(X_ecg, dtype=torch.float32)).detach().numpy())

meta_X = pd.DataFrame({"gam": gam_pred_all, "rf": rf_pred_all})
try:
    meta_pred_proba = meta.predict_proba(meta_X)
    if meta_pred_proba.ndim == 2:
        meta_pred_proba = meta_pred_proba[:, 1]
    else:
        meta_pred_proba = meta_pred_proba
except AttributeError:
    # Handle case where predict_proba might not be available
    meta_pred_proba = meta.predict(meta_X)
meta_pred_binary = (meta_pred_proba > 0.5).astype(int)

meta_accuracy = accuracy_score(y, meta_pred_binary)
meta_precision = precision_score(y, meta_pred_binary)
meta_recall = recall_score(y, meta_pred_binary)
meta_f1 = f1_score(y, meta_pred_binary)
meta_auc = roc_auc_score(y, meta_pred_proba)
meta_mse = mean_squared_error(y, meta_pred_proba)
meta_r2 = r2_score(y, meta_pred_proba)

print(f"   Accuracy: {meta_accuracy:.4f}")
print(f"   Precision: {meta_precision:.4f}")
print(f"   Recall: {meta_recall:.4f}")
print(f"   F1-Score: {meta_f1:.4f}")
print(f"   AUC-ROC: {meta_auc:.4f}")
print(f"   MSE: {meta_mse:.6f}")
print(f"   R²: {meta_r2:.4f}")

# Cross-validation style evaluation on test set
print("\n🔄 CROSS-VALIDATION STYLE EVALUATION ON TEST SET")

# Get test set predictions for ensemble
gam_pred_test = gam.predict_proba(X_tab_test)
if gam_pred_test.ndim == 2:
    gam_pred_test = gam_pred_test[:, 1]
rf_pred_test = rf.predict(z_test)

meta_X_test = pd.DataFrame({"gam": gam_pred_test, "rf": rf_pred_test})
try:
    meta_pred_test_proba = meta.predict_proba(meta_X_test)
    if meta_pred_test_proba.ndim == 2:
        meta_pred_test = meta_pred_test_proba[:, 1]
    else:
        meta_pred_test = meta_pred_test_proba
except AttributeError:
    meta_pred_test = meta.predict(meta_X_test)
meta_pred_test_binary = (meta_pred_test > 0.5).astype(int)

ensemble_accuracy = accuracy_score(y_test, meta_pred_test_binary)
ensemble_precision = precision_score(y_test, meta_pred_test_binary)
ensemble_recall = recall_score(y_test, meta_pred_test_binary)
ensemble_f1 = f1_score(y_test, meta_pred_test_binary)
ensemble_auc = roc_auc_score(y_test, meta_pred_test)
ensemble_mse = mean_squared_error(y_test, meta_pred_test)
ensemble_r2 = r2_score(y_test, meta_pred_test)

print(f"   Accuracy: {ensemble_accuracy:.4f}")
print(f"   Precision: {ensemble_precision:.4f}")
print(f"   Recall: {ensemble_recall:.4f}")
print(f"   F1-Score: {ensemble_f1:.4f}")
print(f"   AUC-ROC: {ensemble_auc:.4f}")
print(f"   MSE: {ensemble_mse:.6f}")
print(f"   R²: {ensemble_r2:.4f}")

# Individual prediction test
print("\n🩺 INDIVIDUAL PREDICTION TEST")

test_cases = [
    {"name": "Young Healthy", "age": 25, "bp": 110, "chol": 180, "hr": 65},
    {"name": "Middle-aged Normal", "age": 45, "bp": 125, "chol": 220, "hr": 72},
    {"name": "High Risk Elderly", "age": 75, "bp": 160, "chol": 280, "hr": 85},
    {"name": "Very High Risk", "age": 80, "bp": 180, "chol": 320, "hr": 95}
]

for case in test_cases:
    # Generate synthetic ECG
    _, ecg = generate_12_lead_ecg(hr=case["hr"]/60)
    features = {}
    for lead, sig in ecg.items():
        features[f"{lead}_mean"] = np.mean(sig)
        features[f"{lead}_std"] = np.std(sig)

    X_ecg_single = np.array(list(features.values())).reshape(1, -1)
    X_tab_single = np.array([[case["age"], case["bp"], case["chol"], case["hr"]]])

    # GAM prediction
    gam_pred = gam.predict_proba(X_tab_single)
    if gam_pred.ndim == 2:
        gam_pred = gam_pred[0, 1]
    else:
        gam_pred = gam_pred[0]

    # Autoencoder encoding
    X_tensor_single = torch.tensor(X_ecg_single, dtype=torch.float32)
    with torch.no_grad():
        Z_single = ae.encoder(X_tensor_single).detach().numpy()

    # RF prediction
    rf_pred = rf.predict(Z_single)[0]

    # Meta model
    meta_X_single = pd.DataFrame({"gam": [gam_pred], "rf": [rf_pred]})
    try:
        meta_pred_single = meta.predict_proba(meta_X_single)
        if meta_pred_single.ndim == 2:
            final_risk = meta_pred_single[0, 1]
        else:
            final_risk = meta_pred_single[0]
    except AttributeError:
        final_risk = meta.predict(meta_X_single)[0]

    risk_category = "Low Risk" if final_risk < 0.3 else "Moderate Risk" if final_risk < 0.7 else "High Risk"

    print(f"{case['name']:20} | Risk: {final_risk:.3f} | Category: {risk_category}")

print("\n" + "="*80)
print("🎉 MODEL TRAINING AND EVALUATION COMPLETE")
print("="*80)
