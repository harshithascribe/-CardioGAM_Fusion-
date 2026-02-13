import sys
sys.path.insert(0, '.')
import pandas as pd
import torch
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from pygam import LogisticGAM, s
import joblib
import pickle
from src.model.autoencoder import ECGAutoencoder



df = pd.read_csv("data/patients_ecg.csv")

ecg_cols = [c for c in df.columns if "lead" in c]
X_ecg = df[ecg_cols].values
X_tab = df[["age","bp","cholesterol","heart_rate"]]
y = df["risk"]

# ---- Autoencoder ----
ae = ECGAutoencoder(X_ecg.shape[1])
opt = optim.Adam(ae.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

X_tensor = torch.tensor(X_ecg, dtype=torch.float32)

for _ in range(100):
    opt.zero_grad()
    recon, z = ae(X_tensor)
    loss = loss_fn(recon, X_tensor)
    loss.backward()
    opt.step()

torch.save(ae.state_dict(), "models/autoencoder.pt")

Z = ae.encoder(X_tensor).detach().numpy()

# ---- GAM ----
gam = LogisticGAM(s(0)+s(1)+s(2)+s(3)).fit(X_tab, y)
gam_pred_proba = gam.predict_proba(X_tab)
if gam_pred_proba.ndim == 2:
    gam_pred = gam_pred_proba[:, 1]
else:
    gam_pred = gam_pred_proba  # 1D array case

# ---- RF on encoded ECG ----
rf = RandomForestRegressor(n_estimators=200)
rf.fit(Z, y)

rf_pred = rf.predict(Z)

# ---- Stacking Meta Model ----
meta_X = pd.DataFrame({
    "gam": gam_pred,
    "rf": rf_pred
})

meta = LogisticRegression()
meta.fit(meta_X, y)

joblib.dump(gam, "models/gam_model.pkl")
joblib.dump(rf, "models/rf_residual.pkl")
joblib.dump(meta, "models/meta_model.pkl")

print("✅ Autoencoder + GAM + RF + Meta model trained")
