import numpy as np
import pandas as pd
from src.ecg.synthetic_ecg import generate_12_lead_ecg
from src.ecg.synthetic_ecg import generate_12_lead_ecg



np.random.seed(42)
records = []

for pid in range(1000):
    age = np.random.randint(25, 80)
    bp = np.random.randint(100, 190)
    chol = np.random.randint(150, 340)
    hr = np.random.randint(60, 120)

    risk = int((age > 55 and bp > 140) or chol > 260)

    _, ecg = generate_12_lead_ecg(hr=hr/60)

    features = {}
    for lead, sig in ecg.items():
        features[f"{lead}_mean"] = np.mean(sig)
        features[f"{lead}_std"] = np.std(sig)

    row = {
        "patient_id": pid,
        "age": age,
        "bp": bp,
        "cholesterol": chol,
        "heart_rate": hr,
        "risk": risk
    }
    row.update(features)
    records.append(row)

df = pd.DataFrame(records)
df.to_csv("data/patients_ecg.csv", index=False)

print("✅ 1000 synthetic ECG patients generated")
