import numpy as np

def ecg_wave(t, hr):
    p = 0.1 * np.sin(2 * np.pi * hr * t)
    qrs = np.exp(-((t - 0.3) ** 2) / (2 * 0.002))
    t_wave = 0.25 * np.sin(2 * np.pi * hr * (t - 0.45))
    return p + qrs + t_wave

def generate_12_lead_ecg(duration=10, fs=500, hr=1.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, duration, duration * fs)
    leads = {}
    for i in range(12):
        noise = np.random.normal(0, 0.01, len(t))
        leads[f"lead_{i+1}"] = ecg_wave(t, hr*(1+i*0.02)) + noise
    return t, leads
