#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from config import WEIGHT_DIR, ROOT

# load trained model
model_path = ROOT / "results" / "trained_ann.h5"
model = load_model(model_path, compile=False)

# ensure target dir
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

# extract & dump
Ws, bs = [], []
for layer in model.layers:
    w_b = layer.get_weights()
    if len(w_b) != 2:
        continue
    w, b = w_b
    w = w.astype(np.float32); b = b.astype(np.float32)
    idx = len(Ws)
    np.save(WEIGHT_DIR / f"W{idx}.npy", w)
    np.save(WEIGHT_DIR / f"b{idx}.npy", b)
    Ws.append(w); bs.append(b)

print(f"✅ Dumped {len(Ws)} weight/bias pairs into {WEIGHT_DIR}")