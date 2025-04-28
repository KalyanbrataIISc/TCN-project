#!/usr/bin/env python3
import numpy as np
from config import WEIGHT_DIR

def build_snn():
    """
    Load all W{i}.npy/b{i}.npy in WEIGHT_DIR and return Ws, bs lists.
    """
    Ws, bs = [], []
    i = 0
    while True:
        wfile = WEIGHT_DIR / f"W{i}.npy"
        bfile = WEIGHT_DIR / f"b{i}.npy"
        if not wfile.exists() or not bfile.exists():
            break
        Ws.append(np.load(wfile))
        bs.append(np.load(bfile))
        i += 1
    if not Ws:
        raise FileNotFoundError(f"No weights found in {WEIGHT_DIR}")
    return Ws, bs

if __name__=="__main__":
    Ws, bs = build_snn()
    print(f"Loaded {len(Ws)} layers from {WEIGHT_DIR}")