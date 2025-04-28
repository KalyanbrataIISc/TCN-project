#!/usr/bin/env python3
# classify_snn_full.py — load ANN & SNN, sweep angles, train H/V classifiers

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import load_model
from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
    mV, Hz, ms
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE    = 20        # 20×20px
SIM_TIME_MS = 500.      # SNN sim per image
MAX_RATE_HZ = 300.      # peak Poisson
BIAS_SCALE  = 0.3       # bias→mV
ANGLE_STEP  = 5         # sweep every 5°

# paths
ANN_MODEL_PATH = "results/trained_ann.h5"
WEIGHT_DIR     = "weights/saved_ann_weights"

# ─── 1) UTILITIES ────────────────────────────────────────────────────────────────
def make_line(angle_deg):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    draw = ImageDraw.Draw(img)
    cx = cy = IMG_SIZE/2
    L = IMG_SIZE * 1.2
    th = np.deg2rad(angle_deg)
    dx, dy = np.cos(th)*(L/2), np.sin(th)*(L/2)
    draw.line([(cx-dx, cy-dy), (cx+dx, cy+dy)],
              fill=0, width=3)
    return np.array(img, dtype=np.uint8)

def lif_eqs():
    return """
    dv/dt     = (-v + I_syn + I_bias)/ (20*ms) : volt
    dI_syn/dt = -I_syn/(20*ms)                : volt
    I_bias    : volt (constant)
    """

# ─── 2) LOAD ANN & WEIGHTS ──────────────────────────────────────────────────────
print("⏳ Loading ANN model…")
ann = load_model(ANN_MODEL_PATH, compile=False)
# extract weights & biases
Ws, bs = [], []
i = 0
while True:
    w_path = os.path.join(WEIGHT_DIR, f"W{i}.npy")
    b_path = os.path.join(WEIGHT_DIR, f"b{i}.npy")
    if not os.path.isfile(w_path):
        break
    Ws.append(np.load(w_path))
    bs.append(np.load(b_path))
    i += 1
print(f"✅ Loaded ANN + {len(Ws)} layers of W/b.")

# ─── 3) RUN SNN ON ONE FLATTENED IMAGE ───────────────────────────────────────────
def run_snn(flat):
    rates = (1.0 - flat) * MAX_RATE_HZ
    # build 4 layers
    sizes = [W.shape[1] for W in Ws]
    groups = []
    for n in sizes:
        G = NeuronGroup(n, lif_eqs(),
                        threshold="v>0.3*mV",
                        reset="v=0*mV",
                        refractory="5*ms",
                        method="euler")
        G.v = 0*mV
        groups.append(G)
    h1,h2,h3,out = groups

    # inject biases
    for G,b in zip(groups, bs):
        b0 = np.pad(b * BIAS_SCALE,
                    (0, max(0,len(G)-len(b))),
                    'constant')[:len(G)]
        G.I_bias = b0 * mV

    net = Network(*groups)

    def connect(pre, post, W):
        S = Synapses(pre, post, 'w:1', on_pre='I_syn_post += w*mV')
        S.connect()
        S.w = W.flatten()
        net.add(S)

    # hidden connections
    connect(h1,h2, Ws[1])
    connect(h2,h3, Ws[2])
    connect(h3,out, Ws[3])
    # input Poisson → h1
    Pg = PoissonGroup(IMG_SIZE*IMG_SIZE, rates=rates*Hz)
    S0 = Synapses(Pg, h1, 'w:1', on_pre='I_syn_post += w*mV')
    S0.connect(); S0.w = Ws[0].flatten()
    net.add(Pg, S0)

    sm = SpikeMonitor(out)
    net.add(sm)
    net.run(SIM_TIME_MS*ms, report=None)

    dur = SIM_TIME_MS/1000.0
    return np.array([sm.count[0]/dur, sm.count[1]/dur], dtype=np.float32)

# ─── 4) BUILD DATASET OVER ANGLES ────────────────────────────────────────────────
angles = np.arange(0, 180, ANGLE_STEP)
X_ann, X_snn, Y = [], [], []
print("⏳ Generating data & collecting ANN/SNN responses…")
for a in angles:
    img = make_line(a)
    flat = img.reshape(-1).astype(np.float32)/255.0
    # ANN output
    out_ann = ann.predict(flat[None,:], verbose=0)[0]
    X_ann.append(out_ann)
    # SNN rates
    X_snn.append(run_snn(flat))
    # label: vertical if within [45°,135°)
    Y.append(1 if (45 <= a < 135) else 0)

X_ann = np.vstack(X_ann)
X_snn = np.vstack(X_snn)
Y     = np.array(Y)

# ─── 5) TRAIN/TEST SPLIT & CLASSIFIERS ──────────────────────────────────────────
X1_train, X1_test, y_train, y_test = train_test_split(
    X_ann, Y, test_size=0.3, random_state=0, stratify=Y)
X2_train, X2_test, _, _       = train_test_split(
    X_snn,  Y, test_size=0.3, random_state=0, stratify=Y)

clf_ann = LogisticRegression().fit(X1_train, y_train)
clf_snn = LogisticRegression().fit(X2_train, y_train)

y1p = clf_ann.predict(X1_test)
y2p = clf_snn.predict(X2_test)

acc_ann = accuracy_score(y_test, y1p)*100
acc_snn = accuracy_score(y_test, y2p)*100

print(f"ANN‐based classifier   accuracy: {acc_ann:.1f}%")
print(f"SNN‐based classifier   accuracy: {acc_snn:.1f}%")

# ─── 6) PLOT RESULTS ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(10,4))

axes[0].scatter(X1_test[:,0]-X1_test[:,1], X1_test[:,1]-X1_test[:,0],
                c=y_test, cmap='coolwarm', s=80, edgecolor='k')
axes[0].set_title(f"ANN features (test) — acc {acc_ann:.1f}%")
axes[0].set_xlabel("vert−horiz")
axes[0].set_ylabel("horiz−vert")

axes[1].scatter(X2_test[:,0]-X2_test[:,1], X2_test[:,1]-X2_test[:,0],
                c=y_test, cmap='coolwarm', s=80, edgecolor='k')
axes[1].set_title(f"SNN rates (test) — acc {acc_snn:.1f}%")
axes[1].set_xlabel("vert_rate−horiz_rate")
axes[1].set_ylabel("horiz_rate−vert_rate")

plt.tight_layout()
plt.show()