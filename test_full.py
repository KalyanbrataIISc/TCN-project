#!/usr/bin/env python3
# test_full.py — end-to-end ANN → SNN on H/V lines

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models
from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
    mV, Hz, ms
)

# ─── hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE    = 20       # 20×20 px images
N_PER_CLASS = 200      # horizontals + verticals
TEST_FRAC   = 0.25     # 25% test split
HIDDEN_SIZE = 32
LR          = 1e-3
EPOCHS      = 30
BATCH_SIZE  = 16

SIM_TIME_MS = 500.     # SNN sim per image
MAX_RATE_HZ = 300.     # peak Poisson rate
BIAS_SCALE  = 0.3      # ANN bias → mV
OUT_FIG     = "snn_comparison.png"

# ─── 1) DATA GENERATION ─────────────────────────────────────────────────────────
def make_line(angle_deg):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    draw = ImageDraw.Draw(img)
    cx = cy = IMG_SIZE/2
    L = IMG_SIZE * 1.2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta)*(L/2), np.sin(theta)*(L/2)
    draw.line([(cx-dx, cy-dy), (cx+dx, cy+dy)],
              fill=0, width=3)
    return np.array(img, dtype=np.uint8)

# build and shuffle angles
angles = np.array([0]*N_PER_CLASS + [90]*N_PER_CLASS)
np.random.shuffle(angles)

# render images & assemble X,Y
imgs = np.stack([make_line(a) for a in angles])
X = imgs.reshape(-1, IMG_SIZE*IMG_SIZE).astype(np.float32)/255.0
Y = np.stack([[1,0] if a==90 else [0,1] for a in angles]).astype(np.float32)

# train/test split
n_test = int(len(X)*TEST_FRAC)
X_test, Y_test, angles_test = X[:n_test], Y[:n_test], angles[:n_test]
X_train, Y_train               = X[n_test:], Y[n_test:]

# ─── 2) BUILD & TRAIN ANN ──────────────────────────────────────────────────────
ann = models.Sequential([
    layers.Input((IMG_SIZE*IMG_SIZE,)),
    layers.Dense(HIDDEN_SIZE, activation="relu"),
    layers.Dense(HIDDEN_SIZE, activation="relu"),
    layers.Dense(HIDDEN_SIZE, activation="relu"),
    layers.Dense(2, activation="linear")
])
ann.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
print("► Training ANN…")
ann.fit(X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        verbose=2)

# extract weights & biases from each dense layer
Ws = []
bs = []
for layer in ann.layers:
    w_b = layer.get_weights()
    if len(w_b)==2:
        Ws.append(w_b[0].astype(np.float32))
        bs.append(w_b[1].astype(np.float32))

# ANN predictions on test
ann_out = ann.predict(X_test, verbose=0)

# ─── 3) BUILD & RUN SNN ─────────────────────────────────────────────────────────
def lif_eqs():
    return """
    dv/dt    = (-v + I_syn + I_bias) / (20*ms) : volt
    dI_syn/dt = -I_syn/(20*ms)                : volt
    I_bias : volt (constant)
    """

snn_rates = []
for idx in range(len(X_test)):
    flat = X_test[idx]        # [0..1]
    rates = (1.0 - flat) * MAX_RATE_HZ  # Hz

    # create neuron groups matching ANN layers
    sizes = [w.shape[1] for w in Ws]  # hidden1, hidden2, hidden3, output
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

    # inject constant biases, scaled from ANN
    for G, b in zip(groups, bs):
        b0 = (b * BIAS_SCALE)
        b0 = np.pad(b0,
                    (0, max(0, len(G)-len(b0))),
                    'constant')[:len(G)]
        G.I_bias = b0 * mV

    net = Network(*groups)

    # helper: fully-connect pre→post with weight matrix Wmat
    def connect(pre, post, Wmat):
        S = Synapses(pre, post, 'w:1', on_pre='I_syn_post += w*mV')
        S.connect()
        S.w = Wmat.flatten()
        net.add(S)

    connect(h1,h2, Ws[1])
    connect(h2,h3, Ws[2])
    connect(h3,out, Ws[3])

    # Poisson input → h1 via Ws[0]
    Pg = PoissonGroup(IMG_SIZE*IMG_SIZE, rates=rates*Hz)
    S0 = Synapses(Pg, h1, 'w:1', on_pre='I_syn_post += w*mV')
    S0.connect()
    S0.w = Ws[0].flatten()
    net.add(Pg, S0)

    sm = SpikeMonitor(out)
    net.add(sm)

    net.run(SIM_TIME_MS * ms, report=None)

    dur = SIM_TIME_MS/1000.0
    snn_rates.append([sm.count[0]/dur, sm.count[1]/dur])

snn_rates = np.array(snn_rates, dtype=np.float32)

# ─── 4) PLOT COMPARISON ─────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))

# ANN scatter (verticalness vs. horizontalness)
plt.subplot(1,2,1)
c = angles_test/90.   # 0=horiz, 1=vert
plt.scatter(ann_out[:,0], ann_out[:,1],
            c=c, cmap='coolwarm', s=50, edgecolor='k')
plt.title("ANN outputs (test)")
plt.xlabel("verticalness")
plt.ylabel("horizontalness")
cb = plt.colorbar()
cb.set_label("class (0=H,1=V)")

# SNN scatter (vertical rate vs. horizontal rate)
plt.subplot(1,2,2)
plt.scatter(snn_rates[:,0], snn_rates[:,1],
            c=c, cmap='coolwarm', s=50, edgecolor='k')
plt.title("SNN firing rates (test)")
plt.xlabel("vertical rate (Hz)")
plt.ylabel("horizontal rate (Hz)")
cb2 = plt.colorbar()
cb2.set_label("class (0=H,1=V)")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print("✅ Saved →", OUT_FIG)
plt.show()