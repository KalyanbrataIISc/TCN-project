#!/usr/bin/env python3
# ann/visualize_ann.py — visualize ANN as a graph of neurons & weighted edges

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("results") / "trained_ann.h5"
OUTPUT_PATH = Path("results") / "plots" / "ann_network_graph.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Load model & extract weights ───────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# Grab only Dense layers with weights
Ws = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) == 2]

# Compute layer sizes
layer_sizes = [W.shape[0] for W in Ws]       # number of pre‐syn neurons
layer_sizes.append(Ws[-1].shape[1])          # add final output layer size

# Flatten all weights to find max absolute value (for linewidth scaling)
all_w = np.concatenate([W.flatten() for W in Ws])
max_w = np.max(np.abs(all_w)) or 1.0

# ─── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
n_layers = len(layer_sizes)
x_spacing = 1.0 / (n_layers - 1)
max_lw = 5.0  # maximum line width

# Draw synapses (edges)
for layer_idx, W in enumerate(Ws):
    pre_n, post_n = W.shape
    x0, x1 = layer_idx * x_spacing, (layer_idx + 1) * x_spacing
    y0s = np.linspace(0, 1, pre_n)
    y1s = np.linspace(0, 1, post_n)
    for i in range(pre_n):
        for j in range(post_n):
            w = W[i, j]
            lw = (abs(w) / max_w) * max_lw
            color = 'blue' if w >= 0 else 'red'
            ax.plot([x0, x1], [y0s[i], y1s[j]],
                    color=color, linewidth=lw, alpha=0.6)

# Draw neurons (nodes)
for layer_idx, n in enumerate(layer_sizes):
    x = layer_idx * x_spacing
    ys = np.linspace(0, 1, n)
    ax.scatter([x]*n, ys, s=100,
               facecolors='white', edgecolors='black', zorder=3)

# Final touches
ax.set_title("ANN network graph (blue=+ weights, red=– weights)")
ax.axis('off')

# Save
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✅ Saved ANN network graph → {OUTPUT_PATH}")