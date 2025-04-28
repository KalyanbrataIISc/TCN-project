#!/usr/bin/env python3
# ann_angle_response.py — plot ANN outputs vs. line angle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from config import PLOTS

# ─── hyperparameters ───────────────────────────────────────────────
IMG_SIZE   = 20                 # image is 20×20 px
MODEL_PATH = "results/trained_ann.h5"  # path to your saved ANN
OUT_PLOT   = PLOTS / "ann_angle_response.png"

# angles at which to test (0° = horizontal, 90° = vertical)
ANGLES = np.linspace(0, 180, 37)  # every 5° from 0 to 180

# ─── helper to draw a single line image ────────────────────────────
def make_line(angle_deg):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    draw = ImageDraw.Draw(img)
    cx = cy = IMG_SIZE / 2
    L = IMG_SIZE * 1.2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta) * (L/2), np.sin(theta) * (L/2)
    draw.line(
        [(cx - dx, cy - dy), (cx + dx, cy + dy)],
        fill=0, width=3
    )
    return np.array(img, dtype=np.uint8)

def main():
    # load model
    model = load_model(MODEL_PATH, compile=False)

    # generate test set
    imgs = np.stack([make_line(a) for a in ANGLES])
    X = imgs.reshape(-1, IMG_SIZE * IMG_SIZE).astype(np.float32) / 255.0

    # run ANN
    preds = model.predict(X, verbose=0)  # shape (len(ANGLES), 2)

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(ANGLES, preds[:,0], '-o', label="Vertical output (neuron 0)")
    plt.plot(ANGLES, preds[:,1], '-s', label="Horizontal output (neuron 1)")
    plt.xlabel("Line angle (°)")
    plt.ylabel("ANN output")
    plt.title("ANN response vs. line angle")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    print(f"✅ Saved → {OUT_PLOT}")
    plt.show()

if __name__ == "__main__":
    main()