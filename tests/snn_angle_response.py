#!/usr/bin/env python3
# angle_response_snn.py — plot SNN firing‐rates vs. line angle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
    mV, Hz, ms
)

# ─── hyperparameters ────────────────────────────────────────────────
IMG_SIZE    = 20                     # image is 20×20 px
MODEL_PATH  = "results/trained_ann.h5"
ANGLES      = np.linspace(0,180,37)  # 0°,5°,10°, …,180°
SIM_TIME_MS = 500.                   # ms per image
MAX_RATE_HZ = 300.                   # peak Poisson
BIAS_SCALE  = 0.3                    # bias→mV
OUT_FIG     = "snn_angle_response.png"

# ─── helper to draw a single‐line image ──────────────────────────────
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

# ─── LIF equations ───────────────────────────────────────────────────
def lif_eqs():
    return """
    dv/dt    = (-v + I_syn + I_bias) / (20*ms) : volt
    dI_syn/dt = -I_syn/(20*ms)                : volt
    I_bias : volt (constant)
    """

def main():
    # 1) load ANN → extract weights & biases
    ann = load_model(MODEL_PATH, compile=False)
    Ws = [lyr.get_weights()[0].astype(np.float32)
          for lyr in ann.layers if len(lyr.get_weights())==2]
    bs = [lyr.get_weights()[1].astype(np.float32)
          for lyr in ann.layers if len(lyr.get_weights())==2]

    # 2) prepare angles→images
    imgs = np.stack([make_line(a) for a in ANGLES])
    X = imgs.reshape(-1, IMG_SIZE*IMG_SIZE).astype(np.float32)/255.0

    # 3) run SNN on each image
    rates_out = []
    for flat in X:
        # encode pixels as Poisson rates
        rates = (1.0 - flat)*MAX_RATE_HZ

        # create 4 groups: h1,h2,h3,out
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
            b0 = (b * BIAS_SCALE)
            b0 = np.pad(b0, (0, max(0, len(G)-len(b0))), 'constant')[:len(G)]
            G.I_bias = b0 * mV

        net = Network(*groups)

        # helper to connect full‐matrix
        def connect(pre, post, Wmat):
            S = Synapses(pre, post, 'w:1', on_pre='I_syn_post += w*mV')
            S.connect()             # all‐to‐all
            S.w = Wmat.flatten()
            net.add(S)

        # hidden‐to‐hidden & hidden‐to‐out
        connect(h1,h2, Ws[1])
        connect(h2,h3, Ws[2])
        connect(h3,out, Ws[3])

        # Poisson input → h1 via W0
        Pg = PoissonGroup(IMG_SIZE*IMG_SIZE, rates=rates*Hz)
        S0 = Synapses(Pg, h1, 'w:1', on_pre='I_syn_post += w*mV')
        S0.connect()
        S0.w = Ws[0].flatten()
        net.add(Pg, S0)

        # record output spikes
        sm = SpikeMonitor(out)
        net.add(sm)

        # run
        net.run(SIM_TIME_MS*ms, report=None)

        dur = SIM_TIME_MS/1000.0
        # neuron 0 → vertical, neuron 1 → horizontal
        rates_out.append([sm.count[0]/dur, sm.count[1]/dur])

    rates_out = np.array(rates_out, dtype=np.float32)

    # 4) plot
    plt.figure(figsize=(6,4))
    plt.plot(ANGLES, rates_out[:,1],
             marker='o', linestyle='None', label="vertical neuron")
    plt.plot(ANGLES, rates_out[:,0],
             marker='s', linestyle='None', label="horizontal neuron")
    plt.xlabel("Line angle (°)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("SNN response vs. line angle")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    print(f"✅ Saved → {OUT_FIG}")
    plt.show()

if __name__=="__main__":
    main()