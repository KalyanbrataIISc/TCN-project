#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import NeuronGroup, PoissonGroup, Synapses, SpikeMonitor, Network, mV, Hz, ms
from config import IMG_SIZE, N_PER_CLASS, TEST_FRAC, MAX_RATE_HZ, SIM_TIME_MS, BIAS_SCALE, OUT_FIG
from snn.build_snn import build_snn
from data.generate_images import load_test_flat

def lif_eqs():
    return """
    dv/dt    = (-v + I_syn + I_bias) / (20*ms) : volt
    dI_syn/dt = -I_syn/(20*ms)                : volt
    I_bias : volt (constant)
    """

def main():
    # load test data
    X_test, Y_test, angles_test = load_test_flat(IMG_SIZE, N_PER_CLASS, TEST_FRAC)
    # load ANN weights
    Ws, bs = build_snn()

    snn_rates = []
    for flat in X_test:
        rates = (1.0 - flat) * MAX_RATE_HZ

        # create and initialize layers
        sizes = [w.shape[1] for w in Ws]
        groups = []
        for n in sizes:
            G = NeuronGroup(
                n, lif_eqs(),
                threshold="v>0.3*mV",
                reset="v=0*mV",
                refractory="5*ms",
                method="euler"
            )
            G.v = 0 * mV
            groups.append(G)
        h1, h2, h3, out = groups

        # inject biases
        for G, b in zip(groups, bs):
            b0 = (b * BIAS_SCALE)
            b0 = np.pad(b0, (0, max(0, len(G)-len(b0))), 'constant')[:len(G)]
            G.I_bias = b0 * mV

        net = Network(*groups)

        # helper to connect full-matrix
        def connect(pre, post, W):
            S = Synapses(pre, post, 'w:1', on_pre='I_syn_post += w*mV')
            S.connect()        # all-to-all
            S.w = W.flatten()
            net.add(S)

        connect(h1, h2, Ws[1])
        connect(h2, h3, Ws[2])
        connect(h3, out, Ws[3])

        # Poisson input → h1
        Pg = PoissonGroup(IMG_SIZE*IMG_SIZE, rates=rates*Hz)
        S0 = Synapses(Pg, h1, 'w:1', on_pre='I_syn_post += w*mV')
        S0.connect(); S0.w = Ws[0].flatten()
        net.add(Pg, S0)

        sm = SpikeMonitor(out)
        net.add(sm)
        net.run(SIM_TIME_MS * ms, report=None)

        dur = SIM_TIME_MS/1000.0
        snn_rates.append([sm.count[0]/dur, sm.count[1]/dur])

    snn_rates = np.array(snn_rates, dtype=np.float32)

    # plot exactly as in test.py (note reversed axes)
    os.makedirs(".", exist_ok=True)
    plt.figure(figsize=(8,4))
    c = angles_test/90.0  # 0=horiz,1=vert
    # plt.subplot(1,2,2)
    plt.scatter(snn_rates[:,1], snn_rates[:,0], c=c, cmap='coolwarm', s=50)
    plt.xlabel("vertical rate (Hz)")
    plt.ylabel("horizontal rate (Hz)")
    cb = plt.colorbar()
    cb.set_label("class (0=H,1=V)")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    print("✅ Saved →", OUT_FIG)
    plt.show()

if __name__=="__main__":
    main()