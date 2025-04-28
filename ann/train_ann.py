#!/usr/bin/env python3
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from ann.model import build_model
from data.generate_images import load_test_flat
from config import IMG_SIZE, N_PER_CLASS, TEST_FRAC, HIDDEN_SIZE, LR, EPOCHS, BATCH_SIZE, ROOT

RESULT_DIR = ROOT / "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def main():
    print("► Loading data…")
    # reuse load_test_flat for training & test split
    X_test, Y_test, _ = load_test_flat(IMG_SIZE, N_PER_CLASS, TEST_FRAC, seed=1)
    X_train, Y_train, _ = load_test_flat(IMG_SIZE, N_PER_CLASS, TEST_FRAC, seed=2)

    print("► Building ANN…")
    model = build_model(IMG_SIZE*IMG_SIZE, HIDDEN_SIZE)
    model.compile(optimizer=Adam(LR), loss="mse")

    print("► Training ANN…")
    history = model.fit(
        np.concatenate([X_train, X_test]),
        np.concatenate([Y_train, Y_test]),
        validation_data=(X_test, Y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2
    )

    # save model
    path = RESULT_DIR / "trained_ann.h5"
    model.save(path)
    print("✅ Saved ANN model to", path)

    # plot loss curve
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend()
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    loss_fig = RESULT_DIR / "plots" / "ann_loss.png"
    plt.savefig(loss_fig, dpi=300)
    print("✅ Saved loss plot to", loss_fig)

if __name__=="__main__":
    main()