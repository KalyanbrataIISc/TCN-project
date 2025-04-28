# TCN-project: ANN-to-SNN Conversion for Line Orientation Classification

This project demonstrates the conversion of a trained Artificial Neural Network (ANN) to a Spiking Neural Network (SNN) for classifying simple line images (horizontal vs. vertical) and analyzing their responses. It includes code for data generation, ANN training, weight export, SNN simulation, and response analysis.

## Project Structure

- `ann/`
  - `model.py` — Defines the ANN architecture.
  - `train_ann.py` — Trains the ANN on generated line images.
  - `save_weights.py` — Exports trained ANN weights to `.npy` files for SNN use.
- `snn/`
  - `build_snn.py` — Loads exported ANN weights for SNN construction.
  - `simulate_snn.py` — Runs the SNN simulation using Brian2.
- `data/`
  - `generate_images.py` — Functions for generating synthetic line images.
- `tests/`
  - `ann_angle_response.py` — Plots ANN output vs. line angle.
  - `snn_angle_response.py` — Plots SNN firing rates vs. line angle.
  - `classify.py` — Compares ANN and SNN classification performance.
- `trial_attempts/` — Standalone scripts for end-to-end experiments.
- `results/` — Saved models and plots.
- `weights/` — Exported ANN weights for SNN.
- `config.py` — Central configuration for hyperparameters and paths.

## Setup

1. **Install dependencies**:
   - Python 3.8+
   - `numpy`, `matplotlib`, `Pillow`, `tensorflow`, `brian2`, `scikit-learn`

2. **Train the ANN**:
   ```bash
   python ann/train_ann.py
   ```

3. **Export ANN weights**:
   ```bash
   python ann/save_weights.py
   ```

4. **Simulate the SNN**:
   ```bash
   python snn/simulate_snn.py
   ```

5. **Run analysis scripts** (optional):
   ```bash
   python tests/ann_angle_response.py
   python tests/snn_angle_response.py
   python tests/classify.py
   ```

## Notes

- All images are synthetic 20×20 px lines at various angles.
- The ANN is a simple 3-layer MLP; the SNN mimics its structure using LIF neurons in Brian2.
- Plots and results are saved in the `results/plots/` directory.

## Citation

If you use this code, please cite the original repository or contact the author.

---
