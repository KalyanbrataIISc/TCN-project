# TCN-project: ANN-to-SNN Conversion for Line Orientation Classification

This project demonstrates the conversion of a trained Artificial Neural Network (ANN) to a Spiking Neural Network (SNN) for classifying simple line images (horizontal vs. vertical) and analyzing their responses. It includes code for data generation, ANN training, weight export, SNN simulation, response analysis, and robustness to background noise.

## Project Structure

- `ann/`
  - `model.py` — Defines the ANN architecture.
  - `train_ann.py` — Trains the ANN on generated line images.
  - `save_weights.py` — Exports trained ANN weights to `.npy` files for SNN use.
  - `visualize_ann.py` — Visualizes the ANN as a network graph.
- `snn/`
  - `build_snn.py` — Loads exported ANN weights for SNN construction.
  - `simulate_snn.py` — Runs the SNN simulation using Brian2 and exported weights.
- `data/`
  - `generate_images.py` — Functions for generating synthetic line images and loading test/train splits.
- `tests/`
  - `ann_angle_response.py` — Plots ANN output vs. line angle.
  - `snn_angle_response.py` — Plots SNN firing rates vs. line angle.
  - `snn_withNoise_response.py` — Plots SNN firing rates vs. line angle with background noise (robustness analysis).
  - `classify.py` — Compares ANN and SNN classification performance using logistic regression.
- `trial_attempts/` — Standalone scripts for end-to-end experiments and prototyping.
- `results/` — Saved models and plots.
- `weights/`
  - `saved_ann_weights/` — Exported ANN weights for SNN (as `.npy` files).
  - `saved_ann_model.h5` — (optional) Saved Keras model.
- `config.py` — Central configuration for hyperparameters and paths.

## Pipeline Overview

1. **Data Generation**: Synthetic 20×20 px line images at various angles are generated using `data/generate_images.py`.
2. **ANN Training**: The ANN is trained on these images using `ann/train_ann.py` (Keras, MLP with 3 hidden layers).
3. **Weight Export**: Trained weights are exported to `.npy` files using `ann/save_weights.py` for SNN use.
4. **SNN Simulation**: The SNN, mimicking the ANN architecture with LIF neurons (Brian2), is constructed and simulated using `snn/simulate_snn.py` and `snn/build_snn.py`.
5. **Analysis & Visualization**: Various scripts in `tests/` analyze and visualize the responses of both ANN and SNN, including robustness to noise and classification accuracy.

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
   python tests/snn_withNoise_response.py
   python tests/classify.py
   ```

## Configuration

All key hyperparameters and paths are set in `config.py` for easy modification and reproducibility.

## Notes

- All images are synthetic 20×20 px lines at various angles.
- The ANN is a simple 3-layer MLP; the SNN mimics its structure using LIF neurons in Brian2.
- The project includes analysis of SNN robustness to background noise and direct comparison of ANN/SNN outputs.
- Plots and results are saved in the `results/plots/` directory.
- Modular design: You can swap out data generation, model, or SNN simulation components as needed.

## Citation

If you use this code, please cite the original repository or contact the author.

---
