from pathlib import Path

# ─── data params ─────────────────
IMG_SIZE    = 20
N_PER_CLASS = 200
TEST_FRAC   = 0.25

# ─── ANN params ──────────────────
HIDDEN_SIZE = 32
LR          = 1e-3
EPOCHS      = 30
BATCH_SIZE  = 16

# ─── SNN params ──────────────────
SIM_TIME_MS = 500.0    # ms
MAX_RATE_HZ = 300.0    # Hz
BIAS_SCALE  = 0.3

# ─── paths & outputs ─────────────
ROOT       = Path(__file__).parent
WEIGHT_DIR = ROOT / "weights" / "saved_ann_weights"
OUT_FIG    = "snn_comparison.png"