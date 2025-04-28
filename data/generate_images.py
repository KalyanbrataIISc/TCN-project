import numpy as np
from PIL import Image, ImageDraw

def make_line(angle_deg, img_size):
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    cx = cy = img_size / 2
    L = img_size * 1.2
    θ = np.deg2rad(angle_deg)
    dx, dy = np.cos(θ)*(L/2), np.sin(θ)*(L/2)
    draw.line([(cx-dx, cy-dy), (cx+dx, cy+dy)], fill=0, width=3)
    return np.array(img, dtype=np.uint8)

def load_test_flat(img_size=20, n_per_class=200, test_frac=0.25, seed=None):
    """
    Returns X_test (N×(img_size²)), Y_test (N×2), angles_test (N,)
    matching exactly the split in your test.py.
    """
    if seed is not None:
        np.random.seed(seed)
    angles = np.array([0]*n_per_class + [90]*n_per_class)
    np.random.shuffle(angles)
    imgs = np.stack([make_line(a, img_size) for a in angles])
    X = imgs.reshape(-1, img_size*img_size).astype(np.float32) / 255.0
    Y = np.stack([[1,0] if a==90 else [0,1] for a in angles]).astype(np.float32)
    n_test = int(len(X) * test_frac)
    return X[:n_test], Y[:n_test], angles[:n_test]