import numpy as np
from PIL import Image
import os
import math

def generate_angled_grating(size=100, cycles=10, angle_deg=45):
    grating = np.zeros((size, size), dtype=int)
    radians = math.radians(angle_deg)
    freq = cycles / size

    for i in range(size):
        for j in range(size):
            x = j * math.cos(radians) + i * math.sin(radians)
            value = 0 if (math.sin(2 * math.pi * freq * x) > 0) else 255
            grating[i, j] = value

    return grating

def save_angled_gratings():
    os.makedirs("testing_images", exist_ok=True)
    angles = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0,
              5, 10, 15, 20, 25, 30, 35, 40, 45,
              50, 55, 60, 65, 70, 75, 80, 85, 90,
              95, 100, 105, 110, 115, 120, 125, 130, 135]
    for angle in angles:
        img_array = generate_angled_grating(size=100, cycles=10, angle_deg=angle)
        img = Image.fromarray(img_array.astype(np.uint8))
        ori = None
        if angle > 45 and angle <= 135:
            ori = "horizontal"
        else:
            ori = "vertical"
        img.save(f"testing_images/{ori}_{angle}.png")

save_angled_gratings()