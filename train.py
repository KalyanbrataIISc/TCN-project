import numpy as np
from PIL import Image
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob

# --- Angled grating generator ---
def generate_angled_grating(size=100, cycles=10, angle_deg=45):
    grating = np.zeros((size, size), dtype=int)
    radians = math.radians(angle_deg)
    freq = cycles / size
    for i in range(size):
        for j in range(size):
            x = j * math.cos(radians) + i * math.sin(radians)
            value = 0 if math.sin(2 * math.pi * freq * x) > 0 else 255
            grating[i, j] = value
    return grating

# --- Updated high-res dataset generation ---
def save_highres_dataset(n_images=1000, size=100, base_folder='grating_images'):
    angles_horizontal = [x for x in range(45, 135)]     # includes horizontal and near-horizontal
    angles_vertical = [x for x in range(-45, 45)]       # includes vertical and near-vertical

    for label, angles in zip(['horizontal', 'vertical'], [angles_horizontal, angles_vertical]):
        folder = os.path.join(base_folder, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_images):
            angle = np.random.choice(angles)
            cycles = np.random.choice([5, 10, 20])
            img_array = generate_angled_grating(size=size, cycles=cycles, angle_deg=angle)
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(os.path.join(folder, f"img_{i}.png"))

# --- Dataset Class ---
class GratingDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, folder in enumerate(['horizontal', 'vertical']):
            for path in glob(os.path.join(root_dir, folder, '*.png')):
                self.samples.append((path, label))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        img = self.transform(img)
        return img, torch.tensor(label)

# --- High-resolution model ---
class HighResNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Training function ---
def train_highres_model():
    dataset = GratingDataset('grating_images')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = HighResNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'grating_model_weights_highres.pth')
    print("Model saved to grating_model_weights_highres.pth")

# --- Run both parts ---
save_highres_dataset()
train_highres_model()