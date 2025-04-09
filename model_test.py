import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Define high-resolution model
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

# Load trained weights
model = HighResNN()
model.load_state_dict(torch.load("grating_model_weights_highres.pth"))
model.eval()

# Transform for 100x100 grayscale images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Inference on angled test images
def run_angled_inference(image_dir="testing_images"):
    for filename in sorted(os.listdir(image_dir)):
        if (filename.startswith("horizontal") or filename.startswith("vertical")) and filename.endswith(".png"):
            path = os.path.join(image_dir, filename)
            img = Image.open(path)
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                prob_vals = probs.numpy()[0]
                prob_str = [f"{p:.4f}" for p in prob_vals]
                pred_class = "Horizontal" if prob_vals[0] > prob_vals[1] else "Vertical"
                print(f"{filename} -> Probabilities: [{prob_str[0]}, {prob_str[1]}] -> Predicted: {pred_class}")

run_angled_inference()