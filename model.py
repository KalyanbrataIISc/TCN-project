import numpy as np
from PIL import Image
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
import argparse

# ===== CONFIGURATION =====
# You can modify these parameters directly or use command line arguments
CONFIG = {
    # Image generation parameters
    "image_size": 20,                  # Size of images (square)
    "cycles_range": [5, 10, 20],        # Possible cycle values for gratings
    
    # Training dataset parameters
    "train_images_per_class": 500,     # Number of training images per class
    # "horizontal_angle_range": [45, 135], # Angle range for horizontal class (inclusive)
    # "vertical_angle_range": [-45, 45],  # Angle range for vertical class (inclusive)
    "horizontal_angle_range": [80, 100],
    "vertical_angle_range": [-10, 10],
    
    # Testing dataset parameters
    # "test_angles": [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0,
    #                 5, 10, 15, 20, 25, 30, 35, 40, 45,
    #                 50, 55, 60, 65, 70, 75, 80, 85, 90,
    #                 95, 100, 105, 110, 115, 120, 125, 130, 135],
    "test_angles": [-45, -40, -35, -30, -25, -20, 20, 25, 30, 35, 40, 45,
                    50, 55, 60, 65, 70, 110, 115, 120, 125, 130, 135],
    
    # Model parameters
    "hidden_layer_size": 16,            # Size of hidden layer
    
    # Training parameters
    "batch_size": 4,                    # Batch size for training
    "learning_rate": 0.001,             # Learning rate for optimizer
    "epochs": 20,                       # Number of training epochs
    
    # File paths
    "train_data_dir": "grating_images", # Directory for training images
    "test_data_dir": "testing_images",  # Directory for test images
    "model_save_path": "grating_model_weights.pth"  # Path to save model weights
}

# ===== MODEL DEFINITION =====
class GratingNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # 2 classes: horizontal and vertical

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ===== DATA GENERATION FUNCTIONS =====
def generate_angled_grating(size=100, cycles=10, angle_deg=45):
    """Generate a grating image at specified angle"""
    grating = np.zeros((size, size), dtype=int)
    radians = math.radians(angle_deg)
    freq = cycles / size

    for i in range(size):
        for j in range(size):
            x = j * math.cos(radians) + i * math.sin(radians)
            value = 0 if (math.sin(2 * math.pi * freq * x) > 0) else 255
            grating[i, j] = value

    return grating

def save_training_dataset(config=CONFIG):
    """Generate and save training dataset"""
    n_images = config["train_images_per_class"]
    size = config["image_size"]
    cycles_range = config["cycles_range"]
    base_folder = config["train_data_dir"]
    h_range = config["horizontal_angle_range"]
    v_range = config["vertical_angle_range"]
    
    # Create angle ranges
    angles_horizontal = list(range(h_range[0], h_range[1] + 1))
    angles_vertical = list(range(v_range[0], v_range[1] + 1))
    
    print(f"Generating training dataset: {n_images} images per class...")
    
    for label, angles in zip(['horizontal', 'vertical'], [angles_horizontal, angles_vertical]):
        folder = os.path.join(base_folder, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_images):
            angle = np.random.choice(angles)
            cycles = np.random.choice(cycles_range)
            img_array = generate_angled_grating(size=size, cycles=cycles, angle_deg=angle)
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(os.path.join(folder, f"img_{i}.png"))
    
    print(f"Training dataset saved to {base_folder}")

def save_test_images(config=CONFIG):
    """Generate and save test images at specific angles"""
    size = config["image_size"]
    test_dir = config["test_data_dir"]
    h_range = config["horizontal_angle_range"]
    
    os.makedirs(test_dir, exist_ok=True)
    angles = config["test_angles"]
    
    print(f"Generating test images at {len(angles)} different angles...")
    
    for angle in angles:
        # Default to middle value in cycles range
        cycles = config["cycles_range"][len(config["cycles_range"]) // 2]
        img_array = generate_angled_grating(size=size, cycles=cycles, angle_deg=angle)
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Determine orientation based on angle ranges
        if angle > h_range[0] - 1 and angle <= h_range[1]:
            ori = "horizontal"
        else:
            ori = "vertical"
            
        img.save(f"{test_dir}/{ori}_{angle}.png")
    
    print(f"Test images saved to {test_dir}")

# ===== DATASET CLASS =====
class GratingDataset(Dataset):
    def __init__(self, root_dir, image_size=100):
        self.samples = []
        self.image_size = image_size
        
        for label, folder in enumerate(['horizontal', 'vertical']):
            for path in glob(os.path.join(root_dir, folder, '*.png')):
                self.samples.append((path, label))
                
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((image_size, image_size)),
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

# ===== TRAINING FUNCTION =====
def train_model(config=CONFIG):
    """Train the model with the given configuration"""
    dataset = GratingDataset(config["train_data_dir"], image_size=config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    input_size = config["image_size"] * config["image_size"]
    model = GratingNN(input_size, config["hidden_layer_size"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print(f"Starting training for {config['epochs']} epochs...")
    
    for epoch in range(config["epochs"]):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model saved to {config['model_save_path']}")
    
    return model

# ===== TESTING FUNCTION =====
def test_model(model=None, config=CONFIG):
    """Test the model on the generated test images"""
    if model is None:
        # Load the model if not provided
        input_size = config["image_size"] * config["image_size"]
        model = GratingNN(input_size, config["hidden_layer_size"])
        model.load_state_dict(torch.load(config["model_save_path"]))
    
    model.eval()
    
    # Define transform for test images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    test_dir = config["test_data_dir"]
    print(f"Testing model on images in {test_dir}...")
    
    results = []
    correct = 0
    total = 0
    
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith(".png"):
            path = os.path.join(test_dir, filename)
            img = Image.open(path)
            tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                prob_vals = probs.numpy()[0]
                prob_str = [f"{p:.4f}" for p in prob_vals]
                pred_class = "Horizontal" if prob_vals[0] > prob_vals[1] else "Vertical"
                
                # Extract true label from filename
                true_label = "Horizontal" if filename.startswith("horizontal") else "Vertical"
                
                # Check if prediction is correct
                is_correct = pred_class == true_label
                if is_correct:
                    correct += 1
                total += 1
                
                result = f"{filename} -> Probabilities: [{prob_str[0]}, {prob_str[1]}] -> Predicted: {pred_class} (Actual: {true_label})"
                print(result)
                results.append(result)
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\nOverall Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return results, accuracy

# ===== MAIN FUNCTION =====
def main():
    parser = argparse.ArgumentParser(description='Grating Image Classification')
    
    # Add command line arguments for all config parameters
    parser.add_argument('--image_size', type=int, default=CONFIG["image_size"], help='Size of images (square)')
    parser.add_argument('--train_images', type=int, default=CONFIG["train_images_per_class"], help='Number of training images per class')
    parser.add_argument('--horizontal_min', type=int, default=CONFIG["horizontal_angle_range"][0], help='Min angle for horizontal class')
    parser.add_argument('--horizontal_max', type=int, default=CONFIG["horizontal_angle_range"][1], help='Max angle for horizontal class')
    parser.add_argument('--vertical_min', type=int, default=CONFIG["vertical_angle_range"][0], help='Min angle for vertical class')
    parser.add_argument('--vertical_max', type=int, default=CONFIG["vertical_angle_range"][1], help='Max angle for vertical class')
    parser.add_argument('--hidden_size', type=int, default=CONFIG["hidden_layer_size"], help='Size of hidden layer')
    parser.add_argument('--batch_size', type=int, default=CONFIG["batch_size"], help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=CONFIG["learning_rate"], help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=CONFIG["epochs"], help='Number of training epochs')
    parser.add_argument('--model_path', type=str, default=CONFIG["model_save_path"], help='Path to save/load model weights')
    parser.add_argument('--train_dir', type=str, default=CONFIG["train_data_dir"], help='Directory for training images')
    parser.add_argument('--test_dir', type=str, default=CONFIG["test_data_dir"], help='Directory for test images')
    
    # Add action arguments
    parser.add_argument('--generate_train', action='store_true', help='Generate training dataset')
    parser.add_argument('--generate_test', action='store_true', help='Generate test images')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--all', action='store_true', help='Run all steps (generate, train, test)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG["image_size"] = args.image_size
    CONFIG["train_images_per_class"] = args.train_images
    CONFIG["horizontal_angle_range"] = [args.horizontal_min, args.horizontal_max]
    CONFIG["vertical_angle_range"] = [args.vertical_min, args.vertical_max]
    CONFIG["hidden_layer_size"] = args.hidden_size
    CONFIG["batch_size"] = args.batch_size
    CONFIG["learning_rate"] = args.learning_rate
    CONFIG["epochs"] = args.epochs
    CONFIG["model_save_path"] = args.model_path
    CONFIG["train_data_dir"] = args.train_dir
    CONFIG["test_data_dir"] = args.test_dir
    
    # Execute requested actions
    if args.all or args.generate_train:
        save_training_dataset(CONFIG)
    
    if args.all or args.generate_test:
        save_test_images(CONFIG)
    
    if args.all or args.train:
        model = train_model(CONFIG)
    else:
        model = None
    
    if args.all or args.test:
        test_model(model, CONFIG)
    
    # If no actions specified, run everything
    if not (args.generate_train or args.generate_test or args.train or args.test or args.all):
        print("No action specified, running complete workflow...")
        save_training_dataset(CONFIG)
        save_test_images(CONFIG)
        model = train_model(CONFIG)
        test_model(model, CONFIG)

if __name__ == "__main__":
    main()