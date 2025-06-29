import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === PARAMETERS ===

# Path to image dataset and label CSV (⚠ Update these paths if needed)
dataset_path = "/path/to/images_light"  # <-- Change to your actual dataset folder
csv_file = os.path.join(dataset_path, "labels.csv")  # Must exist and contain [filename, label]

image_size = (128, 128)  # Resize images to this fixed shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto use GPU if available

# === FEATURE EXTRACTION ===

# Extract both CNN input and handcrafted brightness-based features
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, image_size)

    cnn_input = image_resized.astype(np.float32) / 255.0
    cnn_input = np.expand_dims(cnn_input, axis=0)  # Shape becomes (1, 128, 128)

    height, width = image.shape
    total_pixels = height * width
    mean_brightness = np.mean(image)
    saturated_pixels = np.sum(image >= 230)  # 230+ = saturated
    saturated_percent = (saturated_pixels / total_pixels) * 100

    return cnn_input, np.array([mean_brightness, saturated_percent], dtype=np.float32)

# === DATASET CLASS ===

class HybridBlindingDataset(Dataset):
    def __init__(self, csv_file, dataset_path):
        self.data = pd.read_csv(csv_file)  # Expects columns: filename, label
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = str(self.data.iloc[idx, 0])  # e.g., frame_0001.png
        label = self.data.iloc[idx, 1]          # 0 (not blinded) or 1 (blinded)
        img_path = os.path.join(self.dataset_path, filename)

        cnn_input, handcrafted = extract_features(img_path)

        return (
            torch.tensor(cnn_input, dtype=torch.float32),
            torch.tensor(handcrafted, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

# === HYBRID CNN MODEL DEFINITION ===

class HybridBlindingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # Output: 16 x 64 x 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # Output: 32 x 32 x 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # Output: 64 x 16 x 16
            nn.Flatten()
        )
        # Compute final flattened CNN feature size
        self.flattened_size = 64 * (image_size[0]//8) * (image_size[1]//8)

        # Final fully connected layers (including handcrafted input: 2 dims)
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size + 2, 128), nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output: Probability of blinding (0 to 1)
        )

    def forward(self, image, features):
        cnn_out = self.cnn_branch(image)
        combined = torch.cat((cnn_out, features), dim=1)
        return self.fc(combined)

# === TRAINING FUNCTION ===

def train():
    dataset = HybridBlindingDataset(csv_file, dataset_path)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)  # ⚠ Change batch size if needed

    model = HybridBlindingModel().to(device)
    criterion = nn.BCELoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30  # ⚠ Change epochs depending on performance
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for img_batch, features_batch, labels_batch in train_loader:
            img_batch = img_batch.to(device)
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)  # Shape: [batch, 1]

            outputs = model(img_batch, features_batch)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

        acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss/len(train_loader):.4f}  Accuracy: {acc:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "hybrid_blinding_model.pth")
    print("✅ Model trained and saved as hybrid_blinding_model.pth")

# === INFERENCE FUNCTION ===

def predict(image_path):
    model = HybridBlindingModel()
    model.load_state_dict(torch.load("hybrid_blinding_model.pth", map_location=device))  # ⚠ Adjust path if needed
    model.to(device)
    model.eval()

    cnn_input, handcrafted = extract_features(image_path)
    cnn_input = torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).to(device)
    handcrafted = torch.tensor(handcrafted, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(cnn_input, handcrafted)

    prediction = output.item()
    percentage = prediction * 100
    print(f"Predicted blinding percentage: {percentage:.2f}%")

    if prediction >= 0.5:
        print("⚠ BLINDED")
    else:
        print("✅ NOT BLINDED")

# === MAIN EXECUTION ===

if __name__ == "__main__":
    train()

    # Run a test prediction after training
    test_image = "/path/to/frame_0010.png"  # ⚠ Change to one of your test images
    predict(test_image)

