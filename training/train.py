import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

base_dir = "/content/drive/MyDrive/NATOPS-TSC"
sys.path.append(base_dir)  # Add the project root to Python's module search path

from models.lstm_model import LSTMClassifier
from training.dataset import NATOPSDataset

# Mount Google Drive
base_dir = "/content/drive/MyDrive/NATOPS-TSC/training"
hyperparams_path = os.path.join(base_dir, "best_hyperparams.json")
best_model_path = os.path.join(base_dir, "best_model.pth")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_dataset = NATOPSDataset(train=True)
test_dataset = NATOPSDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load best hyperparameters
with open(hyperparams_path, "r") as f:
    best_hyperparams = json.load(f)

hidden_dim = best_hyperparams["hidden_dim"]
num_layers = best_hyperparams["num_layers"]
bidirectional = best_hyperparams["bidirectional"]
learning_rate = best_hyperparams["learning_rate"]

print(f"Training with best hyperparameters: {best_hyperparams}")

# Define model
input_dim = 51
num_classes = 6
model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, bidirectional).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 50
best_test_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with Test Accuracy: {test_acc:.4f}\n")

print("Training complete!")
print(f"Final Best Test Accuracy: {best_test_acc:.4f}")
