import json
import torch
import itertools
from torch.utils.data import DataLoader
import sys
import os
base_dir = "/content/drive/MyDrive/NATOPS-TSC"
sys.path.append(base_dir)  # Add the project root to Python's module search path

from models.lstm_model import LSTMClassifier
from training.dataset import NATOPSDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os

# Mount Google Drive
base_dir = "/content/drive/MyDrive/NATOPS-TSC/training"
hyperparams_path = os.path.join(base_dir, "best_hyperparams.json")

# Define Hyperparameter Search Space
hidden_dims = [32, 64, 128]
num_layers_list = [1, 2, 3]
bidirectional_list = [True, False]

# Load dataset
train_dataset = NATOPSDataset(train=True)
test_dataset = NATOPSDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0.0
best_params = {}

for hidden_dim, num_layers, bidirectional in itertools.product(hidden_dims, num_layers_list, bidirectional_list):
    print(f"Testing: hidden_dim={hidden_dim}, num_layers={num_layers}, bidirectional={bidirectional}")

    # Initialize model
    input_dim = 51  
    num_classes = 6
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, bidirectional).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs
    for epoch in range(3):  
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
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

    if test_acc > best_acc:
        best_acc = test_acc
        best_params = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "learning_rate": 0.001
        }

        # Save best hyperparameters to Google Drive
        with open(hyperparams_path, "w") as f:
            json.dump(best_params, f)
        print(f"Best hyperparameters updated: {best_params}")

print("Grid Search Complete. Best Hyperparameters:")
print(best_params)
