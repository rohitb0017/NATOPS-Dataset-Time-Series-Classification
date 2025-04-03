import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm_model import LSTMClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
#from models.lstm_model import LSTMClassifier
from training.dataset import NATOPSDataset

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_dataset = NATOPSDataset(train=True)
test_dataset = NATOPSDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model hyperparameters
input_dim = 51       # Features per timestep (Updated to match LSTM model)
hidden_dim = 64       # LSTM hidden units
num_layers = 2        # Number of LSTM layers
num_classes = 6       # Total classes in NATOPS dataset
bidirectional = False # Whether LSTM is bidirectional

# Initialize model
model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, bidirectional).to(device)

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
best_test_acc = 0.0  # Track best test accuracy
best_model_path = "/content/drive/MyDrive/NATOPS-TSC/results/best_model.pth"
best_hyperparams = {}

for epoch in range(num_epochs):
    model.train()  # Set to training mode
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

    # Evaluate on test set after each epoch
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

    # Save model if test accuracy improves
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_hyperparams = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "learning_rate": 0.001
        }
        os.makedirs("/content/drive/MyDrive/NATOPS-TSC/results", exist_ok=True)
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with Test Accuracy: {test_acc:.4f}")
        print(f"Best Hyperparameters: {best_hyperparams}\n")
    else:
        print("No improvement in test accuracy, model not saved.\n")

print("Training complete!")
print(f"Final Best Model Test Accuracy: {best_test_acc:.4f}")
print(f"Final Best Hyperparameters: {best_hyperparams}")
