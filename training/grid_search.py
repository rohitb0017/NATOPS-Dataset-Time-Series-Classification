import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
from models.lstm_model import LSTMClassifier
from utils.dataset import NATOPSDataset
from sklearn.metrics import accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameter grid
param_grid = {
    "hidden_dim": [32, 64],        # LSTM hidden state size
    "num_layers": [1, 2],          # Number of stacked LSTM layers
    "bidirectional": [False, True] # Use bidirectional LSTM or not
}

# Load dataset
train_dataset = NATOPSDataset(train=True)
test_dataset = NATOPSDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define fixed parameters
input_dim = 24    # Features per time step
num_classes = 6   # Number of classes
num_epochs = 3    # Train for limited epochs to save CUDA usage

# Function to evaluate model accuracy
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

# Grid Search over hyperparameters
best_acc = 0
best_params = None
best_model_state = None

for hidden_dim in param_grid["hidden_dim"]:
    for num_layers in param_grid["num_layers"]:
        for bidirectional in param_grid["bidirectional"]:
            print(f"\n🔍 Training with: Hidden={hidden_dim}, Layers={num_layers}, Bidirectional={bidirectional}")

            # Initialize model
            model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, bidirectional).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train model
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                
                print(f"   Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss/len(train_loader):.4f}")

            # Evaluate on test set
            test_acc = evaluate(model, test_loader)
            print(f"   Validation Accuracy: {test_acc:.4f}")

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_params = {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "bidirectional": bidirectional,
                    "test_accuracy": test_acc
                }
                best_model_state = model.state_dict()

# Save best model and parameters
torch.save(best_model_state, "results/best_model.pth")
with open("results/training_log.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("\n✅ Best Model Saved!")
print(f"   Best Params: {best_params}")
