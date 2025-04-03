import torch
from torch.utils.data import Dataset
import numpy as np
import os

# Function to load train/test data
def load_data(train=True):
    data_dir = "/content/drive/MyDrive/NATOPS-TSC/data"  # Adjust based on your directory structure
    file_name = "NATOPS_TRAIN.ts" if train else "NATOPS_TEST.ts"
    file_path = os.path.join(data_dir, file_name)

    # Manually parse .ts file to extract only numerical data
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip metadata lines (lines starting with '@')
    data_lines = [line.strip() for line in lines if not (line.startswith("@") or line.startswith("#"))]

    # Convert to numpy array
    data = np.array([list(map(float, line.replace(":", ",").split(","))) for line in data_lines])

    # Extract features (all but last column)
    X = data[:, :-1]  # Assuming last column is the label
    
    # Reshape X to match expected LSTM input shape
    seq_len = 24  # Adjust based on correct sequence length
    num_features = X.shape[1] // seq_len
    X = X.reshape(-1, seq_len, num_features)  # Shape: [samples, seq_len, features]

    # Extract labels (last column)
    y = data[:, -1]

    # Convert labels to integer and 0-indexed
    y = y.astype(int) - 1  

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print(f"Loaded X shape: {X.shape}")  # Debugging
    print(f"Loaded y shape: {y.shape}")  # Debugging

    return X, y

class NATOPSDataset(Dataset):
    def __init__(self, train=True):
        self.X, self.y = load_data(train=train)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
