import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import MinMaxScaler

# Load NATOPS dataset
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("NATOPS")

# Debug: Print y_train details before conversion
print("y_train sample:", y_train[:5])
print("y_train type:", type(y_train))
print("y_train dtype:", y_train.dtype)

# Ensure y_train and y_test are converted to integers properly
y_train = np.array(y_train, dtype=np.float32).astype(int)  # Convert from float strings to integers
y_test = np.array(y_test, dtype=np.float32).astype(int)

# Convert labels to zero-based index (PyTorch requires this for classification)
y_train -= 1
y_test -= 1

# Debug: Print after conversion
print("y_train after conversion:", y_train[:5])
print("y_train dtype after conversion:", y_train.dtype)

# Normalize time series using Min-Max Scaling (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = np.array([scaler.fit_transform(x) for x in X_train], dtype=np.float32)
X_test = np.array([scaler.transform(x) for x in X_test], dtype=np.float32)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# PyTorch Dataset class
class NATOPSDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders for batching
batch_size = 32
train_dataset = NATOPSDataset(X_train_tensor, y_train_tensor)
test_dataset = NATOPSDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print batch shape for verification
for X_batch, y_batch in train_loader:
    print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
    break
