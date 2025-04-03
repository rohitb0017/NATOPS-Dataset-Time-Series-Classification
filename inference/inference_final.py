import torch
import os
from model import LSTMClassifier  # Import your LSTM model
from dataset import NATOPSDataset  # Import dataset class

# Load test dataset
def load_test_data(test_data_path):
    X_test, y_test = torch.load(test_data_path)
    return X_test, y_test

# Function to run inference
def run_inference(model_path, test_data_path, results_dir):
    # Load trained model
    model = LSTMClassifier(input_dim=24, hidden_dim=64, num_layers=2, num_classes=6, bidirectional=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Load test data
    X_test, y_test = load_test_data(test_data_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "predictions.txt")
    with open(results_path, "w") as f:
        for pred, true_label in zip(predictions, y_test):
            f.write(f"Predicted: {pred.item()}, Actual: {true_label.item()}\n")

    print(f"Inference complete! Results saved in {results_path}")

# Example usage
if __name__ == "__main__":
    run_inference("../models/best_model.pth", "../data/processed/test.pt", "../results")

