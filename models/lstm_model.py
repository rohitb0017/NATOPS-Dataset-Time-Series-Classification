import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=51, hidden_dim=128, num_layers=2, num_classes=6, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size, seq_len, hidden_dim]
        lstm_out = lstm_out[:, -1, :]  # Take last timestep output
        output = self.fc(lstm_out)
        return output
