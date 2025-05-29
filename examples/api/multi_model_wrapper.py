import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Block
class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        return x

# RNN Block
class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNBlock, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x[:, -1, :]  # Last time step

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer(x)
        return x[-1]  # Last token

# Wrapper combining all
class MultiModelWrapper(nn.Module):
    def __init__(self):
        super(MultiModelWrapper, self).__init__()
        self.cnn = CNNBlock(1, 16)
        self.rnn = RNNBlock(16, 32)
        self.transformer = TransformerBlock(32, 4)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.rnn(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = self.fc(x)
        return torch.sigmoid(x) 