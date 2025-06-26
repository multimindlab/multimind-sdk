from multimind.fine_tuning.unified_fine_tuner import AdapterModule
import torch
import torch.nn as nn

# Simple LSTM model
def make_lstm(input_dim, hidden_dim, output_dim):
    class SimpleLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    return SimpleLSTM()

# Dummy adapter (pseudo-code)
class MyAdapter(AdapterModule):
    def forward(self, x):
        # Insert adapter logic here (e.g., bottleneck + nonlinearity)
        return x  # No-op for demo

# Instantiate model and adapter
input_dim, hidden_dim, output_dim = 10, 16, 2
model = make_lstm(input_dim, hidden_dim, output_dim)
adapter = MyAdapter(input_dim, output_dim)

# Plug adapter into model (pseudo-code)
# In practice, you would insert the adapter into the LSTM or after it
x = torch.randn(4, 5, input_dim)
output = adapter(model(x))
print("Adapter output shape:", output.shape) 