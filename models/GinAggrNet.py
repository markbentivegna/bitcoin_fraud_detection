import torch.nn as nn

class GINAggrNet(nn.Module):
    def __init__(self, input_dimension, hidden_layers=128, output_dimension=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dimension, hidden_layers),
            nn.BatchNorm1d(hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, output_dimension),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)