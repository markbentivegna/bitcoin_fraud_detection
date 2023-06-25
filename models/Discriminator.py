import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.dense_layer = nn.Linear(int(input_length), 1);
        self.activation = nn.Sigmoid()
        self.main = nn.Sequential(
            nn.Linear(int(input_length), int(input_length)),
            nn.ReLU(True),
            nn.Linear(int(input_length),1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.activation(self.dense_layer(x))
        # return self.main(x)