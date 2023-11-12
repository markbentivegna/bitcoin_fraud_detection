import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channels_count, features_count):
        super(Generator, self).__init__()

        def get_layer(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d, bias=True), 
                nn.BatchNorm1d(out_d), 
                nn.ReLU()
            )

        # Cleaning up a bit
        self.net = nn.Sequential(
            get_layer(latent_dimension, latent_dimension*8),
            get_layer(latent_dimension*8, latent_dimension*4),
            get_layer(latent_dimension*4, latent_dimension*2),
            nn.Linear(latent_dimension*2, features_count)
        )


    def forward(self,input):
        return self.net(input)