from torch import nn
from models.GANomaly.encoder import Encoder
from models.GANomaly.decoder import Decoder


class Generator(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channel_count, features_count):
        super().__init__()
        self.encoder1 = Encoder(input_dimension, latent_dimension, input_channel_count, features_count)
        self.decoder = Decoder(input_dimension, latent_dimension, input_channel_count, features_count)
        self.encoder2 = Encoder(input_dimension, latent_dimension, input_channel_count, features_count)

    def forward(self, input_tensor):
        latent_input = self.encoder1(input_tensor.unsqueeze(1))
        reconstructed_x = self.decoder(latent_input)
        latent_output = self.encoder2(reconstructed_x)
        return input_tensor, reconstructed_x, latent_input, latent_output
