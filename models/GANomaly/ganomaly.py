import torch
from torch import nn
from models.GANomaly.generator import Generator
from models.GANomaly.discriminator import Discriminator
from models.GANomaly.discriminator_loss import DiscriminatorLoss
from models.GANomaly.generator_loss import GeneratorLoss

class GANomaly(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channels_count, features_count):
        super().__init__()
        self.generator = Generator(input_dimension, latent_dimension, input_channels_count, features_count)
        self.discriminator = Discriminator(input_dimension, input_channels_count, features_count)
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)
        w_adversarial = 1
        w_contextual = 50
        w_encoder = 1
        self.generator_loss = GeneratorLoss(w_adversarial, w_contextual, w_encoder)
        self.discriminator_loss = DiscriminatorLoss()
        
    def weights_init(self, module):
        class_name = module.__class__.__name__
        if class_name.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif class_name.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, batch):
        fake, latent_input, latent_output = self.generator(batch)
        return batch, fake, latent_input, latent_output