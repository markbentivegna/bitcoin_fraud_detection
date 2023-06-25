from torch import nn
from models.GANomaly.encoder import Encoder

class Discriminator(nn.Module):
    def __init__(self, input_dimension, input_channels_count, features_count):
        super().__init__()
        encoder = Encoder(input_dimension, 1, features_count)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('sigmoid', nn.Sigmoid())

    def forward(self, input_tensor):
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.squeeze(1)
        return classifier, features