import math
from torch import nn

class Decoder(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channels_count, features_count, layers=5):
        super().__init__()
        exp_factor = math.ceil(math.log(max(input_dimension) // 2, 2)) - 1
        features_count = features_count * (2**exp_factor)

        self.linear_layer1 = nn.Linear(latent_dimension, features_count, bias=False)
        self.batch_layer1 = nn.BatchNorm1d(features_count)
        self.relu1 = nn.ReLU(True)

        self.inverse_pyramid_features = nn.Sequential()

        for i in range(layers):
            input_features = features_count
            output_features = features_count // 2
            self.inverse_pyramid_features.add_module(f"layer-{i}-linear", nn.Linear(input_features, output_features, bias=False))
            self.inverse_pyramid_features.add_module(f"layer-{i}-batch_norm", nn.BatchNorm1d(output_features))
            self.inverse_pyramid_features.add_module(f"layer-{i}-relu", nn.ReLU(inplace=True))
            features_count = output_features

        self.final_linear_layer = nn.Linear(features_count, input_dimension[1], bias=False)

    def forward(self, input_tensor):
        h = self.batch_layer1(self.linear_layer1(input_tensor))
        h = self.relu1(h)
        h = self.inverse_pyramid_features(h)
        output = self.final_linear_layer(h)
        return output