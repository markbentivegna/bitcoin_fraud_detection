import math
from torch import nn

class Decoder(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channels_count, features_count):
        super().__init__()
        exp_factor = math.ceil(math.log(max(input_dimension) // 2, 2)) - 1
        features_count = features_count * (2**exp_factor)

        # self.conv_layer1 = nn.ConvTranspose1d(latent_dimension, features_count, kernel_size=4, stride=1, bias=False)
        self.conv_layer1 = nn.Linear(latent_dimension, features_count, bias=False)
        # self.batch_layer1 = nn.BatchNorm1d(features_count)
        self.batch_layer1 = nn.BatchNorm1d(features_count)
        self.relu1 = nn.ReLU(True)

        self.inverse_pyramid_features = nn.Sequential()
        pyramid_dimension = max(*input_dimension) // 2

        # while pyramid_dimension > 4:
        for i in range(5):
            input_features = features_count
            output_features = features_count // 2
            # self.pyramid_features.add_module(f"pyramid-{output_features}-conv", nn.ConvTranspose1d(input_features, output_features, kernel_size=4, stride=1, bias=False))
            self.inverse_pyramid_features.add_module(f"pyramid-{output_features}-linear", nn.Linear(input_features, output_features, bias=False))
            # self.pyramid_features.add_module(f"pyramid-{output_features}-batch_norm", nn.BatchNorm1d(output_features))
            self.inverse_pyramid_features.add_module(f"pyramid-{output_features}-batch_norm", nn.BatchNorm1d(output_features))
            self.inverse_pyramid_features.add_module(f"pyramid-{output_features}-relu", nn.ReLU(inplace=True))
            features_count = output_features
            # pyramid_dimension = pyramid_dimension // 2

        # self.final_conv_layer = nn.ConvTranspose1d(features_count,input_channels_count,kernel_size=4,stride=1,bias=False)
        self.final_conv_layer = nn.Linear(features_count, input_dimension[1], bias=False)

    def forward(self, input_tensor):
        h = self.batch_layer1(self.conv_layer1(input_tensor))
        h = self.relu1(h)
        h = self.inverse_pyramid_features(h)
        output = self.final_conv_layer(h)
        return output