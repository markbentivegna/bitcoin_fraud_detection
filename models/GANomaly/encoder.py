from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channel_count, features_count):
        super().__init__()
        # self.conv_layer1 = nn.Conv1d(input_channel_count, features_count, kernel_size=4, stride=1, bias=False)
        self.conv_layer1 = nn.Linear(features_count, features_count, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        # pyramid_dimension = max(*input_dimension)
        self.pyramid_features = nn.Sequential()
        # while pyramid_dimension > 4:
        for i in range(5):
            input_features = features_count
            output_features = features_count * 2
            # self.pyramid_features.add_module(f"pyramid-{output_features}-conv", nn.Conv1d(input_features, output_features, kernel_size=4, stride=1, bias=False))
            self.pyramid_features.add_module(f"pyramid-{output_features}-linear", nn.Linear(input_features, output_features, bias=False))
            # self.pyramid_features.add_module(f"pyramid-{output_features}-batch_norm", nn.BatchNorm1d(output_features))
            self.pyramid_features.add_module(f"pyramid-{output_features}-batch_norm", nn.BatchNorm1d(output_features))
            self.pyramid_features.add_module(f"pyramid-{output_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            features_count = output_features
            # pyramid_dimension = pyramid_dimension // 2

        # self.final_conv_layer = nn.Conv1d(features_count,latent_dimension,kernel_size=4,stride=1,bias=False)
        self.final_conv_layer = nn.Linear(features_count,latent_dimension, bias=False)

    def forward(self, input_tensor):
        h = self.relu1(self.conv_layer1(input_tensor))
        h = self.pyramid_features(h)
        output = self.final_conv_layer(h)
        return output

'''
Slightly cleaner version of above 

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, layers=5):
        super().__init__()

        self.input = nn.Linear(input_dim, input_dim, bias=False)

        def layer_block(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d), 
                nn.BatchNorm1d(out_d), 
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.pyramid_features = nn.Sequential()
        feats = input_dim
        for i in range(layers):
            in_f = feats 
            out_f = feats * 2 
            
            self.pyramid_features.add_module(f'layer-{i}', layer_block(in_f, out_f))
            feats = out_f 

        self.final = nn.Linear(feats, latent_dim, bias=False)

    def forward(self, x): 
        x = self.input(x) 
        x = self.pyramid_features(x)
        return self.final(x) 
'''