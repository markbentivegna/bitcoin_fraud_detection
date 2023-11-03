from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dimension, latent_dimension, features_count, layers=5):
        super().__init__()

        self.input = nn.Linear(features_count, features_count, bias=False)

        def layer_block(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d), 
                nn.BatchNorm1d(out_d), 
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.pyramid_features = nn.Sequential()
        feats = max(*input_dimension)
        for i in range(layers):
            in_f = feats 
            out_f = feats * 2 
            
            self.pyramid_features.add_module(f'layer-{i}', layer_block(in_f, out_f))
            feats = out_f 

        self.final = nn.Linear(feats, latent_dimension, bias=False)

    def forward(self, x): 
        x = self.input(x) 
        x = self.pyramid_features(x)
        return self.final(x) 
