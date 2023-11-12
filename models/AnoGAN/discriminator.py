import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channels_count, features_count):
        super(Discriminator,self).__init__()

        def get_layer(in_d, out_d, bias=True):
            return nn.Sequential(
                nn.Linear(in_d, out_d, bias=bias), 
                nn.BatchNorm1d(out_d), 
                nn.ReLU()
            )

        self.net = nn.Sequential(
            get_layer(features_count, features_count), 
            get_layer(features_count, features_count*2),
            get_layer(features_count*2, 1, bias=True),
            nn.Sigmoid()
        )

        '''
        self.linear_layer1 = nn.Linear(features_count, features_count, bias = False)
        self.batch_layer1 = nn.BatchNorm1d(input_channels_count)
        self.relu1 = nn.ReLU(inplace = True)


        self.linear_layer2 = nn.Linear(features_count , features_count * 2, bias = False)
        self.batch_layer2 = nn.BatchNorm1d(input_channels_count)
        self.relu2 = nn.ReLU(inplace = True)

        self.linear_layer3 = nn.Linear(features_count * 2, 1)
        self.batch_layer3 = nn.BatchNorm1d(input_channels_count)
        self.relu3 = nn.ReLU(inplace = True)

        self.sigmoid = nn.Sigmoid()
        '''

    def forward(self, input):
        return self.net(input)