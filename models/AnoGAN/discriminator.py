import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channels_count, features_count):
        super(Discriminator,self).__init__()

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

    def forward(self, input):
        h = self.relu1(self.batch_layer1(self.linear_layer1(input)))
        h = self.relu2(self.batch_layer2(self.linear_layer2(h)))
        h = self.relu3(self.batch_layer3(self.linear_layer3(h)))
        h = self.sigmoid(h)
        return h.squeeze(1)