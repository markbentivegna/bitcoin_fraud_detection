import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dimension, latent_dimension, input_channels_count, features_count):
        super(Generator, self).__init__()
        self.linear_layer1 = nn.Linear(latent_dimension, latent_dimension * 8, bias = False)
        self.batch_layer1 = nn.BatchNorm1d(input_channels_count)
        self.relu1 = nn.ReLU(inplace = True)

        self.linear_layer2 = nn.Linear(latent_dimension*8, latent_dimension * 4, bias = False)
        self.batch_layer2 = nn.BatchNorm1d(input_channels_count)
        self.relu2 = nn.ReLU(inplace = True)

        self.linear_layer3 = nn.Linear(latent_dimension * 4, latent_dimension * 2, bias = False)
        self.batch_layer3 = nn.BatchNorm1d(input_channels_count)
        self.relu3 = nn.ReLU(inplace = True)
        self.linear_layer4 = nn.Linear(latent_dimension*2, features_count, bias = False)


    def forward(self,input):
        h = self.relu1(self.batch_layer1(self.linear_layer1(input.unsqueeze(1))))
        h = self.relu2(self.batch_layer2(self.linear_layer2(h)))
        h = self.relu3(self.batch_layer3(self.linear_layer3(h)))
        h = self.linear_layer4(h)
        return h