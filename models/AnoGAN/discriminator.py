import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, c_dimension, df_dimension):
        super(Discriminator,self).__init__()

        self.conv_layer1 = nn.Conv1d(c_dimension, df_dimension, 4, bias = False)
        self.elu1 = nn.ELU(inplace = True)


        self.conv_layer2 = nn.Conv1d(df_dimension , df_dimension * 2, 4, bias = False)
        self.batch_layer1 = nn.BatchNorm1d(df_dimension*2)
        self.elu2 = nn.ELU(inplace = True)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(df_dimension * 2, 1)
        self.sigmoid = nn.Sigmoid()

        for module in self.modules():
            if isinstance(module , nn.Conv2d):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input):
        h = self.elu1(self.conv_layer1(input))
        h = self.elu2(self.batch_layer1(self.conv_layer2(h)))
        h = self.linear(self.flatten(self.adaptive_avg_pool(h)))
        out = self.sigmoid(h)
        return out,h