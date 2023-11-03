import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dimension, c_dimension, gf_dimension):
        super(Generator, self).__init__()
        self.transpose_layer1 = nn.ConvTranspose1d(input_dimension, gf_dimension * 8, 4, bias = False)
        self.batch_layer1 = nn.BatchNorm1d(gf_dimension * 8)
        self.elu1 = nn.ELU(inplace = True)


        self.transpose_layer2 = nn.ConvTranspose1d(in_channels= gf_dimension*8 , out_channels= gf_dimension * 4 , kernel_size = 4 , stride = 2 , padding = 1 , bias = False)
        self.batch_layer2 = nn.BatchNorm1d(gf_dimension * 4)
        self.elu2 = nn.ELU(inplace = True)


        self.transpose_layer3 = nn.ConvTranspose1d(in_channels = gf_dimension * 4 , out_channels = gf_dimension * 2 , kernel_size= 4, stride = 2 , padding = 1 , bias = False)
        self.batch_layer3 = nn.BatchNorm1d(gf_dimension * 2)
        self.elu3 = nn.ELU(inplace = True)


        self.transpose_layer4 = nn.ConvTranspose1d(in_channels= gf_dimension*2 , out_channels= 1, kernel_size= 4, stride = 2 , padding = 1 , bias = False)
        self.tanh = nn.Tanh()

        for module in self.modules():
            if isinstance(module , nn.ConvTranspose1d):
                module.weight.data.normal_(mean=0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self,input):
        h = self.elu1(self.batch_layer1(self.transpose_layer1(input.unsqueeze(2))))
        h = self.elu2(self.batch_layer2(self.transpose_layer2(h)))
        h = self.elu3(self.batch_layer3(self.transpose_layer3(h)))
        h = self.transpose_layer4(h)
        out = self.tanh(h)
        return out