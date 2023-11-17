import torch 
from torch import nn 

class SmoothL1LossUncompressed(nn.Module):
    def __init__(self, beta=1) -> None:
        super().__init__()
        self.beta = beta 

    def forward(self, x,y):
        '''
        It really seems like there should be a way to do this but eh
        '''
        out = (x-y).abs()
        mask = out < self.beta 

        out[mask] = (0.5 * out[mask].pow(2)) / self.beta 
        out[~mask] = out[~mask] - 0.5*self.beta 

        return out.mean(dim=-1)