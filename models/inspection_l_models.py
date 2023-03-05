import torch 
from torch import nn 
from torch_geometric.nn import GINConv, GCNConv

class GINAggr(nn.Module):
    '''
    Two layer MLP: 
        Lin -> 128 hidden
        Norm 
        ReLU
        Lin -> 128 out?
        ReLU
    '''
    def __init__(self, in_dim, hidden=128, out=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
class InspectionL(nn.Module):
    '''
    Two-layer GIN with above GINAggr function
    '''
    def __init__(self, in_dim, hidden, out=None):
        super().__init__()

        self.args = (in_dim, hidden)
        self.kwargs = dict(out=out)

        out = hidden if out is None else out
        self.corrupt = Corrupt()
        self.bce = nn.BCELoss()

        self.conv1 = GINConv(GINAggr(in_dim, hidden=hidden))
        self.conv2 = GINConv(GINAggr(hidden, hidden=hidden, out=out))

        # The same as initializing a matrix to multiply with
        # as the discriminator is defined as hws where 
        # h is the embeddings of each node (Bxd) and s is the 
        # mean of all embeddings (1xd) and we want to produce a 
        # Bx1 out so we use hws^T 
        self.disc = nn.Linear(out, out, bias=False)

    def embed(self, x, ei):
        x = self.conv1(x, ei)
        x = self.conv2(x, ei)
        return x
    
    def readout(self, x):
        return torch.sigmoid(
            torch.mean(x, dim=0, keepdim=True)
        )

    def discriminate(self, x, s):
        # @ operator means matrix multiply in torch 
        return torch.sigmoid(
            self.disc(x) @ s.T
        )

    def forward(self, x, ei):
        x_real = self.embed(x,ei)
        x_perturb = self.embed(self.corrupt(x), ei)

        s = self.readout(x_real)
        real_loss = self.discriminate(x_real, s)
        perturb_loss = self.discriminate(x_perturb, s)
        targets = torch.zeros(real_loss.size(0)+perturb_loss.size(0), 1)
        targets[:real_loss.size(0)] = 1.

        loss = self.bce(torch.cat([real_loss, perturb_loss]), targets)
        return loss


class Corrupt(nn.Module):
    '''
    Switch rows in X around
    '''
    def forward(self, x):
        perm = torch.randperm(x.size(0))
        return x[perm]
    
