import torch 
from torch import nn 

class Generator(nn.Module):
    '''
    Given input feature generate the features of nodes
    1 hop away on a random walk (maybe try k-hops later?)
    '''
    def __init__(self, in_dim, hidden, latent):
        super().__init__() 

        self.first = nn.Sequential(
            nn.Linear(in_dim, hidden*2),
            nn.ReLU(),
            nn.Dropout()
        )

        # Don't want final activations as logvar and mu 
        # can both be negative (though maybe it would be okay to do
        # tanh or something to squish into (-1,1)... maybe.)
        self.mu = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, latent),
        )
        self.logvar = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, latent),
        )

        self.decode = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, hidden*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden*2, in_dim),
        )


    def forward(self, x):
        x = self.first(x) 

        mu = self.mu(x)
        logvar = self.logvar(x)

        if self.training:
            z = self.reparam(mu, logvar)
        else:
            z = mu 

        return self.decode(z), self.kld_loss(mu, logvar)

    def sample(self, x, variational=False):
        return self.decode(
            self.encode(x, variational=variational)
        )
        
    def encode(self, x, variational=False):
        x = self.first(x)
        mu = self.mu(x)

        if self.training or variational:
            logvar = self.logvar(x)    
            return self.reparam(mu, logvar)
        else:
            return mu

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def kld_loss(self, mu, logvar):
        return torch.mean(
            -0.5 * torch.sum(
                1 + logvar - mu ** 2 - logvar.exp(), 
                dim=1
            ), dim=0)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden, layers=3):
        super().__init__()

        def layer(ind, outd, act=nn.LeakyReLU):
            return nn.Sequential(
                nn.Linear(ind, outd),
                nn.Dropout(),
                act()
            )

        self.net = nn.Sequential(
            layer(in_dim, hidden),
            *[layer(hidden, hidden) for _ in range(layers-2)], 
            layer(hidden, 1, act=nn.Identity)
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        preds = self.net(x)
        return self.criterion(preds, y)
    
    def inference(self, x):
        return self.net(x)