import torch 
from torch import nn 

def gen_layer(in_d, out_d, drop=0.1, activation=nn.LeakyReLU, act_args=(0.2,)):
    return nn.Sequential(
        nn.Linear(in_d, out_d),
        nn.Dropout(drop),
        nn.BatchNorm1d(out_d, 0.8),
        activation(*act_args)
    )

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim, layers=3, dropout=0.1, alpha=0.5, beta=0.1):
        super().__init__()
        self.args = (latent_dim, hidden_dim, out_dim)
        self.kwargs = dict(layers=layers, dropout=dropout, alpha=alpha, beta=beta)

        self.net = nn.Sequential(
            gen_layer(latent_dim, hidden_dim, drop=dropout),
            *[gen_layer(hidden_dim, hidden_dim, drop=dropout) for _ in range(layers-2)],
            gen_layer(hidden_dim, out_dim, drop=dropout)
        )
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta 

    def forward(self, num_samples):
        z = torch.normal(0., 1., (num_samples, self.latent_dim))
        return self.net(z)

    def loss_fn(self, x_hat, preds):
        encirclement_loss = torch.log(torch.abs(self.alpha - preds)).mean()

        mu = x_hat.mean(dim=0, keepdim=True)
        dispersion_loss = ( 1/(x_hat-mu).pow(2).mean() ) * self.beta

        return encirclement_loss + dispersion_loss
    
class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=3, dropout=0.1, gamma=0.1):
        super().__init__()
        self.args = (in_dim, hidden_dim, out_dim)
        self.kwargs = dict(layers=layers, dropout=dropout)
        
        self.net = nn.Sequential(
            gen_layer(in_dim, hidden_dim, drop=dropout),
            *[gen_layer(hidden_dim, hidden_dim, drop=dropout) for _ in range(layers-2)],
            gen_layer(hidden_dim, out_dim, drop=dropout, activation=nn.Softmax, act_args=(-1,))
        )
        self.gamma = gamma 

    def forward(self, x, labels, weight):
        preds = self.net(x)
        log_loss = -torch.log(preds[:,labels])
        loss = log_loss * weight # Weight guesses on G(z) less
        return loss.mean()
    
    def inference(self, x):
        return self.net(x)