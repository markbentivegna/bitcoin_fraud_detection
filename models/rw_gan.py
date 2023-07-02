import torch 
from torch import nn 


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__() 

        self.multihead = nn.MultiheadAttention(dim, heads, dropout=0.25, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, v, qk):
        val,_ = self.multihead(v,qk,qk)
        val = v + val 
        val = self.norm1(val)

        x = self.proj(val) 
        val = val + x 
        return self.norm2(val)


class RWGenerator(nn.Module):
    '''
    Takes a RW sequence as input, appends a new node to the end
    '''
    def __init__(self, in_dim, hidden, heads, latent_dim=32, layers=3):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(hidden, heads) for _ in range(layers)]
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden, in_dim),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        self.latent = latent_dim
        self.rnd = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Dropout(0.25),
            nn.ReLU()
        )

    def forward(self, rws):
        '''
        Takes as input B x S x d tensor
        representing the d-dimensional node features of nodes from 
        B random walks, of len S, 
        '''

        # Generate random "new" node at the end of the sequence
        z = self.rnd(torch.rand(rws.size(0), 1, self.latent))
        x = self.proj(rws)
        rws = torch.cat([x, z], dim=1)

        # Pass the sequence through n-1 atten mechanisms
        for l in self.layers[:-1]:
            rws = l.forward(rws, rws)
        
        # On the last one, just get the value for the random node 
        new_node = rws[:,-1:,:]
        final = self.layers[-1](new_node, rws)

        return self.out_proj(final)
    

class RWDiscriminator(nn.Module):
    '''
    Takes RW sequence as input and decides if the final node 
    belongs, given its parents
    '''
    def __init__(self, in_dim, hidden, heads, layers=3):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(hidden, heads) for _ in range(layers)]
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, rws, labels):
        '''
        Takes as input B x S x d tensor
        representing the d-dimensional node features of nodes from 
        B random walks, of len S, 
        '''
        x = self.proj(rws)

        # Pass the sequence through n-1 atten mechanisms
        for l in self.layers[:-1]:
            x = l.forward(x,x)
        
        # On the last one, just get the value for the final node 
        test_node = x[:,-1:,:]
        final = self.layers[-1](test_node, x).squeeze(1)
        preds = self.out_proj(final)

        loss = self.loss_fn(preds, labels)
        return loss 