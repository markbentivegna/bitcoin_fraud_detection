import torch 
from torch import nn 
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class SimpleGNNClassifier(nn.Module):
    '''
    Two-layer GNN to predict licit/illicit
    '''
    def __init__(self, in_dim, hidden, out=None, gnn='GIN', weight=None):
        super().__init__()

        self.args = (in_dim, hidden)
        self.kwargs = dict(out=out, gnn=gnn)

        self.out_dim = hidden if out is None else out
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

        if gnn == 'GCN':
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, self.out_dim)
        elif gnn == 'GAT':
            self.conv1 = GATConv(in_dim, hidden//8, heads=8)
            self.conv2 = GATConv(hidden, self.out_dim, heads=8, concat=False)
        elif gnn == 'SAGE':
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, self.out_dim)

        # Two layer just to add some complexity
        self.disc = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, 2),
            nn.Softmax(dim=1)
        )

    def embed(self, x, ei):
        x = torch.relu(self.conv1(x,ei))
        x = torch.relu(self.conv2(x,ei))
        return x  
    
    def predict(self, x, ei):
        z = self.embed(x, ei)
        return self.disc(z)

    def forward(self, x, ei, batch, labels):
        preds = self.predict(x, ei)
        loss = self.loss_fn(preds[batch], labels[batch])
        return loss 