import torch
from torch import nn 
from torch.autograd import Function
from torch_geometric.nn import GCNConv, GATConv

class ReverseGradient(Function):
    '''
    Inverts gradient on backward pass. 
    Used for adversarial training
    '''
    @staticmethod
    def forward(self, x):
        return x
    @staticmethod
    def backward(self, grad_output):
        return (-grad_output)
    

class AdaGNN(nn.Module):
    def __init__(self, in_dim, hidden, gnn='GCN', ts_out=5, lamb=1.0):
        super().__init__()

        # Unclear how many layers from the paper. Assuming 2
        if gnn == 'GCN':
            self.gnn1 = GCNConv(in_dim, hidden)
            self.gnn2 = GCNConv(hidden, hidden)
        elif gnn == 'GAT':
            self.gnn1 = GATConv(in_dim, hidden//8, heads=8)
            self.gnn2 = GATConv(hidden, hidden//8, heads=8)

        # Ditto 
        self.ts_pred = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, ts_out),
            #nn.Softmax(dim=1) XEntropyLoss expects unnormalized logits
        )

        self.class_pred = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, 2),
            #nn.Softmax(dim=1)
        )

        self.reverse_grad = ReverseGradient
        self.lambda_param = lamb # Never specified in the paper either
        self.loss_fn = nn.CrossEntropyLoss()

        self.args = (in_dim, hidden)
        self.kwargs = dict(gnn=gnn, ts_out=ts_out, lamb=lamb)
        self.out_dim=hidden

    def embed(self, x, ei):
        z = torch.relu(self.gnn1(x, ei))
        return self.gnn2(z, ei)

    def forward(self, x, ei, ts_target, node_target, node_mask):
        z = self.embed(x,ei)

        ts = self.ts_pred(self.reverse_grad.apply(z))
        preds = self.class_pred(z)

        ts_loss = self.loss_fn(ts, ts_target)
        classification_loss = self.loss_fn(preds[node_mask], node_target)

        # Returning losses individually for display purposes, but no reason
        # not to add them together. But this should show how the model
        # swings between optimizing ts predictor, and class predictor
        return classification_loss, self.lambda_param*ts_loss 