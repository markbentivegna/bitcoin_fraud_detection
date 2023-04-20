import torch 
from torch import nn
from torch_scatter import (
    scatter_add, scatter_mean, 
    scatter_std, scatter_max, 
    scatter_min
)

'''
Implimenting TGBase temporal graph feature extractor from 
    https://epubs.siam.org/doi/pdf/10.1137/1.9781611977172.73

In the paper they beat SoTA on the BTC data set in terms of AUC
They claim that using their feature generator boosts AUC from 
0.955 -> 0.970 on GraphSAGE models (unclear what params)
trained to be binary classifiers 
'''
class TGBase(nn.Module):
    def __init__(self):
        super().__init__()

    def entropy(self, list):
        if list.size(0) == 0:
            return 0.0
        
        _,p = list.unique(return_counts=True)
        p = p.float()
        p /= p.sum()

        ent = -p*torch.log2(p)
        return ent.sum().item()

    @torch.no_grad() # parameter-free 
    def forward(self, ei, weights=None, ts=None, num_nodes=None):
        cols = [self.struct_feats(ei, num_nodes),]
            
        if weights is not None:
            cols.append(self.edge_feats(weights, ei, num_nodes))
        if ts is not None:
            cols.append(self.edge_feats(ts, ei, num_nodes))
        
        return torch.cat(cols, dim=1)
        

    def struct_feats(self, ei, num_nodes):
        if num_nodes is None:
            num_nodes = ei.max()+1
        
        # Used for degree counting    
        x = torch.ones(ei.size(1),1, dtype=torch.float)

        kwargs = dict(dim=0, dim_size=num_nodes)

        # Individual feats 
        in_deg = scatter_add(x, ei[1].unsqueeze(-1), **kwargs)
        out_deg = scatter_add(x, ei[0].unsqueeze(-1), **kwargs)
        tot_deg = in_deg+out_deg 

        structure_cols = [in_deg, out_deg, tot_deg]
        for i,val in enumerate([in_deg, out_deg, tot_deg]):
            if i == 1:
                ei = ei[[1,0]]
            elif i == 2: 
                ei = torch.cat([ei, ei[[1,0]]], dim=1)

            src = val[ei[0]]
            dst = ei[1].unsqueeze(-1)
            args = (src,dst)

            structure_cols += [
                scatter_add(*args,**kwargs),
                scatter_mean(*args, **kwargs),
                scatter_max(*args, **kwargs)[0],
                scatter_min(*args, **kwargs)[0],
                scatter_std(*args, **kwargs)
            ]

            # There's prob a more efficient way to do this but..
            lists = [[] for _ in range(num_nodes)]
            for j in range(ei.size(1)):
                lists[ei[1][j]].append(val[ei[0][j]])
            
            ent=[]
            for l in lists:
                ent.append([self.entropy(torch.tensor(l))])
            ent = torch.tensor(ent)
            structure_cols.append(ent)

        return torch.cat(structure_cols, dim=1)
    
    def edge_feats(self, val, ei, num_nodes):
        if val.dim() == 1:
            val = val.unsqueeze(-1)

        if num_nodes is None:
            num_nodes = ei.max()+1
        kwargs = dict(dim=0, dim_size=num_nodes)

        feat_cols = []
        for i in range(3):
            if i == 1:
                ei = ei[[1,0]]
            elif i == 2: 
                ei = torch.cat([ei, ei[[1,0]]], dim=1)
                val = val.repeat(2,1)

            src = val
            dst = ei[1].unsqueeze(-1)
            args = (src,dst)

            feat_cols += [
                scatter_add(*args,**kwargs),
                scatter_mean(*args, **kwargs),
                scatter_max(*args, **kwargs)[0],
                scatter_min(*args, **kwargs)[0],
                scatter_std(*args, **kwargs)
            ]

            # There's prob a more efficient way to do this but..
            lists = [[] for _ in range(num_nodes)]
            for j in range(ei.size(1)):
                lists[ei[1][j]].append(val[j])
            
            ent=[]
            for l in lists:
                ent.append([self.entropy(torch.tensor(l))])
            ent = torch.tensor(ent)
            feat_cols.append(ent)

        return torch.cat(feat_cols, dim=1)

from torch_geometric.nn.models import GraphSAGE
class GraphSAGE_TG(nn.Module):
    def __init__(self, in_dim, hidden, out, weight=None):
        super().__init__()

        self.args = (in_dim, hidden, out)
        self.kwargs = dict(weight=weight)
        self.sage = GraphSAGE(in_dim, hidden, 3, out_channels=out)
        self.lin = nn.Sequential(
            nn.Linear(out, 2),
            nn.Softmax(dim=1)
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
    
    def predict(self, x, ei):
        zs = self.sage.forward(x, ei)
        return self.lin(zs)

    def forward(self, x, ei, batch, labels):
        preds = self.predict(x, ei)
        loss = self.loss_fn(preds[batch], labels[batch])
        return loss 


if __name__ == '__main__':
    ei = torch.tensor([
        [0,1,3,3,2],
        [1,2,4,5,1]
    ])
    ew = torch.tensor(
        [1,1,2,2,1],
        dtype=torch.float
    )
    et = torch.tensor(
        [0,0,1,1,2]
    )

    tg = TGBase()
    cols = tg.forward(ei, ew, et)

    print(cols)
    print(cols.size())