from copy import deepcopy
import json 
import os 
from types import SimpleNamespace

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch 
from torch.optim import Adam 

from models.gnn_classifier import SimpleGNNClassifier
from utilities.dataset_util import DatasetUtilityPyTorch

# Don't hog too many resources
torch.set_num_threads(16)

# Same as paper
HYPERPARAMS = SimpleNamespace(
    epochs=300, lr=0.0001, hidden=128,
    estimators=100
)

def get_or_build_data(out_f='resources/graphs.pt', force=False):
    if not os.path.exists(out_f) or force:
        util = DatasetUtilityPyTorch()
        data = util.build_dataset()
        graphs = util.split_subgraphs(data)
        torch.save(graphs, out_f)
    
    else:
        graphs = torch.load(out_f)

    return graphs 

def train(hp, model, batch, graphs): 
    opt = Adam(params=model.parameters(), lr=hp.lr)

    for i,graph_id in enumerate(batch):
        g = graphs[graph_id]
        targets = (g.y - 1).long() # Licit == 1, Illicit == 0
        mask = g.y.nonzero().squeeze(-1)

        for e in range(hp.epochs):
            model.train()
            opt.zero_grad()
            loss = model(g.x, g.edge_index, mask, targets)
            loss.backward()
            opt.step()

            if e % 10 == 0:
                print(f"[{i}-{e}] Loss {loss.item()}")
                out_data = (model.args, model.kwargs, model.state_dict())
                torch.save(out_data, 'saved_models/simple.pt')

    out_data = (model.args, model.kwargs, model.state_dict())
    gnn = model.kwargs['gnn']
    torch.save(out_data, f'saved_models/simple_{gnn}.pt')
    return loss.item()

def get_metrics(y, y_hat, preds, text=''):
    '''
    Return dict of metrics given 
        y: ground truth
        y_hat: predicted labels
        preds: prediction scores (float 0,1)
    '''

    print(confusion_matrix(y, y_hat))

    return {
        'pr'+text: precision_score(y, y_hat),
        're'+text: recall_score(y, y_hat),
        'f1'+text: f1_score(y, y_hat),
        'auc'+text: roc_auc_score(y, preds)
    }


@torch.no_grad()
def evaluate(hp, model, tr, graphs):
    rf = RandomForestClassifier(n_estimators=hp.estimators)

    tr_X, te_X = [],[]
    tr_y, te_y = [],[]
    pred_y = []
    
    model.eval()
    for g in graphs:
        x = model.embed(g.x, g.edge_index)

        # Append model embeddings to original features
        x = torch.cat([x,g.x], dim=1)
        
        # Remove unknown labels
        x = x[g.y != 0]
        y = g.y[g.y != 0]
        y -= 2 # So illicit == -1, licit ==0
        y = -y # So licit = 0 illicit = 1

        if g.ts in tr:
            tr_X.append(x)
            tr_y.append(y)
        else:
            te_X.append(x)
            te_y.append(y)

    # Concat list of matrices together
    tr_X = torch.cat(tr_X,dim=0)
    tr_y = torch.cat(tr_y,dim=0)
    te_X = torch.cat(te_X,dim=0)
    te_y = torch.cat(te_y,dim=0)

    print("Fitting RF")
    rf.fit(tr_X, tr_y)

    y_hat = rf.predict(te_X)
    preds = rf.predict_proba(te_X)[:,1]
    stats = get_metrics(te_y, y_hat, preds)

    print("Fitting RF")
    rf.fit(tr_X[:,:model.out_dim], tr_y)

    y_hat = rf.predict(te_X[:,:model.out_dim])
    preds = rf.predict_proba(te_X[:, :model.out_dim])[:,1]
    stats.update(get_metrics(te_y, y_hat, preds, text='-just_graph'))
    
    print(json.dumps(stats, indent=1))
    return stats

data = get_or_build_data()
model = SimpleGNNClassifier(data[0].x.size(1), HYPERPARAMS.hidden, gnn='GCN')
batch = list(range(35))
loss = train(HYPERPARAMS, model, batch, data)
evaluate(HYPERPARAMS, model, batch, data)