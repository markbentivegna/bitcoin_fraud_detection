from copy import deepcopy
import json 
import os 
from types import SimpleNamespace

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch 
from torch.optim import Adam 
from tqdm import tqdm 

from models.tgbase import TGBase, GraphSAGE_TG
from utilities.dataset_util import DatasetUtilityPyTorch

# Don't hog too many resources
torch.set_num_threads(16)

# Same as paper
HYPERPARAMS = SimpleNamespace(
    epochs=300, lr=0.0001, hidden=128,
    estimators=100
)

def get_or_build_data(out_f='resources/graphs-tgbase.pt', force=False):
    if not os.path.exists(out_f) or force:
        util = DatasetUtilityPyTorch()
        data = util.build_dataset()
        graphs = util.split_subgraphs(data)

        tg = TGBase()
        for g in tqdm(graphs, desc='adding feats'):
            g.x = torch.cat([g.x, tg(g.edge_index)], dim=1)

        torch.save(graphs, out_f)
    
    else:
        graphs = torch.load(out_f)

    return graphs 

def train(hp, model, batch, graphs): 
    opt = Adam(params=model.parameters(), lr=hp.lr)

    for e in range(hp.epochs):
        for i,graph_id in enumerate(batch):
            g = graphs[graph_id]
            targets = -(g.y - 2).long() # Licit == 0, Illicit == 1
            mask = g.y.nonzero().squeeze(-1)
        
            for _ in range(10):
                model.train()
                opt.zero_grad()
                loss = model(g.x, g.edge_index, mask, targets)
                loss.backward()
                opt.step()

        print(f"[{e}] Loss {loss.item()}")
        evaluate(model, batch, [], graphs)
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

    print(text[1:], confusion_matrix(y, y_hat))

    return {
        'pr'+text: precision_score(y, y_hat),
        're'+text: recall_score(y, y_hat),
        'f1'+text: f1_score(y, y_hat),
        'auc'+text: roc_auc_score(y, preds)
    }


@torch.no_grad()
def evaluate(model, tr,va, graphs):
    tr_p, va_p, te_p = [],[],[]
    tr_y, va_y, te_y = [],[],[]
    tr_y_hat, va_y_hat, te_y_hat = [],[],[]
    
    model.eval()
    for g in graphs:
        pred = model.predict(g.x, g.edge_index)
        
        # Remove unknown labels
        pred = pred[g.y != 0]
        y = g.y[g.y != 0]
        y -= 2 # So illicit == -1, licit ==0
        y = -y # So illicit == 1, licit = 0

        p,y_hat = pred.max(dim=1)

        if g.ts in tr:
            tr_p.append(p)
            tr_y.append(y)
            tr_y_hat.append(y_hat)
        elif g.ts in va:
            va_p.append(p)
            va_y.append(y)
            va_y_hat.append(y_hat)
        else:
            te_p.append(p)
            te_y.append(y)
            te_y_hat.append(y_hat)

    stats = dict()
    if len(tr_p):
        tr_p = torch.cat(tr_p,dim=0)
        tr_y = torch.cat(tr_y,dim=0)
        tr_y_hat = torch.cat(tr_y_hat,dim=0)
        stats.update(get_metrics(tr_y, tr_y_hat, tr_p, '-tr'))

    if len(va_p):
        va_p = torch.cat(va_p,dim=0)
        va_y = torch.cat(va_y,dim=0)
        va_y_hat = torch.cat(va_y_hat,dim=0)
        stats.update(get_metrics(va_y, va_y_hat, va_p, '-va'))

    if len(te_p):
        te_p = torch.cat(te_p,dim=0)
        te_y = torch.cat(te_y,dim=0)
        te_y_hat = torch.cat(te_y_hat,dim=0)
        stats.update(get_metrics(te_y, te_y_hat, te_p, '-te'))

    print(json.dumps(stats,indent=1))
    return stats

data = get_or_build_data()
model = GraphSAGE_TG(
    data[0].x.size(1), HYPERPARAMS.hidden, HYPERPARAMS.hidden,
    weight = torch.tensor([0.11, 0.89]) # Using distros from tr/val 
)
batch = list(range(35))
loss = train(HYPERPARAMS, model, batch, data)
evaluate(model, batch,[], data)