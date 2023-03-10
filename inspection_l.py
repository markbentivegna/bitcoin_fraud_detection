from copy import deepcopy
import json 
import os 
from types import SimpleNamespace

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch 
from torch.optim import Adam 

from models.inspection_l_models import InspectionL
from resources.constants import LOCAL_FEATS
from utilities.dataset_util import DatasetUtilityPyTorch

# Don't hog too many resources
torch.set_num_threads(16)

# Same as paper
HYPERPARAMS = SimpleNamespace(
    epochs=300, lr=0.0001, hidden=128,
    estimators=100
)

def get_or_build_data(just_local=False, just_global=False, out_f='resources/graphs.pt', force=False):
    assert not just_global and just_local, 'Graphs have no features when just_local=True and just_global=True'
    
    if not os.path.exists(out_f) or force:
        util = DatasetUtilityPyTorch()
        data = util.build_dataset()
        graphs = util.split_subgraphs(data)

        torch.save(graphs, out_f)
    
    else:
        graphs = torch.load(out_f)

    if just_local:
        for g in graphs:
            g.x = g.x[:, :LOCAL_FEATS]

    if just_global:
        for g in graphs:
            g.x = g.x[:, LOCAL_FEATS:]

    return graphs 

def train_embedder(hp, model, batch, graphs): 
    opt = Adam(params=model.parameters(), lr=hp.lr)

    for i,graph_id in enumerate(batch):
        g = graphs[graph_id]
        
        for e in range(hp.epochs):
            model.train()
            opt.zero_grad()
            loss = model(g.x, g.edge_index)
            loss.backward()
            opt.step()

            if e % 10 == 0:
                print(f"[{i}-{e}] Loss {loss.item()}")
                out_data = (model.args, model.kwargs, model.state_dict())
                torch.save(out_data, 'saved_models/inspection_l.pt')

    out_data = (model.args, model.kwargs, model.state_dict())
    gnn = model.kwargs['gnn']
    torch.save(out_data, f'saved_models/inspection_l_{gnn}.pt')
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

def evaluate_base(hp, tr, graphs):
    rf = RandomForestClassifier(n_estimators=hp.estimators)

    tr_X, te_X = [],[]
    tr_y, te_y = [],[]
    
    for i,g in enumerate(graphs):
        x = g.x
        
        # Remove unknown labels
        x = x[g.y != 0]
        y = g.y[g.y != 0]
        y -= 2 # So illicit == -1, licit ==0
        y = -y # So licit = 0 illicit = 1

        if i in tr:
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
    print(json.dumps(stats, indent=1))
    return stats 


@torch.no_grad()
def evaluate(hp, model, tr, graphs):
    rf = RandomForestClassifier(n_estimators=hp.estimators)

    tr_X, te_X = [],[]
    tr_y, te_y = [],[]
    
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

@torch.no_grad()
def get_false_negatives(hp=HYPERPARAMS):
    args,kwargs,weights = torch.load('saved_models/inspection_l_GIN.pt')
    model = InspectionL(*args, **kwargs)
    model.load_state_dict(weights)
    graphs = get_or_build_data()
    
    rf = RandomForestClassifier(n_estimators=hp.estimators)
    tr_X, te_X = [],[]
    tr_y, te_y = [],[]
    
    te_nodes = []
    te_ts = []
    model.eval()
    tr = range(35)
    for g in graphs:
        x = model.embed(g.x, g.edge_index)

        # Append model embeddings to original features
        x = torch.cat([x,g.x], dim=1)
        
        # Remove unknown labels
        nodes_used = g.y != 0
        x = x[nodes_used]
        y = g.y[nodes_used]
        y -= 2 # So illicit == -1, licit ==0
        y = -y # So licit = 0 illicit = 1

        if g.ts in tr:
            tr_X.append(x)
            tr_y.append(y)
        else:
            te_X.append(x)
            te_y.append(y)

            # For checking out FPs
            te_nodes.append(g.nid_to_node_name[nodes_used])
            te_ts += [g.ts] * nodes_used.sum()

    # Concat list of matrices together
    tr_X = torch.cat(tr_X,dim=0)
    tr_y = torch.cat(tr_y,dim=0)
    te_X = torch.cat(te_X,dim=0)
    te_y = torch.cat(te_y,dim=0)
    te_nodes = torch.cat(te_nodes, dim=0)

    print("Fitting RF")
    rf.fit(tr_X, tr_y)

    y_hat = torch.tensor(rf.predict(te_X))

    fns = (y_hat == 0).logical_and(te_y == 1)
    fn_feats = te_X[fns]
    fn_names = te_nodes[fns]
    fn_times = torch.tensor(te_ts)[fns]

    with open('results/false_negatives.csv', 'w') as f:
        # Header
        f.write('node,ts')
        [f.write(f",f{i}") for i in range(fn_feats.size(1))]
        f.write('\n')

        for i in range(fn_feats.size(0)):
            f.write(f"{fn_names[i]},{fn_times[i]}")
            
            for feat in fn_feats[i]:
                f.write(f",{feat.item()}")
            f.write('\n')

    return fn_feats, fn_names 

def main(gnn, hp):
    # Train embedder
    print(gnn)
    if gnn == 'None': 
        return evaluate_base(hp, list(range(35)), get_or_build_data())

    data = get_or_build_data(**hp.data_kwargs)
    model = InspectionL(data[0].x.size(1), HYPERPARAMS.hidden, gnn=gnn)
    batch = list(range(35))
    loss = train_embedder(hp, model, batch, data)

    # Train supervised RF
    #args,kwargs,weights = torch.load('saved_models/inspection_l_GIN.pt')
    #model = InspectionL(*args, **kwargs)
    #model.load_state_dict(weights)

    stats = evaluate(hp, model, list(range(35)), get_or_build_data(**hp.data_kwargs))
    stats['last_loss'] = loss 

    return stats

if __name__ == '__main__':
    gnn = 'GIN'
    for kw in [{'just_local': True}, {'just_global': True}]:
        hp = deepcopy(HYPERPARAMS)
        hp.data_kwargs = kw 

        tests = [main(gnn, hp) for _ in range(10)]
        df = pd.DataFrame(tests)
        with open('results/inspection_l.txt', 'a') as f:
            f.write(f"{gnn}, {list(kw.keys())[0]}\n")
            f.write(df.to_csv())
            f.write(df.mean().to_csv())
            f.write(df.sem().to_csv())