import json 
from types import SimpleNamespace

import pandas as pd 
import torch 
from torch.optim import Adam 
from torch_cluster import random_walk
from tqdm import tqdm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from models.rw_gan import RWDiscriminator, RWGenerator

torch.set_num_threads(16)
GRAPH_FILE = 'resources/graphs.pt'
ILLICIT = 1
LICIT = 2

X_DIM = 165 
HP = SimpleNamespace(
    g_lr = 1e-3, d_lr = 1e-3, epochs=1000, 
    hidden=512, heads=4, wl = 5, repeat=10
)

def sample_rw(gid, walk_len=5, samples=1):
    graphs = torch.load(GRAPH_FILE)
    g = graphs[gid]
    ei = g.edge_index

    nodes_of_interest = (g.y == ILLICIT).nonzero().squeeze(-1).repeat(samples)
    
    # Switiching row/col because walking "backward" s.t. next step is a 
    # parent node where walk ends with node of interest 
    rws = random_walk(ei[1], ei[0], nodes_of_interest, walk_len)
    
    # Change the order s.t. its a series of nodes ending in node of interest 
    rws = rws.flip(1)
    return g.x[rws]


def train(hp, gen, disc, graph_ids):
    g_opt = Adam(gen.parameters(), lr=hp.g_lr)
    d_opt = Adam(disc.parameters(), lr=hp.d_lr)
    stats = []

    for e in range(hp.epochs):
        for gid in graph_ids:
            # Train generator
            gen.train(); disc.eval()
            disc.require_grad = False 
            gen.require_grad = True 
            g_opt.zero_grad()
            
            rw = sample_rw(gid, walk_len=hp.wl, samples=hp.repeat)
            last_nodes = gen(rw[:,:-1,:])
            gen_rw = torch.cat([rw[:,:-1,:], last_nodes], dim=1)
            
            g_loss = disc(gen_rw, torch.zeros(gen_rw.size(0),1))
            g_loss.backward()
            g_opt.step() 

            # Train discriminator
            gen.eval(); disc.train() 
            disc.require_grad = True 
            gen.require_grad = False 
            d_opt.zero_grad()

            rw = sample_rw(gid, walk_len=hp.wl)
            gen_rw = sample_rw(gid, walk_len=hp.wl)
            last_nodes = gen(gen_rw)
            gen_rw = torch.cat([rw[:,:-1,:], last_nodes], dim=1)
            both_rw = torch.cat([rw, gen_rw], dim=0)

            labels = torch.zeros(rw.size(0)+gen_rw.size(0),1)
            labels[rw.size(0):] = 1. 
            d_loss = disc(both_rw, labels)
            d_loss.backward()
            d_opt.step() 

            print(f'[{e}-{gid}] D-loss: {d_loss.item():0.3f}\tG-loss: {g_loss.item():0.3f}')

        if e % 10 == 0:
            s = eval(gen, hp)
            s['g-loss'] = g_loss.item()
            s['d-loss'] = d_loss.item()
            s['epoch'] = e 
            stats.append(s)

            with open('results/rw_gan.csv', 'w+') as f:
                df = pd.DataFrame(stats)
                df.to_csv(f, index=False)

@torch.no_grad()
def eval(gen, hp):
    gen.eval() 

    tr_X, tr_y = [],[]
    te_X, te_y = [],[]
    test_starts = 35

    graphs = torch.load(GRAPH_FILE)

    print("Building dataset")
    for g in tqdm(graphs):
        mask = (g.y == ILLICIT).logical_or(g.y == LICIT)
        x = g.x[mask]
        y = g.y[mask]
        
        # Set s.t. illicit == 0 instead of 2 for easier AUC calculation later
        y[y == LICIT] = 0 

        # Add synthetic nodes to training set
        if g.ts < test_starts: 
            # Keep adding synthetic nodes until about even
            while y.mean() < 0.5: 
                new_nodes = gen(sample_rw(int(g.ts), walk_len=hp.wl))
                x = torch.cat([x, new_nodes.squeeze(1)], dim=0)
                y = torch.cat([y, torch.ones(new_nodes.size(0))])

            tr_X.append(x)
            tr_y.append(y) 
        
        else: 
            te_X.append(x)
            te_y.append(y)

    tr_X = torch.cat(tr_X, dim=0)
    tr_y = torch.cat(tr_y)
    te_X = torch.cat(te_X, dim=0)
    te_y = torch.cat(te_y)

    print("Training classifier")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=16)
    rf.fit(tr_X, tr_y)
    y_hat = rf.predict(te_X)
    preds = rf.predict_proba(te_X)[:, 1]

    stats = dict(
        acc=accuracy_score(te_y, y_hat), 
        f1=f1_score(te_y, y_hat),
        pr=precision_score(te_y, y_hat),
        re=recall_score(te_y, y_hat),
        auc=roc_auc_score(te_y, preds)
    )

    print(json.dumps(stats, indent=2))
    return stats 

def main(hp):
    disc = RWDiscriminator(X_DIM, hp.hidden, hp.heads)
    gen = RWGenerator(X_DIM, hp.hidden, hp.heads)

    train(hp, gen, disc, range(34))


if __name__ == '__main__':
    main(HP)