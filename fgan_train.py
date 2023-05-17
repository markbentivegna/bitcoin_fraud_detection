import math 
import json 
from types import SimpleNamespace

import pandas as pd 
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
)
import torch 
from torch.optim import Adam 
from tqdm import tqdm 

from models.fgan import Generator, Discriminator
from utilities.dataset_util import DatasetUtilityPyTorch

torch.set_num_threads(16)
HYPERPARAMS = SimpleNamespace(
    d_lr=1e-5, g_lr=1e-3, epochs=1_000, batch_size=10_000,
    latent=64, g_hidden=256, g_layers=3, alpha=0.2, beta=15,
    d_hidden=512, d_layers=3, gamma=0.1
)

ILLICIT=0
LICIT=1

def train(hp, gen, disc, x_tr, y_tr, x_te, y_te):
    g_opt = Adam(gen.parameters(), lr=hp.g_lr)
    d_opt = Adam(disc.parameters(), lr=hp.d_lr)

    n_batches = math.ceil(x_tr.size(0) / hp.batch_size)
    stats=[]
    for e in range(hp.epochs):
        for i in tqdm(range(n_batches)):
            st = hp.batch_size*i 
            en = st + hp.batch_size 
            
            xb = x_tr[st:en]
            yb = y_tr[st:en]

            y_new = torch.zeros(yb.size(0)*2, dtype=torch.long)
            y_new[:yb.size(0)] = yb
            y_new[yb.size(0):] = ILLICIT # For generated anomalies

            # As part of fence gan, weigh preds on generated data less
            weight = torch.tensor(
                [1.]*yb.size(0) + 
                [disc.gamma]*yb.size(0)
            )

            # Train generator
            g_opt.zero_grad()
            disc.requires_grad_(False)
            disc.eval()
            gen.requires_grad_(True)
            gen.train()
            
            x_hat = gen(xb.size(0))
            preds = disc.inference(x_hat)[:,LICIT:] # Try to appear legitimate 
            g_loss = gen.loss_fn(x_hat, preds)
            g_loss.backward()
            g_opt.step() 

            # Train discriminator
            d_opt.zero_grad()
            disc.requires_grad_(True)
            disc.train()
            gen.requires_grad_(False)
            gen.eval()
            
            fake = gen(xb.size(0))
            x = torch.cat([xb, fake], dim=0)
            d_loss = disc(x, y_new, weight)
            d_loss.backward()
            d_opt.step()

        print(f"[{e}] Gen: {g_loss.item():0.4f}\tDisc: {d_loss.item():0.4f}")
        stat = evaluate(disc, x_te, y_te)
        print(json.dumps(stat, indent=1))
        stats.append(stat)

    return pd.DataFrame(stats)

@torch.no_grad()
def evaluate(disc, x, y):
    preds = disc.inference(x)[:,ILLICIT]
    y_hat = (preds>0.5).long()

    return dict(
        pr=precision_score(y, y_hat),
        re=recall_score(y, y_hat),
        f1=f1_score(y, y_hat),
        auc=roc_auc_score(y, preds),
        ap=average_precision_score(y, preds)
    )

def main(hp):
    tr_x = []; tr_y = []
    te_x = []; te_y = []

    data = torch.load('resources/graphs.pt')
    for i,d in enumerate(data):
        valid = d.y != 0
        x = d.x[valid]
        y = d.y[valid]

        # Make y s.t. 0 == ILLICIT, 1 == LICIT
        # Default is 1==illicit, 2==licit
        y -= 1

        if i >= 35:
            te_x.append(x)
            te_y.append(y)
        else:
            tr_x.append(x)
            tr_y.append(y)
    
    tr_x = torch.cat(tr_x)
    tr_y = torch.cat(tr_y)
    te_x = torch.cat(te_x)
    te_y = torch.cat(te_y)


    gen = Generator(
        hp.latent, hp.g_hidden, tr_x.size(1), 
        layers=hp.g_layers, alpha=hp.alpha, beta=hp.beta
    )
    disc = Discriminator(
        tr_x.size(1), hp.d_hidden, 2, 
        layers=hp.d_layers, gamma=hp.gamma
    )
    stats = train(hp, gen, disc, tr_x, tr_y, te_x, te_y)

    with open('results/fgan.csv', 'w+') as f:
        f.write(stats.to_csv())

main(HYPERPARAMS)