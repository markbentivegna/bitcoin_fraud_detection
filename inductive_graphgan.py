import math 
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import (
    roc_auc_score as auc_score, average_precision_score as ap_score
)
import torch 
from torch.optim import Adam 
from torch_geometric.data import Data 
from tqdm import tqdm 

from models.inductive_graph_gan import Generator, Discriminator
from utilities.dataset_util import DatasetUtilityPyTorch

torch.set_num_threads(16)
HYPERPARAMS = SimpleNamespace(
    lr=1e-4, epochs=100, bs=512, n_gen=1,
    hidden=512, latent=64
)

def train(hp, g, g_te, gen: Generator, disc: Discriminator):
    g_opt = Adam(gen.parameters(), lr=hp.lr/10)
    d_opt = Adam(disc.parameters(), lr=hp.lr)

    samples = g.edge_index.size(1)
    n_batches = math.ceil(samples / hp.bs)

    log = []
    i = 0 

    for e in range(hp.epochs):
        idx = torch.randperm(samples)
        for b in tqdm(range(n_batches)):
            src,dst = g.edge_index[:,idx][:, b*hp.bs : (b+1)*hp.bs]
            x_src = g.x[src]
            x_dst = g.x[dst]

            # Train generator 
            g_opt.zero_grad()
            disc.requires_grad_ = False
            disc.eval()
            gen.requires_grad_ = True 
            gen.train() 

            x_gen, kld_loss = gen(x_src)
            recon_loss = disc.forward(
                torch.cat([x_src, x_gen], dim=1), 
                torch.zeros((x_src.size(0),1))
            )
            g_loss = recon_loss + kld_loss
            g_loss.backward()
            g_opt.step()

            # Train discriminator
            d_opt.zero_grad()
            disc.requires_grad_ = True
            disc.train()
            gen.requires_grad_ = False
            gen.eval() 

            x_gen = gen.sample(x_src)

            # Pass discriminator real and fake edge pairs 
            # at same time 
            input_x = torch.cat([
                torch.cat([x_src, x_gen], dim=1),
                torch.cat([x_src, x_dst], dim=1)
            ], dim=0)
            
            labels = torch.zeros((input_x.size(0),1))
            labels[:x_src.size(0)] = 1
            
            d_loss = disc.forward(input_x, labels)
            d_loss.backward()
            d_opt.step()

            auc,ap = eval(g_te, disc)
            log.append({
                'gen': g_loss.item(),
                'disc': d_loss.item(), 
                'iter': i,
                'auc': auc, 
                'ap': ap
            })
            i += 1
        
        print(f'[{e}] G-loss: {g_loss.item():0.3f}, D-loss: {d_loss.item():0.3f}')
        print(f'\tAUC: {auc:0.4f}, AP: {ap:0.4f}')

    return log 

@torch.no_grad()
def eval(g, disc):
    disc.eval()
    input_x = torch.cat([
        g.x[g.edge_index[0]],
        g.x[g.edge_index[1]]
    ], dim=1) 

    # For now, any edge involving a malicious node is classified as
    # malicious... Need some way to translate this to node classificaion
    labels = torch.zeros((g.edge_index.size(1), 1))
    is_mal = (g.y[g.edge_index[0]] == 1).logical_or(g.y[g.edge_index[1]] == 1)
    labels[is_mal] = 1. 

    preds = disc.inference(input_x)
    auc = auc_score(labels, preds)
    ap = ap_score(labels, preds)

    return auc,ap


def prepare_btc_data():
    g = torch.load('resources/btc_graph_unsplit.pt')
    ts, x = g.x[:,0], g.x[:,1:]
    g.x = x; g.ts = ts 

    tr_mask = g.ts[g.edge_index[0]] < 35
    
    tr_ei = g.edge_index[:, tr_mask]
    te_ei = g.edge_index[:, ~tr_mask]

    # Only want benign/unknown interactions in tr set
    is_mal = (g.y[tr_ei[0]] == 1).logical_or(g.y[tr_ei[1]] == 1)
    
    # Give the malicious ones to the test dataset 
    mal_tr = tr_ei[:, is_mal]
    te_ei = torch.cat([mal_tr, te_ei], dim=1)

    tr_ei = tr_ei[:, ~is_mal]

    # Remove unknowns from te for more clear results
    is_unk = (g.y[te_ei[0]] == 0).logical_or(g.y[te_ei[1]] == 0)
    te_ei = te_ei[:, ~is_unk]

    tr_g = Data(g.x, edge_index=tr_ei)
    te_g = Data(g.x, edge_index=te_ei, y=g.y)

    return tr_g, te_g

if __name__ == '__main__':
    tr_g, te_g = prepare_btc_data()
    disc = Discriminator(tr_g.x.size(1)*2, HYPERPARAMS.hidden)
    gen = Generator(tr_g.x.size(1), HYPERPARAMS.hidden, HYPERPARAMS.latent)

    logs = train(HYPERPARAMS, tr_g, te_g, gen, disc)
    logs = pd.DataFrame(logs)
    plt.plot(logs['iter'], logs['gen'], label='Generator')
    plt.plot(logs['iter'], logs['disc'], label='Discriminator')
    plt.ylim(top=5, bottom=0)
    plt.legend()
    plt.savefig('results/gan_loss.png')

    plt.clf()
    plt.plot(logs['iter'], logs['auc'], label='AUC')
    plt.plot(logs['iter'], logs['ap'], label='AP')
    plt.ylim(top=1., bottom=0)
    plt.legend()
    plt.savefig('results/gan_stats.png')

    d_params= disc.state_dict()
    g_params= gen.state_dict() 
    torch.save(
        (d_params, g_params, HYPERPARAMS.hidden, HYPERPARAMS.latent),
        'saved_models/inductive_gan.pt'
    )