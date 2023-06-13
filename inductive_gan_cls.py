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

from models.inductive_graph_gan import GeneratorClassifier, DiscriminatorClassifier
from utilities.dataset_util import DatasetUtilityPyTorch

torch.set_num_threads(16)
HYPERPARAMS = SimpleNamespace(
    lr=1e-5, epochs=10_000, bs=512, n_gen=2,
    hidden=2048, latent=128
)

def train(hp, tr_x, tr_y, te_x, te_y, gen: GeneratorClassifier, disc: DiscriminatorClassifier):
    g_opt = Adam(gen.parameters(), lr=hp.lr)
    d_opt = Adam(disc.parameters(), lr=hp.lr)

    samples = tr_x.size(1)
    n_batches = math.ceil(samples / hp.bs)

    log = []
    i = 0 

    for e in range(hp.epochs):
        idx = torch.randperm(samples)
        tr_x = tr_x[idx]
        tr_y = tr_y[idx]

        for b in tqdm(range(n_batches)):
            mb_x = tr_x[b*hp.bs : (b+1)*hp.bs]
            mb_y = tr_y[b*hp.bs : (b+1)*hp.bs]

            # Train generator 
            g_opt.zero_grad()
            disc.requires_grad_ = False
            disc.eval()
            gen.requires_grad_ = True 
            gen.train() 

            targets = torch.zeros(mb_x.size(0)*hp.n_gen, 3)
            targets[:, 0] = 1.  

            x_gen, kld_loss = gen(targets)
            recon_loss = disc.forward(x_gen, targets)
            g_loss = recon_loss + kld_loss
            g_loss.backward()
            g_opt.step()

            # Train discriminator
            d_opt.zero_grad()
            disc.requires_grad_ = True
            disc.train()
            gen.requires_grad_ = False
            gen.eval() 

            x_gen = gen.sample(targets)
            targets = torch.zeros(mb_x.size(0)*(1+hp.n_gen), 3)
            targets[:x_gen.size(0), 2] = 1. 
            targets[x_gen.size(0) + torch.arange(mb_x.size(0)), mb_y] = 1. 
            
            # Pass discriminator real and fake edge pairs 
            # at same time 
            input_x = torch.cat([
                x_gen, mb_x  
            ], dim=0)
            
            d_loss = disc.forward(input_x, targets)
            d_loss.backward()
            d_opt.step()

            auc,ap = eval(te_x, te_y, disc)
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
def eval(x,y, disc):
    disc.eval()

    # For now, any edge involving a malicious node is classified as
    # malicious... Need some way to translate this to node classificaion
    preds = disc.inference(x)
    auc = auc_score(y, preds)
    ap = ap_score(y, preds)

    return auc,ap

def prepare_btc_data():
    g = torch.load('resources/btc_graph_unsplit.pt')

    # Remove all unk nodes and reindex s.t. 0==benign, 1==mal
    is_unk = g.y == 0 
    x = g.x[~is_unk]
    y = g.y[~is_unk]
    y[y == 2] = 0 
    y = y.long()

    ts, x = x[:,0], x[:,1:]
    tr_mask = ts < 35

    tr_x, tr_y = x[tr_mask], y[tr_mask]
    te_x, te_y = x[~tr_mask], y[~tr_mask]
    
    return tr_x, tr_y, te_x, te_y

if __name__ == '__main__':
    tr_x, tr_y, te_x, te_y = prepare_btc_data()
    disc = DiscriminatorClassifier(
        tr_x.size(1), HYPERPARAMS.hidden//4
    )
    gen = GeneratorClassifier(
        tr_x.size(1), HYPERPARAMS.hidden, HYPERPARAMS.latent
    )

    logs = train(HYPERPARAMS, tr_x, tr_y, te_x, te_y, gen, disc)
    logs = pd.DataFrame(logs)
    plt.plot(logs['iter'], logs['gen'], label='Generator')
    plt.plot(logs['iter'], logs['disc'], label='Discriminator')
    plt.ylim(top=5, bottom=0)
    plt.legend()
    plt.savefig('results/gan_loss_cls.png')

    plt.clf()
    plt.plot(logs['iter'], logs['auc'], label='AUC')
    plt.plot(logs['iter'], logs['ap'], label='AP')
    plt.ylim(top=1., bottom=0)
    plt.legend()
    plt.savefig('results/gan_stats_cls.png')

    d_params= disc.state_dict()
    g_params= gen.state_dict() 
    torch.save(
        (d_params, g_params, HYPERPARAMS.hidden, HYPERPARAMS.latent),
        'saved_models/inductive_gan_cls.pt'
    )