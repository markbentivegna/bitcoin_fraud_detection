from copy import deepcopy
from types import SimpleNamespace

import torch 
from torch.optim import Adam 

from inspection_l import get_or_build_data, evaluate 
from models.adagnn_models import AdaGNN

torch.set_num_threads(16)
HYPERPARAMS = SimpleNamespace(
    hidden=128, estimators=100, lambda_param=0.5,
    epochs=300, lr=0.0001,
    graphs_per_snapshot=10
)

def get_labels(y):
    # Don't train on unlabeled nodes 
    valid_mask = y != 0

    ret_y = y-2     # So illicit == -1, licit ==0
    ret_y = -ret_y  # So licit = 0 illicit = 1

    return ret_y[valid_mask].long(), valid_mask

def train_embedder(hp, model, batch, graphs): 
    opt = Adam(params=model.parameters(), lr=hp.lr)

    for e in range(hp.epochs):
        for i,graph_id in enumerate(batch):
            g = graphs[graph_id]

            # Subtract 1 so first graph has id 0
            ts_target = torch.tensor([(g.ts-1) // hp.graphs_per_snapshot])
            ts_target = ts_target.repeat(g.x.size(0)).long()
            labels, mask = get_labels(g.y)

            model.train()
            opt.zero_grad()
            y_loss, ts_loss = model(g.x, g.edge_index, ts_target, labels, mask)
            (y_loss+ts_loss).backward()
            opt.step()

        print(
            "[%d] Classifier loss: %0.3f; TS loss: %0.3f" %
            (e, y_loss.item(), ts_loss.item())
        )

        if e % 10 == 0:
            out_data = (model.args, model.kwargs, model.state_dict())
            torch.save(out_data, 'saved_models/inspection_l.pt')

    out_data = (model.args, model.kwargs, model.state_dict())
    gnn = model.kwargs['gnn']
    torch.save(out_data, f'saved_models/Ada{gnn}.pt')


def main(gnn, hp):
    data = get_or_build_data()
    model = AdaGNN(data[0].x.size(1), hp.hidden, gnn=gnn, lamb=hp.lambda_param)
    batch = list(range(35))
    loss = train_embedder(hp, model, batch, data)

    # Train supervised RF
    #args,kwargs,weights = torch.load('saved_models/inspection_l_GIN.pt')
    #model = InspectionL(*args, **kwargs)
    #model.load_state_dict(weights)

    return evaluate(hp, model, list(range(35)), get_or_build_data())


if __name__ == '__main__':
    for gnn in ['GCN', 'GAT']:
        for lamb in [1., 0.5, 0]:
            hp = deepcopy(HYPERPARAMS)
            hp.lambda_param = lamb
            tests = [main(gnn, hp) for _ in range(5)]

            df = pd.DataFrame(tests)
            with open('results/adagnn.txt', 'a') as f:
                f.write(f"\n{gnn}, lambda={lamb}\n")
                f.write(df.to_csv())
                f.write(df.mean().to_csv())
                f.write(df.sem().to_csv())