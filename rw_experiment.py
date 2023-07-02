from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from resources import constants
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from tqdm.notebook import tqdm
from torch_geometric.data import Data

UPPER_BOUND = constants.FEAT_DIM + 1

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
local_features_matrix = actual_labels_graphs.x[:,2:UPPER_BOUND].numpy()
graph_labels = actual_labels_graphs.y.numpy()

gan_dataset_df = pd.read_csv(f"{constants.GAN_ELLIPTIC_DATASET}", index_col=0)
actual_nodes_df = gan_dataset_df[gan_dataset_df["165"] == 0].iloc[:,:-1]
generated_nodes_df = gan_dataset_df[gan_dataset_df["165"] == 1].iloc[:,:-1]
x_tensor = torch.tensor(gan_dataset_df.to_numpy()[:,:UPPER_BOUND + 1])
edges_tensor = actual_labels_graphs.edge_index
y_tensor = torch.tensor(gan_dataset_df.to_numpy()[:,UPPER_BOUND + 1])
gan_graph = Data(x=x_tensor, edge_index=edges_tensor, y=y_tensor,num_nodes=x_tensor.size(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Node2Vec(actual_labels_graphs.edge_index.T.long(), embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10,num_negative_samples=1,p=1,q=1,sparse=True).to(device)
loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

for epoch in range(20):
    model.train()
    total_loss = 0
    for positive_random_walk, negative_random_walk in loader:
        optimizer.zero_grad()
        loss = model.loss(positive_random_walk.to(device), negative_random_walk.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    iter_loss = total_loss / len(loader)
    print(f"Epoch: {epoch:02d}, Loss: {iter_loss:.4f}")

p_walks = [p_walk for (p_walk, _) in loader]
n_walks = [n_walk for (_, n_walk) in loader]
print("foo")