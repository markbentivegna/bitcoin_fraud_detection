from utilities.dataset_util import DatasetUtility
from resources import constants
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv

dataset_util = DatasetUtility()
dataset = dataset_util.get_dataset()
# edge_list = dataset_util.get_edge_list().to_numpy()

# N = constants.N_NODES
# adj_matrix = np.zeros((N,N))
# adj_matrix[edge_list[:,0], edge_list[:,1]] = 1

# pytorch_dataset_util = DatasetUtilityPyTorch()

# dataset = pytorch_dataset_util.build_dataset()


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 16)
        self.conv3 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.long().T

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].long())
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
print(f"Number of illicit predictions: {(pred[data.test_mask] == 0).sum().item()}")
print(f"Number of licit predictions: {(pred[data.test_mask] == 1).sum().item()}")
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')