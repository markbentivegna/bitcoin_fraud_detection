from utilities.dataset_util import DatasetUtility, DatasetUtilityPyTorch
from resources import constants
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim import Adam 
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GCNConv, GATConv
import torch.nn.functional as F
import time

start_time = time.time()
dataset_util = DatasetUtility()
dataset = dataset_util.get_dataset(filter_labeled=True)
subgraphs = dataset_util.split_subgraphs(dataset)
end_time = time.time()
print(f"Took {end_time-start_time} to compute subgraphs")

class GINAggrNet(nn.Module):
    def __init__(self, input_dimension, hidden_layers=128, output_dimension=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dimension, hidden_layers),
            nn.BatchNorm1d(hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, output_dimension),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)

class InspectionL(nn.Module):
    def __init__(self, input_dimension, hidden_layers, output_dimension, gnn):
        super().__init__()
        self.output_dimension = output_dimension
        
        if gnn == 'GIN':
            self.conv_layer1 = GINConv(GINAggrNet(input_dimension, hidden_layers=hidden_layers))
            self.conv_layer2 = GINConv(GINAggrNet(hidden_layers, hidden_layers=hidden_layers, output_dimension=output_dimension))
        elif gnn == 'GCN':
            self.conv_layer1 = GCNConv(input_dimension, hidden_layers)
            self.conv_layer2 = GCNConv(hidden_layers, output_dimension)
        elif gnn == 'GAT':
            self.conv_layer1 = GATConv(input_dimension, hidden_layers//8, heads=8)
            self.conv_layer2 = GATConv(hidden_layers, output_dimension, heads=8, concat=False)

        self.bce = nn.BCELoss()
        self.discriminator = nn.Linear(output_dimension, output_dimension, bias=False)

    def embed(self, x, edge_index):
        x = self.conv_layer1(x, edge_index)
        x = self.conv_layer2(x, edge_index)
        return x
    
    def readout(self, x):
        return torch.sigmoid(
            torch.mean(x, dim=0, keepdim=True)
        )
    
    def discriminate(self, x, s):
        return torch.sigmoid(
            self.discriminator(x) @ s.T
        )
    
    def corrupt(self, x):
        perm = torch.randperm(x.size(0))
        return x[perm]
    
    def forward(self, x, ei):
        x_real = self.embed(x,ei)
        x_corrupted = self.embed(self.corrupt(x), ei)

        s = self.readout(x_real)
        real_loss = self.discriminate(x_real, s)
        corrupted_loss = self.discriminate(x_corrupted, s)

        targets = torch.zeros(real_loss.size(0)+corrupted_loss.size(0), 1)
        targets[:real_loss.size(0)] = 1.

        loss = self.bce(torch.cat([real_loss, corrupted_loss]), targets)
        return loss

    
model = InspectionL(subgraphs[0].x.size(1), 128, 128, gnn="GIN")

optimizer = Adam(params=model.parameters(), lr=0.0001)

for graph_id in range(35):
    graph = subgraphs[graph_id]
        
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        loss = model(graph.x, graph.edge_index)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[{graph_id}-{epoch}] Loss {loss.item()}")

train_graphs, test_graphs = train_test_split(subgraphs, train_size = 0.7, shuffle=False)

random_forest = RandomForestClassifier(n_estimators=100)
x_train, y_train, x_test, y_test = [], [], [], []
for graph in train_graphs:
    x_train.append(model.embed(graph.x, graph.edge_index))
    y_train.append(graph.y)
for graph in test_graphs:
    x_test.append(model.embed(graph.x, graph.edge_index))
    y_test.append(graph.y)

x_train = torch.cat(x_train,dim=0)
y_train = torch.cat(y_train,dim=0)
x_test = torch.cat(x_test,dim=0)
y_test = torch.cat(y_test,dim=0)

model.eval()

random_forest.fit(x_train.detach().numpy(), y_train.detach().numpy())

y_hat = random_forest.predict(x_test.detach().numpy())
predictions = random_forest.predict_proba(x_test.detach().numpy())[:,1]
precision = precision_score(y_test, y_hat)
recall = recall_score(y_test, y_hat)
f1 = f1_score(y_test, y_hat)
roc_auc = roc_auc_score(y_test, predictions)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
print(f"roc_auc: {roc_auc}")
print(confusion_matrix(y_test, y_hat))