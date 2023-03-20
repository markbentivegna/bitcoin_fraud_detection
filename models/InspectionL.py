from torch_geometric.nn import GINConv, GCNConv, GATConv
import torch.nn as nn
import torch
from models.GinAggrNet import GINAggrNet

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
    
    def forward(self, x, edge_index):
        x_real = self.embed(x,edge_index)
        x_corrupted = self.embed(self.corrupt(x), edge_index)

        s = self.readout(x_real)
        real_loss = self.discriminate(x_real, s)
        corrupted_loss = self.discriminate(x_corrupted, s)

        targets = torch.zeros(real_loss.size(0)+corrupted_loss.size(0), 1)
        targets[:real_loss.size(0)] = 1.

        loss = self.bce(torch.cat([real_loss, corrupted_loss]), targets)
        return loss

