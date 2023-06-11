from torch_geometric.nn import GINConv, GCNConv, GATConv
import torch
from torch import nn
from models.GinAggrNet import GINAggrNet
from models.ReverseGradient import ReverseGradient

class AdaGNN(nn.Module):
    def __init__(self, input_dimension, hidden_layers, output_dimension, timestmap_outputs=5, gnn="GAT"):
        super().__init__()

        if gnn == 'GIN':
            self.conv_layer1 = GINConv(GINAggrNet(input_dimension, hidden_layers=hidden_layers))
            self.conv_layer2 = GINConv(GINAggrNet(hidden_layers, hidden_layers=hidden_layers, output_dimension=output_dimension))
        elif gnn == 'GCN':
            self.conv_layer1 = GCNConv(input_dimension, hidden_layers)
            self.conv_layer2 = GCNConv(hidden_layers, output_dimension)
        elif gnn == 'GAT':
            self.conv_layer1 = GATConv(input_dimension, hidden_layers//8, heads=8)
            self.conv_layer2 = GATConv(hidden_layers, output_dimension, heads=8, concat=False)

        self.timestamp_predictor = nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(), 
            nn.Linear(hidden_layers, timestmap_outputs)
        )

        self.class_predictor = nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(), 
            nn.Linear(hidden_layers, 2)
        )

        self.regularization_parameter = 1.0
        self.reverse_gradient = ReverseGradient
        self.loss_function = nn.CrossEntropyLoss()

    def embed(self, x, edge_index):
        x = self.conv_layer1(x, edge_index)
        x = self.conv_layer2(x, edge_index)
        return x
    
    def forward(self, x, edge_index, timestamp_target, node_target, node_mask):
        x_embedded = self.embed(x, edge_index)

        timestamp_prediction = self.timestamp_predictor(self.reverse_gradient.apply(x_embedded))
        class_predictions = self.class_predictor(x_embedded)

        timestamp_loss = self.loss_function(timestamp_prediction, timestamp_target)
        classification_loss = self.loss_function(class_predictions[node_mask], node_target[node_mask].type(torch.LongTensor))

        return classification_loss + self.regularization_parameter*timestamp_loss 