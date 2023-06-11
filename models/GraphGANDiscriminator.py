import torch
from torch import nn
import numpy as np

class GraphGANDiscriminator(nn.Module):
    def __init__(self, n_node, initial_node_embedding):
        super().__init__()
        self.n_node = n_node
        self.initial_node_embedding = torch.from_numpy(initial_node_embedding)
        self.embedding_matrix = nn.Parameter(torch.FloatTensor(initial_node_embedding.shape))
        self.bias_vector = nn.Parameter(torch.FloatTensor(n_node))

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_matrix.data = self.initial_node_embedding
        self.bias_vector.data = torch.zeros([self.n_node])

    def forward(self, node_id, node_neighbor_id):
        node_embedding = self.embedding_matrix[node_id]
        node_neighbor_embedding = self.embedding_matrix[node_neighbor_id]
        bias = self.bias_vector[node_neighbor_id]
        score = torch.sum(node_embedding * node_neighbor_embedding, dim=1) + bias
        score = torch.clamp(score, -10, 10)
        
        return score, node_embedding, node_neighbor_embedding, bias

    def get_reward(self, score):
        reward = torch.log(1 + torch.exp(score))
        return reward