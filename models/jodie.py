import torch.nn as nn
from torch.nn import functional as F
import torch
import math

class NormalLinear(nn.Linear):
    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.normal_(0, std)


class Jodie(nn.Module):
    def __init__(self, input_dimension, embedding_dimension, users_count, items_count):
        super().__init__()
        self.initial_item_embedding = nn.Parameter(torch.Tensor(embedding_dimension))
        self.initial_user_embedding = nn.Parameter(torch.Tensor(embedding_dimension))

        self.item_rnn = nn.RNNCell(input_dimension + embedding_dimension + 1, embedding_dimension)
        self.user_rnn = nn.RNNCell(input_dimension + embedding_dimension + 1, embedding_dimension)

        self.linear_layer1 = nn.Linear(embedding_dimension, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(users_count + items_count + embedding_dimension * 2, items_count + embedding_dimension)
        self.embedding_layer = NormalLinear(1, embedding_dimension)

    def forward(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select == 'item_update':
            input = torch.cat([user_embeddings, timediffs, features], dim=1)
            item_embedding_output = self.item_rnn(input, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input = torch.cat([item_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input, user_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs.to(torch.float)))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out
    