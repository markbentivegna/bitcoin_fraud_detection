from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from resources import constants
import torch
from torch import nn
from models.Generator import Generator
from models.Discriminator import Discriminator
import numpy as np
from typing import List

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
local_features_matrix = actual_labels_graphs.x[:,2:constants.LOCAL_FEATS].numpy()
graph_labels = actual_labels_graphs.y.numpy()

training_steps = 500
max_int = 128
batch_size = 1024
input_length = int(constants.LOCAL_FEATS - 2)

generator = Generator(input_length)
discriminator = Discriminator(input_length)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

def balanced_dataset_sampler(local_features_matrix, graph_labels):
    graph_labels = np.reshape(graph_labels,(len(graph_labels),-1))
    full_dataset = np.concatenate((local_features_matrix,graph_labels),axis=1)
    illicit_rows = np.where(full_dataset[:,91]==1)[0]
    licit_rows = np.where(full_dataset[:,91]==0)[0]
    random_illicit_nodes = np.random.choice(illicit_rows, size=batch_size//2)
    random_licit_nodes = np.random.choice(licit_rows, size=batch_size//2)
    random_indices = np.concatenate((random_illicit_nodes,random_licit_nodes),axis=0)
    sampled_dataset = full_dataset[random_indices]
    np.random.shuffle(sampled_dataset)
    return sampled_dataset[:,:91], sampled_dataset[:,91]


loss = nn.BCELoss()
for i in range(training_steps):
    generator_optimizer.zero_grad()

    random_min = np.percentile(local_features_matrix,1)
    random_max = np.percentile(local_features_matrix, 99)
    random_noise = np.random.uniform(low=random_min, high=random_max, size=(batch_size, local_features_matrix.shape[1])) * torch.rand((batch_size, local_features_matrix.shape[1])).numpy()
    generated_data = generator(torch.tensor(random_noise).float())
    true_data, true_labels = balanced_dataset_sampler(local_features_matrix, graph_labels)
    true_data = torch.tensor(true_data).float()
    true_labels = torch.tensor(true_labels).float().unsqueeze(1)

    # Train the generator
    generator_discriminator_out = discriminator(generated_data)
    generator_loss = loss(generator_discriminator_out, true_labels)
    generator_loss.backward()
    generator_optimizer.step()

    # Train the discriminator
    discriminator_optimizer.zero_grad()
    true_discriminator_out = discriminator(true_data)
    true_discriminator_loss = loss(true_discriminator_out, true_labels)

    generator_discriminator_out = discriminator(generated_data.detach())
    generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size).unsqueeze(1))
    discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
    discriminator_loss.backward()
    discriminator_optimizer.step()
    if i % 100 == 0:
        print(f"train step {i} has discriminator_loss: {discriminator_loss}")
        print(f"train step {i} has generator_discriminator_loss: {generator_discriminator_loss}")
        print(f"train step {i} has true_discriminator_loss: {true_discriminator_loss}")
        print(f"train step {i} has generator_loss: {generator_loss}")

    if i == training_steps - 1:
        print("foo")
