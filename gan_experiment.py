from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from resources import constants
import torch
from torch import nn
from models.Generator import Generator
from models.Discriminator import Discriminator
import math
import numpy as np
from typing import List

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
local_features_matrix = actual_labels_graphs.x[:,2:constants.LOCAL_FEATS].numpy()
graph_labels = actual_labels_graphs.y.numpy()
def convert_float_matrix_to_int_list(float_matrix: np.array, threshold: float = 0.5):
    return [
        int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold
    ]

def generate_even_data(max_int: int, batch_size: int=16):
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data

def create_binary_list_from_int(number: int) -> List[int]:
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]

training_steps = 500
max_int = 128
# batch_size = 256
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

    # Create noisy input for generator
    # Need float type instead of int
    noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
    random_min = np.percentile(local_features_matrix,1)
    random_max = np.percentile(local_features_matrix, 99)
    random_noise = np.random.uniform(low=random_min, high=random_max, size=(batch_size, local_features_matrix.shape[1])) * torch.rand((batch_size, local_features_matrix.shape[1])).numpy()
    # generated_data = generator(noise)
    generated_data = generator(torch.tensor(random_noise).float())
    # Generate examples of even real data
    # true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
    true_data, true_labels = balanced_dataset_sampler(local_features_matrix, graph_labels)
    true_data = torch.tensor(true_data).float()
    true_labels = torch.tensor(true_labels).float().unsqueeze(1)

    # Train the generator
    # We invert the labels here and don't train the discriminator because we want the generator
    # to make things the discriminator classifies as true.
    generator_discriminator_out = discriminator(generated_data)
    generator_loss = loss(generator_discriminator_out, true_labels)
    generator_loss.backward()
    generator_optimizer.step()

    # Train the discriminator on the true/generated data
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
