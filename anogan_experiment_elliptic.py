from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from sklearn.preprocessing import MinMaxScaler
from resources import constants
import torch
from torch.utils.data import DataLoader
from torch import nn
from models.AnoGAN.generator import Generator
from models.AnoGAN.discriminator import Discriminator
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# UPPER_BOUND = constants.LOCAL_FEATS
UPPER_BOUND = constants.FEAT_DIM + 1
TIMESTAMP_INDEX = 2
REAL_LABEL = 1
FAKE_LABEL = 0

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
local_features_matrix_actual = actual_labels_graphs.x.numpy()
graph_labels_actual = actual_labels_graphs.y.numpy()
local_features_matrix = predicted_labels_graphs.x.numpy()
graph_labels = predicted_labels_graphs.y.numpy()

training_steps = 25
batch_size = 256
input_length = int(UPPER_BOUND )
generated_illicit_nodes = []
def balanced_dataset_sampler(dataset, labels,batch_size,filter_illicit=False):
    labels = np.reshape(labels,(len(labels),-1))
    full_dataset = np.concatenate((dataset,labels),axis=1)
    illicit_rows = np.where(full_dataset[:,UPPER_BOUND]==1)[0]
    licit_rows = np.where(full_dataset[:,UPPER_BOUND]==0)[0]
    if filter_illicit:
        random_illicit_nodes = np.random.choice(illicit_rows, size=batch_size)
        random_indices = random_illicit_nodes
    else:
        random_illicit_nodes = np.random.choice(illicit_rows, size=batch_size//2)
        random_licit_nodes = np.random.choice(licit_rows, size=batch_size//2)
        random_indices = np.concatenate((random_illicit_nodes,random_licit_nodes),axis=0)
    sampled_dataset = full_dataset[random_indices]
    np.random.shuffle(sampled_dataset)
    return sampled_dataset[:,:UPPER_BOUND], sampled_dataset[:,UPPER_BOUND]

timestamps = np.unique(local_features_matrix[:,TIMESTAMP_INDEX])

z_dimension = 128
gf_dimension = 64
df_dimension = 64
c_dimension = 1
percentiles = [0,10,25,50,75,90,99]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
for timestamp in timestamps:
    generator = Generator(z_dimension, c_dimension, gf_dimension).to(device)
    discriminator = Discriminator(c_dimension, df_dimension).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    generator_losses = []
    discriminator_losses = []

    iter_features = local_features_matrix[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp][:,1:] 
    iter_labels = graph_labels[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp]

    iter_features = iter_features[iter_labels == 0]
    iter_labels = iter_labels[iter_labels == 0]

    iter_dataset = []
    for i in range(len(iter_features)):
        iter_dataset.append([iter_features[i], iter_labels[i]])
    train_dataloader = DataLoader(dataset=iter_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"TIMESTAMP: {timestamp}")
    for epoch in range(training_steps):
        epoch_generator_loss = []
        epoch_discriminator_loss = []
        epoch_discriminator_real_loss = []
        epoch_discriminator_fake_loss = []
        for i, data in enumerate(train_dataloader):
            discriminator.zero_grad()

            real_data = data[0].to(device)
            real_batch_size = real_data.size(0)
            label = torch.full((real_batch_size,), REAL_LABEL, dtype=torch.float, device=device)
            output = discriminator(real_data.unsqueeze(1))[0].view(-1)

            discriminator_real_error = criterion(output, label)
            discriminator_real_error.backward()
            
            discriminator_x = output.mean().item()

            noise = torch.randn(real_batch_size, z_dimension, device=device)

            fake_data = generator(noise)
            label.fill_(FAKE_LABEL)

            output = discriminator(fake_data.detach())[0].view(-1)
            discriminator_fake_error = criterion(output, label)
            discriminator_fake_error.backward()
            
            d_gz_1 = output.mean().item()
            discriminator_error = discriminator_real_error + discriminator_fake_error

            discriminator_optimizer.step()

            generator.zero_grad()
            label.fill_(REAL_LABEL)

            output = discriminator(fake_data)[0].view(-1)

            generator_error = criterion(output, label)
            generator_error.backward()
            g_gz_2 = output.mean().item()
            generator_optimizer.step()

            epoch_generator_loss.append(generator_error.item())
            epoch_discriminator_real_loss.append(discriminator_real_error.item())
            epoch_discriminator_fake_loss.append(discriminator_fake_error.item())
            epoch_discriminator_loss.append(discriminator_error.item())
            generator_losses.append(generator_error.item())
            discriminator_losses.append(discriminator_error.item())
        # print(f"EPOCH: {epoch} GENERATOR LOSS: {np.mean(epoch_generator_loss)} DISCRIMINATOR LOSS: {np.mean(epoch_discriminator_loss)}")
        print(f"EPOCH: {epoch} DISCRIMINATOR REAL LOSS: {np.mean(epoch_discriminator_real_loss)} DISCRIMINATOR FAKE LOSS: {np.mean(epoch_discriminator_fake_loss)}")
