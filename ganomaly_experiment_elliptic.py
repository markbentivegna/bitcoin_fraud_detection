from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from sklearn.preprocessing import MinMaxScaler
from resources import constants
import torch
from torch.utils.data import DataLoader
from torch import nn
from models.AnoGAN.generator import Generator
from models.AnoGAN.discriminator import Discriminator
from models.GANomaly.ganomaly import GANomaly
from models.GANomaly.discriminator_loss import DiscriminatorLoss
from models.GANomaly.generator_loss import GeneratorLoss
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
input_dimension = (1, local_features_matrix.shape[1] - 1)
latent_dimension = 128
input_channels_count = 1
features_count = local_features_matrix.shape[1] - 1

percentiles = [0,10,25,50,75,90,99]
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
w_adversarial = 1
w_contextual = 50
w_encoder = 1
device = "cpu"
for timestamp in timestamps:
    ganomaly = GANomaly(input_dimension, latent_dimension, input_channels_count, features_count).to(device)
    generator_loss = GeneratorLoss(w_adversarial, w_contextual, w_encoder)
    discriminator_loss = DiscriminatorLoss()
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
            batch, fake, latent_input, latent_output = ganomaly(data[0].to(device))
            fake = fake.squeeze(1)
            pred_real, _ = ganomaly.discriminator(batch)
   
            pred_fake, _ = ganomaly.discriminator(fake.detach())
            discrim_loss = discriminator_loss(pred_real, pred_fake)
        
            pred_fake, _ = ganomaly.discriminator(fake)
            gen_loss = generator_loss(latent_input, latent_output, batch, fake, pred_real, pred_fake)
            print(f"EPOCH: {epoch} DISCRIMINATOR_LOSS: {discrim_loss} GENERATOR_LOSS: {gen_loss}")