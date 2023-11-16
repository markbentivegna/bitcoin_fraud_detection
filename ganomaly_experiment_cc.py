from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from resources import constants
import torch
from torch import nn
import os
from models.GANomaly.generator import Generator
from models.GANomaly.discriminator import Discriminator
from models.GANomaly.generator_loss import GeneratorLoss
from models.GANomaly.discriminator_loss import DiscriminatorLoss
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, accuracy_score

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
creditcard_df = dataset_util.load_dataset(constants.CREDITCARD_DATASET)
creditcard_dataset = creditcard_df.drop("label",axis=1).to_numpy()
creditcard_labels = creditcard_df["label"].to_numpy()
batch_size = 256
training_steps = 1
LABEL_INDEX = 30
LICIT_LABEL = 0
ILLICIT_LABEL = 1
input_length = creditcard_df.shape[1] - 1

input_dimension = (1, input_length)
latent_dimension = 128
input_channels_count = 1
features_count = input_length

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

generator = Generator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)
discriminator = Discriminator(input_dimension, input_channels_count, features_count).to(device)
generator_loss = GeneratorLoss()
discriminator_loss = DiscriminatorLoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)


iter_features = creditcard_dataset[creditcard_labels == LICIT_LABEL]
iter_labels = creditcard_labels[creditcard_labels == LICIT_LABEL]
scaler = MinMaxScaler()
scaler.fit(iter_features)
iter_features = scaler.transform(iter_features)

iter_dataset = []
for i in range(len(iter_features)):
    iter_dataset.append([iter_features[i], iter_labels[i]])

train_dataloader = DataLoader(dataset=iter_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model_filename = f"saved_models/GANomaly_generator_cc_{training_steps}.pt"
if os.path.isfile(model_filename):
    weights = torch.load(model_filename)
    generator.load_state_dict(weights)
else:
    for epoch in range(training_steps):
        for i, data in enumerate(train_dataloader):
            generator_optimizer.zero_grad()
            batch, fake, latent_input, latent_output = generator(torch.Tensor.float(data[0]).to(device))
            fake = fake.squeeze(1)
            pred_fake, _ = discriminator(fake.detach())
            pred_real, _ = discriminator(batch.detach())
            gen_loss = generator_loss(latent_input, latent_output, batch, fake, pred_real.detach(), pred_fake.detach())
            gen_loss.backward()
            generator_optimizer.step()

            pred_fake, _ = discriminator(fake.detach())
            discrim_loss = discriminator_loss(pred_real, pred_fake)
            discrim_loss.backward()
            discriminator_optimizer.step()
            print(f"EPOCH: {epoch} DISCRIMINATOR_LOSS: {discrim_loss} GENERATOR_LOSS: {gen_loss}")
        if epoch % 25 == 0 or epoch == training_steps - 1:
            continue
            # torch.save(generator.state_dict(), f"saved_models/GANomaly_generator_cc_{epoch + 1}.pt")
            # torch.save(discriminator.state_dict(), f"saved_models/GANomaly_discriminator_cc_{epoch + 1}.pt")

anomaly_score = nn.SmoothL1Loss()
_, _, licit_latent_input, licit_latent_output = generator(torch.Tensor(iter_features[iter_labels == LICIT_LABEL]).to(device))
licit_anomaly_score = anomaly_score(licit_latent_input, licit_latent_output)

_, _, illicit_latent_input, illicit_latent_output = generator(torch.Tensor(iter_features[iter_labels == ILLICIT_LABEL]).to(device))
illicit_anomaly_score = anomaly_score(illicit_latent_input, illicit_latent_output)
print(f"LICIT ANOMALY SCORE: {licit_anomaly_score}")
print(f"ILLICIT ANOMALY SCORE: {illicit_anomaly_score}")