from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from resources import constants
import torch
from torch import nn
import os
from models.AnoGAN.generator import Generator
from models.AnoGAN.discriminator import Discriminator
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
training_steps = 10
REAL_LABEL = 1
FAKE_LABEL = 0
LABEL_INDEX = 30
LICIT_LABEL = 0
ILLICIT_LABEL = 1
input_length = creditcard_df.shape[1] - 1

input_dimension = (1, input_length)
latent_dimension = 16
input_channels_count = 1
features_count = input_length

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

iter_features = creditcard_dataset[creditcard_labels == LICIT_LABEL]
iter_labels = creditcard_labels[creditcard_labels == LICIT_LABEL]
scaler = MinMaxScaler()
scaler.fit(iter_features)

iter_features = scaler.transform(iter_features)

iter_dataset = []
for i in range(len(iter_features)):
    iter_dataset.append([iter_features[i], iter_labels[i]])

train_dataloader = DataLoader(dataset=iter_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
generator = Generator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)
discriminator = Discriminator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
criterion = nn.BCELoss()

model_filename = f"saved_models/GANomaly_generator_cc_{training_steps}.pt"
if os.path.isfile(model_filename):
    weights = torch.load(model_filename)
    generator.load_state_dict(weights)
else:
    for epoch in range(training_steps):
        for i, data in enumerate(train_dataloader):
            noise = torch.randn(batch_size, latent_dimension, device=device)
            fake_data = generator(noise)
                
            fake_label = torch.full((batch_size,), FAKE_LABEL, dtype=torch.float, device=device)
            real_label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=device)

            random_fake_output = discriminator(fake_data)
            real_output = discriminator(torch.Tensor.float(data[0]).to(device).unsqueeze(1))
                
            discriminator_real_error = criterion(real_output.squeeze(1), real_label)
            discriminator_fake_error = criterion(random_fake_output.squeeze(1), fake_label)
            discriminator_error = discriminator_real_error + discriminator_fake_error

            discriminator_optimizer.zero_grad()
            discriminator_error.backward()
            discriminator_optimizer.step()

            generated_fake_output = discriminator(fake_data.detach())
            generator_error = criterion(generated_fake_output.squeeze(1), real_label)

            generator_optimizer.zero_grad()
            generator_error.backward()
            generator_optimizer.step() 
        print(f"EPOCH: {epoch} DISCRIMINATOR ERROR: {discriminator_error} GENERATOR ERROR: {generator_error}")
            # if epoch % 25 == 0 or epoch == training_steps - 1:
            #     torch.save(generator.state_dict(), f"saved_models/AnoGAN_generator_cc_{epoch + 1}.pt")
            #     torch.save(discriminator.state_dict(), f"saved_models/AnoGAN_discriminator_cc_{epoch + 1}.pt")
licit_iter_features = torch.Tensor(creditcard_dataset[creditcard_labels == LICIT_LABEL]).to(device)
illicit_iter_features = torch.Tensor(creditcard_dataset[creditcard_labels == ILLICIT_LABEL]).to(device)
    
LAMBDA = 0.1

noise = torch.randn(licit_iter_features.shape[0], latent_dimension, device=device)
generated_output = generator(noise)
residual_loss = residual_loss = torch.sum(torch.abs(licit_iter_features - generated_output.squeeze(1)))
latent_generated_output = discriminator(generated_output)
latent_real = discriminator(licit_iter_features.unsqueeze(1))
discrimination_loss = torch.sum(torch.abs(latent_real - latent_generated_output))
licit_anomaly_score = (1-LAMBDA) * residual_loss + (LAMBDA) * discrimination_loss
print(f"LICIT ANOMALY SCORE: {licit_anomaly_score}")

noise = torch.randn(illicit_iter_features.shape[0], latent_dimension, device=device)
generated_output = generator(noise)
residual_loss = residual_loss = torch.sum(torch.abs(illicit_iter_features - generated_output))
latent_generated_output = discriminator(generated_output)
latent_real = discriminator(illicit_iter_features.unsqueeze(1))
discrimination_loss = torch.sum(torch.abs(latent_real - latent_generated_output))
illicit_anomaly_score = (1-LAMBDA) * residual_loss + (LAMBDA) * discrimination_loss
print(f"ILLICIT ANOMALY SCORE: {illicit_anomaly_score}")