from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from sklearn.preprocessing import MinMaxScaler
from resources import constants
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from models.AnoGAN.generator import Generator
from models.AnoGAN.discriminator import Discriminator
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

UPPER_BOUND = constants.FEAT_DIM + 1
TIMESTAMP_INDEX = 2
REAL_LABEL = 1
FAKE_LABEL = 0
LICIT_LABEL = 0
ILLICIT_LABEL = 1

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)

# Flipping back and forth from np to torch causes float/double casting issues
local_features_matrix_actual = actual_labels_graphs.x
graph_labels_actual = actual_labels_graphs.y
local_features_matrix = predicted_labels_graphs.x
graph_labels = predicted_labels_graphs.y

training_steps = 100 #200
batch_size = 256

timestamps = predicted_labels_graphs.ts.unique()

input_dimension = (1, local_features_matrix.size(1))
latent_dimension = 128
input_channels_count = 1
features_count = local_features_matrix.size(1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
results_array = []

get_noise = lambda : torch.randn(batch_size, latent_dimension, device=device)                

for timestamp in timestamps:
    generator = Generator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)
    discriminator = Discriminator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    mask = predicted_labels_graphs.ts == timestamp
    iter_features = local_features_matrix[mask] 
    iter_labels = graph_labels[mask]

    iter_features = iter_features[iter_labels == LICIT_LABEL]
    iter_labels = iter_labels[iter_labels == LICIT_LABEL]
    scaler = MinMaxScaler()
    scaler.fit(iter_features)
    iter_features = torch.from_numpy(scaler.transform(iter_features))

    iter_dataset = TensorDataset(iter_features, iter_labels)
    train_dataloader = DataLoader(dataset=iter_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"TIMESTAMP: {timestamp}")
    percentiles = [0,10,25,50,75,90,99]
    generator_model_filename = f"saved_models/AnoGAN_generator_{timestamp}_{training_steps}.pt"
    discriminator_model_filename = f"saved_models/AnoGAN_discriminator_{timestamp}_{training_steps}.pt"
    if os.path.isfile(generator_model_filename):
        generator_weights = torch.load(generator_model_filename)
        discriminator_weights = torch.load(discriminator_model_filename)
        generator.load_state_dict(generator_weights)
        discriminator.load_state_dict(discriminator_weights)
    else:
        for epoch in range(training_steps):
            for i, (x,y) in enumerate(train_dataloader):
                x = x.float()
                fake_label = torch.full((batch_size,1), FAKE_LABEL, dtype=torch.float, device=device)
                real_label = torch.full((batch_size,1), REAL_LABEL, dtype=torch.float, device=device)

                # Discriminator training
                discriminator_optimizer.zero_grad()

                fake_data = generator(get_noise())
                fake_pred = discriminator(fake_data)
                real_pred = discriminator(x.to(device))

                discriminator_error = criterion(
                    torch.cat([real_pred, fake_pred]), 
                    torch.cat([real_label, fake_label])
                )

                discriminator_error.backward()
                discriminator_optimizer.step()

                # Generator training
                generator_optimizer.zero_grad()

                fake_data = generator(get_noise())
                generated_fake_output = discriminator(fake_data)
                generator_error = criterion(generated_fake_output, real_label)

                generator_error.backward()
                generator_optimizer.step() 
                
                if epoch == training_steps - 1:
                    torch.save(generator.state_dict(), f"saved_models/AnoGAN_generator_{timestamp}_{epoch + 1}.pt")
                    torch.save(discriminator.state_dict(), f"saved_models/AnoGAN_discriminator_{timestamp}_{epoch + 1}.pt")

            print(f"EPOCH: {epoch} DISCRIMINATOR_LOSS: {discriminator_error.item()} GENERATOR_LOSS: {generator_error.item()}")

    iter_features = local_features_matrix[mask] 
    iter_features = torch.from_numpy(scaler.transform(iter_features))
    iter_features = iter_features.float() 

    iter_labels = graph_labels[mask]
    licit_iter_features = iter_features[iter_labels == LICIT_LABEL].to(device)
    illicit_iter_features = iter_features[iter_labels == ILLICIT_LABEL].to(device)
    
    LAMBDA = 0.1

    noise = torch.randn(licit_iter_features.size(0), latent_dimension, device=device)
    generated_output = generator(noise)
    residual_loss = torch.mean(torch.abs(licit_iter_features - generated_output))
    latent_generated_output = discriminator(generated_output)
    latent_real = discriminator(licit_iter_features)
    discrimination_loss = torch.mean(torch.abs(latent_real - latent_generated_output))
    licit_anomaly_score = (1-LAMBDA) * residual_loss + (LAMBDA) * discrimination_loss
    print(f"LICIT ANOMALY SCORE: {licit_anomaly_score}")

    noise = torch.randn(illicit_iter_features.size(0), latent_dimension, device=device)
    generated_output = generator(noise)
    residual_loss = residual_loss = torch.mean(torch.abs(illicit_iter_features - generated_output))
    latent_generated_output = discriminator(generated_output)
    latent_real = discriminator(illicit_iter_features)
    discrimination_loss = torch.mean(torch.abs(latent_real - latent_generated_output))
    illicit_anomaly_score = (1-LAMBDA) * residual_loss + (LAMBDA) * discrimination_loss
    print(f"ILLICIT ANOMALY SCORE: {illicit_anomaly_score}")
    results_array.append({
        "Timestamp": timestamp,
        "Licit Count": iter_features[iter_labels == LICIT_LABEL].size(0),
        "Illicit Count": iter_features[iter_labels == ILLICIT_LABEL].size(0),
        "Licit Anomaly Score": licit_anomaly_score.item(),
        "Illicit Anomaly Score": illicit_anomaly_score.item()
    })

results_filename = f"AnoGAN_results_file.csv"
results_df = pd.DataFrame(results_array)
results_df.to_csv(f"results/{results_filename}",index=False)

plt.plot(timestamps, results_df["Licit Anomaly Score"], color="blue", label="Licit Transactions")
plt.plot(timestamps, results_df["Illicit Anomaly Score"], color="red", label="Illict Transactions")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Score")
plt.title("AnoGAN Results on Elliptic Dataset")
plt.savefig("results/AnoGAN.png")
