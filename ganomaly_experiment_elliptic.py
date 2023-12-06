from utilities.dataset_util import DatasetUtility
from sklearn.preprocessing import MinMaxScaler
from resources import constants
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from utilities.results_util import ResultsUtility
from models.GANomaly.generator_loss import GeneratorLoss
from models.GANomaly.discriminator_loss import DiscriminatorLoss
from models.GANomaly.generator import Generator
from models.GANomaly.discriminator import Discriminator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
import numpy as np
from utilities.loss_fns import SmoothL1LossUncompressed
import matplotlib.pyplot as plt

TRAINING_SPLIT_INDEX = 35
torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
test_labels_graphs = actual_labels_graphs
actual_labels_graphs = predicted_labels_graphs

training_steps = 50
batch_size = 256

input_dimension = (1, actual_labels_graphs.x.shape[1])
latent_dimension = 128
input_channels_count = 1
features_count = actual_labels_graphs.x.shape[1]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

scaler = MinMaxScaler()
scaler.fit(actual_labels_graphs.x)
scaled_features = scaler.transform(actual_labels_graphs.x)

illicit_anomaly_scores = []
licit_anomaly_scores = []
roc_scores = []
results_filename = f"results/GANamolyUNLABELED.csv"
timestamps = actual_labels_graphs.ts.unique().numpy().astype(int)
# for timestamp in timestamps:
for timestamp in range(6,np.max(timestamps)):
    for layers in range(1,6):
        generator_filename = f"saved_models/GANomaly_generator_{timestamp}_{training_steps}_{layers}_UNLABELED1.pt"
        discriminator_filename = f"saved_models/GANomaly_discriminator_{timestamp}_{training_steps}_{layers}_UNLABELED1.pt"
        # generator_filename = "foo"
        training_mask = (actual_labels_graphs.ts == timestamp) & (actual_labels_graphs.y == 0)
        generator = Generator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)
        discriminator = Discriminator(input_dimension, input_channels_count, features_count).to(device)
        generator_loss = GeneratorLoss()
        discriminator_loss = DiscriminatorLoss()
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

        iter_features = scaled_features[training_mask]
        iter_labels = actual_labels_graphs.y[training_mask]
        iter_dataset = []
        for i in range(len(iter_features)):
            iter_dataset.append([iter_features[i], iter_labels[i]])

        train_dataloader = DataLoader(dataset=iter_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        if os.path.isfile(generator_filename):
            generator_weights = torch.load(generator_filename)
            generator.load_state_dict(generator_weights)
        else:
            for epoch in range(training_steps):  
                generator_losses = []
                discriminator_losses = []
                for i, (x,y) in enumerate(train_dataloader):
                    x = x.float()
                    for _ in range(5):
                        generator_optimizer.zero_grad()
                        batch, fake, latent_input, latent_output = generator(x.to(device))

                        pred_fake, _ = discriminator(fake)
                        pred_real, _ = discriminator(batch)
                        gen_loss = generator_loss(latent_input, latent_output, batch, fake, pred_real, pred_fake)
                        gen_loss.backward()
                        generator_optimizer.step()
                            
                    discriminator_optimizer.zero_grad()
                    pred_fake, _ = discriminator(fake.detach())
                    pred_real, _ = discriminator(batch) 
                    discrim_loss = discriminator_loss(pred_real, pred_fake)
                    discrim_loss.backward()
                    discriminator_optimizer.step()

                    generator_losses.append(gen_loss.item())
                    discriminator_losses.append(discrim_loss.item())

                if epoch % 25 == 0 or epoch == training_steps - 1:
                    torch.save(generator.state_dict(), f"saved_models/GANomaly_generator_{timestamp}_{epoch + 1}_{layers}_UNLABELED1.pt")
                    torch.save(discriminator.state_dict(), f"saved_models/GANomaly_discriminator_{timestamp}_{epoch + 1}_{layers}_UNLABELED1.pt")
                    print(f"EPOCH: {epoch} DISCRIMINATOR_LOSS: {np.mean(discriminator_losses)} GENERATOR_LOSS: {np.mean(generator_losses)}")
        generator.eval()
        discriminator.eval()
        print(f"TIMESTAMP: {timestamp}, LAYERS: {layers}")
        scaled_test_features = scaler.transform(test_labels_graphs.x)
        test_mask = (test_labels_graphs.ts == min(timestamp + 1, max(timestamps)))
        licit_mask = (test_labels_graphs.ts == min(timestamp + 1, max(timestamps))) & (test_labels_graphs.y == 0)
        illicit_mask = (test_labels_graphs.ts == min(timestamp + 1, max(timestamps))) & (test_labels_graphs.y == 1)

        anomaly_score = SmoothL1LossUncompressed()
        with torch.no_grad():
            _, _, licit_latent_input, licit_latent_output = generator(torch.Tensor(scaled_test_features[licit_mask]).float().to(device))
            licit_anamoly_score = anomaly_score(licit_latent_input, licit_latent_output)

            _, _, illicit_latent_input, illicit_latent_output = generator(torch.Tensor(scaled_test_features[illicit_mask]).float().to(device))
            illicit_anamoly_score = anomaly_score(illicit_latent_input, illicit_latent_output)

            # unlabeled_features = predicted_labels_graphs.x[predicted_labels_graphs.ts == timestamp]
            # scaled_unlabeled_features = scaler.transform(unlabeled_features)
            # _, _, unlabeled_latent_input, unlabeled_latent_output = generator(torch.Tensor(unlabeled_features).float().to(device))
        scores = torch.cat([licit_anamoly_score, illicit_anamoly_score])
        y = torch.zeros(scores.size(0))
        y[-illicit_anamoly_score.size(0):] = 1.
        roc_score = roc_auc_score(y, scores.to('cpu'))
        average_precision = average_precision_score(y, scores.to('cpu'))
        print(f"ROC SCORE: {roc_score}")
        print(f"AVERAGE PRECISION SCORE: {average_precision}")
        print(f"LICIT ANAMOLY SCORE: {licit_anamoly_score.mean()}")
        print(f"ILLICIT ANAMOLY SCORE: {illicit_anamoly_score.mean()}")

        licit_anomaly_scores.append(licit_anamoly_score)
        illicit_anomaly_scores.append(illicit_anamoly_score)
        roc_scores.append(roc_score)
        results_object = {
            "timestamp": timestamp,
            "layers": layers,
            "roc_score": roc_score,
            "average_precision": average_precision,
            "licit_anamoly_score": licit_anamoly_score.mean().item(),
            "illicit_anamoly_score": illicit_anamoly_score.mean().item()
        }
        if results_util.results_file_exists(results_filename):
            pd.DataFrame([results_object]).to_csv(results_filename, index=False, mode='a',header=False)
        else:
            results_util.create_results_file(results_object, results_filename)
        torch.save((scores,y), f'scores/{timestamp}.pt')

print("foo")