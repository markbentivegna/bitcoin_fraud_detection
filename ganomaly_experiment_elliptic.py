from utilities.dataset_util import DatasetUtility
from sklearn.preprocessing import MinMaxScaler
from resources import constants
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from models.GANomaly.generator_loss import GeneratorLoss
from models.GANomaly.discriminator_loss import DiscriminatorLoss
from models.GANomaly.generator import Generator
from models.GANomaly.discriminator import Discriminator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, accuracy_score

TRAINING_SPLIT_INDEX = 35
torch.set_num_threads(16)
dataset_util = DatasetUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)

training_steps = 10
batch_size = 256

input_dimension = (1, actual_labels_graphs.x.shape[1])
latent_dimension = 128
input_channels_count = 1
features_count = actual_labels_graphs.x.shape[1]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
generator = Generator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)
discriminator = Discriminator(input_dimension, input_channels_count, features_count).to(device)
generator_loss = GeneratorLoss()
discriminator_loss = DiscriminatorLoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

scaler = MinMaxScaler()
scaler.fit(actual_labels_graphs.x)
scaled_features = scaler.transform(actual_labels_graphs.x)

timestamps = actual_labels_graphs.ts.unique().numpy().astype(int)
for timestamp in range(TRAINING_SPLIT_INDEX, max(timestamps)):
    training_mask = (actual_labels_graphs.ts < timestamp) & (actual_labels_graphs.y == 0)

    iter_features = scaled_features[training_mask]
    iter_labels = actual_labels_graphs.y[training_mask]
    iter_dataset = []
    for i in range(len(iter_features)):
        iter_dataset.append([iter_features[i], iter_labels[i]])

    train_dataloader = DataLoader(dataset=iter_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model_filename = f"saved_models/GANomaly_generator_{timestamp}_{training_steps}.pt"
    if os.path.isfile(model_filename):
        weights = torch.load(model_filename)
        generator.load_state_dict(weights)
    else:
        for epoch in range(training_steps):  
            for i, (x,y) in enumerate(train_dataloader):
                x = x.float()
                generator_optimizer.zero_grad()
                batch, fake, latent_input, latent_output = generator(x.to(device))

                pred_fake, _ = discriminator(fake)
                pred_real, _ = discriminator(batch)
                gen_loss = generator_loss(latent_input, latent_output, batch, fake, pred_real, pred_fake)
                gen_loss.backward()
                generator_optimizer.step()
                        
                discriminator_optimizer.zero_grad()
                _, fake, _,_ = generator(x.to(device))
                pred_fake, _ = discriminator(fake)
                pred_real, _ = discriminator(batch) 
                discrim_loss = discriminator_loss(pred_real, pred_fake)
                discrim_loss.backward()
                discriminator_optimizer.step()
                # print(f"EPOCH: {epoch} DISCRIMINATOR_LOSS: {discrim_loss.item()} GENERATOR_LOSS: {gen_loss.item()}")

            if epoch % 25 == 0 or epoch == training_steps - 1:
                torch.save(generator.state_dict(), f"saved_models/GANomaly_generator_{timestamp}_{epoch + 1}.pt")
                torch.save(discriminator.state_dict(), f"saved_models/GANomaly_discriminator_{timestamp}_{epoch + 1}.pt")

    # generator.eval()
    
    print(f"TIMESTAMP: {timestamp}")
    testing_mask = actual_labels_graphs.ts == timestamp
    licit_mask = (actual_labels_graphs.ts == timestamp) & (actual_labels_graphs.y == 0)
    illicit_mask = (actual_labels_graphs.ts == timestamp) & (actual_labels_graphs.y == 1)

    anomaly_score = nn.SmoothL1Loss()
    _, _, licit_latent_input, licit_latent_output = generator(torch.Tensor.float(torch.from_numpy(scaled_features[licit_mask])).to(device))
    licit_anamoly_score = anomaly_score(licit_latent_input, licit_latent_output)

    _, _, illicit_latent_input, illicit_latent_output = generator(torch.Tensor.float(torch.from_numpy(scaled_features[illicit_mask])).to(device))
    illicit_anamoly_score = anomaly_score(illicit_latent_input, illicit_latent_output)

    # anomaly_scores = []
    # for feature in scaled_features[testing_mask]:
    #     _, _, latent_input, latent_output = generator(torch.Tensor.float(torch.from_numpy(feature).unsqueeze(0)).to(device))
    #     anomaly_scores.append(anomaly_score(latent_input, latent_output).item())
    # roc_score = roc_auc_score(actual_labels_graphs.y[testing_mask], anomaly_scores)
    # average_precision = average_precision_score(actual_labels_graphs.y[testing_mask], anomaly_scores)
    # print(f"ROC SCORE: {roc_score}")
    # print(f"AVERAGE PRECISION SCORE: {average_precision}")
    print(f"LICIT ANAMOLY SCORE: {licit_anamoly_score}")
    print(f"ILLICIT ANAMOLY SCORE: {illicit_anamoly_score}")