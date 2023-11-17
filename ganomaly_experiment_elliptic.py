from utilities.dataset_util import DatasetUtility
from sklearn.preprocessing import MinMaxScaler
from resources import constants
import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from torch import nn
from models.GANomaly.generator_loss import GeneratorLoss
from models.GANomaly.discriminator_loss import DiscriminatorLoss
from models.GANomaly.generator import Generator
from models.GANomaly.discriminator import Discriminator
import numpy as np
from utilities.loss_fns import SmoothL1LossUncompressed

ILLICIT_LABEL = 1
LICIT_LABEL = 0
torch.set_num_threads(16)
dataset_util = DatasetUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)

# Why are we converting back to numpy? Why are we making 
# two variables that hold the same data?
local_features_matrix_actual = actual_labels_graphs.x
local_features_matrix = actual_labels_graphs.x
graph_labels = actual_labels_graphs.y

training_steps = 10
batch_size = 256
generated_illicit_nodes = []

timestamps = np.unique(actual_labels_graphs.ts.numpy()).astype(int)
input_dimension = (1, local_features_matrix.shape[1])
latent_dimension = 128
input_channels_count = 1
features_count = local_features_matrix.shape[1]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
for timestamp in timestamps:
    generator = Generator(input_dimension, latent_dimension, input_channels_count, features_count).to(device)
    discriminator = Discriminator(input_dimension, input_channels_count, features_count).to(device)
    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    mask = actual_labels_graphs.ts == timestamp
    iter_features = local_features_matrix[mask]
    iter_labels = graph_labels[mask]
    licit_iter_features = iter_features[iter_labels == LICIT_LABEL]
    
    # This should be in the model somewhere. Makes it difficult to 
    # Rerun on arbitrary new data
    scaler = MinMaxScaler()
    scaler.fit(licit_iter_features)
    licit_iter_features = scaler.transform(licit_iter_features)
    iter_features = scaler.transform(iter_features)

    # Then we convert the numpy matrix back to a torch matrix?
    iter_dataset = []
    for i in range(len(licit_iter_features)):
        iter_dataset.append([licit_iter_features[i], LICIT_LABEL])

    train_dataloader = DataLoader(dataset=iter_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"TIMESTAMP: {timestamp}")
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

                # This is why it's not learning. You're detatching the tensors so 
                # how much it fools the discriminator isn't backpropping to the generator
                #pred_fake, _ = discriminator(fake.detach())
                #pred_real, _ = discriminator(batch.detach())
                #gen_loss = generator_loss(latent_input, latent_output, batch, fake, pred_real.detach(), pred_fake.detach())
                
                pred_fake, _ = discriminator(fake)
                pred_real, _ = discriminator(batch)
                gen_loss = generator_loss(latent_input, latent_output, batch, fake, pred_real, pred_fake)
                gen_loss.backward()
                generator_optimizer.step()
                
                # This is also why it's not learning. Need to zero disc's grad 
                # because it has loss from the generator accumulated from the last 
                # call to .backward() 
                discriminator_optimizer.zero_grad()
                pred_fake, _ = discriminator(fake.detach())
                # Have to call again. Autograd gets angry if you call backward on the same grad twice
                pred_real, _ = discriminator(batch) 
                discrim_loss = discriminator_loss(pred_real, pred_fake)
                discrim_loss.backward()
                discriminator_optimizer.step()

            if epoch % 25 == 0 or epoch == training_steps - 1:
                torch.save(generator.state_dict(), f"saved_models/GANomaly_generator_{timestamp}_{epoch + 1}.pt")
                torch.save(discriminator.state_dict(), f"saved_models/GANomaly_discriminator_{timestamp}_{epoch}.pt")
            print(f"EPOCH: {epoch} DISCRIMINATOR_LOSS: {discrim_loss.item()} GENERATOR_LOSS: {gen_loss.item()}")
    
    # Load in next timestamp 
    new_mask = actual_labels_graphs.ts == timestamp + 1 
    iter_features = local_features_matrix[new_mask].float()
    iter_labels = graph_labels[new_mask]

    anomaly_score = SmoothL1LossUncompressed()
    with torch.no_grad():
        _, _, licit_latent_input, licit_latent_output = generator(iter_features[iter_labels == LICIT_LABEL].to(device))
        licit_anamoly_score = anomaly_score(licit_latent_input, licit_latent_output)

        _, _, illicit_latent_input, illicit_latent_output = generator(iter_features[iter_labels == ILLICIT_LABEL].to(device))
        illicit_anamoly_score = anomaly_score(illicit_latent_input, illicit_latent_output)

    scores = torch.cat([licit_anamoly_score, illicit_anamoly_score])
    y = torch.zeros(scores.size(0))
    y[-illicit_anamoly_score.size(0):] = 1.

    print(f"AUC: {roc_auc_score(y, scores)}")
    print(f"AP: {average_precision_score(y, scores)}")
    print(f"LICIT ANAMOLY SCORE: {licit_anamoly_score.mean().item()}")
    print(f"ILLICIT ANAMOLY SCORE: {illicit_anamoly_score.mean().item()}")

    torch.save((scores,y), f'scores/{timestamp}.pt')