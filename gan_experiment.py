from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from sklearn.preprocessing import MinMaxScaler
from resources import constants
import torch
from torch import nn
from models.Generator import Generator
from models.Discriminator import Discriminator
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# UPPER_BOUND = constants.LOCAL_FEATS
UPPER_BOUND = constants.FEAT_DIM + 1
TIMESTAMP_INDEX = 2

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
local_features_matrix = actual_labels_graphs.x.numpy()
graph_labels = actual_labels_graphs.y.numpy()

training_steps = 500
batch_size = 1024
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

for timestamp in timestamps:
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    iter_dataset = local_features_matrix[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp][:,1:]
    iter_labels = graph_labels[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp]
    iter_batch_size = iter_dataset[iter_labels == 0].shape[0] - iter_dataset[iter_labels == 1].shape[0]
    scaler = MinMaxScaler()
    scaler.fit(iter_dataset[iter_labels == 1])
    iter_dataset = scaler.transform(iter_dataset)
    loss = nn.BCELoss()
    for i in range(training_steps):
        generator_optimizer.zero_grad()

        random_min = 0
        random_max = 1
        random_noise = np.random.uniform(low=random_min, high=random_max, size=(iter_batch_size, iter_dataset.shape[1])) * torch.rand((iter_batch_size, iter_dataset.shape[1])).numpy()
        generated_data = generator(torch.tensor(random_noise).float())
        true_data, true_labels = balanced_dataset_sampler(iter_dataset, iter_labels, iter_batch_size, filter_illicit=True)
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
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(iter_batch_size).unsqueeze(1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
        if i % 100 == 0:
            print(f"TIMESTMAP {timestamp} train step {i} has discriminator_loss: {discriminator_loss}")
            print(f"TIMESTMAP {timestamp} train step {i} has generator_loss: {generator_loss}")

        if i == training_steps - 1:
            generated_illicit_nodes.append(scaler.inverse_transform(generated_data.detach().numpy()))

percentiles = [0,10,25,50,75,90,99]
for timestamp in timestamps:
    iter_features = local_features_matrix[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp][:,1:]
    iter_labels = graph_labels[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp]
    iter_illicit_features = iter_features[iter_labels == 1]
    print(f"TIMESTAMP {timestamp}")
    print(f"ACTUAL    : {[np.percentile(iter_illicit_features, x).round(decimals=2) for x in percentiles]}")
    print(f"GENERATED : {[np.percentile(generated_illicit_nodes[int(timestamp)-1], x).round(decimals=2) for x in percentiles]}")
full_generated_dataset = np.empty(shape=[0,UPPER_BOUND + 3])
for timestamp in timestamps:
    new_illicit_matrix = np.insert(generated_illicit_nodes[int(timestamp - 1)], TIMESTAMP_INDEX, timestamp,axis=1)
    new_illicit_matrix = np.insert(new_illicit_matrix, UPPER_BOUND + 1, 1, axis=1)
    new_illicit_matrix = np.insert(new_illicit_matrix, UPPER_BOUND + 2, 1, axis=1)

    # iter_actual_matrix = local_features_matrix[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp]
    # iter_actual_labels = graph_labels[local_features_matrix[:,TIMESTAMP_INDEX] == timestamp]
    # iter_dataset = np.insert(iter_actual_matrix, UPPER_BOUND + 1, iter_actual_labels, axis=1)
    # iter_dataset = np.insert(iter_dataset, UPPER_BOUND + 2, 0, axis=1)
    
    # iter_dataset = np.concatenate((new_illicit_matrix, iter_dataset), axis=0)
    full_generated_dataset = np.concatenate((full_generated_dataset, new_illicit_matrix),axis=0)
full_actual_dataset = np.insert(actual_labels_graphs.x, UPPER_BOUND + 1, actual_labels_graphs.y, axis=1)
full_actual_dataset = np.insert(full_actual_dataset, UPPER_BOUND + 2, 0, axis=1)
full_dataset = np.concatenate((full_actual_dataset, full_generated_dataset), axis=0)
X_generated, y_generated = full_dataset[:,:UPPER_BOUND + 1], full_dataset[:,UPPER_BOUND + 1]
pd.DataFrame(full_dataset).reset_index().drop(columns={0},axis=1).to_csv(f"{constants.GAN_ELLIPTIC_DATASET}")
X_actual, y_actual = local_features_matrix, graph_labels
y_test_full = np.array([])
pred_full = np.array([])
pred_threshold_swap = 0.3
print(f"PROB SWITCH THRESHOLD: {pred_threshold_swap}")
for timestamp in range(35, int(np.max(timestamps))):
    print(f"TIMESTAMP: {timestamp}")
    if timestamp >= 44:
        X_train = X_generated[(X_generated[:,TIMESTAMP_INDEX] > 42) & (X_generated[:,TIMESTAMP_INDEX] < timestamp)]
        y_train = y_generated[(X_generated[:,TIMESTAMP_INDEX] > 42) & (X_generated[:,TIMESTAMP_INDEX] < timestamp)]
    else:
        X_train = X_generated[X_generated[:,TIMESTAMP_INDEX] < timestamp]
        y_train = y_generated[X_generated[:,TIMESTAMP_INDEX] < timestamp]
    X_test = X_actual[X_actual[:,TIMESTAMP_INDEX] == timestamp]
    y_test = y_actual[X_actual[:,TIMESTAMP_INDEX] == timestamp]
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    if timestamp >= 43:
        pred_prob = clf.predict_proba(X_test)
        pred[pred_prob[:,1] > pred_threshold_swap] = 1
    y_test_full = np.concatenate((y_test_full,y_test),axis=0)
    pred_full = np.concatenate((pred_full,pred),axis=0)
    print(f"precision: {precision_score(y_test, pred)}")
    print(f"recall: {recall_score(y_test, pred)}")
    print(f"f1: {f1_score(y_test, pred)}")
print("FINAL RESULTS")
print(f"precision: {precision_score(y_test_full, pred_full)}")
print(f"recall   : {recall_score(y_test_full, pred_full)}")
print(f"f1       : {f1_score(y_test_full, pred_full)}")
print(f"roc_auc  : {roc_auc_score(y_test_full, pred_full)}")
