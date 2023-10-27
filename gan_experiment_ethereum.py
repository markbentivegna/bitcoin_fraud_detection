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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
ethereum_df = dataset_util.load_dataset(constants.ETHEREUM_DATASET)
ethereum_dataset = ethereum_df.drop("label",axis=1).to_numpy()
ethereum_labels = ethereum_df["label"].to_numpy()
input_length = ethereum_dataset.shape[1]
generated_illicit_transactions = np.empty(shape=[0,input_length])
training_steps = 1000
LABEL_INDEX = ethereum_dataset.shape[1]
batch_size = ethereum_labels[ethereum_labels == 1].shape[0]
generator = Generator(input_length)
discriminator = Discriminator(input_length)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
loss = nn.BCELoss()

scaler = MinMaxScaler()
scaler.fit(ethereum_dataset[ethereum_labels == 1])
iter_dataset = scaler.transform(ethereum_dataset)
# iter_dataset = ethereum_dataset
print("foo")

for i in range(training_steps):
    generator_optimizer.zero_grad()

    random_min = iter_dataset.min()
    random_max = iter_dataset.max()
    random_noise = np.random.uniform(low=random_min, high=random_max, size=(batch_size, iter_dataset.shape[1])) * torch.rand((batch_size, iter_dataset.shape[1])).numpy()
    generated_data = generator(torch.tensor(random_noise).float())
    true_data, true_labels = iter_dataset[ethereum_labels == 1], np.ones(iter_dataset[ethereum_labels == 1].shape[0])
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
    generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size).unsqueeze(1))
    discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
    discriminator_loss.backward()
    discriminator_optimizer.step()
    if i % 100 == 0:
        print(f"Discriminator_loss at epoch {i}: {discriminator_loss}")
        print(f"Generator_loss at epoch {i}: {generator_loss}")

    if i == training_steps - 1:
        generated_illicit_transactions = np.concatenate((generated_illicit_transactions,scaler.inverse_transform(generated_data.detach().numpy())), axis=0)
        # generated_illicit_transactions = np.concatenate((generated_illicit_transactions,generated_data.detach().numpy()), axis=0)
percentiles = [0,10,25,50,75,90,99]
print(f"ACTUAL    : {[np.percentile(ethereum_dataset[ethereum_labels == 1], x).round(decimals=2) for x in percentiles]}")
print(f"GENERATED : {[np.percentile(generated_illicit_transactions, x).round(decimals=2) for x in percentiles]}")

ethereum_dataset = np.insert(ethereum_dataset, LABEL_INDEX, np.zeros(ethereum_dataset.shape[0]), axis=1)
generated_illicit_transactions = np.insert(generated_illicit_transactions, LABEL_INDEX, np.ones(generated_illicit_transactions.shape[0]), axis=1)
full_dataset = np.concatenate((ethereum_dataset, generated_illicit_transactions),axis=0)
full_labels = np.concatenate((ethereum_labels, np.ones(generated_illicit_transactions.shape[0])),axis=0)

X_train, X_test, y_train, y_test = train_test_split(full_dataset, full_labels, train_size=0.7, shuffle=True)
actual_indices = (X_test[:,LABEL_INDEX] == 0)
X_test = X_test[actual_indices][:,:LABEL_INDEX]
y_test = y_test[actual_indices]

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train[:,:LABEL_INDEX], y_train)
pred = clf.predict(X_test)
print(f"GAN GROUP")
print(f"precision: {precision_score(y_test, pred)}")
print(f"recall: {recall_score(y_test, pred)}")
print(f"f1: {f1_score(y_test, pred)}")
print(f"roc_auc  : {roc_auc_score(y_test, pred)}")
print(f"accuracy  : {accuracy_score(y_test, pred)}")

X_train, X_test, y_train, y_test = train_test_split(ethereum_df.drop("label",axis=1), ethereum_df["label"], train_size=0.7, shuffle=True)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(f"CONTROL GROUP")
print(f"precision: {precision_score(y_test, pred)}")
print(f"recall: {recall_score(y_test, pred)}")
print(f"f1: {f1_score(y_test, pred)}")
print(f"roc_auc  : {roc_auc_score(y_test, pred)}")
print(f"accuracy  : {accuracy_score(y_test, pred)}")

