from resources import constants
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.nn import functional as F
import torch.nn as nn
import os
import gc
from models.jodie import Jodie

class FraudTransactionUtility:
    def __init__(self):
        pass

    def _to_tensor(self, array):
        return torch.Tensor(array)

    def generate_graph(self, fraud_transaction_df):
        edges_tensor = self._to_tensor(fraud_transaction_df[["CUSTOMER_ID", "TERMINAL_ID"]].to_numpy())
        customers_tensor = self._to_tensor(fraud_transaction_df[["CUSTOMER_ID"]].to_numpy())
        terminals_tensor = self._to_tensor(fraud_transaction_df[["TERMINAL_ID"]].to_numpy())
    
    def _preprocess_dataset(self, fraud_transaction_df):
        selected_columns = ["TRANSACTION_ID","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_TIME_SECONDS", "CUSTOMER_DELTA", "VENDOR_DELTA", "CUSTOMER_PREVIOUS_ITEM"]
        fraud_delta_filename = f"{constants.FRAUD_TRANSACTION_DATASET}/{constants.FRAUD_DELTA_FILE}"
        if not self._file_exists(fraud_delta_filename):
            customer_ids = fraud_transaction_df["CUSTOMER_ID"].unique()
            user_timestamp_diffs = pd.Series(int)
            customer_vendor_series = pd.Series(int)
            for customer_id in customer_ids:
                current_timestamp_diffs = (fraud_transaction_df[fraud_transaction_df["CUSTOMER_ID"] == customer_id]["TX_TIME_SECONDS"] - fraud_transaction_df[fraud_transaction_df["CUSTOMER_ID"] == customer_id]["TX_TIME_SECONDS"].shift()).fillna(0).astype(int)
                previous_vendors = (fraud_transaction_df[fraud_transaction_df["CUSTOMER_ID"] == customer_id])["TERMINAL_ID"].shift().fillna(0).astype(int)
                user_timestamp_diffs = pd.concat([user_timestamp_diffs, current_timestamp_diffs])
                customer_vendor_series = pd.concat([customer_vendor_series,previous_vendors])
            fraud_transaction_df["CUSTOMER_DELTA"] = user_timestamp_diffs.iloc[1:]
            fraud_transaction_df["CUSTOMER_PREVIOUS_ITEM"] = customer_vendor_series.iloc[1:]
            
            vendor_ids = fraud_transaction_df["TERMINAL_ID"].unique()
            vendor_timestamp_diffs = pd.Series(int)
            for vendor_id in vendor_ids:
                current_timestamp_diffs = (fraud_transaction_df[fraud_transaction_df["TERMINAL_ID"] == vendor_id]["TX_TIME_SECONDS"] - fraud_transaction_df[fraud_transaction_df["TERMINAL_ID"] == vendor_id]["TX_TIME_SECONDS"].shift()).fillna(0).astype(int)
                vendor_timestamp_diffs = pd.concat([vendor_timestamp_diffs,current_timestamp_diffs])
            fraud_transaction_df["VENDOR_DELTA"] = vendor_timestamp_diffs.iloc[1:]
            fraud_transaction_df[selected_columns].to_csv(fraud_delta_filename)
        return pd.read_csv(fraud_delta_filename, index_col=0)

    def get_t_batches(self, fraud_transaction_df):
        fraud_tbatch_filename = f"{constants.FRAUD_TRANSACTION_DATASET}/{constants.TBATCH_FRAUD_FILE}"
        if not self._file_exists(fraud_tbatch_filename):
            preprocessed_df = self._preprocess_dataset(fraud_transaction_df)
            preprocessed_df["CUSTOMER_TBATCH"] = -1
            preprocessed_df["VENDOR_TBATCH"] = -1
            for index, row in preprocessed_df.iterrows():
                customer_tbatch = (preprocessed_df[preprocessed_df["CUSTOMER_ID"] == row["CUSTOMER_ID"]])["CUSTOMER_TBATCH"].shift().fillna(0).astype(int)
                vendor_tbatch = (preprocessed_df[preprocessed_df["TERMINAL_ID"] == row["TERMINAL_ID"]])["VENDOR_TBATCH"].shift().fillna(0).astype(int)
                tbatch_to_insert = max(customer_tbatch[index], vendor_tbatch[index]) + 1
                preprocessed_df.loc[index, "CUSTOMER_TBATCH"] = tbatch_to_insert
                preprocessed_df.loc[index, "VENDOR_TBATCH"] = tbatch_to_insert
                # tbatch_starttime = row["TX_TIME_SECONDS"]
                if index % 10000 == 0:
                    print(f"Generated t-batches for {index} interactions")
            preprocessed_df.to_csv(fraud_tbatch_filename)
        tbatch_df = pd.read_csv(fraud_tbatch_filename, index_col=0)
        return tbatch_df
    
    def _file_exists(self, filename):
        return os.path.isfile(filename)

    def get_dataset(self):
        selected_columns = ["TRANSACTION_ID","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_TIME_SECONDS"]
        feature_columns = ["TX_AMOUNT", "TX_TIME_SECONDS"]
        fraud_transaction_df = pd.read_csv(f"{constants.FRAUD_TRANSACTION_DATASET}/{constants.FRAUD_TRANSACTION_FILE}",index_col=0)
        fraud_transaction_df["CUSTOMER_ID"] = fraud_transaction_df["CUSTOMER_ID"] + 1
        fraud_transaction_df["TERMINAL_ID"] = fraud_transaction_df["TERMINAL_ID"] + 1
        fraud_transaction_df["month"] = pd.DatetimeIndex(fraud_transaction_df["TX_DATETIME"]).month
        fraud_transaction_df["day"] = pd.DatetimeIndex(fraud_transaction_df["TX_DATETIME"]).day
        fraud_transaction_df["hour"] = pd.DatetimeIndex(fraud_transaction_df["TX_DATETIME"]).hour
        fraud_transaction_df.drop(["TX_FRAUD", "TX_DATETIME"],axis=1,inplace=True)
        fraud_transaction_df = fraud_transaction_df.rename(columns={"TX_FRAUD_SCENARIO": "label"})
        t_batches_df =  self.get_t_batches(fraud_transaction_df)
        vendor_embedding_static = torch.autograd.Variable(torch.eye(t_batches_df["TERMINAL_ID"].max() + 1)).cuda()
        user_embedding_static = torch.autograd.Variable(torch.eye(t_batches_df["CUSTOMER_ID"].max() + 1)).cuda()
        initial_user_embedding = nn.Parameter(F.normalize(torch.rand(128), dim=0)).cuda()
        initial_item_embedding = nn.Parameter(F.normalize(torch.rand(128), dim=0)).cuda()
        user_embeddings = initial_user_embedding.repeat(t_batches_df["CUSTOMER_ID"].max() + 1, 1)
        vendor_embeddings = initial_item_embedding.repeat(t_batches_df["TERMINAL_ID"].max() + 1, 1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DEVICE: {device}")
        model = Jodie(2, 128, t_batches_df["CUSTOMER_ID"].max() + 1, t_batches_df["TERMINAL_ID"].max() + 1).to(device)
        model.initial_user_embedding = nn.Parameter(initial_user_embedding)
        model.initial_item_embedding = nn.Parameter(initial_item_embedding)
        MSELoss = nn.MSELoss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        for epoch in range(50):
            user_embeddings_timeseries = torch.autograd.Variable(torch.Tensor(t_batches_df.shape[0], 128)).cuda()
            vendor_embeddings_timeseries = torch.autograd.Variable(torch.Tensor(t_batches_df.shape[0], 128)).cuda()
            optimizer.zero_grad()
            total_loss, loss, total_interaction_count = 0, 0, 0
            # for tbatch in range(1, t_batches_df["CUSTOMER_TBATCH"].nunique()):
            for tbatch in range(1, 1000):
                current_tbatch_users = torch.LongTensor(t_batches_df[t_batches_df["CUSTOMER_TBATCH"] == tbatch]["CUSTOMER_ID"].to_numpy()).cuda()
                current_tbatch_vendors = torch.LongTensor(t_batches_df[t_batches_df["VENDOR_TBATCH"] == tbatch]["TERMINAL_ID"].to_numpy()).cuda()
                current_tbatch_interactions = torch.LongTensor(t_batches_df[t_batches_df["CUSTOMER_TBATCH"] == tbatch]["TRANSACTION_ID"].to_numpy())
                current_tbatch_timestamps = torch.LongTensor(t_batches_df[t_batches_df["CUSTOMER_TBATCH"] == tbatch]["TX_TIME_SECONDS"].to_numpy())
                # current_tbatch_labels = torch.LongTensor(fraud_transaction_df.merge(t_batches_df)[fraud_transaction_df.merge(t_batches_df)["CUSTOMER_TBATCH"] == tbatch]["label"].to_numpy()).cuda()
                current_tbatches_previous_vendor = torch.LongTensor(t_batches_df[t_batches_df["CUSTOMER_TBATCH"] == tbatch]["CUSTOMER_PREVIOUS_ITEM"].to_numpy())
                current_features = torch.LongTensor(t_batches_df[t_batches_df["VENDOR_TBATCH"] == tbatch][feature_columns].to_numpy()).cuda()

                feature_tensor = torch.autograd.Variable(current_features).cuda()
                user_timediffs_tensor = torch.autograd.Variable(current_tbatch_timestamps).unsqueeze(1).cuda()
                vendor_embedding_previous = vendor_embeddings[current_tbatches_previous_vendor,:].cuda()
                user_embedding_input = user_embeddings[current_tbatch_users,:].cuda()
                user_projected_embedding = model.forward(user_embedding_input, vendor_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                
                user_merchant_embedding = torch.cat([user_projected_embedding, vendor_embedding_previous, vendor_embedding_static[current_tbatches_previous_vendor,:], user_embedding_static[current_tbatch_users,:]], dim=1)
                predicted_vendor_embedding = model.predict_item_embedding(user_merchant_embedding)
                

                vendor_embedding_input = vendor_embeddings[current_tbatch_vendors,:].cuda()
                
                loss += MSELoss(predicted_vendor_embedding, torch.cat([vendor_embedding_input, vendor_embedding_static[current_tbatch_vendors,:]], dim=1).detach()).item()

                user_embedding_output = model.forward(user_embedding_input, vendor_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                vendor_embedding_output = model.forward(user_embedding_input, vendor_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='item_update')

                vendor_embeddings[current_tbatch_vendors,:] = vendor_embedding_output
                user_embeddings[current_tbatch_users,:] = user_embedding_output  

                user_embeddings_timeseries[current_tbatch_interactions,:] = user_embedding_output
                vendor_embeddings_timeseries[current_tbatch_interactions,:] = vendor_embedding_output

                loss += MSELoss(vendor_embedding_output, vendor_embedding_input.detach()).item()
                loss += MSELoss(user_embedding_output, user_embedding_input.detach()).item()
                if tbatch % 100 == 0:
                    print(f"COMPLETED TBATCH: {tbatch}")
                    gc.collect()
                torch.cuda.empty_cache()
                del current_features
                del current_tbatch_interactions
                del current_tbatch_timestamps
                del current_tbatch_users
                del current_tbatch_vendors
                del current_tbatches_previous_vendor
                del user_projected_embedding
                del predicted_vendor_embedding
                del user_embedding_output
                del vendor_embedding_output
                del user_embedding_input
                del vendor_embedding_input
                del user_merchant_embedding
                del vendor_embedding_previous
                del user_timediffs_tensor

            print(f"COMPLETED EPOCH: {epoch}")
            total_loss += loss
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = 0
            vendor_embeddings.detach_()
            user_embeddings.detach_()
            vendor_embeddings_timeseries.detach_() 
            user_embeddings_timeseries.detach_()
                   
            print("foo")
        # self.generate_graph(fraud_transaction_df)