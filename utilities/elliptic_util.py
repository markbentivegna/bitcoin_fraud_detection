from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import torch
import os
import numpy as np
from torch_geometric.data import Data

from resources import constants


class EllipticUtility:
    def __init__(self):
        self._initialize_dataframes()

    def _initialize_dataframes(self):
        self.edgelist_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.EDGELIST_FILE}")
        self.edgelist_df.rename(columns={
            "txId1": "source",
            "txId2": "target"
        }, inplace=True)
        self.features_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE}")
        self.features_df.rename(columns={
            "1": "timestamp",
            "230425980": "transaction_id"
        }, inplace=True)
        self.labels_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.CLASSES_FILE}")
        self.labels_df.rename(columns={
            "txId": "transaction_id"
        }, inplace=True)

    def get_dataset(self, filter_labeled=True):
        dataset_df = self._init_dataset_df(filter_labeled=filter_labeled)
        filename = constants.EDGES_LABELED_FILENAME if filter_labeled else constants.EDGES_INDEXED_FILENAME
        if self._file_exists(filename):
            edges_df = self.get_edges_csv(labeled=filter_labeled)
        else:
            edges_df = self.get_edge_list(generate_csv=True, is_labeled=filter_labeled)
        train_mask_tensor, val_mask_tensor, test_mask_tensor = self._get_mask_tensors(dataset_df)
        edges_tensor = self._to_tensor(edges_df.to_numpy())
        x_tensor = self._to_tensor(dataset_df.reset_index().drop("class",axis=1).to_numpy())
        y_tensor = self._to_tensor(dataset_df["class"].to_numpy().astype(int))

        return Data(
            x=x_tensor, edge_index=edges_tensor, y=y_tensor,
            num_nodes=x_tensor.size(0), train_mask=train_mask_tensor,
            val_mask=val_mask_tensor, test_mask=test_mask_tensor
        )

    def _init_dataset_df(self, filter_labeled=True):
        filename = constants.DATASET_FILENAME
        if self._file_exists(filename):
            return pd.read_csv(filename)
        dataset_df = self.features_df.merge(self.labels_df,how="left").dropna()
        if filter_labeled:
            dataset_df = self.features_df.merge(self.labels_df[self.labels_df["class"] != "unknown"],how="left").dropna()
        dataset_df["class"] = pd.to_numeric(dataset_df["class"].str.replace("unknown", "0").replace("suspicious", "1"))
        dataset_df["class"] -= 2
        dataset_df["class"] = dataset_df["class"].abs()
        dataset_df = self.predict_timestamps(dataset_df)
        dataset_df.sort_values(by=["timestamp"]).to_csv(filename,index=False)
        return dataset_df.sort_values(by=["timestamp"])

    def _get_mask_tensors(self,dataset_df):
        train_mask_tensor = self._to_tensor(np.where(dataset_df['timestamp'] <= constants.TRAIN_TIMESTAMP, True, False)).bool()
        val_mask_tensor = self._to_tensor(np.where((dataset_df['timestamp'] > constants.TRAIN_TIMESTAMP) & (dataset_df['timestamp'] <= constants.TEST_TIMESTAMP), True, False)).bool()
        test_mask_tensor = self._to_tensor(np.where(dataset_df['timestamp'] > constants.TEST_TIMESTAMP, True, False)).bool()
        return train_mask_tensor, val_mask_tensor, test_mask_tensor

    def _to_tensor(self, array):
        return torch.Tensor(array)

    def _file_exists(self, filename):
        return os.path.isfile(filename)

    def get_edges_csv(self,labeled=False):
        if labeled:
            return pd.read_csv(f"{constants.EDGES_LABELED_FILENAME}")
        else:
            return pd.read_csv(f"{constants.EDGES_INDEXED_FILENAME}")

    def get_edge_list(self, generate_csv=False, is_labeled=False):
        features_df = self.features_df
        filename = constants.EDGES_LABELED_FILENAME if is_labeled else constants.EDGES_INDEXED_FILENAME
        if is_labeled:
            features_df = self.features_df.merge(self.labels_df[self.labels_df["class"] != "unknown"],how="left").dropna().reset_index().drop("index",axis=1)
        edges_df = self._get_edges_indexed(features_df)
        if generate_csv:
            edges_df.to_csv(f"{filename}",index=False)
        return edges_df
    
    def _get_edges_indexed(self, features_df):
        source_df = features_df.reset_index()[["index", "transaction_id"]].rename(columns={"index": "source_index","transaction_id": "source"})
        target_df = features_df.reset_index()[["index", "transaction_id"]].rename(columns={"index": "target_index","transaction_id": "target"})
        return self.edgelist_df.merge(source_df.drop_duplicates(subset=['source']),how="left").merge(target_df.drop_duplicates(subset=['target']),how="left").dropna().astype(int)[["source_index", "target_index"]]

    
    def get_transaction_count(self):
        features_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE}")
        return features_df.shape[0]
    
    def lookup_node_index(self, node_id):
        return self.features_df[self.features_df.transaction_id == node_id].index[0]
    
    def predict_timestamps(self, dataset_df):
        test_size = dataset_df[dataset_df["timestamp"] == -1].shape[0]
        X_train, X_test, y_train, _ = train_test_split(dataset_df.drop("timestamp",axis=1), dataset_df["timestamp"], test_size=test_size,shuffle=False)
        y_train -= 1
        classifier = xgb.XGBClassifier()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict_proba(X_test)
        y_pred = classifier.predict(X_test)
        y_pred += 1
        X_test["timestamp"] = y_pred
        pred_df = dataset_df.merge(X_test, how="right")
        dataset_df = dataset_df[dataset_df["timestamp"] != -1]
        return pd.concat([dataset_df, pred_df])

    def split_subgraphs(self, data):
        timestamps = data.x[data.edge_index.T[0].type(torch.LongTensor)][:,1]
        spans = data.x[:,2].unique()

        graphs = []
        for span in spans:
            nodes = data.x[data.x[:,2] == span]
            min_index = nodes.type(torch.LongTensor)[:,0].min()
            max_index = nodes.type(torch.LongTensor)[:,0].max()
            mask = (data.edge_index[:,0] <= max_index) & (data.edge_index[:,1] <= max_index) & (data.edge_index[:,0] >= min_index) & (data.edge_index[:,1] >= min_index)
            edge_indexes = data.edge_index[mask].type(torch.LongTensor).unique(return_inverse=True)[1]
            
            x = torch.cat([nodes[:,3:]],dim=1)
            y = data.y[data.x[:,2] == span]
            graphs.append(Data(
                x=x, y=y, edge_index=edge_indexes.T, 
                timestamp=span.item()
            ))

        return graphs 