from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pandas as pd
import torch
import os
import numpy as np
from torch_geometric.data import Data

from resources import constants


class EllipticUtility:
    DONT_TRAIN_COLS = ['transaction_id', 'timestamp', 'class']
    def __init__(self):
        self._initialize_dataframes()

    def _initialize_dataframes(self):
        self.edgelist_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.EDGELIST_FILE_ORIGINAL}")
        self.edgelist_df.rename(columns={
            "txId1": "source",
            "txId2": "target"
        }, inplace=True)
        self.features_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE_ORIGINAL}")
        self.features_df.rename(columns={
            "1": "timestamp",
            "230425980": "transaction_id"
        }, inplace=True)
        self.labels_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.CLASSES_FILE_ORIGINAL}")
        self.labels_df.rename(columns={
            "txId": "transaction_id"
        }, inplace=True)
    
    def preprocess(self, df):
        # Slice out transaction_id, timestamp, and class
        train_cols = [c for c in df.columns if c not in self.DONT_TRAIN_COLS]
        return df[train_cols].to_numpy()
        
    def train_classifier(self, dataset_df, classifier):
        train_df = dataset_df[dataset_df["class"] != 2]
        x_train = self.preprocess(train_df)
        y_train = train_df["class"].to_numpy()

        classifier.fit(x_train, y_train)
        return classifier

    def attempt_labeling(self, filename, classifier):
        dataset_df = self._init_dataset_df(filter_labeled=False)
        classifier = self.train_classifier(dataset_df, classifier)
        
        # Slight optimization
        unlabeled = dataset_df[dataset_df['class'] == 2]
        idx = unlabeled.index 
        pred_labels = classifier.predict(self.preprocess(unlabeled))
        dataset_df.loc[idx, 'class'] = pred_labels
        '''
        for index, row in dataset_df.iterrows():
            if row["class"] == 2:
                dataset_df.loc[index, "class"] = classifier.predict([row.drop("class")])[0]
        '''

        dataset_df.to_csv(f"{filename}")
        return dataset_df
    
    def lookup_classifier_name(self, classifier):
        if type(classifier) is RandomForestClassifier:
            return "RF"
        if type(classifier) is xgb.XGBClassifier:
            return "XGB"

    def get_datasets(self, filter_labeled=True, classifier=RandomForestClassifier(n_estimators=100)):
        predicted_labels_filename = f"{constants.BITCOIN_DATASET_DIR}/{constants.LABELED_ELLIPTIC_DATASET}_{self.lookup_classifier_name(classifier)}.csv"
        dataset_df = self._init_dataset_df(filter_labeled=filter_labeled)
        if not self._file_exists(predicted_labels_filename):
            predicted_labels_df = self.attempt_labeling(predicted_labels_filename, classifier)
        else:
            predicted_labels_df = pd.read_csv(predicted_labels_filename, index_col=0)
        actual_labels_graph = self.get_dataset_graph(dataset_df)
        predicted_labels_graph = self.get_dataset_graph(predicted_labels_df)
        return actual_labels_graph, predicted_labels_graph

    def get_dataset_graph(self, dataset_df,filter_labeled=True):
        # edges_filename = constants.EDGES_LABELED_FILENAME if filter_labeled else constants.EDGES_INDEXED_FILENAME
        edges_df = self.get_edge_list(dataset_df, generate_csv=False, is_labeled=filter_labeled)

        train_mask_tensor, val_mask_tensor, test_mask_tensor = self._get_mask_tensors(dataset_df)
        edges_tensor = torch.from_numpy(edges_df.to_numpy())
        x_tensor = torch.from_numpy(self.preprocess(dataset_df))
        ts = torch.from_numpy(dataset_df['timestamp'].to_numpy())

        '''
        if filter_labeled:
            x_tensor = self._to_tensor(dataset_df.reset_index().reset_index().drop("index",axis=1).drop("class",axis=1).to_numpy())
        else:
            x_tensor = self._to_tensor(dataset_df.reset_index().drop("class",axis=1).to_numpy())
        '''

        y_tensor = torch.from_numpy(dataset_df["class"].to_numpy().astype(int))

        return Data(
            x=x_tensor, edge_index=edges_tensor, y=y_tensor,
            num_nodes=x_tensor.size(0), train_mask=train_mask_tensor,
            val_mask=val_mask_tensor, test_mask=test_mask_tensor, ts=ts
        )

    def _init_dataset_df(self, filter_labeled=True):
        dataset_df = self.features_df.merge(self.labels_df,how="left").dropna()
        if filter_labeled:
            dataset_df = self.features_df.merge(self.labels_df[self.labels_df["class"] != "unknown"],how="left").dropna()
        dataset_df["class"] = pd.to_numeric(dataset_df["class"].str.replace("unknown", "0").replace("suspicious", "1"))
        dataset_df["class"] -= 2
        dataset_df["class"] = dataset_df["class"].abs()
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

    def get_edge_list(self, dataset_df, generate_csv=False, is_labeled=False):
        features_df = dataset_df.reset_index().drop("index",axis=1)
        filename = constants.EDGES_LABELED_FILENAME if is_labeled else constants.EDGES_INDEXED_FILENAME
        # if is_labeled:
        #     features_df = features_df.merge(self.labels_df[self.labels_df["class"] != "unknown"],how="left").dropna().reset_index().drop("index",axis=1)
        edges_df = self._get_edges_indexed(features_df)
        if generate_csv:
            edges_df.to_csv(f"{filename}",index=False)
        return edges_df
    
    def _get_edges_indexed(self, features_df):
        source_df = features_df.reset_index()[["index", "transaction_id"]].rename(columns={"index": "source_index","transaction_id": "source"})
        target_df = features_df.reset_index()[["index", "transaction_id"]].rename(columns={"index": "target_index","transaction_id": "target"})
        return self.edgelist_df.merge(source_df.drop_duplicates(subset=['source']),how="left").merge(target_df.drop_duplicates(subset=['target']),how="left").dropna().astype(int)[["source_index", "target_index"]]

    
    def get_transaction_count(self):
        features_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE_ORIGINAL}")
        return features_df.shape[0]
    
    def lookup_node_index(self, node_id):
        return self.features_df[self.features_df.transaction_id == node_id].index[0]
    
    def predict_timestamps(self, dataset_df):
        test_size = dataset_df[dataset_df["timestamp"] == -1].shape[0]
        X_train, X_test, y_train, _ = train_test_split(dataset_df.drop("timestamp",axis=1), dataset_df["timestamp"], test_size=test_size,shuffle=False)
        y_train -= 1
        classifier = xgb.XGBClassifier()
        classifier.fit(X_train, y_train)
        _ = classifier.predict_proba(X_test)
        y_pred = classifier.predict(X_test)
        y_pred += 1
        X_test["timestamp"] = y_pred
        pred_df = dataset_df.merge(X_test, how="right")
        dataset_df = dataset_df[dataset_df["timestamp"] != -1]
        return pd.concat([dataset_df, pred_df])

    def generate_train_and_test_graphs(self, data):
        train_nodes = data.x[data.x[:,2] <= 35]
        train_min_index = train_nodes.type(torch.LongTensor)[:,0].min()
        train_max_index = train_nodes.type(torch.LongTensor)[:,0].max()
        train_mask = (data.edge_index[:,0] <= train_max_index) & (data.edge_index[:,1] <= train_max_index) & (data.edge_index[:,0] >= train_min_index) & (data.edge_index[:,1] >= train_min_index)
        train_edge_indexes = data.edge_index[train_mask].type(torch.LongTensor).unique(return_inverse=True)[1]
            
        train_x = torch.cat([train_nodes[:,3:]],dim=1)
        train_y = data.y[data.x[:,2] <= 35]
        train_graph = Data(x=train_x, y=train_y, edge_index=train_edge_indexes.T)
        
        test_nodes = data.x[data.x[:,2] > 35]
        test_min_index = test_nodes.type(torch.LongTensor)[:,0].min()
        test_max_index = test_nodes.type(torch.LongTensor)[:,0].max()
        test_mask = (data.edge_index[:,0] <= test_max_index) & (data.edge_index[:,1] <= test_max_index) & (data.edge_index[:,0] >= test_min_index) & (data.edge_index[:,1] >= test_min_index)
        test_edge_indexes = data.edge_index[test_mask].type(torch.LongTensor).unique(return_inverse=True)[1]
            
        test_x = torch.cat([test_nodes[:,3:]],dim=1)
        test_y = data.y[data.x[:,2] > 35]
        test_graph = Data(x=test_x, y=test_y, edge_index=test_edge_indexes.T)
        return train_graph, test_graph
    
    def split_subgraphs(self, data):
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