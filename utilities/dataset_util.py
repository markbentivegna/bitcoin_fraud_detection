import pandas as pd
import torch
import os
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm 

from resources import constants
from utilities.graph_util import GraphUtility


class DatasetUtility:
    def __init__(self):
        self.graph_util = GraphUtility()
        self._initialize_dataframes()

    def get_u_tuples(self):
        return self.graph_util.get_u_graph()

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

    def get_dataset(self):
        dataset_df = self._init_dataset_df()
        if self._file_exists(f"{constants.EDGES_LABELED_FILENAME}"):
            edges_df = self.get_edges_csv(labeled=True)
        else:
            edges_df = self.get_edge_list(generate_csv=True, is_labeled=True)
        train_mask_tensor, val_mask_tensor, test_mask_tensor = self._get_mask_tensors(dataset_df)
        edges_tensor = self._to_tensor(edges_df.to_numpy())
        x_tensor = self._to_tensor(dataset_df.drop("class",axis=1).to_numpy())
        y_tensor = self._to_tensor(dataset_df["class"].to_numpy().astype(int))

        return Data(
            x=x_tensor, edge_index=edges_tensor, y=y_tensor,
            num_nodes=x_tensor.size(0), train_mask=train_mask_tensor,
            val_mask=val_mask_tensor, test_mask=test_mask_tensor
        )

    def _init_dataset_df(self):
        dataset_df = self.features_df.merge(self.labels_df[self.labels_df["class"] != "unknown"],how="left").dropna()
        dataset_df["class"] = pd.to_numeric(dataset_df["class"])
        dataset_df["class"] -= 1
        return dataset_df

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
        filename = constants.EDGES_INDEXED_FILENAME
        if is_labeled:
            features_df = self.features_df.merge(self.labels_df[self.labels_df["class"] != "unknown"],how="left").dropna().reset_index().drop("index",axis=1)
            filename = constants.EDGES_LABELED_FILENAME
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


class DatasetUtilityPyTorch():
    DATA = constants.BITCOIN_DATASET_DIR + '/'
    def __init__(self):
        self.nodemap = dict()
        self.nid = 0 


    def _get_or_add(self, node):
        if not node in self.nodemap:
            self.nodemap[node] = self.nid 
            self.nid += 1
        
        return self.nodemap[node]


    def build_edge_list(self):
        f = open(self.DATA + constants.EDGELIST_FILE, 'r')
        f.readline() # Skip header

        srcs,dsts = [],[]
        prog = tqdm(desc='Building edge list', total=constants.N_EDGES)
        line = f.readline()
        while(line):
            src,dst = line.strip().split(',')

             # Convert to sequential IDs
            src = self._get_or_add(src) 
            dst = self._get_or_add(dst) 

            srcs.append(src) 
            dsts.append(dst)
            
            prog.update()
            line = f.readline()

        f.close()
        prog.close()
        return torch.tensor([srcs,dsts])
    

    def build_features(self):
        f = open(self.DATA + constants.FEATURES_FILE, 'r')
        x = torch.zeros(constants.N_NODES, constants.FEAT_DIM)

        prog = tqdm(desc='Building features', total=constants.N_NODES)
        line = f.readline()
        while(line):
            tokens = line.strip().split(',')
            node, feats = tokens[0], tokens[1:]
            feats = torch.tensor([float(f) for f in feats])

            # This *shouldn't* throw a key error assuming
            # build_edge_list was called first
            nid = self.nodemap[node]
            x[nid] = feats 

            prog.update()
            line = f.readline()

        f.close() 
        prog.close()
        return x 
    

    def build_labels(self):
        f = open(self.DATA + constants.CLASSES_FILE, 'r') 
        f.readline() # Skip headers 
        ys = torch.zeros(constants.N_NODES)

        prog = tqdm(desc='Building ground truth', total=constants.N_NODES)
        line = f.readline()

        # Tiny bit faster than if statements
        ymap = {'unknown':0, '1':1, '2':2}

        while(line):
            node,y = line.strip().split(',')
            ys[self.nodemap[node]] = ymap[y]

            prog.update()
            line = f.readline()

        f.close()
        prog.close()
        return ys 
    
    def build_dataset(self):
        ei = self.build_edge_list()
        
        # These two can be run in parallel but build_labels takes
        # <1s so it's not really worth it
        x = self.build_features()
        y = self.build_labels()
        
        return Data(
            x=x, edge_index=ei, y=y,
            num_nodes=x.size(0)
        )


    def split_subgraphs(self, data):
        '''
        Time stamps associated with nodes in each edge
        supposed to be fully connected components accross each 
        timestamp. Want to pull out individual subgraphs spanning
        specific timestamps
        '''
        times = data.x[data.edge_index[0]][:,0]
        spans = times.unique()

        graphs = []
        for span in spans:
            ei = data.edge_index[:, times==span]
            nodes, reindex = ei.unique(return_inverse=True)

            # Don't bother including timestamp. It's superfluous
            x = data.x[nodes][:,1:]
            y = data.y[nodes]
            graphs.append(Data(x=x, y=y, edge_index=reindex, ts=span.item()))

        return graphs 