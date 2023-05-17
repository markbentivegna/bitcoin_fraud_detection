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

    def get_dataset(self, filter_labeled=True):
        dataset_df = self._init_dataset_df(filter_labeled=filter_labeled)
        filename = constants.EDGES_LABELED_FILENAME if filter_labeled else constants.EDGES_INDEXED_FILENAME
        if self._file_exists(filename):
            edges_df = self.get_edges_csv(labeled=filter_labeled)
        else:
            edges_df = self.get_edge_list(generate_csv=True, is_labeled=filter_labeled)
        train_mask_tensor, val_mask_tensor, test_mask_tensor = self._get_mask_tensors(dataset_df)
        edges_tensor = self._to_tensor(edges_df.to_numpy())
        x_tensor = self._to_tensor(dataset_df.drop("class",axis=1).to_numpy())
        y_tensor = self._to_tensor(dataset_df["class"].to_numpy().astype(int))

        return Data(
            x=x_tensor, edge_index=edges_tensor, y=y_tensor,
            num_nodes=x_tensor.size(0), train_mask=train_mask_tensor,
            val_mask=val_mask_tensor, test_mask=test_mask_tensor
        )

    def _init_dataset_df(self, filter_labeled=True):
        dataset_df = self.features_df.merge(self.labels_df,how="left").dropna()
        if filter_labeled:
            dataset_df = self.features_df.merge(self.labels_df[self.labels_df["class"] != "unknown"],how="left").dropna()
        dataset_df["class"] = pd.to_numeric(dataset_df["class"].str.replace("unknown", "0"))
        dataset_df["class"] -= 2
        dataset_df["class"] = dataset_df["class"].abs()
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
    
    def split_subgraphs(self, data):
        timestamps = data.x[data.edge_index.T[0].type(torch.LongTensor)][:,1]
        spans = data.x[:,1].unique()

        graphs = []
        for span in spans:
            selected_edge_indexes = data.edge_index[timestamps==span,:]
            nodes, edge_indexes = selected_edge_indexes.unique(return_inverse=True)
            nodes = nodes.type(torch.LongTensor)

            x = torch.cat([data.x[:, 2:]], dim=1)[nodes]
            y = data.y[nodes]
            graphs.append(Data(
                x=x, y=y, edge_index=edge_indexes.T, 
                timestamp=span.item()
            ))

        return graphs 


class DatasetUtilityPyTorch():
    def __init__(self, dataset=constants.BITCOIN_DATASET_DIR):
        self.DATA = dataset + '/'
        self.nodemap = dict()
        self.nid = 0 
        self.nid_to_node_name = []

        self.N_EDGES = constants.N_EDGES if dataset==constants.BITCOIN_DATASET_DIR \
            else constants.N_AUGMENTED_EDGES


    def _get_or_add(self, node):
        if not node in self.nodemap:
            self.nodemap[node] = self.nid 
            self.nid_to_node_name.append(int(node))
            self.nid += 1
        
        return self.nodemap[node]


    def build_edge_list(self):
        f = open(self.DATA + constants.EDGELIST_FILE, 'r')
        f.readline() # Skip header

        srcs,dsts = [],[]
        prog = tqdm(desc='Building edge list', total=self.N_EDGES)
        line = f.readline()
        while(line):
            src,dst = line.strip().split(',')

            # I think any new transaction from the augmented data
            # involves completely new nodes for both src and dst
            # E.g. the first transaction with a negative is listed as 
            # -28784671,152039820
            # but there is no record of 152039820. There is, however
            # a record associated with node -152039820 so I think that is
            # the (poorly documented) implication
            if src.startswith('-'):
                dst = '-'+dst

             # Convert to sequential IDs
            src = self._get_or_add(src) 
            dst = self._get_or_add(dst) 

            srcs.append(src) 
            dsts.append(dst)
            
            prog.update()
            line = f.readline()

        f.close()
        prog.close()

        # All nodes now added. Finalize by making it a tensor
        self.nid_to_node_name = torch.tensor(self.nid_to_node_name)
        return torch.tensor([srcs,dsts])
    

    def build_features(self):
        f = open(self.DATA + constants.FEATURES_FILE, 'r')
        x = torch.zeros(len(self.nodemap), constants.FEAT_DIM)

        prog = tqdm(desc='Building features', total=len(self.nodemap))
        line = f.readline()
        while(line):
            tokens = line.strip().split(',')
            node, feats = tokens[0], tokens[1:]
            feats = torch.tensor([float(f) for f in feats])

            # This *shouldn't* throw a key error assuming
            # build_edge_list was called first
            if (nid := self.nodemap.get(node)) is None:
                print("Skipping feature for node %s that doesn't appear in edgelist" % node)
            x[nid] = feats 

            prog.update()
            line = f.readline()

        f.close() 
        prog.close()
        return x 
    

    def build_labels(self):
        f = open(self.DATA + constants.CLASSES_FILE, 'r') 
        f.readline() # Skip headers 
        ys = torch.zeros(len(self.nodemap))

        prog = tqdm(desc='Building ground truth', total=len(self.nodemap))
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
            graphs.append(Data(
                x=x, y=y, edge_index=reindex, 
                ts=span.item(), nid_to_node_name=self.nid_to_node_name[nodes]
            ))

        return graphs 