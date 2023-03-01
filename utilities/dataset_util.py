import pandas as pd
import torch
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
        self.features_df = pd.read_csv(f"{constants.WORKING_DIR}/{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE}")
        self.features_df.rename(columns={
            "1": "timestamp",
            "230425980": "transaction_id"
        }, inplace=True)

    def get_edge_list(self):
        self.edgelist_df.to_csv(constants.EDGE_LIST_FILENAME)
        return self.edgelist_df.to_numpy()
    
    def get_transaction_count(self):
        features_df = pd.read_csv(f"{constants.WORKING_DIR}/{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE}")
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