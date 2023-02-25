import pandas as pd
import numpy as np
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
