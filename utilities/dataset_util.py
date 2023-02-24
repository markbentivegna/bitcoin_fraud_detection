import pandas as pd
import numpy as np
from resources import constants
from utilities.graph_util import GraphUtility

class DatasetUtility:
    def __init__(self):
        self.graph_util = GraphUtility()

    def get_u_tuples(self):
        return self.graph_util.get_u_graph()

    def get_adjacency_matrix(self):
        edgelist_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.EDGELIST_FILE}")
        features_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE}")
        features_df.rename(columns={
            "1": "timestamp",
            "230425980": "transaction_id"
        }, inplace=True)
        N = edgelist_df.shape[0]
        adj_matrix =  np.zeros((N, N))
        for _, row in edgelist_df.iterrows():
            try:
                source = int(features_df[features_df["transaction_id"] == int(row[0])].index[0])
                destination = int(features_df[features_df["transaction_id"] == int(row[1])].index[0])
                adj_matrix[source][destination] = 1
            except Exception as e:
                print(e)

        # np.savetxt(constants.ADJ_MATRIX_FILENAME, adj_matrix)
        return adj_matrix