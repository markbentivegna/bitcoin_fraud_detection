import json
import pandas as pd
import os
from resources import constants

class GraphUtility():
    def __init__(self):
        pass

    def get_u_graph(self):
        if self.graph_file_exists():
            return self.get_u_graph_tuples()
        else:
            return self.generate_u_graph()

    def graph_file_exists(self):
        path_exists = os.path.isfile(constants.U_GRAPH_FILENAME)
        if path_exists:
            return True
        return False

    def get_u_graph_tuples(self):
        u_tuples_list = []
        with open (constants.U_GRAPH_FILENAME, "r") as f:
            u_dict = json.load(f)
            u_tuples_list = u_dict["graph"]
        return u_tuples_list

    def generate_u_graph(self):
        # local_features_count = 94
        # aggregate_features_count = 72
        edgelist_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.EDGELIST_FILE}")
        edgelist_df.rename(columns={
            "txId1": "transaction_id"
        }, inplace=True)
        features_df = pd.read_csv(f"{constants.BITCOIN_DATASET_DIR}/{constants.FEATURES_FILE}")
        features_df.rename(columns={
            "1": "timestamp",
            "230425980": "transaction_id"
        }, inplace=True)
        # local_columns = list(features_df.drop(["timestamp", "transaction_id"],axis=1).columns)[:local_features_count]
        # aggregate_columns = list(features_df.drop(["timestamp", "transaction_id"],axis=1).columns)[-aggregate_features_count:]
        features_df["timestamp_group"] = features_df["timestamp"] // 10 + 1

        u_graph = edgelist_df.merge(features_df, on="transaction_id",how="left")[["transaction_id", "txId2", "timestamp"]].dropna().values.tolist()
        self.write_graph_file(u_graph)
        return u_graph

    def write_graph_file(self, u_graph):
        u_dict = {"graph": u_graph}
        with open(constants.U_GRAPH_FILENAME, 'w') as f:
            json.dump(u_dict, f)
