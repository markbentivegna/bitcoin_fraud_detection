from resources import constants
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utilities.elliptic_util import EllipticUtility
from utilities.babd_util import BABDUtility
from utilities.bhd_util import BHDUtility
from utilities.creditcard_util import CreditCardUtility
from utilities.fraud_transaction_util import FraudTransactionUtility
from utilities.promo_code_util import PromoCodeUtility
from torch_geometric import utils
import networkx as nx
import matplotlib.pyplot as plt

class DatasetUtility:
    def __init__(self):
        # self.graph_util = GraphUtility()
        self.elliptic_util = EllipticUtility()
        self.babd_util = BABDUtility()
        self.bhd_util = BHDUtility()
        self.creditcard_util = CreditCardUtility()
        self.fraud_transaction_util = FraudTransactionUtility()
        self.promo_code_util = PromoCodeUtility()

    # def get_u_tuples(self):
    #     return self.graph_util.get_u_graph()

    def load_dataset(self, dataset):
        if dataset is not constants.ELLIPTIC_DATASET:
            full_dataset = self.get_dataset(dataset)
        else:
            actual_labels_graphs, predicted_labels_graphs = self.get_dataset(dataset)
            return actual_labels_graphs, predicted_labels_graphs
            # return self.split_subgraphs(actual_labels_graphs, predicted_labels_graphs, dataset)
        return full_dataset
    
    def get_dataset(self, dataset, filter_labeled=True):
        if dataset == constants.ELLIPTIC_DATASET:
            return self.elliptic_util.get_datasets(filter_labeled=filter_labeled)
        elif dataset == constants.BABD_DATASET:
            return self.babd_util.get_dataset()
        elif dataset == constants.BHD_DATASET:
            return self.bhd_util.get_dataset()
        elif dataset == constants.CREDITCARD_DATASET:
            return self.creditcard_util.get_dataset()
        elif dataset == constants.FRAUD_TRANSACTION_DATASET:
            return self.fraud_transaction_util.get_dataset()

    def split_subgraphs(self, data, dataset):
        if dataset == constants.ELLIPTIC_DATASET:
            return self.elliptic_util.split_subgraphs(data)
        elif dataset == constants.BABD_DATASET:
            return data
        elif dataset == constants.BHD_DATASET:
            return data
        elif dataset == constants.CREDITCARD_DATASET:
            return data
        elif dataset == constants.FRAUD_TRANSACTION_DATASET:
            return data