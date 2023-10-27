from resources import constants
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utilities.elliptic_util import EllipticUtility
from utilities.babd_util import BABDUtility
from utilities.bhd_util import BHDUtility
from utilities.creditcard_util import CreditCardUtility
from utilities.fraud_transaction_util import FraudTransactionUtility
from utilities.promo_code_util import PromoCodeUtility
from utilities.ethereum_transaction_util import EthereumTransactionUtility
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
        self.ethereum_util = EthereumTransactionUtility()

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
    
    def get_dataset(self, dataset):
        if dataset == constants.ELLIPTIC_DATASET:
            return self.elliptic_util.get_datasets(classifier=RandomForestClassifier(n_estimators=100))
        elif dataset == constants.BABD_DATASET:
            return self.babd_util.get_dataset()
        elif dataset == constants.BHD_DATASET:
            return self.bhd_util.get_dataset()
        elif dataset == constants.CREDITCARD_DATASET:
            return self.creditcard_util.get_dataset()
        elif dataset == constants.FRAUD_TRANSACTION_DATASET:
            return self.fraud_transaction_util.get_dataset()
        elif dataset == constants.PROMO_CODE_DATASET:
            return self.promo_code_util.get_dataset()
        elif dataset == constants.ETHEREUM_DATASET:
            return self.ethereum_util.get_dataset()
        
    def get_elliptic_graphs(self, dataset):
        data = self.get_dataset(dataset, filter_labeled=False)
        if dataset == constants.ELLIPTIC_DATASET:
            return self.elliptic_util.generate_train_and_test_graphs(data)
        else:
            return None

    def split_subgraphs(self, unlabeled_dataset, labeled_dataset, dataset):
        if dataset == constants.ELLIPTIC_DATASET:
            return self.elliptic_util.split_subgraphs(unlabeled_dataset), self.elliptic_util.split_subgraphs(labeled_dataset)