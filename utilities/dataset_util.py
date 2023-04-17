from resources import constants
from utilities.graph_util import GraphUtility
from utilities.elliptic_util import EllipticUtility
from utilities.babd_util import BABDUtility
from utilities.bhd_util import BHDUtility
from utilities.creditcard_util import CreditCardUtility
from utilities.fraud_transaction_util import FraudTransactionUtility


class DatasetUtility:
    def __init__(self):
        # self.graph_util = GraphUtility()
        self.elliptic_util = EllipticUtility()
        self.babd_util = BABDUtility()
        self.bhd_util = BHDUtility()
        self.creditcard_util = CreditCardUtility()
        self.fraud_transaction_util = FraudTransactionUtility()

    # def get_u_tuples(self):
    #     return self.graph_util.get_u_graph()

    def load_dataset(self, dataset):
        full_dataset = self.get_dataset(dataset, filter_labeled=False)
        return self.split_subgraphs(full_dataset, dataset)

    def get_dataset(self, dataset, filter_labeled=True):
        if dataset == constants.ELLIPTIC_DATASET:
            return self.elliptic_util.get_dataset(filter_labeled=filter_labeled)
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