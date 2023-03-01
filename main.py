from utilities.dataset_util import DatasetUtilityPyTorch, DatasetUtility
from resources import constants
import numpy as np
dataset_util = DatasetUtility()
edge_list = dataset_util.get_edge_list()

N = constants.N_NODES
adj_matrix = np.zeros((N,N))
adj_matrix[edge_list[:,0], edge_list[:,1]] = 1
pytorch_dataset_util = DatasetUtilityPyTorch()

dataset = pytorch_dataset_util.build_dataset()