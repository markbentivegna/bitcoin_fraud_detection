from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from utilities.model_util import ModelUtility
from resources import constants
from models.GraphGAN import GraphGAN
from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np

MODEL = "GraphGAN"

torch.set_num_threads(16)
dataset_util = DatasetUtility()
results_util = ResultsUtility()
model_util = ModelUtility(MODEL)
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
labeled_nodes_count = actual_labels_graphs.x.shape[0]

generator_embedding_matrix = np.random.rand(labeled_nodes_count, 128)
discriminator_embedding_matrix = np.random.rand(labeled_nodes_count, 128)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
model = GraphGAN(labeled_nodes_count, actual_labels_graphs, generator_embedding_matrix, discriminator_embedding_matrix).to(device)
model.train(device)
print("foo")