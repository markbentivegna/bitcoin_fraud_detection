from utilities.dataset_util import DatasetUtility
import os
import pandas as pd
import torch
from resources import constants
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ResultsUtility:
    def __init__(self):
        pass

    def evaluate_performance(self, y_test, y_hat, predictions):
        return {
            "precision": precision_score(y_test, y_hat),#, average="micro"),
            "recall": recall_score(y_test, y_hat),#, average="micro"),
            "f1": f1_score(y_test, y_hat),#, average="micro"),
            "roc_auc": roc_auc_score(y_test, predictions),
            "confusion_matrix": confusion_matrix(y_test, y_hat)
        }

    def results_file_exists(self, results_filename):
        return os.path.isfile(f"{results_filename}") 

    def create_results_file(self, results_dict, reuslts_filename, flat=True):
        if flat:
            pd.DataFrame([results_dict]).to_csv(f"{reuslts_filename}", header=list(results_dict.keys()))
        else:
            pd.DataFrame(results_dict).to_csv(f"{reuslts_filename}", header=list(results_dict.keys()))

    def record_results(self, model, gnn, classifier, hidden_layers, output_layers, performance_dict, results_file="results/results.csv"):
        results_dict = {
            "model": model,
            "gnn": gnn,
            "classifier": classifier,
            "hidden_layers": hidden_layers,
            "output_dimension": output_layers,
            "precision": performance_dict["precision"],
            "recall": performance_dict["recall"],
            "f1": performance_dict["f1"],
            "roc_auc": performance_dict["roc_auc"],
            "true_negative": int(performance_dict["confusion_matrix"].flatten()[0]),
            "false_positive": int(performance_dict["confusion_matrix"].flatten()[1]),
            "false_negative": int(performance_dict["confusion_matrix"].flatten()[2]),
            "true_positive": int(performance_dict["confusion_matrix"].flatten()[3]),
        }
        if not self.results_file_exists(results_file):
            self.create_results_file(results_dict,results_file)
        else:
            pd.DataFrame([results_dict]).to_csv(results_file, mode='a', header=False)

    def record_mispredictions(self, y_test, y_hat, model, gnn, classifier, false_positives_file="results/false_positive_nodes.csv", false_negatives_file="results/false_negative_nodes.csv"):
        dataset_util = DatasetUtility()
        labeled_dataset = dataset_util.get_dataset(constants.ELLIPTIC_DATASET)[0]
        false_positive_mask = (torch.tensor(y_hat) == 1) & (y_test == 0)
        false_negative_mask = (torch.tensor(y_hat) == 0) & (y_test == 1)
        test_split_index = false_positive_mask.shape[0]
        false_positive_nodes = labeled_dataset.x[-test_split_index:][false_positive_mask][:,0]
        false_negative_nodes = labeled_dataset.x[-test_split_index:][false_negative_mask][:,0]
        false_positives_dict = {
            "model": model,
            "gnn": gnn,
            "classifier": classifier,
            "node_id": false_positive_nodes.flatten().detach().numpy().astype(int)
        }
        false_negatives_dict = {
            "model": model,
            "gnn": gnn,
            "classifier": classifier,
            "node_id": false_negative_nodes.flatten().detach().numpy().astype(int)
        }
        if not self.results_file_exists(false_positives_file):
            self.create_results_file(false_positives_dict,false_positives_file,flat=False)
        else:
            pd.DataFrame(false_positives_dict).to_csv(false_positives_file, mode='a', header=False)
        
        if not self.results_file_exists(false_negatives_file):
            self.create_results_file(false_negatives_dict,false_negatives_file,flat=False)
        else:
            pd.DataFrame(false_negatives_dict).to_csv(false_negatives_file, mode='a', header=False)
