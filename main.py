from utilities.dataset_util import DatasetUtility
from models.InspectionL import InspectionL
from models.AdaGNN import AdaGNN
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim import Adam 
import torch
import os
import json
import pandas as pd

TRAIN_GRAPH_SIZE = 35
MODEL = "InspectionL"

def load_dataset():
    dataset_util = DatasetUtility()
    full_dataset = dataset_util.get_dataset(filter_labeled=False)
    return dataset_util.split_subgraphs(full_dataset)

def save_model(filename, model, optimizer, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, f"saved_models/{filename}")
def load_model(filename):
    return torch.load(f"saved_models/{filename}")

def checkpoint_exists(checkpoint_filename):
    return os.path.isfile(f"saved_models/{checkpoint_filename}") 

def results_file_exists(results_filename):
    return os.path.isfile(f"{results_filename}") 

def create_results_file(results_dict, reuslts_filename):
    results_columns = ["gnn","classifier","hidden_layers","output_dimension","precision","recall","f1","roc_auc","true_negative","false_positive","false_negative","true_positive"]
    pd.DataFrame([results_dict]).to_csv(f"{reuslts_filename}", header=results_columns)

def train_embedder(full_subgraphs, model, gnn, epochs=300):
    checkpoint_filename = f"{MODEL}_{gnn}_{epochs}.pt"
    optimizer = Adam(params=model.parameters(), lr=0.0001)
    initial_epoch = 0
    if checkpoint_exists(checkpoint_filename):
        checkpoint = load_model(checkpoint_filename)
        initial_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for graph_id in range(TRAIN_GRAPH_SIZE):
        graph = full_subgraphs[graph_id]
        timestamp_target = torch.tensor([(graph.timestamp-1) // 10])
        timestamp_target = timestamp_target.repeat(graph.x.size(0)).long()
        labels, timestamp_mask = graph.y, graph.y != 2
        for epoch in range(initial_epoch, epochs):
            model.train()
            optimizer.zero_grad()
            if MODEL == "AdaGNN":
                loss = model(graph.x, graph.edge_index, timestamp_target, labels, timestamp_mask)
            else:
                loss = model(graph.x, graph.edge_index)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch + 1 == epochs:
                print(f"[{graph_id}-{epoch}] Loss {loss.item()}")
                save_model(checkpoint_filename, model, optimizer, epoch)

def classifier_predictions(x_train, y_train, x_test, classifier):
    classifier.fit(x_train.detach().numpy(), y_train.detach().numpy())
    y_hat = classifier.predict(x_test.detach().numpy())
    predictions = classifier.predict_proba(x_test.detach().numpy())[:,1]
    return y_hat, predictions

def evaluate_performance(y_test, y_hat, predictions):
    return {
        "precision": precision_score(y_test, y_hat),
        "recall": recall_score(y_test, y_hat),
        "f1": f1_score(y_test, y_hat),
        "roc_auc": roc_auc_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, y_hat)
    }

def load_classifier_datasets(full_subgraphs, model):
    train_graphs, test_graphs = train_test_split(full_subgraphs, train_size = 0.7, shuffle=False)
    x_train, y_train, x_test, y_test = [], [], [], []
    model.eval()
    for graph in train_graphs:
        x_train.append(torch.cat([model.embed(graph.x, graph.edge_index), graph.x],dim=1)[graph.y != 2])
        y_train.append(graph.y[graph.y != 2])
    for graph in test_graphs:
        x_test.append(torch.cat([model.embed(graph.x, graph.edge_index), graph.x],dim=1)[graph.y != 2])
        y_test.append(graph.y[graph.y != 2])
    x_train = torch.cat(x_train,dim=0)
    y_train = torch.cat(y_train,dim=0)
    x_test = torch.cat(x_test,dim=0)
    y_test = torch.cat(y_test,dim=0)
    return x_train, y_train, x_test, y_test

def record_results(gnn, classifier, hidden_layers, output_layers, performance_dict, results_file="results/results.csv"):
    results_dict = {
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
    if not results_file_exists(results_file):
        create_results_file(results_dict,results_file)
    else:
        pd.DataFrame([results_dict]).to_csv(results_file, mode='a', header=False)


for gnn in ["GAT", "GIN", "GCN"]:
    HIDDEN_LAYERS = 128
    OUTPUT_DIMENSION = 128
    full_subgraphs = load_dataset()
    if MODEL == "AdaGNN":
        model = AdaGNN(full_subgraphs[0].x.size(1), HIDDEN_LAYERS, OUTPUT_DIMENSION, gnn=gnn)
    else:
        model = InspectionL(full_subgraphs[0].x.size(1), HIDDEN_LAYERS, OUTPUT_DIMENSION, gnn=gnn)
    train_embedder(full_subgraphs, model, gnn)

    x_train, y_train, x_test, y_test = load_classifier_datasets(full_subgraphs, model)

    y_hat, predictions = classifier_predictions(x_train, y_train, x_test, xgb.XGBClassifier())
    xgb_results_dict = evaluate_performance(y_test, y_hat, predictions)
    record_results(gnn, "XGBoost", HIDDEN_LAYERS, OUTPUT_DIMENSION, xgb_results_dict)
    # print(f"XGBClassifier results: {json.dumps(xgb_results_dict, indent=2)}")
    y_hat, predictions = classifier_predictions(x_train, y_train, x_test, RandomForestClassifier(n_estimators=100))
    rf_results_dict = evaluate_performance(y_test, y_hat, predictions)
    record_results(gnn, "RandomForest", HIDDEN_LAYERS, OUTPUT_DIMENSION, rf_results_dict)
    # print(f"RandomForest results: {json.dumps(rf_results_dict, indent=2)}")
    