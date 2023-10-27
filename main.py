from utilities.dataset_util import DatasetUtility
from utilities.results_util import ResultsUtility
from utilities.model_util import ModelUtility
from models.InspectionL import InspectionL
from models.AdaGNN import AdaGNN
from torch.optim import Adam 
from resources import constants
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import torch
import networkx as nx
from torch_geometric import utils
import numpy as np

MODEL = "InspectionL"
torch.set_num_threads(16)

<<<<<<< HEAD
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
=======
def classifier_predictions(x_train, y_train, x_test, classifier, oversample=False):
    licit_indices = np.where(y_train == 0)[0]
    illicit_indices = np.where(y_train == 1)[0]
    if oversample:
        for _ in range(10):
            random_illicit_index = np.random.choice(illicit_indices, 75)
            random_licit_index = np.random.choice(licit_indices, 100)
            training_subset = np.concatenate((random_illicit_index.astype(int), random_licit_index.astype(int)), axis=0)
            classifier.fit(x_train[training_subset], y_train[training_subset])
    else:
        classifier.fit(x_train, y_train)
    y_hat = classifier.predict(x_test)
    predictions = classifier.predict_proba(x_test)[:,1]
>>>>>>> 0171899 (feat: Adding GAN)
    return y_hat, predictions

# def load_train_test_classifier_dataset(train_graph, test_graph):
#     model.eval()
#     x_train = (torch.cat([model.embed(train_graph.x, train_graph.edge_index), train_graph.x],dim=1)[train_graph.y != 2])
#     y_train = (train_graph.y[train_graph.y != 2])
#     x_test = (torch.cat([model.embed(test_graph.x, test_graph.edge_index), test_graph.x],dim=1)[test_graph.y != 2])
#     y_test = (test_graph.y[test_graph.y != 2])
#     return x_train, y_train, x_test, y_test

def load_rolling_classifier_datasets(full_subgraphs, include_embeddings=True):
    train_graphs = full_subgraphs[:-1]
    test_graphs = [full_subgraphs[-1]]
    x_train, y_train, x_test, y_test = [], [], [], []
    model.eval()
    for i in range(len(train_graphs)):
        if include_embeddings:
            x_train.append(torch.cat([model.embed(train_graphs[i].x, train_graphs[i].edge_index), train_graphs[i].x],dim=1)[train_graphs[i].y != 2])
        else:
            x_train.append(torch.cat([train_graphs[i].x[:,:constants.LOCAL_FEATS]],dim=1)[train_graphs[i].y != 2])
        y_train.append(train_graphs[i].y[train_graphs[i].y != 2])
    for i in range(len(test_graphs)):
        if include_embeddings:
            x_test.append(torch.cat([model.embed(test_graphs[i].x, test_graphs[i].edge_index), test_graphs[i].x],dim=1)[test_graphs[i].y != 2])
        else:
            x_test.append(torch.cat([test_graphs[i].x[:,:constants.LOCAL_FEATS]],dim=1)[test_graphs[i].y != 2])
        y_test.append(test_graphs[i].y[test_graphs[i].y != 2])
    x_train = torch.cat(x_train,dim=0)
    y_train = torch.cat(y_train,dim=0)
    x_test = torch.cat(x_test,dim=0)
    y_test = torch.cat(y_test,dim=0)
    return x_train, y_train, x_test, y_test

def load_temporal_classifier_datasets(full_subgraphs, include_embeddings=True):
    train_graphs, test_graphs = train_test_split(full_subgraphs, train_size = 0.7, shuffle=False)
    x_train, y_train, x_test, y_test = [], [], [], []
    model.eval()
    for graph in train_graphs:
        if include_embeddings:
            x_train.append(torch.cat([model.embed(graph.x, graph.edge_index), graph.x],dim=1)[graph.y != 2])
        else:
            x_train.append(torch.cat([graph.x[:,:constants.LOCAL_FEATS]],dim=1)[graph.y != 2])
        y_train.append(graph.y[graph.y != 2])
    for graph in test_graphs:
        if include_embeddings:
            x_test.append(torch.cat([model.embed(graph.x, graph.edge_index), graph.x],dim=1)[graph.y != 2])
        else:
            x_test.append(torch.cat([graph.x[:,:constants.LOCAL_FEATS]],dim=1)[graph.y != 2])
        y_test.append(graph.y[graph.y != 2])
    x_train = torch.cat(x_train,dim=0)
    y_train = torch.cat(y_train,dim=0)
    x_test = torch.cat(x_test,dim=0)
    y_test = torch.cat(y_test,dim=0)
    return x_train, y_train, x_test, y_test

<<<<<<< HEAD
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
=======
def control_group_simulation(gnn, subgraphs, classifier, classifier_name):
    x_train, y_train, x_test, y_test = load_temporal_classifier_datasets(subgraphs)
    y_hat, predictions = classifier_predictions(x_train.detach().numpy(), y_train.detach().numpy(), x_test.detach().numpy(), RandomForestClassifier(n_estimators=100))
    rf_results_dict = results_util.evaluate_performance(y_test, y_hat, predictions)
    get_performance_by_timestamp(subgraphs, y_test, y_hat, predictions)
    print("foo")
>>>>>>> 0171899 (feat: Adding GAN)

def get_performance_by_timestamp(subgraphs, y_test, y_hat, predictions):
    test_graphs = subgraphs[constants.TRAIN_GRAPH_SIZE:]
    offset = 0
    final_results_dict = {
        "precision": [],
        "recall": [],
        "f1": []
    }
    for graph in test_graphs:
        graph_samples_count = graph.y.shape[0]
        timestamp_predictions = y_hat[offset:offset+graph_samples_count]
        results_dict = results_util.evaluate_performance(y_test[offset:offset+graph_samples_count], timestamp_predictions, timestamp_predictions)
        final_results_dict["precision"].append(results_dict["precision"])
        final_results_dict["recall"].append(results_dict["recall"])
        final_results_dict["f1"].append(results_dict["f1"])
        offset += graph_samples_count
    return final_results_dict


def rolling_classifier_simulation(gnn, actual_labels_graphs, predicted_labels_graphs, classifier, classifier_name):
    y_test_full, y_hat_full, predictions_full = [], [], []
    results_array = []
    for i in range(constants.TRAIN_GRAPH_SIZE - 1, len(actual_labels_graphs)):
    # for i in range(42, len(actual_labels_graphs)):
        current_graphs = predicted_labels_graphs[:i]
        current_graphs.append(actual_labels_graphs[i])
        if i > 41:
            current_graphs = actual_labels_graphs[:i]
            current_graphs.append(actual_labels_graphs[i])
            x_train, y_train, x_test, y_test = load_rolling_classifier_datasets(current_graphs[41:])
            y_hat, predictions = classifier_predictions(x_train.detach().numpy(), y_train.detach().numpy(), x_test.detach().numpy(), classifier,oversample=True)
        else:
            x_train, y_train, x_test, y_test = load_rolling_classifier_datasets(current_graphs)
            y_hat, predictions = classifier_predictions(x_train.detach().numpy(), y_train.detach().numpy(), x_test.detach().numpy(), classifier)
        results_dict = results_util.evaluate_performance(y_test, y_hat, predictions)
        results_dict["index"] = i
        results_array.append(results_dict)
        y_test_full = np.concatenate((y_test_full,y_test), axis=0 )
        y_hat_full = np.concatenate((y_hat_full,y_hat), axis=0 )
        predictions_full = np.concatenate((predictions_full,predictions), axis=0 )
        if results_dict['precision'] <= 0.5 or results_dict['recall'] <= 0.5:
            print("failure")
        print(f"TESTING ON INDEX: {i} PRECISION: {results_dict['precision']} RECALL: {results_dict['recall']} F1: {results_dict['f1']}")
    final_results_dict = results_util.evaluate_performance(y_test_full, y_hat_full, predictions_full)
    results_util.record_results(MODEL, gnn, classifier_name, constants.HIDDEN_LAYERS, constants.OUTPUT_DIMENSION, final_results_dict)
    results_util.record_mispredictions(y_test, y_hat, MODEL, gnn, classifier_name)

dataset_util = DatasetUtility()
results_util = ResultsUtility()
model_util = ModelUtility(MODEL)
actual_labels_graphs, predicted_labels_graphs = dataset_util.load_dataset(constants.ELLIPTIC_DATASET)
# babd_dataset = dataset_util.load_dataset(constants.BABD_DATASET)
# bhd_dataset = dataset_util.load_dataset(constants.BHD_DATASET)
# creditcard_dataset = dataset_util.load_dataset(constants.CREDITCARD_DATASET)
# fraud_transaction_dataset = dataset_util.load_dataset(constants.FRAUD_TRANSACTION_DATASET)


print("foo")



# promo_code_dataset = dataset_util.load_dataset(constants.PROMO_CODE_DATASET)

# x_train, x_test, y_train, y_test = train_test_split(fraud_transaction_dataset.drop("label",axis=1), fraud_transaction_dataset["label"], train_size = 0.7, shuffle=True)

# y_hat, predictions = classifier_predictions(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), xgb.XGBClassifier())
# xgb_results_dict = results_util.evaluate_performance(y_test, y_hat, predictions)
# results_util.record_results("NA", "NA", "XGBoost_PC", "NA", "NA", xgb_results_dict)
# #XGBoost accuracy:
# y_hat, predictions = classifier_predictions(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), RandomForestClassifier(n_estimators=100))
# rf_results_dict = results_util.evaluate_performance(y_test, y_hat, predictions)
# results_util.record_results("NA", "NA", "RandomForest_PC", "NA", "NA", rf_results_dict)
# #RF Accuracy: 

for gnn in ["GAT", "GIN", "GCN"]:
    if MODEL == "AdaGNN":
        model = AdaGNN(actual_labels_graphs[0].x.size(1), constants.HIDDEN_LAYERS, constants.OUTPUT_DIMENSION, gnn=gnn)
    else:
<<<<<<< HEAD
        model = InspectionL(full_subgraphs[0].x.size(1), HIDDEN_LAYERS, OUTPUT_DIMENSION, gnn=gnn)
    train_embedder(full_subgraphs, model, gnn)
=======
        model = InspectionL(actual_labels_graphs[0].x.size(1), constants.HIDDEN_LAYERS, constants.OUTPUT_DIMENSION, gnn=gnn)
        # model = InspectionL(167, constants.HIDDEN_LAYERS, constants.OUTPUT_DIMENSION, gnn=gnn)
    model_util.train_embedder(predicted_labels_graphs, model, gnn,epochs=3)
    control_group_simulation(gnn, predicted_labels_graphs, RandomForestClassifier(n_estimators=100), gnn)
    rolling_classifier_simulation(gnn, actual_labels_graphs, predicted_labels_graphs, xgb.XGBClassifier(), "XGBoost")
    rolling_classifier_simulation(gnn, actual_labels_graphs, predicted_labels_graphs, RandomForestClassifier(n_estimators=100), "RandomForest")
    # rolling_classifier_simulation(gnn, actual_labels_graphs, predicted_labels_graphs, BaggingClassifier(base_estimator=AdaBoostClassifier(), n_estimators=100, random_state=0), "AdaBoost", model)
>>>>>>> 0171899 (feat: Adding GAN)

    # y_hat, predictions = classifier_predictions(x_train.detach().numpy(), y_train.detach().numpy(), x_test.detach().numpy(), RandomForestClassifier(n_estimators=100))
    # rf_results_dict = results_util.evaluate_performance(y_test, y_hat, predictions)
    # results_util.record_results(MODEL, gnn, "RandomForest", constants.HIDDEN_LAYERS, constants.OUTPUT_DIMENSION, rf_results_dict)
    # results_util.record_mispredictions(y_test, y_hat, MODEL, gnn, "RandomForest")

#     x_train, y_train, x_test, y_test = load_classifier_datasets(full_subgraphs, model,include_embeddings=False)
    
#     y_hat, predictions = classifier_predictions(x_train.detach().numpy(), y_train.detach().numpy(), x_test.detach().numpy(), xgb.XGBClassifier())
#     xgb_no_embed_results_dict = results_util.evaluate_performance(y_test, y_hat, predictions)
#     results_util.record_results("NA", "NA", "XGBoost_NO_EMBED", "NA", "NA", xgb_no_embed_results_dict)
#     results_util.record_mispredictions(y_test, y_hat, "NA", "NA", "XGBoost_NO_EMBED")

#     y_hat, predictions = classifier_predictions(x_train.detach().numpy(), y_train.detach().numpy(), x_test.detach().numpy(), RandomForestClassifier(n_estimators=100))
#     rf_no_embed_results_dict = results_util.evaluate_performance(y_test, y_hat, predictions)
#     results_util.record_results("NA", "NA", "RandomForest_NO_EMBED", "NA", "NA", rf_no_embed_results_dict)
#     results_util.record_mispredictions(y_test, y_hat, "NA", "NA", "RandomForest_NO_EMBED")