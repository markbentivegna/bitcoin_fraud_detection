from resources import constants
from torch.optim import Adam 
import torch
import os


class ModelUtility:
    def __init__(self, model):
        self.MODEL = model

    def save_model(self, filename, model, optimizer, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, f"saved_models/{filename}")

    def load_model(self, filename):
        return torch.load(f"saved_models/{filename}")

    def checkpoint_exists(self, checkpoint_filename):
        return os.path.isfile(f"saved_models/{checkpoint_filename}") 

    def train_embedder(self, full_subgraphs, model, gnn, epochs=300):
        for graph_id in range(len(full_subgraphs)):
            checkpoint_filename = f"{self.MODEL}_{graph_id}_{gnn}_{epochs}.pt"
            optimizer = Adam(params=model.parameters(), lr=0.0001)
            initial_epoch = 0
            if self.checkpoint_exists(checkpoint_filename):
                checkpoint = self.load_model(checkpoint_filename)
                initial_epoch = checkpoint["epoch"] + 1
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            # graph = full_subgraphs[graph_id]
            graph = full_subgraphs
            # timestamp_target = torch.tensor([(graph.timestamp-1) // 10])
            # timestamp_target = timestamp_target.repeat(graph.x.size(0)).long()
            labels, timestamp_mask = graph.y, graph.y != 2
            for epoch in range(initial_epoch, epochs):
                model.train()
                optimizer.zero_grad()
                if self.MODEL == "AdaGNN":
                    print("foo")
                    # loss = model(graph.x, graph.edge_index, timestamp_target, labels, timestamp_mask)
                else:
                    loss = model(graph.x, graph.edge_index)
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0 or epoch + 1 == epochs:
                    print(f"[{graph_id}-{epoch}] Loss {loss.item()}")
                    self.save_model(checkpoint_filename, model, optimizer, epoch)
