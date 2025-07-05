import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EdgeModel(nn.Module):
    
    def __init__(self, node_in_dim, edge_in_dim):
        super().__init__()
        self.node_conv1 = SAGEConv(node_in_dim, 64)
        self.node_conv2 = SAGEConv(64, 64)
        self.edge_mlp = nn.Sequential(
            nn.Linear(64*2 + edge_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.node_conv1(x, edge_index))
        x = torch.relu(self.node_conv2(x, edge_index))

        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=1)  

        out = self.edge_mlp(edge_features)
        return torch.sigmoid(out).squeeze(-1)
    

class GraphTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        loader_train,
        loader_val,
        epochs,
        device = "cpu",
        threshold = 0.7,
    ):
        self.model = model.to(device)
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.device = device
        self.epochs = epochs
        self.threshold = threshold

        self.optimizer = optimizer
        self.criterion  = criterion

    def _train_epoch(self):
        self.model.train()
        
        tot_loss = 0.0
        num_batches = 0

        for g in self.loader_train:
            g = g.to(self.device)
            self.optimizer.zero_grad()
            
            y_pred = self.model(g.x, g.edge_index, g.edge_attr)
            y_true = g.y.float()
            loss = self.criterion(y_pred, y_true)

            loss.backward()
            self.optimizer.step()

            tot_loss += loss.item()
            num_batches += 1

        return tot_loss / num_batches


    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        y_true = [] 
        y_prob = []

        for g in self.loader_val:
            g = g.to(self.device)
            p_hat = self.model(g.x, g.edge_index, g.edge_attr)
            y_true.append(g.y.cpu())
            y_prob.append(p_hat.cpu())

        y_true = torch.cat(y_true)
        y_prob  = torch.cat(y_prob)
        y_pred = (y_prob > self.threshold).int()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return acc, prec, recall, f1
       
       
    @torch.no_grad()    
    def predict(self, loader):
        self.model.eval()
        preds, trues = [], []

        for g in loader:
            g = g.to(self.device)
            pred = self.model(g.x, g.edge_index, g.edge_attr)
            preds.append(pred.cpu())
            trues.append(g.y.cpu())

        return torch.cat(preds), torch.cat(trues)


    def fit(self):

        for epoch in range(self.epochs):
            loss = self._train_epoch()
            acc, prec, recall, f1 = self._validate()

            print(
                f"Epoch {epoch:03d} | loss={loss:.4f} | "
                f"acc={acc:.3f} | prec={prec:.3f} | rec={recall:.3f} | f1={f1:.3f}"
            )