import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from sklearn.metrics import recall_score


class SAGE_model(nn.Module):
    
    def __init__(self, in_dim):
        super().__init__()
        self.c1 = SAGEConv(in_dim, 128)
        self.c2 = SAGEConv(128, 64)
        self.act = nn.ReLU()
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x, edge_index):
        x = self.act(self.c1(x, edge_index))
        x = self.act(self.c2(x, edge_index)) 
        x = torch.sigmoid(self.fc(x))
        return x.squeeze(-1)
    

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

            p_hat = self.model(g.x, g.edge_index)
            y_true = g.y.float()
            loss = self.criterion(p_hat, y_true)

            loss.backward()
            self.optimizer.step()

            tot_loss += loss.item()
            num_batches += 1

        return tot_loss / num_batches


    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        y_true = [] 
        y_hat = []

        for g in self.loader_val:
            g = g.to(self.device)
            p_hat = self.model(g.x, g.edge_index)
            y_true.append(g.y.cpu())
            y_hat.append((p_hat > self.threshold).cpu())

        y_true = torch.cat(y_true)
        y_hat  = torch.cat(y_hat)
        return recall_score(y_true, y_hat)


    def fit(self):

        for epoch in range(self.epochs):
            loss   = self._train_epoch()
            recall = self._validate()

            print(f"Epoch {epoch:03d} | loss={loss:.4f} | recall_binding={recall:.3f}")
