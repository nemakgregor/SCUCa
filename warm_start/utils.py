import torch
from torch.utils.data import DataLoader, TensorDataset

def make_gru_loader(X_seq, X_static, Y_seq, batch_size=64, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(X_static, dtype=torch.float32),
            torch.tensor(Y_seq, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle)