import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, n_features: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        n_in = n_features
        for i, n_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(n_in, n_out, kernel_size, dilation, dropout))
            n_in = n_out
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.network(x)
        last_step = out[:, :, -1]
        return torch.sigmoid(self.head(last_step)).squeeze(-1)

def make_windows(features_arr, labels_arr, window: int):
    X, y = [], []
    for i in range(window, len(features_arr)):
        X.append(features_arr[i - window : i])
        y.append(labels_arr[i])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

import numpy as np

def train_tcn(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cpu",
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    loss_fn = nn.BCELoss()
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device))
                val_loss = loss_fn(val_pred, y_val.to(device)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model

def predict_tcn(model: nn.Module, X: torch.Tensor, device: str = "cpu", batch_size: int = 128) -> np.ndarray:
    model.eval()
    model.to(device)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            p = model(xb.to(device)).cpu().numpy()
            preds.append(np.atleast_1d(p))
    return np.concatenate(preds) if preds else np.array([])
