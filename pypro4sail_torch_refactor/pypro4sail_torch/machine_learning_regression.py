"""PyTorch regression utilities replacing the legacy scikit-learn training path.

The LUT simulators should call the tensorized radiative-transfer kernels in this
package. This file focuses on GPU training/inference for regression models.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressorTorch(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden_sizes: Sequence[int] = (128, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        last = n_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list


def _standardize_fit(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-12)
    return mean, std


def train_reg(X_array, y_array, *, hidden_sizes=(128, 128), epochs=200, batch_size=1024, lr=1e-3,
              weight_decay=0.0, validation_split=0.2, device=None, dtype=torch.float32,
              standardize=True, num_workers=0):
    """Train an MLP regressor on CPU or GPU.

    Returns ``model, metadata, history``. ``metadata`` includes feature/target
    standardization tensors and can be fed to ``test_reg``.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = torch.as_tensor(X_array, dtype=dtype, device=device)
    y = torch.as_tensor(y_array, dtype=dtype, device=device)
    if y.ndim == 1:
        y = y[:, None]

    if standardize:
        x_mean, x_std = _standardize_fit(X)
        y_mean, y_std = _standardize_fit(y)
        Xn = (X - x_mean) / x_std
        yn = (y - y_mean) / y_std
    else:
        x_mean = torch.zeros(1, X.shape[1], device=device, dtype=dtype); x_std = torch.ones_like(x_mean)
        y_mean = torch.zeros(1, y.shape[1], device=device, dtype=dtype); y_std = torch.ones_like(y_mean)
        Xn, yn = X, y

    n = Xn.shape[0]
    n_val = int(n * validation_split)
    perm = torch.randperm(n, device=device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = TensorDataset(Xn[train_idx], yn[train_idx])
    val_X, val_y = Xn[val_idx], yn[val_idx]
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = MLPRegressorTorch(X.shape[1], y.shape[1], hidden_sizes=hidden_sizes).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    hist = TrainingHistory(train_loss=[], val_loss=[])
    for _ in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]
        hist.train_loss.append(running / max(seen, 1))
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X), val_y).detach() if n_val else torch.tensor(float("nan"), device=device)
        hist.val_loss.append(float(val_loss))

    metadata = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "standardize": standardize}
    return model, metadata, hist


@torch.no_grad()
def test_reg(X_array, model, metadata=None, *, device=None, dtype=torch.float32, batch_size=8192):
    """Run batched inference with a trained ``MLPRegressorTorch``."""
    device = torch.device(device or next(model.parameters()).device)
    X = torch.as_tensor(X_array, dtype=dtype, device=device)
    if metadata is None:
        metadata = {}
    x_mean = metadata.get("x_mean", torch.zeros(1, X.shape[1], device=device, dtype=dtype)).to(device=device, dtype=dtype)
    x_std = metadata.get("x_std", torch.ones(1, X.shape[1], device=device, dtype=dtype)).to(device=device, dtype=dtype)
    y_mean = metadata.get("y_mean", torch.zeros(1, 1, device=device, dtype=dtype)).to(device=device, dtype=dtype)
    y_std = metadata.get("y_std", torch.ones_like(y_mean)).to(device=device, dtype=dtype)
    outs = []
    model.eval()
    for start in range(0, X.shape[0], batch_size):
        xb = (X[start:start + batch_size] - x_mean) / x_std
        outs.append(model(xb) * y_std + y_mean)
    return torch.cat(outs, dim=0)
