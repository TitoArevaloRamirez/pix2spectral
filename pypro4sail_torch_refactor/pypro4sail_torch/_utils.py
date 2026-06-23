import math
from typing import Optional

import torch


def infer_device_dtype(*values, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    """Infer a PyTorch device/dtype from the first tensor in values."""
    for value in values:
        if torch.is_tensor(value):
            return value.device if device is None else device, value.dtype if dtype is None else dtype
    return torch.device("cpu") if device is None else torch.device(device), torch.float64 if dtype is None else dtype


def as_tensor(value, *, like=None, device=None, dtype=None):
    if like is not None and torch.is_tensor(like):
        device = like.device if device is None else device
        dtype = like.dtype if dtype is None else dtype
    if torch.is_tensor(value):
        return value.to(device=device or value.device, dtype=dtype or value.dtype)
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64
    return torch.as_tensor(value, device=device, dtype=dtype)


def deg2rad(x):
    return x * (math.pi / 180.0)


def rad2deg(x):
    return x * (180.0 / math.pi)


def scalar_like(value) -> bool:
    return (not torch.is_tensor(value)) or value.ndim == 0


def column(x):
    """Return scalar/1D input as (..., 1) for broadcasting over wavelengths."""
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def align_to(x, ref):
    """Append singleton dimensions to x until it can broadcast with ref."""
    while x.ndim < ref.ndim:
        x = x.unsqueeze(-1)
    return x


def maybe_squeeze_batch(x, single: bool):
    if single and torch.is_tensor(x) and x.ndim > 0 and x.shape[0] == 1:
        return x.squeeze(0)
    return x
