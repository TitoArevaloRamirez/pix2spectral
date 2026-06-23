from functools import lru_cache
from importlib import resources
from typing import Tuple

import numpy as np
import torch


@lru_cache(maxsize=32)
def _load_spectra_cached(device_str: str, dtype_str: str) -> Tuple[torch.Tensor, ...]:
    dtype = getattr(torch, dtype_str)
    data_path = resources.files(__package__).joinpath("prospect_d_spectra.txt")
    arr = np.loadtxt(data_path, dtype=np.float64)
    # Original file columns: wl, nr, kab, kcar, kant, kbrown, kw, km.
    wl, nr, kab, kcar, kant, kbrown, kw, km = arr.T
    tensors = [torch.as_tensor(x, dtype=dtype, device=torch.device(device_str))
               for x in (wl, nr, kab, kcar, kbrown, kw, km, kant)]
    return tuple(tensors)


def get_spectra(device=None, dtype=torch.float64) -> Tuple[torch.Tensor, ...]:
    """Return PROSPECT-D spectral constants as tensors on the requested device.

    Returns ``wl, refr_index, Cab_k, Car_k, Cbrown_k, Cw_k, Cm_k, Ant_k``.
    """
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    if isinstance(dtype, str):
        dtype_name = dtype
    else:
        dtype_name = str(dtype).split(".")[-1]
    return _load_spectra_cached(str(device), dtype_name)
