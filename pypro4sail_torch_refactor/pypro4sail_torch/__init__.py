"""PyTorch refactor of pyPro4SAIL computational kernels.

This package keeps the scientific API close to the original NumPy code while
moving the PROSPECT-D, 4SAIL, coupled Pro4SAIL, costs, and radiation helpers to
PyTorch tensors. All tensor inputs preserve their device, so CUDA tensors stay
on GPU. Spectral text libraries are loaded once and cached as tensors per
(device, dtype).
"""

from .spectral_library import get_spectra
from .prospect import prospectd, prospectd_wl, prospectd_vec
from .four_sail import foursail, foursail_wl, calc_lidf_campbell, calc_lidf_verhoef
from .pypro4sail import run, run_tir, CalcStephanBoltzmann

__all__ = [
    "get_spectra",
    "prospectd", "prospectd_wl", "prospectd_vec",
    "foursail", "foursail_wl", "calc_lidf_campbell", "calc_lidf_verhoef",
    "run", "run_tir", "CalcStephanBoltzmann",
]
