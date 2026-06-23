# pyPro4SAIL -> PyTorch refactor notes

## Executive summary

The original zip is a NumPy/SciPy/scikit-learn implementation of PROSPECT-D + 4SAIL with hand-coded Jacobians, CPU-side CMA-ES, LUT generation, and sensor/soil text libraries. The GPU-critical code is concentrated in `prospect.py`, `four_sail.py`, `pypro4sail.py`, `cost_functions.py`, `prospect_jacobian.py`, `four_sail_jacobian.py`, and `radiation_helpers.py`.

This refactor creates a new package named `pypro4sail_torch`. It preserves the main scientific APIs while replacing NumPy arrays with PyTorch tensors, making the kernels usable on CPU or CUDA. It also removes the SciPy `expi` dependency from PROSPECT by using the polynomial `trans_approx` already present in the legacy code.

## How to use

```python
import torch
import pypro4sail_torch as p4s

# CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

wl, canopy = p4s.run(
    torch.tensor(1.5, device=device, dtype=dtype),  # N
    torch.tensor(40.0, device=device, dtype=dtype), # Cab
    torch.tensor(8.0, device=device, dtype=dtype),  # Car
    torch.tensor(0.0, device=device, dtype=dtype),  # Cbrown
    torch.tensor(0.01, device=device, dtype=dtype), # Cw/EWT
    torch.tensor(0.009, device=device, dtype=dtype),# Cm/LMA
    torch.tensor(1.0, device=device, dtype=dtype),  # Ant
    torch.tensor(3.0, device=device, dtype=dtype),  # LAI
    torch.tensor(0.01, device=device, dtype=dtype), # hotspot
    torch.tensor(30.0, device=device, dtype=dtype), # solar zenith
    torch.tensor(180.0, device=device, dtype=dtype),# solar azimuth
    torch.tensor(10.0, device=device, dtype=dtype), # view zenith
    torch.tensor(180.0, device=device, dtype=dtype),# view azimuth
    57.0,
    skyl=0.2,
    device=device,
    dtype=dtype,
)

# Autograd works
Cab = torch.tensor(40.0, device=device, dtype=dtype, requires_grad=True)
_, canopy = p4s.run(1.5, Cab, 8.0, 0.0, 0.01, 0.009, 1.0, 3.0, 0.01, 30.0, 180.0, 10.0, 180.0, 57.0, device=device, dtype=dtype)
loss = canopy[100:150].mean()
loss.backward()
print(Cab.grad)
```

## File-by-file analysis

| Original file | Role in original zip | Main CPU / PyTorch problems found | Refactor in `pypro4sail_torch` |
|---|---|---|---|
| `__init__.py` | Exposes `get_spectra()` and a global `spectral_lib`. | Imports assume package naming that conflicts with `pypro4sail.py`; global NumPy arrays are CPU-only. | New `__init__.py` exposes tensor kernels and cached tensor spectral data. |
| `spectral_library.py` | Loads PROSPECT-D spectral constants from `prospect_d_spectra.txt`. | Uses `pkgutil.get_data('pyPro4Sail', ...)`, which is case-sensitive and breaks in the extracted package; returns NumPy arrays only. | New `spectral_library.py` uses `importlib.resources`, caches by `(device, dtype)`, returns tensors. |
| `prospect.py` | Leaf optical RTM. Core GPU candidate. | Uses NumPy, SciPy `expi`, in-place masking, CPU arrays, separate scalar/vector functions. | New `prospect.py` is tensorized, CUDA-compatible, differentiable, batched, and numerically matches legacy output. |
| `four_sail.py` | Canopy RTM. Core GPU candidate. | Uses NumPy/in-place masks, scalar branches, duplicated scalar/vector paths, hard Py6S import although Py6S is only needed by one atmospheric helper. | New `four_sail.py` is tensorized and batched; `foursail`, `foursail_vec`, and `foursail_wl` share one implementation. Py6S-specific atmospheric simulation is intentionally not part of the GPU kernel. |
| `pypro4sail.py` | Coupled PROSPECT + 4SAIL runner and thermal runner. | Reads soil file every call, mixes file I/O with model execution, returns NumPy arrays. | New `pypro4sail.py` accepts tensors, supports `device`/`dtype`, and allows direct `soil_reflectance` input to avoid file I/O in training loops. |
| `radiation_helpers.py` | Campbell simplified radiation helpers. | NumPy math and in-place NaN repair. | New `radiation_helpers.py` uses torch and keeps outputs on input device. |
| `cost_functions.py` | Objective functions for inversion. | Python loops, hard-coded hand Jacobian modules, NumPy arrays, non-differentiable paths. | New `cost_functions.py` computes losses as tensors and uses `torch.autograd.grad` for Jacobians. |
| `prospect_jacobian.py` | Hand-coded PROSPECT Jacobian. | Large, fragile symbolic derivative implementation; depends on SciPy and NumPy. | Replaced with autograd wrappers. For many wavelengths, prefer differentiating scalar losses rather than materializing full Jacobian. |
| `four_sail_jacobian.py` | Hand-coded 4SAIL Jacobian. | Large symbolic derivative implementation, hard to maintain, CPU-bound. | Replaced with autograd wrappers for key Jacobian use cases. For full spectral Jacobians, consider `torch.func.jacrev`/`vmap` in downstream code. |
| `machine_learning_regression.py` | scikit-learn models, LUT generation, PCA/scalers, multiprocessing. | scikit-learn models are CPU-bound; LUT generation calls NumPy RTM; multiprocessing conflicts with GPU batching. | New `machine_learning_regression.py` provides a PyTorch MLP training/inference path. LUT generation should now call tensorized `prospect`/`four_sail` directly in batches. |
| `cma.py` | Standalone CMA-ES optimizer. | 6,880-line CPU optimizer; not a tensor kernel. Porting CMA internals to GPU usually gives little benefit because objective evaluation dominates. | Preserved as `cma_numpy_legacy.py`. Recommended approach: keep optimizer on CPU, evaluate candidate batches through the torch kernels. |
| spectral text libraries | Soil and sensor response text files. | Data files are fine; they should be loaded once, not during inner loops. | Copied unchanged into package data. |

## Numerical validation performed

Against a locally patched legacy import of the original code:

- PROSPECT-D `rho` max absolute difference: `4.55e-15`
- PROSPECT-D `tau` max absolute difference: `3.89e-15`
- Campbell LIDF max absolute difference: `2.78e-16`
- 4SAIL selected outputs (`tss`, `rdd`, `rdot`, `rsot`) max absolute differences: about `1e-15` to `4e-15`

These checks used float64 on CPU. GPU float32 will be faster but less numerically identical.

## Important design choices

1. **Do not create tensors inside loops without a device.** All public functions infer `device`/`dtype` from tensor inputs or accept explicit `device=` and `dtype=`.
2. **Use batched tensors for speed.** Best performance comes from passing arrays of parameters and wavelengths through one call rather than looping in Python.
3. **Autograd replaces hand Jacobians.** The legacy Jacobian files are difficult to maintain and CPU-bound. In PyTorch, gradients of losses are the natural API.
4. **Keep file I/O out of training loops.** Pass `soil_reflectance=` directly when optimizing or training.
5. **External models stay external.** The legacy Py6S-dependent atmospheric helper is not GPU-refactored because it calls an external radiative-transfer model, not tensor math.

## Recommended next step

For production use, build your inversion/training loop around batched calls:

```python
# shape: (batch,)
Cab = torch.linspace(20, 80, 4096, device="cuda")
N = torch.full_like(Cab, 1.5)
Car = torch.full_like(Cab, 8.0)
Cbrown = torch.zeros_like(Cab)
Cw = torch.full_like(Cab, 0.01)
Cm = torch.full_like(Cab, 0.009)
Ant = torch.ones_like(Cab)

wl, rho_leaf, tau_leaf = pypro4sail_torch.prospect.prospectd_vec(N, Cab, Car, Cbrown, Cw, Cm, Ant)
```
