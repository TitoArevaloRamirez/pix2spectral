"""Differentiable PyTorch cost functions for PROSPECT-D and PROSAIL inversion."""

from typing import Dict, Iterable, Sequence

import torch

from . import four_sail, prospect
from ._utils import as_tensor, infer_device_dtype

PARAMS_PROSPECT = prospect.params_prospect
PARAMS_PROSAIL = four_sail.params_prosail


def _unscale_params(x0, obj_param: Sequence[str], fixed_values, scale, param_list, *, device=None, dtype=None) -> Dict[str, torch.Tensor]:
    x0 = as_tensor(x0, device=device, dtype=dtype)
    scale = as_tensor(scale, device=x0.device, dtype=x0.dtype)
    fixed_values = list(fixed_values.values()) if isinstance(fixed_values, dict) else list(fixed_values)
    out = {}
    i = 0; j = 0
    for p in param_list:
        if p in obj_param:
            out[p] = x0[i] * scale[i, 1] + scale[i, 0]
            i += 1
        else:
            out[p] = as_tensor(fixed_values[j], device=x0.device, dtype=x0.dtype)
            j += 1
    return out


def _wl_indices(wls, wl_grid):
    wls_t = as_tensor(wls, like=wl_grid).round().to(torch.long)
    return (wls_t - wl_grid[0].round().to(torch.long)).clamp(0, wl_grid.numel() - 1)


def cost_prospectd(x0, ObjParam, FixedValues, rho_leaf, wls, scale):
    device, dtype = infer_device_dtype(x0, rho_leaf, scale)
    p = _unscale_params(x0, ObjParam, FixedValues, scale, PARAMS_PROSPECT, device=device, dtype=dtype)
    wl, rho, _tau = prospect.prospectd(p["N_leaf"], p["Cab"], p["Car"], p["Cbrown"], p["Cw"], p["Cm"], p["Ant"], device=device, dtype=dtype)
    idx = _wl_indices(wls, wl)
    pred = rho[..., idx]
    obs = as_tensor(rho_leaf, device=device, dtype=dtype)
    return 0.5 * torch.mean((pred - obs).square())


def cost_prospectd_wl(x0, ObjParam, FixedValues, rho_leaf, wls, scale):
    return cost_prospectd(x0, ObjParam, FixedValues, rho_leaf, wls, scale)


def cost_jac_prospectd(x0, ObjParam, FixedValues, rho_leaf, wls, scale):
    x = as_tensor(x0).detach().clone().requires_grad_(True)
    y = cost_prospectd(x, ObjParam, FixedValues, rho_leaf, wls, scale)
    (grad,) = torch.autograd.grad(y, x, create_graph=False)
    return y, grad


def cost_prosail(x0, ObjParam, FixedValues, n_obs, rho_canopy, vza, sza, psi, skyl, rsoil, wls, scale):
    device, dtype = infer_device_dtype(x0, rho_canopy, skyl, rsoil, scale)
    p = _unscale_params(x0, ObjParam, FixedValues, scale, PARAMS_PROSAIL, device=device, dtype=dtype)
    wl, rho_leaf, tau_leaf = prospect.prospectd(p["N_leaf"], p["Cab"], p["Car"], p["Cbrown"], p["Cw"], p["Cm"], p["Ant"], device=device, dtype=dtype)
    idx = _wl_indices(wls, wl)
    rho_leaf = rho_leaf[..., idx]
    tau_leaf = tau_leaf[..., idx]
    rsoil_t = as_tensor(rsoil, device=device, dtype=dtype)
    if rsoil_t.numel() != idx.numel():
        rsoil_t = rsoil_t[..., idx]
    obs = as_tensor(rho_canopy, device=device, dtype=dtype)
    vza = as_tensor(vza, device=device, dtype=dtype).reshape(-1)
    sza = as_tensor(sza, device=device, dtype=dtype).reshape(-1)
    psi = as_tensor(psi, device=device, dtype=dtype).reshape(-1)
    skyl = as_tensor(skyl, device=device, dtype=dtype)
    lidf = four_sail.calc_lidf_campbell(p["leaf_angle"], device=device, dtype=dtype)
    preds = []
    for obs_idx in range(int(n_obs)):
        out = four_sail.foursail(p["LAI"], p["hotspot"], lidf, sza[obs_idx], vza[obs_idx], psi[obs_idx], rho_leaf, tau_leaf, rsoil_t)
        pred = out[14] * skyl[obs_idx] + out[17] * (1.0 - skyl[obs_idx])
        preds.append(pred)
    pred = torch.stack(preds, dim=0)
    return 0.5 * torch.mean((pred - obs).square())


def cost_prosail_wl(x0, ObjParam, FixedValues, n_obs, rho_canopy, vza, sza, psi, skyl, rsoil, wls, scale):
    return cost_prosail(x0, ObjParam, FixedValues, n_obs, rho_canopy, vza, sza, psi, skyl, rsoil, wls, scale)


def cost_jac_prosail(x0, ObjParam, FixedValues, n_obs, rho_canopy, vza, sza, psi, skyl, rsoil, wls, scale):
    x = as_tensor(x0).detach().clone().requires_grad_(True)
    y = cost_prosail(x, ObjParam, FixedValues, n_obs, rho_canopy, vza, sza, psi, skyl, rsoil, wls, scale)
    (grad,) = torch.autograd.grad(y, x, create_graph=False)
    return y, grad
