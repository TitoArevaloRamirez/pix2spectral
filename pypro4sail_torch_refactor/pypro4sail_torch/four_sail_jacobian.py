"""Autograd replacements for legacy hand-coded 4SAIL Jacobian functions."""

import torch

from . import four_sail
from ._utils import as_tensor

params_SAIL = four_sail.params_sail
params_prosail = four_sail.params_prosail


def JacCalcLIDF_Campbell(alpha, n_elements=18):
    a = as_tensor(alpha).detach().clone().requires_grad_(True)
    lidf = four_sail.calc_lidf_campbell(a, n_elements=n_elements)
    rows = []
    for y in lidf.reshape(-1):
        (g,) = torch.autograd.grad(y, a, retain_graph=True)
        rows.append(g)
    return lidf, torch.stack(rows).reshape(lidf.shape)


def JacFourSAIL(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil, Delta_rho=None, Delta_tau=None):
    lai_t = as_tensor(lai).detach().clone().requires_grad_(True)
    hotspot_t = as_tensor(hotspot).detach().clone().requires_grad_(True)
    rho_t = as_tensor(rho).detach().clone().requires_grad_(True)
    tau_t = as_tensor(tau).detach().clone().requires_grad_(True)
    outputs = four_sail.foursail(lai_t, hotspot_t, lidf, tts, tto, psi, rho_t, tau_t, rsoil)
    vars_ = (lai_t, hotspot_t, rho_t, tau_t)
    jac = []
    for out in outputs:
        flat_rows = []
        for yi in out.reshape(-1):
            grads = torch.autograd.grad(yi, vars_, retain_graph=True, allow_unused=True)
            flat_rows.append(tuple(torch.zeros_like(v) if g is None else g for g, v in zip(grads, vars_)))
        jac.append(flat_rows)
    return outputs, jac


def volscatt(tts, tto, psi, ttl):
    return four_sail.volscatt(tts, tto, psi, ttl)


def define_geometric_constant(tts, tto, psi):
    return four_sail.define_geometric_constant(tts, tto, psi)
