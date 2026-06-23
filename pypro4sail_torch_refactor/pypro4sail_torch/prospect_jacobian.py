"""Autograd replacements for legacy hand-coded PROSPECT Jacobian functions."""

import torch

from . import prospect
from ._utils import as_tensor

params_prospect = prospect.params_prospect


def JacProspectD(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant):
    vals = [as_tensor(v).detach().clone().requires_grad_(True) for v in (Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant)]
    wl, rho, tau = prospect.prospectd(*vals)
    jac_rho = []
    jac_tau = []
    for y, store in ((rho, jac_rho), (tau, jac_tau)):
        flat = y.reshape(-1)
        rows = []
        for yi in flat:
            grads = torch.autograd.grad(yi, vals, retain_graph=True, allow_unused=True)
            rows.append(torch.stack([torch.zeros_like(vals[i]) if g is None else g for i, g in enumerate(grads)]))
        store.append(torch.stack(rows).reshape(*y.shape, len(vals)))
    return wl, rho, tau, jac_rho[0], jac_tau[0]


def JacProspectD_wl(wl, Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant):
    vals = [as_tensor(v).detach().clone().requires_grad_(True) for v in (Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant)]
    wlo, rho, tau = prospect.prospectd_wl(wl, *vals)
    jac_rho = torch.stack(torch.autograd.grad(rho, vals, retain_graph=True, allow_unused=True))
    jac_tau = torch.stack(torch.autograd.grad(tau, vals, retain_graph=True, allow_unused=True))
    return wlo, rho, tau, jac_rho, jac_tau


def tav(theta, ref):
    return prospect.tav(theta, ref)


def tav_wl(theta, ref):
    return prospect.tav_wl(theta, ref)
