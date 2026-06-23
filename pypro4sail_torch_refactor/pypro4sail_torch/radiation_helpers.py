"""PyTorch version of Campbell radiation helper functions."""

import math
import torch

from ._utils import as_tensor, deg2rad, infer_device_dtype

TAUD_STEP_SIZE_DEG = 5


def calc_K_be_Campbell(theta, x_lad=1, radians=False):
    device, dtype = infer_device_dtype(theta, x_lad)
    theta = as_tensor(theta, device=device, dtype=dtype)
    x_lad = as_tensor(x_lad, device=device, dtype=dtype)
    if not radians:
        theta = deg2rad(theta)
    return torch.sqrt(x_lad.square() + torch.tan(theta).square()) / (x_lad + 1.774 * (x_lad + 1.182).pow(-0.733))


def _calc_taud(x_lad, lai):
    device, dtype = infer_device_dtype(x_lad, lai)
    x_lad = as_tensor(x_lad, device=device, dtype=dtype)
    lai = as_tensor(lai, device=device, dtype=dtype)
    taud = torch.zeros_like(lai + x_lad)
    for angle in range(0, 90, TAUD_STEP_SIZE_DEG):
        angle_rad = torch.as_tensor(math.radians(angle), device=device, dtype=dtype)
        akd = calc_K_be_Campbell(angle_rad, x_lad, radians=True)
        taub = torch.exp(-akd * lai)
        taud = taud + taub * torch.cos(angle_rad) * torch.sin(angle_rad) * math.radians(TAUD_STEP_SIZE_DEG)
    return 2.0 * taud


def calc_spectra_Cambpell(lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad=1, lai_eff=None):
    device, dtype = infer_device_dtype(lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad)
    lai = as_tensor(lai, device=device, dtype=dtype)
    sza = as_tensor(sza, device=device, dtype=dtype)
    rho_leaf = as_tensor(rho_leaf, device=device, dtype=dtype)
    tau_leaf = as_tensor(tau_leaf, device=device, dtype=dtype)
    rho_soil = as_tensor(rho_soil, device=device, dtype=dtype)
    x_lad = as_tensor(x_lad, device=device, dtype=dtype)
    lai_eff = lai if lai_eff is None else as_tensor(lai_eff, device=device, dtype=dtype)

    amean_sqrt = torch.sqrt(torch.clamp(1.0 - rho_leaf - tau_leaf, min=0.0))
    taud = _calc_taud(x_lad, lai)
    akd = -torch.log(torch.clamp(taud, min=torch.finfo(dtype).tiny)) / lai
    rcpy = (1.0 - amean_sqrt) / (1.0 + amean_sqrt)
    rdcpy = 2.0 * akd * rcpy / (akd + 1.0)
    expfac = amean_sqrt * akd * lai
    neg_exp, d_neg_exp = torch.exp(-expfac), torch.exp(-2.0 * expfac)
    taudt = ((rdcpy.square() - 1.0) * neg_exp) / ((rdcpy * rho_soil - 1.0) + rdcpy * (rdcpy - rho_soil) * d_neg_exp)
    fact = ((rdcpy - rho_soil) / (rdcpy * rho_soil - 1.0)) * d_neg_exp
    albd = (rdcpy + fact) / (1.0 + rdcpy * fact)

    akb = calc_K_be_Campbell(sza, x_lad)
    rbcpy = 2.0 * akb * rcpy / (akb + 1.0)
    expfac = amean_sqrt * akb * lai_eff
    neg_exp, d_neg_exp = torch.exp(-expfac), torch.exp(-2.0 * expfac)
    taubt = ((rbcpy.square() - 1.0) * neg_exp) / ((rbcpy * rho_soil - 1.0) + rbcpy * (rbcpy - rho_soil) * d_neg_exp)
    fact = ((rbcpy - rho_soil) / (rbcpy * rho_soil - 1.0)) * d_neg_exp
    albb = (rbcpy + fact) / (1.0 + rbcpy * fact)

    taubt = torch.where(torch.isnan(taubt), torch.ones_like(taubt), taubt)
    taudt = torch.where(torch.isnan(taudt), torch.ones_like(taudt), taudt)
    albb = torch.where(torch.isnan(albb), rho_soil, albb)
    albd = torch.where(torch.isnan(albd), rho_soil, albd)
    return albb, albd, taubt, taudt


def leafangle_2_chi(alpha):
    alpha = as_tensor(alpha)
    return (deg2rad(alpha) / 9.65).pow(1.0 / -1.65) - 3.0
