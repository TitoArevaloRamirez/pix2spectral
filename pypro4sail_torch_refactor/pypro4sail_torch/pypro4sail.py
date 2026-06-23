"""Coupled PROSPECT-D + 4SAIL PyTorch interface."""

from pathlib import Path

import numpy as np
import torch

from . import four_sail, prospect
from ._utils import as_tensor, infer_device_dtype

SOIL_FOLDER = Path(__file__).parent / "spectra" / "soil_spectral_library"
DEFAULT_SOIL = "ProSAIL_WetSoil.txt"
SB = 5.670373e-8

PLANOPHILE = (1, 0)
ERECTOPHILE = (-1, 0)
PLAGIOPHILE = (0, -1)
EXTREMOPHILE = (0, 1)
SPHERICAL = (-0.35, -0.15)
UNIFORM = (0, 0)


def _soil_tensor(soilType=DEFAULT_SOIL, *, device=None, dtype=torch.float64, soil_reflectance=None):
    if soil_reflectance is not None:
        return as_tensor(soil_reflectance, device=device, dtype=dtype)
    arr = np.loadtxt(SOIL_FOLDER / soilType, dtype=np.float64)
    return torch.as_tensor(arr[:, 1], device=device, dtype=dtype)


def _lidf_tensor(LIDF, *, device=None, dtype=torch.float64):
    if isinstance(LIDF, (tuple, list)):
        if len(LIDF) != 2:
            raise ValueError("Verhoef LIDF must be a pair (LIDFa, LIDFb).")
        return four_sail.calc_lidf_verhoef(LIDF[0], LIDF[1], device=device, dtype=dtype)
    return four_sail.calc_lidf_campbell(LIDF, device=device, dtype=dtype)


def run(N, chloro, caroten, brown, EWT, LMA, Ant, LAI, hot_spot, solar_zenith, solar_azimuth,
        view_zenith, view_azimuth, LIDF, skyl=0.2, soilType=DEFAULT_SOIL, *, soil_reflectance=None,
        device=None, dtype=None):
    """Run coupled PROSPECT-D + 4SAIL and return ``wl, canopy_reflectance``.

    Pass tensors on CUDA (or set ``device='cuda'``) to execute the kernels on GPU.
    ``soil_reflectance`` can be supplied directly to avoid file I/O in training
    loops.
    """
    device, dtype = infer_device_dtype(N, chloro, caroten, brown, EWT, LMA, Ant, LAI, hot_spot,
                                       solar_zenith, solar_azimuth, view_zenith, view_azimuth, skyl,
                                       device=device, dtype=dtype)
    rsoil = _soil_tensor(soilType, device=device, dtype=dtype, soil_reflectance=soil_reflectance)
    lidf = _lidf_tensor(LIDF, device=device, dtype=dtype)
    wl, rho_leaf, tau_leaf = prospect.prospectd(N, chloro, caroten, brown, EWT, LMA, Ant, device=device, dtype=dtype)
    psi = torch.abs(as_tensor(solar_azimuth, device=device, dtype=dtype) - as_tensor(view_azimuth, device=device, dtype=dtype))
    outputs = four_sail.foursail(LAI, hot_spot, lidf, solar_zenith, view_zenith, psi, rho_leaf, tau_leaf, rsoil)
    rdot, rsot = outputs[14], outputs[17]
    skyl_t = as_tensor(skyl, device=device, dtype=dtype)
    rho_canopy = rdot * skyl_t + rsot * (1.0 - skyl_t)
    return wl, rho_canopy


def CalcStephanBoltzmann(T_K):
    T_K = as_tensor(T_K)
    return SB * T_K.pow(4.0)


def run_tir(emisVeg, emisSoil, T_Veg, T_Soil, LAI, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth,
            LIDF, T_VegSunlit=None, T_SoilSunlit=None, T_atm=0, *, device=None, dtype=None):
    device, dtype = infer_device_dtype(emisVeg, emisSoil, T_Veg, T_Soil, LAI, hot_spot, solar_zenith, solar_azimuth,
                                       view_zenith, view_azimuth, T_atm, device=device, dtype=dtype)
    emisVeg = as_tensor(emisVeg, device=device, dtype=dtype)
    emisSoil = as_tensor(emisSoil, device=device, dtype=dtype)
    rsoil = 1.0 - emisSoil
    rho_leaf = 1.0 - emisVeg
    tau_leaf = torch.zeros_like(rho_leaf)
    lidf = _lidf_tensor(LIDF, device=device, dtype=dtype)
    psi = torch.abs(as_tensor(solar_azimuth, device=device, dtype=dtype) - as_tensor(view_azimuth, device=device, dtype=dtype))
    outputs = four_sail.foursail(LAI, hot_spot, lidf, solar_zenith, view_zenith, psi, rho_leaf, tau_leaf, rsoil)
    tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso = outputs
    tso = tsstoo + tss * (tdo + rsoil * rdd * too) / (1.0 - rsoil * rdd)
    gammad = 1.0 - rdd - tdd
    gammao = 1.0 - rdo - tdo - too
    ttot = (too + tdo) / (1.0 - rsoil * rdd)
    gammaot = gammao + ttot * rsoil * gammad
    gammasot = gammaso + ttot * rsoil * gammasdf
    aeev = gammaot
    aees = ttot * emisSoil
    Hvc = CalcStephanBoltzmann(as_tensor(T_Veg, device=device, dtype=dtype))
    Hgc = CalcStephanBoltzmann(as_tensor(T_Soil, device=device, dtype=dtype))
    Hsky = CalcStephanBoltzmann(as_tensor(T_atm, device=device, dtype=dtype))
    Hvh = Hvc if T_VegSunlit is None else CalcStephanBoltzmann(as_tensor(T_VegSunlit, device=device, dtype=dtype))
    Hgh = Hgc if T_SoilSunlit is None else CalcStephanBoltzmann(as_tensor(T_SoilSunlit, device=device, dtype=dtype))
    Lw = (rdot * Hsky + (aeev * Hvc + gammasot * emisVeg * (Hvh - Hvc) + aees * Hgc + tso * emisSoil * (Hgh - Hgc))) / torch.pi
    TB_obs = (torch.pi * Lw / SB).pow(0.25)
    emiss = 1.0 - rdot
    return Lw, TB_obs, emiss


# Backwards-compatible legacy spelling.
def run_TIR(*args, **kwargs):
    return run_tir(*args, **kwargs)
