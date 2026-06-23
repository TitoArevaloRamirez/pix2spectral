"""PyTorch implementation of PROSPECT-D.

The original code used NumPy plus SciPy's ``expi``. PyTorch does not currently
provide Ei/E1 on all builds, so this module uses the polynomial approximation
already present in the legacy source (``trans_approx``). It is differentiable
almost everywhere and runs on CUDA.
"""

import torch

from ._utils import as_tensor, column, infer_device_dtype, maybe_squeeze_batch, scalar_like
from .spectral_library import get_spectra

params_prospect = ("N_leaf", "Cab", "Car", "Cbrown", "Cw", "Cm", "Ant")


def _polyval(coeffs, x):
    y = torch.zeros_like(x) + coeffs[0]
    for c in coeffs[1:]:
        y = y * x + c
    return y


def trans_approx(k: torch.Tensor) -> torch.Tensor:
    """Approximate leaf internal transmittance from absorption coefficient ``k``.

    This is the legacy PROSPECT approximation rewritten with ``torch.where`` so
    it can be executed and differentiated on GPU.
    """
    eps = torch.finfo(k.dtype).tiny
    k_safe = torch.clamp(k, min=eps)

    xx1 = 0.5 * k_safe - 1.0
    coeffs1 = [
        -3.60311230482612224e-13,
        3.46348526554087424e-12,
        -2.99627399604128973e-11,
        2.57747807106988589e-10,
        -2.09330568435488303e-9,
        1.59501329936987818e-8,
        -1.13717900285428895e-7,
        7.55292885309152956e-7,
        -4.64980751480619431e-6,
        2.63830365675408129e-5,
        -1.37089870978830576e-4,
        6.47686503728103400e-4,
        -2.76060141343627983e-3,
        1.05306034687449505e-2,
        -3.57191348753631956e-2,
        1.07774527938978692e-1,
        -2.96997075145080963e-1,
    ]
    yy1 = (_polyval(coeffs1, xx1) * xx1 + 8.64664716763387311e-1) * xx1 + 7.42047691268006429e-1
    yy1 = yy1 - torch.log(k_safe)
    trans1 = (1.0 - k_safe) * torch.exp(-k_safe) + k_safe.square() * yy1

    xx2 = 14.5 / (k_safe + 3.25) - 1.0
    coeffs2 = [
        -1.62806570868460749e-12,
        -8.95400579318284288e-13,
        -4.08352702838151578e-12,
        -1.45132988248537498e-11,
        -8.35086918940757852e-11,
        -2.13638678953766289e-10,
        -1.10302431467069770e-9,
        -3.67128915633455484e-9,
        -1.66980544304104726e-8,
        -6.11774386401295125e-8,
        -2.70306163610271497e-7,
        -1.05565006992891261e-6,
        -4.72090467203711484e-6,
        -1.95076375089955937e-5,
        -9.16450482931221453e-5,
        -4.05892130452128677e-4,
        -2.14213055000334718e-3,
    ]
    yy2 = ((_polyval(coeffs2, xx2) * xx2 - 1.06374875116569657e-2) * xx2
           - 8.50699154984571871e-2) * xx2 + 9.23755307807784058e-1
    yy2 = torch.exp(-k_safe) * yy2 / k_safe
    trans2 = (1.0 - k_safe) * torch.exp(-k_safe) + k_safe.square() * yy2

    trans = torch.where(k <= 0.0, torch.ones_like(k),
                        torch.where(k <= 4.0, trans1,
                                    torch.where(k <= 85.0, trans2, torch.zeros_like(k))))
    return torch.clamp(trans, min=0.0, max=1.0)


def tav(theta, ref):
    """Average transmittivity at the leaf surface."""
    theta_value = float(theta) if not torch.is_tensor(theta) or theta.ndim == 0 else None
    if theta_value is not None:
        theta_rad = torch.as_tensor(theta_value * torch.pi / 180.0, dtype=ref.dtype, device=ref.device)
        if theta_value == 0.0:
            return 4.0 * ref / (ref + 1.0).square()
        theta_is_half_pi = abs(theta_value - 90.0) < 1e-12
    else:
        theta_rad = as_tensor(theta, like=ref) * torch.pi / 180.0
        theta_is_half_pi = False

    r2 = ref.square()
    rp = r2 + 1.0
    rm = r2 - 1.0
    a = (ref + 1.0).square() / 2.0
    k = -(r2 - 1.0).square() / 4.0
    ds = torch.sin(theta_rad)
    if theta_is_half_pi:
        b1 = torch.zeros_like(ref)
    else:
        b1 = torch.sqrt(torch.clamp((ds.square() - rp / 2.0).square() + k, min=0.0))
    k2 = k.square()
    rm2 = rm.square()
    b2 = ds.square() - rp / 2.0
    b = torch.clamp(b1 - b2, min=torch.finfo(ref.dtype).tiny)
    ts = (k2 / (6.0 * b.pow(3.0)) + k / b - b / 2.0) - (k2 / (6.0 * a.pow(3.0)) + k / a - a / 2.0)
    tp1 = -2.0 * r2 * (b - a) / rp.square()
    tp2 = -2.0 * r2 * rp * torch.log(b / a) / rm2
    tp3 = r2 * (b.reciprocal() - a.reciprocal()) / 2.0
    tp4 = 16.0 * r2.square() * (r2.square() + 1.0) * torch.log(
        (2.0 * rp * b - rm2) / (2.0 * rp * a - rm2)
    ) / (rp.pow(3.0) * rm2)
    tp5 = 16.0 * r2.pow(3.0) * (
        (2.0 * rp * b - rm2).reciprocal() - (2.0 * rp * a - rm2).reciprocal()
    ) / rp.pow(3.0)
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    return (ts + tp) / (2.0 * ds.square())


def tav_wl(theta, ref):
    return tav(theta, ref)


def refl_trans_one_layer(alpha, nr, tau):
    talf = tav(alpha, nr)
    ralf = 1.0 - talf
    t12 = tav(90.0, nr)
    r12 = 1.0 - t12
    t21 = t12 / nr.square()
    r21 = 1.0 - t21
    denom = 1.0 - r21.square() * tau.square()
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t
    return r, t, Ra, Ta


def reflectance_n_layers_stokes(r, t, Ra, Ta, Nleaf):
    D = torch.sqrt(torch.clamp((1.0 + r + t) * (1.0 + r - t) * (1.0 - r + t) * (1.0 - r - t), min=0.0))
    eps = torch.finfo(r.dtype).eps
    a = (1.0 + r.square() - t.square() + D) / (2.0 * torch.clamp(r, min=eps))
    b = (1.0 - r.square() + t.square() + D) / (2.0 * torch.clamp(t, min=eps))
    bNm1 = b.pow(Nleaf - 1.0)
    bN2 = bNm1.square()
    a2 = a.square()
    denom = a2 * bN2 - 1.0
    Rsub = a * (bN2 - 1.0) / denom
    Tsub = bNm1 * (a2 - 1.0) / denom
    zero_abs = r + t >= 1.0
    T_zero = t / (t + (1.0 - t) * (Nleaf - 1.0))
    Rsub = torch.where(zero_abs, 1.0 - T_zero, Rsub)
    Tsub = torch.where(zero_abs, T_zero, Tsub)
    denom2 = 1.0 - Rsub * r
    tran = Ta * Tsub / denom2
    refl = Ra + Ta * Rsub * t / denom2
    return refl, tran


def _prepare_params(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, *, device=None, dtype=None):
    device, dtype = infer_device_dtype(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, device=device, dtype=dtype)
    vals = [as_tensor(v, device=device, dtype=dtype) for v in (Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant)]
    single = all(scalar_like(v) for v in vals)
    return [column(v) for v in vals], single, device, dtype


def prospectd(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, *, device=None, dtype=None, squeeze=True):
    """Run PROSPECT-D using PyTorch tensors.

    Scalar inputs return wavelength, rho, tau with spectral shape ``(2101,)``.
    Batched one-dimensional inputs return ``rho`` and ``tau`` with shape
    ``(batch, 2101)``. All outputs are tensors on the inferred/requested device.
    """
    (Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant), single, device, dtype = _prepare_params(
        Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, device=device, dtype=dtype
    )
    wl, nr, Cab_k, Car_k, Cbrown_k, Cw_k, Cm_k, Ant_k = get_spectra(device=device, dtype=dtype)
    k = (Cab * Cab_k + Car * Car_k + Cbrown * Cbrown_k + Cw * Cw_k + Cm * Cm_k + Ant * Ant_k) / Nleaf
    k = torch.clamp(k, min=0.0)
    trans = trans_approx(k)
    rho, tau, Ra, Ta = refl_trans_one_layer(40.0, nr, trans)
    rho, tau = reflectance_n_layers_stokes(rho, tau, Ra, Ta, Nleaf)
    if squeeze:
        rho = maybe_squeeze_batch(rho, single)
        tau = maybe_squeeze_batch(tau, single)
    return wl, rho, tau


def prospectd_vec(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, *, device=None, dtype=None):
    return prospectd(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, device=device, dtype=dtype, squeeze=False)


def prospectd_wl(wl, Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, *, device=None, dtype=None):
    wavelength, rho, tau = prospectd(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant, device=device, dtype=dtype)
    wl_t = as_tensor(wl, like=wavelength).round().to(torch.long)
    idx = (wl_t - wavelength[0].round().to(torch.long)).clamp(0, wavelength.numel() - 1)
    return wavelength[idx], rho[..., idx], tau[..., idx]
