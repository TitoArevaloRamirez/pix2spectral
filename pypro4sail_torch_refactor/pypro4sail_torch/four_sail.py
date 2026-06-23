"""PyTorch implementation of core 4SAIL canopy radiative transfer functions."""

from pathlib import Path
import math

import torch

from ._utils import align_to, as_tensor, deg2rad, infer_device_dtype

params_sail = ("LAI", "hotspot", "leaf_angle")
params_prosail = ("N_leaf", "Cab", "Car", "Cbrown", "Cw", "Cm", "Ant", "LAI", "hotspot", "leaf_angle")

SOIL_LIBRARY = Path(__file__).parent / "spectra" / "soil_spectral_library"
SRF_LIBRARY = Path(__file__).parent / "spectra" / "sensor_response_functions"


def calc_lidf_verhoef(a, b, n_elements=18, *, device=None, dtype=torch.float64):
    """Verhoef bimodal LIDF. This branch is generally used with fixed constants."""
    a_f = float(a.detach().cpu()) if torch.is_tensor(a) else float(a)
    b_f = float(b.detach().cpu()) if torch.is_tensor(b) else float(b)
    freq = 1.0
    step = 90.0 / n_elements
    lidf = []
    angles = [i * step for i in reversed(range(n_elements))]
    for angle in angles:
        tl1 = math.radians(angle)
        if a_f > 1.0:
            f = 1.0 - math.cos(tl1)
        else:
            eps = 1e-8
            delx = 1.0
            x = 2.0 * tl1
            p = float(x)
            while delx >= eps:
                y = a_f * math.sin(x) + 0.5 * b_f * math.sin(2.0 * x)
                dx = 0.5 * (y - x + p)
                x = x + dx
                delx = abs(dx)
            f = (2.0 * y + p) / math.pi
        freq = freq - f
        lidf.append(freq)
        freq = float(f)
    return torch.as_tensor(list(reversed(lidf)), device=device, dtype=dtype)


def calc_lidf_campbell(alpha, n_elements=18, *, device=None, dtype=None):
    """Campbell ellipsoidal LIDF as torch tensor.

    ``alpha`` can be scalar or batched. Scalar output shape is ``(18,)``; batched
    output shape is ``(18, batch)``.
    """
    device, dtype = infer_device_dtype(alpha, device=device, dtype=dtype)
    alpha = as_tensor(alpha, device=device, dtype=dtype).reshape(-1)
    excent = torch.exp(-1.6184e-5 * alpha.pow(3.0) + 2.1145e-3 * alpha.square() - 1.2390e-1 * alpha + 3.2491)
    freq = []
    step = 90.0 / n_elements
    eps = torch.finfo(dtype).eps
    for i in range(n_elements):
        tl1 = torch.as_tensor(math.radians(i * step), device=device, dtype=dtype)
        tl2 = torch.as_tensor(math.radians((i + 1.0) * step), device=device, dtype=dtype)
        x1 = excent / torch.sqrt(1.0 + excent.square() * torch.tan(tl1).square())
        x2 = excent / torch.sqrt(1.0 + excent.square() * torch.tan(tl2).square())
        spherical = torch.abs(excent - 1.0) <= 1e-12
        spherical_freq = torch.abs(torch.cos(tl1) - torch.cos(tl2)).expand_as(excent)
        alph = excent / torch.sqrt(torch.clamp(torch.abs(1.0 - excent.square()), min=eps))
        alph2 = alph.square()
        x12 = x1.square()
        x22 = x2.square()
        alpx1 = torch.sqrt(alph2 + x12)
        alpx2 = torch.sqrt(alph2 + x22)
        dum_gt = x1 * alpx1 + alph2 * torch.log(torch.clamp(x1 + alpx1, min=eps))
        val_gt = torch.abs(dum_gt - (x2 * alpx2 + alph2 * torch.log(torch.clamp(x2 + alpx2, min=eps))))
        almx1 = torch.sqrt(torch.clamp(alph2 - x12, min=0.0))
        almx2 = torch.sqrt(torch.clamp(alph2 - x22, min=0.0))
        ratio1 = torch.clamp(x1 / torch.clamp(alph, min=eps), -1.0 + 1e-12, 1.0 - 1e-12)
        ratio2 = torch.clamp(x2 / torch.clamp(alph, min=eps), -1.0 + 1e-12, 1.0 - 1e-12)
        dum_lt = x1 * almx1 + alph2 * torch.asin(ratio1)
        val_lt = torch.abs(dum_lt - (x2 * almx2 + alph2 * torch.asin(ratio2)))
        val = torch.where(spherical, spherical_freq, torch.where(excent > 1.0, val_gt, val_lt))
        freq.append(val)
    freq_t = torch.stack(freq, dim=0)
    lidf = freq_t / freq_t.sum(dim=0, keepdim=True)
    if lidf.shape[1] == 1:
        return lidf[:, 0]
    return lidf


def calc_lidf_campbell_vec(alpha, n_elements=18):
    return calc_lidf_campbell(alpha, n_elements=n_elements)


def volscatt(tts, tto, psi, ttl):
    device, dtype = infer_device_dtype(tts, tto, psi, ttl)
    tts = as_tensor(tts, device=device, dtype=dtype)
    tto = as_tensor(tto, device=device, dtype=dtype)
    psi = as_tensor(psi, device=device, dtype=dtype)
    ttl = as_tensor(ttl, device=device, dtype=dtype)
    pi = torch.as_tensor(math.pi, device=device, dtype=dtype)

    cts = torch.cos(deg2rad(tts)); cto = torch.cos(deg2rad(tto))
    sts = torch.sin(deg2rad(tts)); sto = torch.sin(deg2rad(tto))
    cospsi = torch.cos(deg2rad(psi)); psir = deg2rad(psi)
    cttl = torch.cos(deg2rad(ttl)); sttl = torch.sin(deg2rad(ttl))
    cs = cttl * cts; co = cttl * cto
    ss = sttl * sts; so = sttl * sto

    cosbts = torch.where(torch.abs(ss) > 1e-6, -cs / ss, torch.zeros_like(cs) + 5.0)
    cosbto = torch.where(torch.abs(so) > 1e-6, -co / so, torch.zeros_like(co) + 5.0)

    bts = torch.where(torch.abs(cosbts) < 1.0, torch.acos(torch.clamp(cosbts, -1.0, 1.0)), torch.zeros_like(cosbts) + pi)
    ds = torch.where(torch.abs(cosbts) < 1.0, ss, cs)
    chi_s = 2.0 / pi * ((bts - pi * 0.5) * cs + torch.sin(bts) * ss)

    bto = torch.where(
        torch.abs(cosbto) < 1.0,
        torch.acos(torch.clamp(cosbto, -1.0, 1.0)),
        torch.where(tto < 90.0, torch.zeros_like(cosbto) + pi, torch.zeros_like(cosbto)),
    )
    do_ = torch.where(
        torch.abs(cosbto) < 1.0,
        so,
        torch.where(tto < 90.0, co, -co),
    )
    chi_o = 2.0 / pi * ((bto - pi * 0.5) * co + torch.sin(bto) * so)

    btran1 = torch.abs(bts - bto)
    btran2 = pi - torch.abs(bts + bto - pi)
    bt1 = torch.where(psir <= btran1, psir, btran1)
    bt2 = torch.where(psir <= btran1, btran1, torch.where(psir <= btran2, psir, btran2))
    bt3 = torch.where(psir <= btran1, btran2, torch.where(psir <= btran2, btran2, psir))

    t1 = 2.0 * cs * co + ss * so * cospsi
    t2 = torch.where(bt2 > 0.0,
                     torch.sin(bt2) * (2.0 * ds * do_ + ss * so * torch.cos(bt1) * torch.cos(bt3)),
                     torch.zeros_like(bt2))
    denom = 2.0 * pi.square()
    frho = torch.clamp(((pi - bt2) * t1 + t2) / denom, min=0.0)
    ftau = torch.clamp((-bt2 * t1 + t2) / denom, min=0.0)
    return chi_s, chi_o, frho, ftau


def volscatt_vec(tts, tto, psi, ttl):
    return volscatt(tts, tto, psi, ttl)


def weighted_sum_over_lidf(lidf, tts, tto, psi):
    device, dtype = infer_device_dtype(lidf, tts, tto, psi)
    lidf = as_tensor(lidf, device=device, dtype=dtype)
    tts = as_tensor(tts, device=device, dtype=dtype)
    tto = as_tensor(tto, device=device, dtype=dtype)
    psi = as_tensor(psi, device=device, dtype=dtype)
    cts = torch.cos(deg2rad(tts)); cto = torch.cos(deg2rad(tto)); ctscto = cts * cto
    ks = torch.zeros_like(cts); ko = torch.zeros_like(cto)
    bf = torch.zeros_like(ctscto); sob = torch.zeros_like(ctscto); sof = torch.zeros_like(ctscto)
    n_angles = int(lidf.shape[0])
    angle_step = 90.0 / n_angles
    for i in range(n_angles):
        ttl = (i * angle_step) + (angle_step * 0.5)
        cttl = math.cos(math.radians(ttl))
        chi_s, chi_o, frho, ftau = volscatt(tts, tto, psi, ttl)
        ksli = chi_s / cts
        koli = chi_o / cto
        sobli = frho * math.pi / ctscto
        sofli = ftau * math.pi / ctscto
        bfli = cttl ** 2
        w = lidf[i]
        ks = ks + ksli * w
        ko = ko + koli * w
        bf = bf + bfli * w
        sob = sob + sobli * w
        sof = sof + sofli * w
    return ks, ko, bf, sob, sof


def weighted_sum_over_lidf_vec(lidf, tts, tto, psi):
    return weighted_sum_over_lidf(lidf, tts, tto, psi)


def jfunc1(k, l, t):
    del_ = (k - l) * t
    denom = k - l
    eps = torch.finfo(del_.dtype).eps
    regular = (torch.exp(-l * t) - torch.exp(-k * t)) / torch.where(torch.abs(denom) > eps, denom, torch.ones_like(denom) * eps)
    series = 0.5 * t * (torch.exp(-k * t) + torch.exp(-l * t)) * (1.0 - del_.square() / 12.0)
    return torch.where(torch.abs(del_) > 1e-3, regular, series)


def jfunc1_vec(k, l, t):
    return jfunc1(k, l, t)


def jfunc2(k, l, t):
    denom = k + l
    eps = torch.finfo(denom.dtype).eps
    return (1.0 - torch.exp(-denom * t)) / torch.where(torch.abs(denom) > eps, denom, torch.ones_like(denom) * eps)


def jfunc1_wl(k, l, t):
    return jfunc1(k, l, t)


def jfunc2_wl(k, l, t):
    return jfunc2(k, l, t)


def define_geometric_constant(tts, tto, psi):
    tants = torch.tan(deg2rad(tts))
    tanto = torch.tan(deg2rad(tto))
    cospsi = torch.cos(deg2rad(psi))
    return torch.sqrt(torch.clamp(tants.square() + tanto.square() - 2.0 * tants * tanto * cospsi, min=0.0))


def hotspot_calculations(hotspot, lai, ko, ks, dso, tss):
    hotspot = as_tensor(hotspot, like=lai)
    eps = torch.finfo(lai.dtype).eps
    hotspot_safe = torch.where(hotspot > 0.0, hotspot, torch.ones_like(hotspot))
    alf = torch.where(hotspot > 0.0, (dso / hotspot_safe) * 2.0 / (ks + ko), torch.ones_like(lai) * 1e36)

    pure = (lai > 0.0) & (torch.abs(alf) <= eps)
    tsstoo_pure = tss
    sumint_pure = (1.0 - tss) / torch.clamp(ks * lai, min=eps)

    fhot = lai * torch.sqrt(torch.clamp(ko * ks, min=0.0))
    x1 = torch.zeros_like(fhot); y1 = torch.zeros_like(fhot); f1 = torch.ones_like(fhot)
    fint = (1.0 - torch.exp(-alf)) * 0.05
    sumint = torch.zeros_like(fhot)
    for istep in range(1, 21):
        if istep < 20:
            x2 = -torch.log(torch.clamp(1.0 - istep * fint, min=eps)) / alf
        else:
            x2 = torch.ones_like(fhot)
        y2 = -(ko + ks) * lai * x2 + fhot * (1.0 - torch.exp(-alf * x2)) / alf
        f2 = torch.exp(y2)
        denom = y2 - y1
        incr = (f2 - f1) * (x2 - x1) / torch.where(torch.abs(denom) > eps, denom, torch.ones_like(denom) * eps)
        sumint = sumint + incr
        x1, y1, f1 = x2, y2, f2
    sumint = torch.nan_to_num(sumint, nan=0.0, posinf=0.0, neginf=0.0)
    tsstoo = f1
    sumint = torch.where(pure, sumint_pure, sumint)
    tsstoo = torch.where(pure, tsstoo_pure, tsstoo)
    return tsstoo, sumint


def hotspot_calculations_vec(hotspot, lai, ko, ks, dso, tss):
    return hotspot_calculations(hotspot, lai, ko, ks, dso, tss)


def foursail(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil):
    """Run 4SAIL on PyTorch tensors.

    ``rho``, ``tau`` and ``rsoil`` can be spectral tensors. Scalar/batched canopy
    parameters are automatically broadcast over the spectral dimension.
    """
    device, dtype = infer_device_dtype(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil)
    rho = as_tensor(rho, device=device, dtype=dtype)
    tau = as_tensor(tau, device=device, dtype=dtype)
    rsoil = as_tensor(rsoil, device=device, dtype=dtype)
    lai0 = as_tensor(lai, device=device, dtype=dtype)
    hotspot0 = as_tensor(hotspot, device=device, dtype=dtype)
    tts0 = as_tensor(tts, device=device, dtype=dtype)
    tto0 = as_tensor(tto, device=device, dtype=dtype)
    psi0 = as_tensor(psi, device=device, dtype=dtype)
    lidf0 = as_tensor(lidf, device=device, dtype=dtype)

    ks_g, ko_g, bf_g, sob_g, sof_g = weighted_sum_over_lidf(lidf0, tts0, tto0, psi0)
    lai_pos = torch.clamp(lai0, min=torch.as_tensor(1e-12, device=device, dtype=dtype))

    ks = align_to(ks_g, rho); ko = align_to(ko_g, rho); bf = align_to(bf_g, rho)
    sob = align_to(sob_g, rho); sof = align_to(sof_g, rho)
    lai_s = align_to(lai_pos, rho); hotspot_s = align_to(hotspot0, rho)
    tts_s = align_to(tts0, rho); tto_s = align_to(tto0, rho); psi_s = align_to(psi0, rho)

    sdb = 0.5 * (ks + bf); sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf); dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1.0 + bf); ddf = 0.5 * (1.0 - bf)
    sigb = torch.clamp(ddb * rho + ddf * tau, min=1e-36)
    sigf = torch.clamp(ddf * rho + ddb * tau, min=1e-36)
    att = 1.0 - sigf
    m = torch.sqrt(torch.clamp(att.square() - sigb.square(), min=0.0))
    sb = sdb * rho + sdf * tau; sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau; vf = dof * rho + dob * tau
    w = sob * rho + sof * tau

    e1 = torch.exp(-m * lai_s); e2 = e1.square()
    rinf = (att - m) / sigb; rinf2 = rinf.square(); re = rinf * e1
    denom = 1.0 - rinf2 * e2
    J1ks = jfunc1(ks, m, lai_s); J2ks = jfunc2(ks, m, lai_s)
    J1ko = jfunc1(ko, m, lai_s); J2ko = jfunc2(ko, m, lai_s)
    Pss = (sf + sb * rinf) * J1ks; Qss = (sf * rinf + sb) * J2ks
    Pv = (vf + vb * rinf) * J1ko; Qv = (vf * rinf + vb) * J2ko
    tdd = (1.0 - rinf2) * e1 / denom
    rdd = rinf * (1.0 - e2) / denom
    tsd = (Pss - re * Qss) / denom; rsd = (Qss - re * Pss) / denom
    tdo = (Pv - re * Qv) / denom; rdo = (Qv - re * Pv) / denom
    gammasdf = (1.0 + rinf) * (J1ks - re * J2ks) / denom
    gammasdb = (1.0 + rinf) * (-re * J1ks + J2ks) / denom

    tss = torch.exp(-ks * lai_s); too = torch.exp(-ko * lai_s)
    z = jfunc2(ks, ko, lai_s)
    g1 = (z - J1ks * too) / (ko + m)
    g2 = (z - J1ko * tss) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1; Tv2 = (vf + vb * rinf) * g2
    T1 = Tv1 * (sf + sb * rinf); T2 = Tv2 * (sf * rinf + sb)
    T3 = (rdo * Qss + tdo * Pss) * rinf
    rsod = (T1 + T2 - T3) / (1.0 - rinf2)
    T4 = Tv1 * (1.0 + rinf); T5 = Tv2 * (1.0 + rinf)
    T6 = (rdo * J2ks + tdo * J1ks) * (1.0 + rinf) * rinf
    gammasod = (T4 + T5 - T6) / (1.0 - rinf2)

    dso = define_geometric_constant(tts_s, tto_s, psi_s)
    tsstoo, sumint = hotspot_calculations(hotspot_s, lai_s, ko, ks, dso, tss)
    rsos = w * lai_s * sumint
    gammasos = ko * lai_s * sumint
    rso = rsos + rsod
    gammaso = gammasos + gammasod

    dn = torch.clamp(1.0 - rsoil * rdd, min=1e-36)
    rddt = rdd + tdd * rsoil * tdd / dn
    rsdt = rsd + (tsd + tss) * rsoil * tdd / dn
    rdot = rdo + tdd * rsoil * (tdo + too) / dn
    rsodt = ((tss + tsd) * tdo + (tsd + tss * rsoil * rdd) * too) * rsoil / dn
    rsost = rso + tsstoo * rsoil
    rsot = rsost + rsodt

    zero = torch.zeros_like(rsoil)
    one = torch.ones_like(rsoil)
    lai_zero = align_to(as_tensor(lai, device=device, dtype=dtype) <= 0.0, rho)
    outputs = [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
               rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]
    zero_lai_outputs = [one, one, one, zero, one, zero, zero, zero, zero,
                        zero, zero, zero, rsoil, rsoil, rsoil, zero, rsoil, rsoil, zero, zero, zero]
    return [torch.where(lai_zero, zlo, out) for out, zlo in zip(outputs, zero_lai_outputs)]


def foursail_vec(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil):
    return foursail(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil)


def foursail_wl(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil):
    return foursail(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil)


# Backwards-compatible typo from the legacy module.
def fousail_wl(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil):
    return foursail_wl(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil)


def calc_sun_angles(lat, lon, stdlon, doy, ftime):
    device, dtype = infer_device_dtype(lat, lon, stdlon, doy, ftime)
    lat, lon, stdlon, doy, ftime = [as_tensor(v, device=device, dtype=dtype) for v in (lat, lon, stdlon, doy, ftime)]
    declination = 0.409 * torch.sin((2.0 * math.pi * doy / 365.0) - 1.39)
    EOT = 0.258 * torch.cos(declination) - 7.416 * torch.sin(declination) - 3.648 * torch.cos(2.0 * declination) - 9.228 * torch.sin(2.0 * declination)
    LC = (stdlon - lon) / 15.0
    solar_time = ftime - ((-EOT / 60.0) + LC)
    w = (solar_time - 12.0) * 15.0
    sin_theta = torch.cos(deg2rad(w)) * torch.cos(declination) * torch.cos(deg2rad(lat)) + torch.sin(declination) * torch.sin(deg2rad(lat))
    sun_elev = torch.asin(torch.clamp(sin_theta, -1.0, 1.0))
    sza = 90.0 - sun_elev * 180.0 / math.pi
    cos_phi = (torch.sin(declination) * torch.cos(deg2rad(lat)) - torch.cos(deg2rad(w)) * torch.cos(declination) * torch.sin(deg2rad(lat))) / torch.cos(sun_elev)
    saa = torch.where(w <= 0.0, torch.acos(torch.clamp(cos_phi, -1.0, 1.0)) * 180.0 / math.pi,
                      360.0 - torch.acos(torch.clamp(cos_phi, -1.0, 1.0)) * 180.0 / math.pi)
    return sza, saa
