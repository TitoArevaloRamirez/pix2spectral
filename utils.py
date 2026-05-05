import torch
import config
from torchvision.utils import save_image

import numpy as np
from scipy.optimize import minimize

from pyPro4Sail.cost_functions import cost_prospectd

from pypro4sail import prospect

from scipy.optimize import least_squares

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def invert_prospect_parameters(
    rho_leaf,
    wls,
    x0=None,
    bounds=None,
    ant=0.0,
    method="L-BFGS-B",
    options=None,
):
    """
    Invert PROSPECT-D parameters from a known leaf reflectance spectrum.

    Estimates:
        N_leaf, Cab, Car, Cbrown, Cw, Cm

    Parameters
    ----------
    rho_leaf : array-like, shape [M]
        Measured leaf reflectance at wavelengths `wls`.

    wls : array-like, shape [M]
        Wavelengths corresponding to rho_leaf.
        These should match PROSPECT wavelengths, usually integer nm values
        between 400 and 2500.

    x0 : array-like, optional, shape [6]
        Initial physical parameter guess:
            [N_leaf, Cab, Car, Cbrown, Cw, Cm]

        If None, a generic green-leaf guess is used.

    bounds : list[tuple], optional
        Physical parameter bounds:
            [(N_min, N_max), (Cab_min, Cab_max), ...]

        If None, broad default bounds are used.

    ant : float
        Anthocyanin value used as fixed parameter.
        PROSPECT-D includes Ant, but here we keep it fixed.

    method : str
        scipy.optimize.minimize method.

    options : dict
        Options passed to scipy.optimize.minimize.

    Returns
    -------
    best_params : dict
        Estimated physical parameters.

    result : scipy.optimize.OptimizeResult
        Full scipy optimization result.

    Notes
    -----
    pypro4sail's cost_prospectd expects scaled parameters x in [0, 1].
    This wrapper lets you work with physical parameter bounds and returns
    physical parameters.
    """

    rho_leaf = np.asarray(rho_leaf, dtype=float).squeeze()
    wls = np.asarray(wls, dtype=float).squeeze()

    if rho_leaf.ndim != 1:
        raise ValueError(f"rho_leaf must be 1D, got shape {rho_leaf.shape}")

    if wls.ndim != 1:
        raise ValueError(f"wls must be 1D, got shape {wls.shape}")

    if rho_leaf.shape[0] != wls.shape[0]:
        raise ValueError(
            f"rho_leaf and wls must have same length. "
            f"Got {rho_leaf.shape[0]} and {wls.shape[0]}"
        )

    # PROSPECT-D parameter names used by pypro4sail.
    # In cost_functions.py, ObjParam is compared against
    # prospect_jacobian.params_prospect.
    obj_param = [
        "N_leaf",
        "Cab",
        "Car",
        "Cbrown",
        "Cw",
        "Cm",
    ]

    # Ant is fixed because this wrapper estimates only the six parameters above.
    # cost_prospectd fills parameters not in ObjParam from FixedValues.
    fixed_values = [ant]

    if bounds is None:
        bounds = [
            (1.0, 100.0),  # N_leaf, dimensionless
            (0.0, 1500.0),  # Cab, ug/cm^2
            (0.0, 100.0),  # Car, ug/cm^2
            (0.0, 100.0),  # Cbrown, dimensionless
            (1e-5, 1),  # Cw, cm
            (1e-4, 1),  # Cm, g/cm^2
        ]

    bounds = np.asarray(bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    span = upper - lower

    if np.any(span <= 0):
        raise ValueError("Each bound must satisfy upper > lower.")

    # pypro4sail scale format:
    #   physical_value = x_scaled * scale + minimum
    scale = [(float(lo), float(hi - lo)) for lo, hi in bounds]

    if x0 is None:
        x0_physical = np.array(
            [
                1.5,  # N_leaf
                40.0,  # Cab
                8.0,  # Car
                0.0,  # Cbrown
                0.015,  # Cw
                0.009,  # Cm
            ],
            dtype=float,
        )
    else:
        x0_physical = np.asarray(x0, dtype=float).squeeze()

    if x0_physical.shape[0] != 6:
        raise ValueError(f"x0 must have 6 values, got {x0_physical.shape[0]}")

    # Convert physical x0 to scaled [0, 1].
    x0_scaled = (x0_physical - lower) / span
    x0_scaled = np.clip(x0_scaled, 0.0, 1.0)

    scaled_bounds = [(0.0, 1.0)] * 6

    if options is None:
        options = {
            "maxiter": 1000,
            "ftol": 1e-12,
        }

    def objective(x_scaled):
        return cost_prospectd(
            x_scaled,
            obj_param,
            fixed_values,
            rho_leaf,
            wls,
            scale,
        )

    result = minimize(
        objective,
        x0_scaled,
        method=method,
        bounds=scaled_bounds,
        options=options,
    )

    best_scaled = result.x
    best_physical = lower + best_scaled * span

    best_params = {
        "N_leaf": float(best_physical[0]),
        "Cab": float(best_physical[1]),
        "Car": float(best_physical[2]),
        "Cbrown": float(best_physical[3]),
        "Cw": float(best_physical[4]),
        "Cm": float(best_physical[5]),
        "Ant": float(ant),
        "cost": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
    }

    return best_params, result

def invert_prospectd_interpolated(
    rho_measured,
    wl_measured,
    x0=None,
    bounds=None,
    ant=0.0,
    fit_ranges=((400, 1350), (1450, 1800), (1950, 2500)),
    n_restarts=20,
    seed=123,
):
    """
    Invert PROSPECT-D parameters from a measured leaf reflectance spectrum.

    Estimates:
        N_leaf, Cab, Car, Cbrown, Cw, Cm

    Parameters
    ----------
    rho_measured : array-like, shape [M]
        Measured reflectance. Should be in [0, 1].

    wl_measured : array-like, shape [M]
        Measured wavelengths in nm.

    x0 : array-like, optional, shape [6]
        Initial guess:
            [N_leaf, Cab, Car, Cbrown, Cw, Cm]

    bounds : tuple, optional
        (lower, upper), each shape [6].

    ant : float
        Fixed anthocyanin value for PROSPECT-D.

    fit_ranges : tuple of wavelength ranges
        Wavelength intervals used for fitting.
        Default excludes common noisy water absorption regions.

    Returns
    -------
    params : dict
        Best estimated parameters.

    result : scipy.optimize.OptimizeResult
        Best least_squares result.

    fitted : np.ndarray
        Fitted reflectance interpolated to wl_measured.
    """

    rho_measured = np.asarray(rho_measured, dtype=float).squeeze()
    wl_measured = np.asarray(wl_measured, dtype=float).squeeze()

    if rho_measured.ndim != 1:
        raise ValueError(f"rho_measured must be 1D, got {rho_measured.shape}")

    if wl_measured.ndim != 1:
        raise ValueError(f"wl_measured must be 1D, got {wl_measured.shape}")

    if rho_measured.shape[0] != wl_measured.shape[0]:
        raise ValueError(
            f"rho_measured and wl_measured must have same length. "
            f"Got {rho_measured.shape[0]} and {wl_measured.shape[0]}"
        )

    # Convert percent reflectance to fraction if needed.
    if np.nanmax(rho_measured) > 2.0:
        rho_measured = rho_measured / 100.0

    rho_measured = np.clip(rho_measured, 0.0, 1.0)

    if bounds is None:
        lower = np.array([1.0, 0.0, 0.0, 0.0, 1e-5, 1e-4], dtype=float)
        upper = np.array([100.0, 1500.0, 500.0, 100.0, 1, 1], dtype=float)
    else:
        lower, upper = bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

    if x0 is None:
        x0 = np.array([1.5, 40.0, 8.0, 0.0, 0.015, 0.009], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)

    def valid_mask():
        mask = np.isfinite(wl_measured) & np.isfinite(rho_measured)

        range_mask = np.zeros_like(mask, dtype=bool)
        for lo, hi in fit_ranges:
            range_mask |= (wl_measured >= lo) & (wl_measured <= hi)

        return mask & range_mask

    mask = valid_mask()

    if mask.sum() < 10:
        raise ValueError("Too few wavelengths available after masking.")

    def forward(params):
        N_leaf, Cab, Car, Cbrown, Cw, Cm = params

        wl_model, rho_model, tau_model = prospect.prospectd(
            N_leaf,
            Cab,
            Car,
            Cbrown,
            Cw,
            Cm,
            ant,
        )

        wl_model = np.asarray(wl_model, dtype=float)
        rho_model = np.asarray(rho_model, dtype=float)

        rho_interp = np.interp(wl_measured, wl_model, rho_model)
        return rho_interp

    def residual(params):
        rho_fit = forward(params)

        # Plain residual in reflectance units.
        res = rho_fit[mask] - rho_measured[mask]

        return res

    rng = np.random.default_rng(seed)

    initial_guesses = [np.clip(x0, lower, upper)]

    for _ in range(max(0, n_restarts - 1)):
        initial_guesses.append(lower + rng.random(6) * (upper - lower))

    best_result = None
    best_rmse = np.inf

    for init in initial_guesses:
        result = least_squares(
            residual,
            x0=init,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=5000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        rmse = np.sqrt(np.mean(residual(result.x) ** 2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_result = result

    p = best_result.x
    fitted = forward(p)

    params = {
        "N_leaf": float(p[0]),
        "Cab": float(p[1]),
        "Car": float(p[2]),
        "Cbrown": float(p[3]),
        "Cw": float(p[4]),
        "Cm": float(p[5]),
        "Ant": float(ant),
        "rmse": float(best_rmse),
        "success": bool(best_result.success),
        "message": str(best_result.message),
    }

    return params, best_result, fitted

def invert_prospect_parameters_by_regions(
    rho_measured,
    wl_measured,
    regions=None,
    x0=None,
    bounds=None,
    ant=0.0,
    n_restarts=20,
    seed=123,
    loss="soft_l1",
    f_scale=0.02,
    max_nfev=5000,
    verbose=True,
):
    """
    Invert PROSPECT-D parameters by fitting spectral regions.

    Estimates:
        N_leaf, Cab, Car, Cbrown, Cw, Cm

    Parameters
    ----------
    rho_measured : array-like, shape [M]
        Measured leaf reflectance spectrum.
        Expected scale is [0, 1]. If max > 2, it is assumed to be percent
        reflectance and divided by 100.

    wl_measured : array-like, shape [M]
        Wavelengths corresponding to rho_measured, in nm.

    regions : list of dicts
        Spectral regions used for fitting.

        Example:
            regions = [
                {"name": "VIS", "range": (400, 700), "weight": 2.0},
                {"name": "NIR", "range": (700, 1300), "weight": 1.0},
                {"name": "SWIR1", "range": (1450, 1800), "weight": 1.0},
                {"name": "SWIR2", "range": (1950, 2500), "weight": 1.0},
            ]

        The residuals from each region are normalized by the number of bands
        in that region, then multiplied by region weight. This prevents long
        regions from dominating only because they contain more wavelength samples.

    x0 : array-like, shape [6], optional
        Initial guess:
            [N_leaf, Cab, Car, Cbrown, Cw, Cm]

    bounds : tuple, optional
        Tuple (lower, upper), where each is shape [6].

    ant : float
        Fixed anthocyanin parameter for PROSPECT-D.

    n_restarts : int
        Number of random restarts.

    seed : int
        Random seed for random restarts.

    loss : str
        scipy.optimize.least_squares loss.
        Good options:
            "linear", "soft_l1", "huber", "cauchy"

    f_scale : float
        Robust-loss scale.

    max_nfev : int
        Maximum number of function evaluations per restart.

    Returns
    -------
    best_params : dict
        Estimated parameters and fitting diagnostics.

    best_result : scipy.optimize.OptimizeResult
        Best scipy result.

    rho_fitted : np.ndarray, shape [M]
        Fitted PROSPECT reflectance interpolated to wl_measured.

    region_report : dict
        Per-region RMSE and number of bands.
    """

    rho_measured = np.asarray(rho_measured, dtype=float).squeeze()
    wl_measured = np.asarray(wl_measured, dtype=float).squeeze()

    if rho_measured.ndim != 1:
        raise ValueError(f"rho_measured must be 1D, got {rho_measured.shape}")

    if wl_measured.ndim != 1:
        raise ValueError(f"wl_measured must be 1D, got {wl_measured.shape}")

    if rho_measured.shape[0] != wl_measured.shape[0]:
        raise ValueError(
            "rho_measured and wl_measured must have same length. "
            f"Got {rho_measured.shape[0]} and {wl_measured.shape[0]}."
        )

    # Convert percent reflectance to [0, 1] if needed.
    if np.nanmax(rho_measured) > 2.0:
        rho_measured = rho_measured / 100.0

    rho_measured = np.clip(rho_measured, 0.0, 1.0)

    if regions is None:
        regions = [
            {"name": "VIS", "range": (400.0, 700.0), "weight": 2.0},
            {"name": "NIR", "range": (700.0, 1300.0), "weight": 1.0},
            # Common water absorption/noisy areas are often excluded:
            # 1350–1450 and 1800–1950.
            {"name": "SWIR1", "range": (1450.0, 1800.0), "weight": 1.0},
            {"name": "SWIR2", "range": (1950.0, 2500.0), "weight": 1.0},
        ]

    if bounds is None:
        lower = np.array([1.0, 0.0, 0.0, 0.0, 1e-5, 1e-4], dtype=float)
        upper = np.array([100.0, 1500.0, 500.0, 100.0, 1, 1], dtype=float)
    else:
        lower, upper = bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

    if lower.shape[0] != 6 or upper.shape[0] != 6:
        raise ValueError("Bounds must contain 6 lower and 6 upper values.")

    if np.any(upper <= lower):
        raise ValueError("Each upper bound must be greater than lower bound.")

    if x0 is None:
        x0 = np.array([1.5, 40.0, 8.0, 0.0, 0.015, 0.009], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).squeeze()

    if x0.shape[0] != 6:
        raise ValueError(f"x0 must have 6 values, got {x0.shape[0]}.")

    x0 = np.clip(x0, lower, upper)

    # Build masks once.
    region_masks = []
    for region in regions:
        name = region["name"]
        lo, hi = region["range"]
        weight = float(region.get("weight", 1.0))

        mask = (
            np.isfinite(wl_measured)
            & np.isfinite(rho_measured)
            & (wl_measured >= lo)
            & (wl_measured <= hi)
        )

        n_bands = int(mask.sum())

        if n_bands == 0:
            if verbose:
                print(f"Warning: region {name} has no wavelength samples.")
            continue

        region_masks.append(
            {
                "name": name,
                "mask": mask,
                "weight": weight,
                "n_bands": n_bands,
                "range": (lo, hi),
            }
        )

    if len(region_masks) == 0:
        raise ValueError("No valid spectral regions available for fitting.")

    def forward(params):
        """
        Run PROSPECT-D and interpolate reflectance to measured wavelengths.
        """
        N_leaf, Cab, Car, Cbrown, Cw, Cm = params

        wl_model, rho_model, tau_model = prospect.prospectd(
            N_leaf,
            Cab,
            Car,
            Cbrown,
            Cw,
            Cm,
            ant,
        )

        wl_model = np.asarray(wl_model, dtype=float).squeeze()
        rho_model = np.asarray(rho_model, dtype=float).squeeze()

        rho_interp = np.interp(wl_measured, wl_model, rho_model)
        return rho_interp

    def residual_by_regions(params):
        """
        Concatenate weighted residuals region by region.

        Each region is normalized by sqrt(n_bands), so a region with many bands
        does not dominate only because it is longer.
        """
        rho_fit = forward(params)
        all_residuals = []

        for region in region_masks:
            mask = region["mask"]
            weight = region["weight"]
            n_bands = region["n_bands"]

            res = rho_fit[mask] - rho_measured[mask]

            # Region-balanced residual.
            res = weight * res / np.sqrt(n_bands)

            all_residuals.append(res)

        return np.concatenate(all_residuals)

    rng = np.random.default_rng(seed)

    initial_guesses = [x0]
    for _ in range(max(0, n_restarts - 1)):
        random_x0 = lower + rng.random(6) * (upper - lower)
        initial_guesses.append(random_x0)

    best_result = None
    best_cost = np.inf

    for i, init in enumerate(initial_guesses):
        result = least_squares(
            residual_by_regions,
            x0=init,
            bounds=(lower, upper),
            method="trf",
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        cost = result.cost

        if verbose:
            rho_tmp = forward(result.x)
            total_rmse = np.sqrt(np.mean((rho_tmp - rho_measured) ** 2))
            print(
                f"restart {i + 1:02d}/{len(initial_guesses)} | "
                f"cost={cost:.8e} | "
                f"full_rmse={total_rmse:.8f} | "
                f"params={result.x}"
            )

        if cost < best_cost:
            best_cost = cost
            best_result = result

    p = best_result.x
    rho_fitted = forward(p)

    # Region diagnostics without weighting or normalization.
    region_report = {}
    for region in region_masks:
        name = region["name"]
        mask = region["mask"]

        err = rho_fitted[mask] - rho_measured[mask]
        rmse = np.sqrt(np.mean(err ** 2))
        mae = np.mean(np.abs(err))

        region_report[name] = {
            "range": region["range"],
            "weight": region["weight"],
            "n_bands": region["n_bands"],
            "rmse": float(rmse),
            "mae": float(mae),
        }

    full_err = rho_fitted - rho_measured
    full_rmse = np.sqrt(np.mean(full_err ** 2))
    full_mae = np.mean(np.abs(full_err))

    best_params = {
        "N_leaf": float(p[0]),
        "Cab": float(p[1]),
        "Car": float(p[2]),
        "Cbrown": float(p[3]),
        "Cw": float(p[4]),
        "Cm": float(p[5]),
        "Ant": float(ant),
        "cost": float(best_result.cost),
        "full_rmse": float(full_rmse),
        "full_mae": float(full_mae),
        "success": bool(best_result.success),
        "message": str(best_result.message),
    }

    return best_params, best_result, rho_fitted, region_report




