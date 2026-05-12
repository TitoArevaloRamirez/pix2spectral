# test_conditional_prospect_flow.py
#
# Evaluate a trained conditional flow-matching model on a test CSV.
#
# Outputs:
#   1. Terminal table with quantitative metrics per stage.
#   2. Plot with one subplot per stage:
#        - average measured test spectrum: solid line
#        - measured std: shaded area
#        - 5 generated spectra sampled from the flow model
#
# Example:
#   python test_flow2spectral.py \
#       --test_csv_path /path/to/avocado_test.csv \
#       --checkpoint_path checkpoints/avocado_conditional_flow_best.pt \
#       --root_dir "/media/usr3/Expansion/Data/EstradaDataset/Avocado/Multispectral Images/" \
#       --species Avocado \
#       --stage all \
#       --test_cache_path cache/avocado_test_prospect_data.npz \
#       --n_samples_per_stage 5 \
#       --sampling_steps 100 \
#       --force_recompute_cache

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flow_matching.solver import ODESolver

from dataset import (
    BAND_KEYS,
    MultiSpectralCSVPatchDataset,
    normalize_stage_name,
    patch_collate_fn,
)
from pypro4sail import prospect
from utils import invert_prospect_parameters


PARAMETER_NAMES = [
    "N_leaf",
    "Cab",
    "Car",
    "Cbrown",
    "Cw",
    "Cm",
    "Ant",
]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class TestConfig:
    test_csv_path: str
    checkpoint_path: str

    root_dir: str | None = None
    species: str | None = None
    stage: str | None = "all"

    patch_h: int = 32
    patch_w: int = 32
    stride_h: int | None = 16
    stride_w: int | None = 16
    black_thr: float = 0.0

    wavelength_min: float = 400.0
    wavelength_max: float = 2500.0
    wavelength_count: int = 2101

    test_cache_path: str = "cache/test_conditional_prospect_data.npz"
    force_recompute_cache: bool = False
    n_fit_samples: int = -1

    pixel_scale: float = 255.0
    include_patch_count: bool = False

    batch_size: int = 64
    num_workers: int = 2

    n_samples_per_stage: int = 5
    sampling_steps: int = 100
    sample_mode: str = "mean"  # "mean" or "random"
    stage_order: str = "fresh,stage1,stage2,stage3,dry"

    save_plot_path: str | None = None
    no_show: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def make_wavelengths(cfg: TestConfig) -> np.ndarray:
    return np.linspace(
        cfg.wavelength_min,
        cfg.wavelength_max,
        cfg.wavelength_count,
        dtype=np.float64,
    )


def make_feature_names(include_patch_count: bool = True) -> List[str]:
    names = []

    for band in BAND_KEYS:
        names.extend(
            [
                f"{band}_mean",
                f"{band}_std",
                f"{band}_texture",
            ]
        )

        if include_patch_count:
            names.append(f"{band}_patch_count")

    return names


def patch_texture_energy(patches: torch.Tensor) -> torch.Tensor:
    """
    Average gradient-energy texture descriptor.

    patches:
        [N, 1, H, W]
    """
    if patches.numel() == 0 or patches.shape[0] == 0:
        return torch.tensor(0.0)

    dx = patches[..., :, 1:] - patches[..., :, :-1]
    dy = patches[..., 1:, :] - patches[..., :-1, :]

    return torch.sqrt((dx**2).mean() + (dy**2).mean() + 1e-12)


def summarize_one_band_patches(
    patches: torch.Tensor,
    pixel_scale: float,
    include_patch_count: bool,
) -> List[float]:
    """
    Summarize all ROI patches for one band of one sample.

    Returns:
        [mean, std, texture, optional patch_count]
    """
    if patches.numel() == 0 or patches.shape[0] == 0:
        values = [0.0, 0.0, 0.0]

        if include_patch_count:
            values.append(0.0)

        return values

    x = patches.float()

    if pixel_scale is not None and pixel_scale > 0:
        x = x / float(pixel_scale)

    flat = x.reshape(-1)

    values = [
        float(flat.mean().item()),
        float(flat.std(unbiased=False).item()),
        float(patch_texture_energy(x).item()),
    ]

    if include_patch_count:
        values.append(float(patches.shape[0]))

    return values


def extract_condition_features_from_batch(
    band_dict: Dict[str, List[torch.Tensor]],
    pixel_scale: float,
    include_patch_count: bool,
) -> np.ndarray:
    """
    Convert one dataloader batch of patch dictionaries into condition features.

    Returns:
        features: [B, F]
    """
    batch_size = len(band_dict[BAND_KEYS[0]])
    rows = []

    for sample_idx in range(batch_size):
        row = []

        for band_name in BAND_KEYS:
            row.extend(
                summarize_one_band_patches(
                    patches=band_dict[band_name][sample_idx],
                    pixel_scale=pixel_scale,
                    include_patch_count=include_patch_count,
                )
            )

        rows.append(row)

    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------------
# Dataset/cache
# ---------------------------------------------------------------------

def print_stage_distribution(stages: np.ndarray, title: str = "Stage distribution") -> None:
    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])
    unique, counts = np.unique(stages_clean, return_counts=True)

    print("\n" + title)
    print("-" * len(title))

    for stage, count in zip(unique, counts):
        print(f"{stage}: {count}")


def build_test_dataset(cfg: TestConfig) -> MultiSpectralCSVPatchDataset:
    return MultiSpectralCSVPatchDataset(
        csv_path=os.path.expanduser(cfg.test_csv_path),
        root_dir=cfg.root_dir,
        species=cfg.species,
        stage=cfg.stage,
        patch_h=cfg.patch_h,
        patch_w=cfg.patch_w,
        stride_h=cfg.stride_h,
        stride_w=cfg.stride_w,
        black_thr=cfg.black_thr,
    )


def collect_spectra_features_stages(
    cfg: TestConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        spectra:       [N, M]
        features:      [N, F]
        stages:        [N]
        feature_names: list[str]
    """
    dataset = build_test_dataset(cfg)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=patch_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    feature_names = make_feature_names(cfg.include_patch_count)

    all_spectra = []
    all_features = []
    all_stages = []

    for batch_idx, (band_dict, spec, stages) in enumerate(loader):
        features = extract_condition_features_from_batch(
            band_dict=band_dict,
            pixel_scale=cfg.pixel_scale,
            include_patch_count=cfg.include_patch_count,
        )

        all_features.append(features)
        all_spectra.append(spec.cpu().numpy().astype(np.float32))
        all_stages.extend([normalize_stage_name(s) for s in stages])

        print(
            f"[test] feature extraction batch "
            f"{batch_idx + 1:04d}/{len(loader):04d} | "
            f"features={features.shape} | spectra={tuple(spec.shape)}"
        )

    spectra_np = np.concatenate(all_spectra, axis=0).astype(np.float32)
    features_np = np.concatenate(all_features, axis=0).astype(np.float32)
    stages_np = np.asarray(all_stages, dtype=str)

    return spectra_np, features_np, stages_np, feature_names


def select_balanced_indices_by_stage(
    stages: np.ndarray,
    n_fit_samples: int,
    seed: int = 123,
) -> np.ndarray:
    """
    Optional balanced subset for fast testing.
    If n_fit_samples < 0, uses all test samples.
    """
    n_total = len(stages)

    if n_fit_samples is None or n_fit_samples < 0 or n_fit_samples >= n_total:
        return np.arange(n_total, dtype=int)

    rng = np.random.default_rng(seed)
    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])

    unique_stages = np.unique(stages_clean)
    n_stages = len(unique_stages)

    base_per_stage = max(1, n_fit_samples // n_stages)
    remainder = n_fit_samples % n_stages

    selected = []

    for i, stage_name in enumerate(unique_stages):
        idx = np.where(stages_clean == stage_name)[0]
        rng.shuffle(idx)

        k = base_per_stage + (1 if i < remainder else 0)
        k = min(k, len(idx))

        selected.extend(idx[:k].tolist())

    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)

    return selected


def invert_spectra_to_prospect_params(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Invert measured test spectra into PROSPECT parameters.

    Returns:
        params: [N, 7]
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be [N, M], got {spectra.shape}")

    if spectra.shape[1] != wavelengths.shape[0]:
        raise ValueError(
            f"Spectrum length {spectra.shape[1]} does not match "
            f"wavelength count {wavelengths.shape[0]}."
        )

    params = []

    for idx in range(spectra.shape[0]):
        rho = spectra[idx].astype(np.float64)

        best_params, result = invert_prospect_parameters(
            rho_leaf=rho,
            wls=wavelengths,
        )

        row = [
            best_params["N_leaf"],
            best_params["Cab"],
            best_params["Car"],
            best_params["Cbrown"],
            best_params["Cw"],
            best_params["Cm"],
            best_params.get("Ant", 0.0),
        ]

        params.append(row)

        print(
            f"[test] inverted {idx + 1:04d}/{spectra.shape[0]:04d} | "
            f"success={result.success} | "
            f"cost={float(result.fun):.8g} | "
            f"params={np.asarray(row)}"
        )

    return np.asarray(params, dtype=np.float32)


def load_or_build_test_arrays(
    cfg: TestConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        spectra:       [N, M]
        features:      [N, F]
        params:        [N, 7]
        stages:        [N]
        feature_names: list[str]
    """
    cache_path = os.path.expanduser(cfg.test_cache_path)
    wavelengths = make_wavelengths(cfg)

    if os.path.exists(cache_path) and not cfg.force_recompute_cache:
        data = np.load(cache_path, allow_pickle=True)

        spectra = data["spectra"].astype(np.float32)
        features = data["features"].astype(np.float32)
        params = data["params"].astype(np.float32)
        stages = data["stages"].astype(str)
        feature_names = data["feature_names"].tolist()

        print(f"[test] Loaded cached arrays from {cache_path}")
        print(f"[test] spectra shape:  {spectra.shape}")
        print(f"[test] features shape: {features.shape}")
        print(f"[test] params shape:   {params.shape}")
        print(f"[test] stages shape:   {stages.shape}")
        print_stage_distribution(stages, "[test] cached stage distribution")

        return spectra, features, params, stages, feature_names

    spectra, features, stages, feature_names = collect_spectra_features_stages(cfg)

    selected_indices = select_balanced_indices_by_stage(
        stages=stages,
        n_fit_samples=cfg.n_fit_samples,
        seed=cfg.seed,
    )

    spectra = spectra[selected_indices]
    features = features[selected_indices]
    stages = stages[selected_indices]

    print_stage_distribution(stages, "[test] stage distribution after subset")

    params = invert_spectra_to_prospect_params(
        spectra=spectra,
        wavelengths=wavelengths,
    )

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    np.savez(
        cache_path,
        spectra=spectra,
        features=features,
        params=params,
        stages=stages,
        feature_names=np.asarray(feature_names),
        parameter_names=np.asarray(PARAMETER_NAMES),
        wavelengths=wavelengths,
    )

    print(f"[test] Saved cached arrays to {cache_path}")

    return spectra, features, params, stages, feature_names


# ---------------------------------------------------------------------
# Model definition: same architecture as training
# ---------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        if not isinstance(dim, int):
            raise TypeError(f"time_dim must be int, got {type(dim)}")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]

        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], got {t.shape}")

        half = self.dim // 2
        device = t.device

        freqs = torch.exp(
            -np.log(10000.0)
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class ConditionalProspectVelocityMLP(nn.Module):
    """
    Conditional velocity model:
        v_theta(x_t, t, condition)
    """

    def __init__(
        self,
        param_dim: int,
        condition_dim_in: int,
        hidden: int = 128,
        depth: int = 3,
        time_dim: int = 64,
        condition_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.param_dim = param_dim
        self.condition_dim_in = condition_dim_in
        self.hidden = hidden
        self.depth = depth
        self.time_dim = time_dim
        self.condition_dim = condition_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, condition_dim),
            nn.SiLU(),
        )

        layers = []
        dim = param_dim + hidden + condition_dim

        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden

        layers.append(nn.Linear(hidden, param_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be [B, D], got {x.shape}")

        if condition.ndim != 2:
            raise ValueError(f"condition must be [B, F], got {condition.shape}")

        if t.ndim == 0:
            t = t.expand(x.shape[0])

        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {t.shape}")

        if x.shape[0] != condition.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has {x.shape[0]}, "
                f"condition has {condition.shape[0]}"
            )

        if x.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has {x.shape[0]}, t has {t.shape[0]}"
            )

        #t_emb = self.time_embed(t)
        t_scalar = t[:, None]
        t_emb = self.time_embed(t_scalar)
        #t_emb = self.time_embed(t)
        c_emb = self.condition_embed(condition)

        h = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(h)


class ConditionedVelocityWrapper(nn.Module):
    """
    Wrapper for ODESolver, which expects velocity_model(x, t).
    """

    def __init__(
        self,
        base_model: ConditionalProspectVelocityMLP,
        condition: torch.Tensor,
    ):
        super().__init__()
        self.base_model = base_model
        self.condition = condition

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.base_model(
            x=x,
            t=t,
            condition=self.condition,
        )


# ---------------------------------------------------------------------
# Checkpoint loading and sampling
# ---------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ConditionalProspectVelocityMLP, dict]:
    """
    Load a trained conditional flow model checkpoint.

    Important:
        PyTorch >= 2.6 changed torch.load default behavior to weights_only=True.
        Since our checkpoint stores numpy arrays and metadata, we must use
        weights_only=False.

    Only use weights_only=False for checkpoints you trust.
    """
    checkpoint_path = os.path.expanduser(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = [
        "model_state_dict",
        "param_dim",
        "condition_dim_in",
        "hidden",
        "depth",
        "time_dim",
        "condition_dim",
        "param_mean",
        "param_std",
        "feature_mean",
        "feature_std",
    ]

    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError(
            "Checkpoint is missing required keys: "
            + ", ".join(missing)
        )

    model = ConditionalProspectVelocityMLP(
        param_dim=int(checkpoint["param_dim"]),
        condition_dim_in=int(checkpoint["condition_dim_in"]),
        hidden=int(checkpoint["hidden"]),
        depth=int(checkpoint["depth"]),
        time_dim=int(checkpoint["time_dim"]),
        condition_dim=int(checkpoint["condition_dim"]),
        dropout=0.0,  # disable dropout during evaluation
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "best_val_mse" in checkpoint:
        print(f"Checkpoint best_val_mse: {checkpoint['best_val_mse']}")

    return model, checkpoint


@torch.no_grad()
def sample_conditional_flow(
    model: ConditionalProspectVelocityMLP,
    conditions_norm: np.ndarray,
    cfg: TestConfig,
) -> np.ndarray:
    """
    Sample normalized PROSPECT parameters conditioned on patch features.

    conditions_norm:
        [B, F]
    """
    device = torch.device(cfg.device)

    model.eval()

    condition = torch.from_numpy(conditions_norm).float().to(device)
    n_samples = condition.shape[0]

    wrapper = ConditionedVelocityWrapper(
        base_model=model,
        condition=condition,
    ).to(device)

    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(
        n_samples,
        model.param_dim,
        device=device,
        dtype=torch.float32,
    )

    time_grid = torch.linspace(
        0.0,
        1.0,
        cfg.sampling_steps + 1,
        device=device,
        dtype=torch.float32,
    )

    generated = solver.sample(
        x_init=x_init,
        step_size=1.0 / cfg.sampling_steps,
        method="euler",
        time_grid=time_grid,
    )

    return generated.detach().cpu().numpy()


def apply_normalization(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def denormalize_array(
    x_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return x_norm * std + mean


def sanitize_params_with_train_bounds(
    params: np.ndarray,
    train_params_raw: np.ndarray | None,
    margin_fraction: float = 0.25,
) -> np.ndarray:
    """
    Clamp generated parameters using train parameter quantile bounds if available.
    """
    if train_params_raw is None:
        return params

    q_low = np.percentile(train_params_raw, 1, axis=0, keepdims=True)
    q_high = np.percentile(train_params_raw, 99, axis=0, keepdims=True)
    margin = margin_fraction * (q_high - q_low)

    lower = q_low - margin
    upper = q_high + margin

    lower[:, 0] = np.maximum(lower[:, 0], 1.0)
    lower[:, 1:] = np.maximum(lower[:, 1:], 0.0)

    return np.clip(params, lower, upper)


# ---------------------------------------------------------------------
# PROSPECT decoding and metrics
# ---------------------------------------------------------------------

def prospect_reflectance_from_params(
    params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    params:
        [7] = [N_leaf, Cab, Car, Cbrown, Cw, Cm, Ant]
    """
    wl_model, rho_model, _ = prospect.prospectd(
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
    )

    return np.asarray(wl_model), np.asarray(rho_model)


def spectra_from_params(
    params: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Decode generated PROSPECT parameters to reflectance spectra.

    params:
        [N, 7]

    returns:
        spectra: [N, M]
    """
    spectra = []

    for row in params:
        wl_model, rho_model = prospect_reflectance_from_params(row)
        rho_interp = np.interp(wavelengths, wl_model, rho_model)
        spectra.append(rho_interp)

    return np.asarray(spectra, dtype=np.float32)


def spectral_angle_mapper(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Spectral angle mapper in radians.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    numerator = np.sum(y_true * y_pred, axis=1)
    denominator = (
        np.linalg.norm(y_true, axis=1)
        * np.linalg.norm(y_pred, axis=1)
        + eps
    )

    cos_theta = np.clip(numerator / denominator, -1.0, 1.0)
    angles = np.arccos(cos_theta)

    return float(np.mean(angles))


def compute_stage_metrics(
    measured_spectra: np.ndarray,
    measured_params: np.ndarray,
    generated_spectra: np.ndarray,
    generated_params: np.ndarray,
) -> Dict[str, float]:
    """
    Compare measured test spectra/params to generated spectra/params.

    generated_* should be aligned to measured samples or repeated conditions.
    """
    spectral_error = generated_spectra - measured_spectra
    param_error = generated_params - measured_params

    metrics = {
        "n": measured_spectra.shape[0],
        "spectral_rmse": float(np.sqrt(np.mean(spectral_error**2))),
        "spectral_mae": float(np.mean(np.abs(spectral_error))),
        "spectral_sam_rad": spectral_angle_mapper(measured_spectra, generated_spectra),
        "param_rmse": float(np.sqrt(np.mean(param_error**2))),
        "param_mae": float(np.mean(np.abs(param_error))),
    }

    return metrics


def print_metrics_table(stage_metrics: Dict[str, Dict[str, float]]) -> None:
    headers = [
        "stage",
        "n",
        "spectral_rmse",
        "spectral_mae",
        "sam_rad",
        "param_rmse",
        "param_mae",
    ]

    rows = []

    for stage_name, metrics in stage_metrics.items():
        rows.append(
            [
                stage_name,
                int(metrics["n"]),
                metrics["spectral_rmse"],
                metrics["spectral_mae"],
                metrics["spectral_sam_rad"],
                metrics["param_rmse"],
                metrics["param_mae"],
            ]
        )

    col_widths = [max(len(h), 12) for h in headers]

    print("\nTest metrics by stage")
    print("---------------------")
    print(
        f"{headers[0]:<{col_widths[0]}} "
        f"{headers[1]:>{col_widths[1]}} "
        f"{headers[2]:>{col_widths[2]}} "
        f"{headers[3]:>{col_widths[3]}} "
        f"{headers[4]:>{col_widths[4]}} "
        f"{headers[5]:>{col_widths[5]}} "
        f"{headers[6]:>{col_widths[6]}}"
    )

    for row in rows:
        print(
            f"{row[0]:<{col_widths[0]}} "
            f"{row[1]:>{col_widths[1]}} "
            f"{row[2]:>{col_widths[2]}.6f} "
            f"{row[3]:>{col_widths[3]}.6f} "
            f"{row[4]:>{col_widths[4]}.6f} "
            f"{row[5]:>{col_widths[5]}.6f} "
            f"{row[6]:>{col_widths[6]}.6f}"
        )


# ---------------------------------------------------------------------
# Stage condition selection and evaluation
# ---------------------------------------------------------------------

def select_stage_condition(
    features_norm: np.ndarray,
    stages: np.ndarray,
    stage_name: str,
    mode: str,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """
    Select one representative normalized condition vector for a stage.

    Returns:
        condition: [1, F]
        reference_index: int
    """
    rng = np.random.default_rng(seed)
    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])
    stage_clean = normalize_stage_name(stage_name)

    idx = np.where(stages_clean == stage_clean)[0]

    if len(idx) == 0:
        raise ValueError(f"No samples found for stage: {stage_clean}")

    stage_features = features_norm[idx]

    if mode == "mean":
        condition = stage_features.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(stage_features - condition, axis=1)
        ref_index = int(idx[int(np.argmin(distances))])

    elif mode == "random":
        ref_index = int(rng.choice(idx))
        condition = features_norm[ref_index : ref_index + 1]

    else:
        raise ValueError("mode must be 'mean' or 'random'.")

    return condition.astype(np.float32), ref_index

def reconstruct_real_spectra_from_inverted_params(
    params: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct spectra from the inverted PROSPECT parameters that were
    estimated from the real measured spectral signatures.

    params:
        [N, 7]

    returns:
        reconstructed_spectra: [N, M]
    """
    return spectra_from_params(
        params=params,
        wavelengths=wavelengths,
    )


def generate_for_each_test_sample(
    model: ConditionalProspectVelocityMLP,
    features_norm: np.ndarray,
    checkpoint: dict,
    cfg: TestConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one parameter vector and spectrum per test sample.

    This is used for quantitative metrics.
    """
    param_mean = checkpoint["param_mean"]
    param_std = checkpoint["param_std"]

    if param_mean is None or param_std is None:
        raise ValueError("Checkpoint does not contain param_mean/param_std.")

    generated_norm = sample_conditional_flow(
        model=model,
        conditions_norm=features_norm,
        cfg=cfg,
    )

    generated_params = denormalize_array(
        x_norm=generated_norm,
        mean=param_mean,
        std=param_std,
    )

    train_params_raw = checkpoint.get("params_raw_train", None)

    generated_params = sanitize_params_with_train_bounds(
        generated_params,
        train_params_raw=train_params_raw,
    )

    wavelengths = make_wavelengths(cfg)
    generated_spectra = spectra_from_params(
        params=generated_params,
        wavelengths=wavelengths,
    )

    return generated_params, generated_spectra

def generate_stage_plot_samples_from_real_patches(
    model: ConditionalProspectVelocityMLP,
    features_norm: np.ndarray,
    stages: np.ndarray,
    checkpoint: dict,
    cfg: TestConfig,
    stage_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate spectra using real patch-feature conditions from a given stage.

    Unlike the previous version, this uses different real test image patch
    conditions from the selected stage.

    Returns:
        generated_params:  [K, 7]
        generated_spectra: [K, M]
        selected_indices:  [K]
    """
    rng = np.random.default_rng(cfg.seed)

    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])
    stage_clean = normalize_stage_name(stage_name)

    idx = np.where(stages_clean == stage_clean)[0]

    if len(idx) == 0:
        raise ValueError(f"No samples found for stage: {stage_clean}")

    k = min(cfg.n_samples_per_stage, len(idx))
    selected_indices = rng.choice(idx, size=k, replace=False)

    conditions = features_norm[selected_indices].astype(np.float32)

    generated_norm = sample_conditional_flow(
        model=model,
        conditions_norm=conditions,
        cfg=cfg,
    )

    generated_params = denormalize_array(
        x_norm=generated_norm,
        mean=checkpoint["param_mean"],
        std=checkpoint["param_std"],
    )

    generated_params = sanitize_params_with_train_bounds(
        generated_params,
        train_params_raw=checkpoint.get("params_raw_train", None),
    )

    wavelengths = make_wavelengths(cfg)

    generated_spectra = spectra_from_params(
        params=generated_params,
        wavelengths=wavelengths,
    )

    return generated_params, generated_spectra, selected_indices



def generate_stage_plot_samples(
    model: ConditionalProspectVelocityMLP,
    features_norm: np.ndarray,
    stages: np.ndarray,
    checkpoint: dict,
    cfg: TestConfig,
    stage_name: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate n_samples_per_stage spectra using one representative condition
    for the selected stage.
    """
    condition, ref_index = select_stage_condition(
        features_norm=features_norm,
        stages=stages,
        stage_name=stage_name,
        mode=cfg.sample_mode,
        seed=cfg.seed,
    )

    conditions = np.repeat(condition, cfg.n_samples_per_stage, axis=0)

    generated_norm = sample_conditional_flow(
        model=model,
        conditions_norm=conditions,
        cfg=cfg,
    )

    generated_params = denormalize_array(
        x_norm=generated_norm,
        mean=checkpoint["param_mean"],
        std=checkpoint["param_std"],
    )

    generated_params = sanitize_params_with_train_bounds(
        generated_params,
        train_params_raw=checkpoint.get("params_raw_train", None),
    )

    wavelengths = make_wavelengths(cfg)
    generated_spectra = spectra_from_params(
        params=generated_params,
        wavelengths=wavelengths,
    )

    return generated_params, generated_spectra, ref_index


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_stage_spectra(
    wavelengths: np.ndarray,
    measured_spectra: np.ndarray,
    reconstructed_real_spectra: np.ndarray,
    stages: np.ndarray,
    generated_plot_spectra_by_stage: Dict[str, np.ndarray],
    stage_order: List[str],
    cfg: TestConfig,
) -> None:
    """
    Plot, for each stage:
      - mean measured real spectrum
      - measured std as shaded area
      - mean reconstructed spectrum from inverted real PROSPECT params
      - 5 generated spectra from the flow model
    """
    available_stages = [
        s for s in stage_order
        if s in generated_plot_spectra_by_stage
    ]

    n_stages = len(available_stages)

    if n_stages == 0:
        print("No available stages to plot.")
        return

    ncols = 1
    nrows = n_stages

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(11, 3.4 * nrows),
        sharex=True,
    )

    if n_stages == 1:
        axes = [axes]

    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])

    for ax, stage_name in zip(axes, available_stages):
        idx = np.where(stages_clean == stage_name)[0]

        if len(idx) == 0:
            continue

        # --------------------------------------------
        # Measured real spectra
        # --------------------------------------------
        stage_measured = measured_spectra[idx]
        mean_measured = stage_measured.mean(axis=0)
        std_measured = stage_measured.std(axis=0)

        ax.plot(
            wavelengths,
            mean_measured,
            linewidth=2.2,
            label=f"Measured mean: {stage_name}",
        )

        ax.fill_between(
            wavelengths,
            mean_measured - std_measured,
            mean_measured + std_measured,
            alpha=0.22,
            label="Measured ±1 std",
        )

        # --------------------------------------------
        # Reconstructed spectra from inverted real params
        # --------------------------------------------
        stage_reconstructed = reconstructed_real_spectra[idx]
        mean_reconstructed = stage_reconstructed.mean(axis=0)
        std_reconstructed = stage_reconstructed.std(axis=0)

        ax.plot(
            wavelengths,
            mean_reconstructed,
            linestyle="--",
            linewidth=2.0,
            label="Reconstructed from inverted real params",
        )

        ax.fill_between(
            wavelengths,
            mean_reconstructed - std_reconstructed,
            mean_reconstructed + std_reconstructed,
            alpha=0.12,
            label="Reconstructed ±1 std",
        )

        # --------------------------------------------
        # Generated spectra from the flow model
        # --------------------------------------------
        generated_spectra = generated_plot_spectra_by_stage[stage_name]

        for i in range(generated_spectra.shape[0]):
            ax.plot(
                wavelengths,
                generated_spectra[i],
                linewidth=1.2,
                alpha=0.9,
                label=f"Flow sample {i + 1}" if i == 0 else None,
            )

        ax.set_title(f"Stage: {stage_name} | n={len(idx)}")
        ax.set_ylabel("Reflectance")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Wavelength [nm]")

    fig.tight_layout()

    if cfg.save_plot_path is not None:
        os.makedirs(os.path.dirname(cfg.save_plot_path) or ".", exist_ok=True)
        fig.savefig(cfg.save_plot_path, dpi=200)
        print(f"Saved plot to: {cfg.save_plot_path}")

    if not cfg.no_show:
        plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(cfg: TestConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    wavelengths = make_wavelengths(cfg)

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
    )

    param_mean = checkpoint.get("param_mean", None)
    param_std = checkpoint.get("param_std", None)
    feature_mean = checkpoint.get("feature_mean", None)
    feature_std = checkpoint.get("feature_std", None)

    if param_mean is None or param_std is None:
        raise ValueError("Checkpoint is missing param_mean or param_std.")

    if feature_mean is None or feature_std is None:
        raise ValueError("Checkpoint is missing feature_mean or feature_std.")

    spectra, features, params, stages, feature_names = load_or_build_test_arrays(cfg)

    reconstructed_real_spectra = reconstruct_real_spectra_from_inverted_params(
        params=params,
        wavelengths=wavelengths,)


    checkpoint_feature_names = checkpoint.get("feature_names", None)

    if checkpoint_feature_names is not None and list(checkpoint_feature_names) != list(feature_names):
        print("\nWarning: feature names from test file and checkpoint differ.")
        print("Checkpoint feature names:")
        print(checkpoint_feature_names)
        print("Test feature names:")
        print(feature_names)

    print_stage_distribution(stages, "[test] loaded stage distribution")

    features_norm = apply_normalization(
        x=features,
        mean=feature_mean,
        std=feature_std,
    )

    params_norm = apply_normalization(
        x=params,
        mean=param_mean,
        std=param_std,
    )

    print("\nTest array summary")
    print("------------------")
    print(f"spectra:      {spectra.shape}")
    print(f"features:     {features.shape}")
    print(f"features_norm:{features_norm.shape}")
    print(f"params:       {params.shape}")
    print(f"params_norm:  {params_norm.shape}")

    # ------------------------------------------------------------
    # Quantitative evaluation
    # ------------------------------------------------------------
    generated_params, generated_spectra = generate_for_each_test_sample(
        model=model,
        features_norm=features_norm,
        checkpoint=checkpoint,
        cfg=cfg,
    )

    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])
    requested_stage_order = [normalize_stage_name(s) for s in cfg.stage_order.split(",")]

    available_stage_order = [
        s for s in requested_stage_order
        if np.any(stages_clean == s)
    ]

    # Include any extra stage not listed in stage_order.
    for s in np.unique(stages_clean):
        if s not in available_stage_order:
            available_stage_order.append(s)

    stage_metrics = {}

    for stage_name in available_stage_order:
        idx = np.where(stages_clean == stage_name)[0]

        if len(idx) == 0:
            continue

        stage_metrics[stage_name] = compute_stage_metrics(
            measured_spectra=spectra[idx],
            measured_params=params[idx],
            generated_spectra=generated_spectra[idx],
            generated_params=generated_params[idx],
        )

    print_metrics_table(stage_metrics)

    # ------------------------------------------------------------
    # Plotting samples
    # ------------------------------------------------------------
    generated_plot_spectra_by_stage = {}

    for stage_name in available_stage_order:
        try:
            _, generated_stage_spectra, selected_indices = generate_stage_plot_samples_from_real_patches(
                model=model,
                features_norm=features_norm,
                stages=stages,
                checkpoint=checkpoint,
                cfg=cfg,
                stage_name=stage_name,
            )
            
            generated_plot_spectra_by_stage[stage_name] = generated_stage_spectra
            
            print(
                f"Generated {generated_stage_spectra.shape[0]} plot samples "
                f"for stage {stage_name} using real patch indices {selected_indices.tolist()}"
            )

        except ValueError as exc:
            print(f"Skipping plot samples for {stage_name}: {exc}")

    plot_stage_spectra(
        wavelengths=wavelengths,
        measured_spectra=spectra,
        reconstructed_real_spectra=reconstructed_real_spectra,
        stages=stages,
        generated_plot_spectra_by_stage=generated_plot_spectra_by_stage,
        stage_order=available_stage_order,
        cfg=cfg,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/usr3/Expansion/Data/EstradaDataset/Avocado/Multispectral Images/",
    )
    parser.add_argument("--species", type=str, default="Avocado")
    parser.add_argument("--stage", type=str, default="all")

    parser.add_argument("--patch_h", type=int, default=32)
    parser.add_argument("--patch_w", type=int, default=32)
    parser.add_argument("--stride_h", type=int, default=16)
    parser.add_argument("--stride_w", type=int, default=16)
    parser.add_argument("--black_thr", type=float, default=0.0)

    parser.add_argument("--wavelength_min", type=float, default=400.0)
    parser.add_argument("--wavelength_max", type=float, default=2500.0)
    parser.add_argument("--wavelength_count", type=int, default=2101)

    parser.add_argument(
        "--test_cache_path",
        type=str,
        default="cache/test_conditional_prospect_data.npz",
    )
    parser.add_argument("--force_recompute_cache", action="store_true")
    parser.add_argument("--n_fit_samples", type=int, default=-1)

    parser.add_argument("--pixel_scale", type=float, default=255.0)

    parser.add_argument(
        "--include_patch_count",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--n_samples_per_stage", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--sample_mode", type=str, default="mean", choices=["mean", "random"])
    parser.add_argument("--stage_order", type=str, default="fresh,stage1,stage2,stage3,dry")

    parser.add_argument("--save_plot_path", type=str, default=None)
    parser.add_argument("--no_show", action="store_true")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    return TestConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
