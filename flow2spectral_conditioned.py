# flow2spectral_conditioned_with_validation.py
#
# Conditional flow matching over PROSPECT parameters using multispectral patch statistics.
#
# Adds:
#   - validation CSV support
#   - separate train/validation caches
#   - train-only normalization statistics
#   - validation MSE/RMSE/alignment
#   - best checkpoint by validation MSE

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from flow_matching.path import CondOTProbPath
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


@dataclass
class TrainConfig:
    train_csv_path: str
    val_csv_path: str

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

    train_cache_path: str = "cache/train_conditional_prospect_data.npz"
    val_cache_path: str = "cache/val_conditional_prospect_data.npz"
    force_recompute_cache: bool = False
    n_fit_samples: int = -1
    n_val_fit_samples: int = -1

    pixel_scale: float = 255.0
    include_patch_count: bool = False

    batch_size: int = 64
    val_batch_size: int = 128
    epochs: int = 10000
    lr: float = 1e-4
    weight_decay: float = 1e-4

    hidden: int = 64
    depth: int = 3
    time_dim: int = 32
    condition_dim: int = 32
    dropout: float = 0.05

    grad_clip: float = 1.0
    log_every: int = 100
    val_every: int = 100
    checkpoint_every: int = 5000
    val_repeats: int = 3

    save_path: str = "checkpoints/conditional_prospect_flow_final.pt"
    best_save_path: str = "checkpoints/conditional_prospect_flow_best.pt"

    sampling_steps: int = 100
    sample_mode: str = "mean"
    stage_order: str = "fresh,stage1,stage2,stage3,dry"

    plot: bool = False

    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123
    log_path: str = "logs/conditional_flow_training_log.txt"
    early_stop_patience: int = 1000
    min_delta: float = 1e-4


def plot_parameter_statistics(
    params: np.ndarray,
    param_names=None,
    title: str = "Parameter statistics",
    ylabel: str = "Value",
):
    """
    Plot mean parameter values as a bar chart with std as error bars.

    Parameters
    ----------
    params : np.ndarray
        Array of shape [N, P], where N is number of samples and
        P is number of parameters.

    param_names : list[str] or None
        Names of the parameters. If None, generic names are used.

    title : str
        Plot title.

    ylabel : str
        Y-axis label.
    """
    params = np.asarray(params, dtype=np.float64)

    if params.ndim != 2:
        raise ValueError(f"params must have shape [N, P], got {params.shape}")

    n_samples, n_params = params.shape

    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    if len(param_names) != n_params:
        raise ValueError(
            f"len(param_names)={len(param_names)} does not match "
            f"number of parameters={n_params}"
        )

    means = params.mean(axis=0)
    stds = params.std(axis=0)

    x = np.arange(n_params)

    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, param_names, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_log(path: str, message: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def make_wavelengths(cfg: TrainConfig) -> np.ndarray:
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


def print_stage_distribution(stages: np.ndarray, title: str = "Stage distribution") -> None:
    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])
    unique, counts = np.unique(stages_clean, return_counts=True)

    print("\n" + title)
    print("-" * len(title))
    for stage, count in zip(unique, counts):
        print(f"{stage}: {count}")


def build_multispectral_dataset(
    csv_path: str,
    cfg: TrainConfig,
) -> MultiSpectralCSVPatchDataset:
    return MultiSpectralCSVPatchDataset(
        csv_path=os.path.expanduser(csv_path),
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
    csv_path: str,
    cfg: TrainConfig,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset = build_multispectral_dataset(csv_path=csv_path, cfg=cfg)

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
            f"[{split_name}] feature extraction batch "
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
    n_fit_samples: int = -1,
) -> np.ndarray:
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be [N, M], got {spectra.shape}")

    if spectra.shape[1] != wavelengths.shape[0]:
        raise ValueError(
            f"Spectrum length {spectra.shape[1]} does not match "
            f"wavelength count {wavelengths.shape[0]}."
        )

    if n_fit_samples is None or n_fit_samples < 0:
        n_fit_samples = spectra.shape[0]

    n_fit_samples = min(int(n_fit_samples), spectra.shape[0])

    params = []

    for idx in range(n_fit_samples):
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
            f"inverted {idx + 1:04d}/{n_fit_samples:04d} | "
            f"success={result.success} | "
            f"cost={float(result.fun):.8g} | "
            f"params={np.asarray(row)}"
        )

    return np.asarray(params, dtype=np.float32)


def load_or_build_arrays_for_split(
    csv_path: str,
    cache_path: str,
    cfg: TrainConfig,
    split_name: str,
    n_fit_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    cache_path = os.path.expanduser(cache_path)
    wavelengths = make_wavelengths(cfg)

    if os.path.exists(cache_path) and not cfg.force_recompute_cache:
        data = np.load(cache_path, allow_pickle=True)

        spectra = data["spectra"].astype(np.float32)
        features = data["features"].astype(np.float32)
        params = data["params"].astype(np.float32)
        stages = data["stages"].astype(str)
        feature_names = data["feature_names"].tolist()

        print(f"[{split_name}] Loaded cached arrays from {cache_path}")
        print(f"[{split_name}] spectra shape:  {spectra.shape}")
        print(f"[{split_name}] features shape: {features.shape}")
        print(f"[{split_name}] params shape:   {params.shape}")
        print(f"[{split_name}] stages shape:   {stages.shape}")
        print_stage_distribution(stages, f"[{split_name}] cached stage distribution")

        return spectra, features, params, stages, feature_names

    spectra, features, stages, feature_names = collect_spectra_features_stages(
        csv_path=csv_path,
        cfg=cfg,
        split_name=split_name,
    )

    selected_indices = select_balanced_indices_by_stage(
        stages=stages,
        n_fit_samples=n_fit_samples,
        seed=cfg.seed,
    )

    spectra = spectra[selected_indices]
    features = features[selected_indices]
    stages = stages[selected_indices]

    print_stage_distribution(stages, f"[{split_name}] stage distribution after subset")

    params = invert_spectra_to_prospect_params(
        spectra=spectra,
        wavelengths=wavelengths,
        n_fit_samples=-1,
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

    print(f"[{split_name}] Saved cached arrays to {cache_path}")

    return spectra, features, params, stages, feature_names


def normalize_array(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    x_norm = (x - mean) / std

    return x_norm.astype(np.float32), mean, std


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


def make_data_driven_param_bounds(
    params_raw: np.ndarray,
    margin_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    q_low = np.percentile(params_raw, 1, axis=0, keepdims=True)
    q_high = np.percentile(params_raw, 99, axis=0, keepdims=True)

    margin = margin_fraction * (q_high - q_low)

    lower = q_low - margin
    upper = q_high + margin

    lower[:, 0] = np.maximum(lower[:, 0], 1.0)
    lower[:, 1:] = np.maximum(lower[:, 1:], 0.0)

    return lower.astype(np.float32), upper.astype(np.float32)


def sanitize_prospect_params_data_driven(
    params: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(params, lower, upper)


def select_stage_conditions(
    features_norm: np.ndarray,
    stages: np.ndarray,
    stage_order: List[str],
    mode: str = "mean",
    seed: int = 123,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    stages_clean = np.asarray([normalize_stage_name(s) for s in stages])

    print_stage_distribution(stages_clean, "Stage distribution used for sampling")

    conditions = []
    selected_stage_names = []
    selected_indices = []

    for stage in stage_order:
        stage_clean = normalize_stage_name(stage)
        idx = np.where(stages_clean == stage_clean)[0]

        if len(idx) == 0:
            print(f"Warning: no samples found for stage '{stage_clean}'. Skipping.")
            continue

        stage_features = features_norm[idx]

        if mode == "mean":
            condition = stage_features.mean(axis=0, keepdims=True)
            distances = np.linalg.norm(stage_features - condition, axis=1)
            selected_idx = int(idx[int(np.argmin(distances))])

        elif mode == "random":
            selected_idx = int(rng.choice(idx))
            condition = features_norm[selected_idx : selected_idx + 1]

        else:
            raise ValueError("mode must be 'mean' or 'random'.")

        conditions.append(condition)
        selected_stage_names.append(stage_clean)
        selected_indices.append(selected_idx)

    if not conditions:
        raise ValueError("No valid stage conditions were selected.")

    return (
        np.concatenate(conditions, axis=0).astype(np.float32),
        selected_stage_names,
        np.asarray(selected_indices, dtype=int),
    )


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
            #SinusoidalTimeEmbedding(time_dim),
            #nn.Linear(time_dim, hidden),
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
        c_emb = self.condition_embed(condition)

        h = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(h)


class ConditionedVelocityWrapper(nn.Module):
    def __init__(
        self,
        base_model: ConditionalProspectVelocityMLP,
        condition: torch.Tensor,
    ):
        super().__init__()
        self.base_model = base_model
        self.condition = condition

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.base_model(x=x, t=t, condition=self.condition)


def make_tensor_loader(
    params_norm: np.ndarray,
    features_norm: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(params_norm).float(),
        torch.from_numpy(features_norm).float(),
    )

    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def compute_flow_metrics(
    model: ConditionalProspectVelocityMLP,
    loader: DataLoader,
    path: CondOTProbPath,
    mse: nn.Module,
    device: torch.device,
    train_mode: bool,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> Tuple[float, float, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    all_mse = []
    all_rmse = []
    all_alignment = []

    context = torch.enable_grad() if train_mode else torch.no_grad()

    with context:
        for x_1_cpu, condition_cpu in loader:
            x_1 = x_1_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            condition = condition_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=device)

            path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

            pred_velocity = model(
                x=path_sample.x_t,
                t=path_sample.t,
                condition=condition,
            )

            loss = mse(pred_velocity, path_sample.dx_t)
            rmse = torch.sqrt(loss + 1e-8)

            alignment = F.cosine_similarity(
                pred_velocity.flatten(start_dim=1),
                path_sample.dx_t.flatten(start_dim=1),
                dim=1,
            ).mean()

            if train_mode:
                if optimizer is None:
                    raise ValueError("optimizer is required when train_mode=True")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            all_mse.append(float(loss.item()))
            all_rmse.append(float(rmse.item()))
            all_alignment.append(float(alignment.item()))

    return (
        float(np.mean(all_mse)),
        float(np.mean(all_rmse)),
        float(np.mean(all_alignment)),
    )


def evaluate_conditional_flow(
    model: ConditionalProspectVelocityMLP,
    val_loader: DataLoader,
    path: CondOTProbPath,
    mse: nn.Module,
    device: torch.device,
    repeats: int = 3,
) -> Tuple[float, float, float]:
    """
    Validation is stochastic because x_0 and t are sampled.
    Average several passes to reduce noise.
    """
    metrics = []

    for _ in range(max(1, repeats)):
        metrics.append(
            compute_flow_metrics(
                model=model,
                loader=val_loader,
                path=path,
                mse=mse,
                device=device,
                train_mode=False,
            )
        )

    metrics_np = np.asarray(metrics, dtype=np.float64)
    return tuple(metrics_np.mean(axis=0).tolist())


def save_checkpoint(
    path: str,
    model: ConditionalProspectVelocityMLP,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    param_mean: np.ndarray | None,
    param_std: np.ndarray | None,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
    feature_names: List[str] | None,
    params_raw_train: np.ndarray | None,
    features_raw_train: np.ndarray | None,
    params_raw_val: np.ndarray | None = None,
    features_raw_val: np.ndarray | None = None,
    best_val_mse: float | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "global_step": global_step,
        "param_dim": model.param_dim,
        "condition_dim_in": model.condition_dim_in,
        "hidden": model.hidden,
        "depth": model.depth,
        "time_dim": model.time_dim,
        "condition_dim": model.condition_dim,
        "parameter_names": PARAMETER_NAMES,
        "feature_names": feature_names,
        "param_mean": param_mean,
        "param_std": param_std,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "params_raw_train": params_raw_train,
        "features_raw_train": features_raw_train,
        "params_raw_val": params_raw_val,
        "features_raw_val": features_raw_val,
        "best_val_mse": best_val_mse,
    }

    torch.save(payload, path)
    print(f"saved checkpoint to {path}")


def train_conditional_flow(
    train_params_norm: np.ndarray,
    train_features_norm: np.ndarray,
    val_params_norm: np.ndarray,
    val_features_norm: np.ndarray,
    cfg: TrainConfig,
    param_mean: np.ndarray,
    param_std: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_names: List[str],
    train_params_raw: np.ndarray,
    train_features_raw: np.ndarray,
    val_params_raw: np.ndarray,
    val_features_raw: np.ndarray,
) -> Tuple[ConditionalProspectVelocityMLP, torch.optim.Optimizer, Dict[str, list]]:
    device = torch.device(cfg.device)

    train_loader = make_tensor_loader(
        params_norm=train_params_norm,
        features_norm=train_features_norm,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        shuffle=True,
    )

    val_loader = make_tensor_loader(
        params_norm=val_params_norm,
        features_norm=val_features_norm,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        device=device,
        shuffle=False,
    )

    param_dim = train_params_norm.shape[1]
    condition_dim_in = train_features_norm.shape[1]

    model = ConditionalProspectVelocityMLP(
        param_dim=param_dim,
        condition_dim_in=condition_dim_in,
        hidden=cfg.hidden,
        depth=cfg.depth,
        time_dim=cfg.time_dim,
        condition_dim=cfg.condition_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    path = CondOTProbPath()
    mse = nn.MSELoss()

    history = {
        "epoch": [],
        "step": [],
        "train_mse": [],
        "train_rmse": [],
        "train_alignment": [],
        "val_mse": [],
        "val_rmse": [],
        "val_alignment": [],
    }

    best_val_mse = float("inf")
    global_step = 0
    epochs_without_improvement = 0
    global_step = 0

    os.makedirs(os.path.dirname(cfg.log_path) or ".", exist_ok=True)
    with open(cfg.log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,step,train_mse,train_rmse,train_alignment,"
            "val_mse,val_rmse,val_alignment,best_val_mse\n"
        )

    for epoch in range(1, cfg.epochs + 1):
        train_mse, train_rmse, train_alignment = compute_flow_metrics(
            model=model,
            loader=train_loader,
            path=path,
            mse=mse,
            device=device,
            train_mode=True,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
        )

        global_step += len(train_loader)

        should_validate = (
            epoch == 1
            or epoch % cfg.val_every == 0
            or epoch == cfg.epochs
        )

        if should_validate:
            val_mse, val_rmse, val_alignment = evaluate_conditional_flow(
                model=model,
                val_loader=val_loader,
                path=path,
                mse=mse,
                device=device,
                repeats=cfg.val_repeats,
            )
        else:
            val_mse = np.nan
            val_rmse = np.nan
            val_alignment = np.nan

        if should_validate:
            improved = val_mse < (best_val_mse - cfg.min_delta)
        
            if improved:
                best_val_mse = val_mse
                epochs_without_improvement = 0
        
                save_checkpoint(
                    path=cfg.best_save_path,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    param_mean=param_mean,
                    param_std=param_std,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    feature_names=feature_names,
                    params_raw_train=train_params_raw,
                    features_raw_train=train_features_raw,
                    params_raw_val=val_params_raw,
                    features_raw_val=val_features_raw,
                    best_val_mse=best_val_mse,
                )
        
                print(
                    f"New best validation MSE: {best_val_mse:.8f} "
                    f"at epoch {epoch}"
                )
        
            else:
                epochs_without_improvement += cfg.val_every
        
                print(
                    f"No validation improvement for "
                    f"{epochs_without_improvement} epochs."
                )
        
            if (
                cfg.early_stop_patience > 0
                and epochs_without_improvement >= cfg.early_stop_patience
            ):
                message = (
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_mse={best_val_mse:.8f}. "
                    f"No improvement for {epochs_without_improvement} epochs."
                )
        
                print(message)
                write_log(cfg.log_path, message)
                break

        history["epoch"].append(epoch)
        history["step"].append(global_step)
        history["train_mse"].append(train_mse)
        history["train_rmse"].append(train_rmse)
        history["train_alignment"].append(train_alignment)
        history["val_mse"].append(val_mse)
        history["val_rmse"].append(val_rmse)
        history["val_alignment"].append(val_alignment)

        if epoch == 1 or epoch % cfg.log_every == 0 or should_validate:
            message = (
                f"epoch {epoch:06d} | "
                f"step {global_step:07d} | "
                f"train_mse {train_mse:.8f} | "
                f"train_rmse {train_rmse:.8f} | "
                f"train_alignment {train_alignment:.6f} | "
                f"val_mse {val_mse:.8f} | "
                f"val_rmse {val_rmse:.8f} | "
                f"val_alignment {val_alignment:.6f} | "
                f"best_val_mse {best_val_mse:.8f}"
            )

            print(message)
            write_log(cfg.log_path, message)

            with open(cfg.log_path.replace(".txt", ".csv"), "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{global_step},"
                    f"{train_mse},{train_rmse},{train_alignment},"
                    f"{val_mse},{val_rmse},{val_alignment},{best_val_mse}\n"
                )

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
            root, ext = os.path.splitext(cfg.save_path)
            ckpt_path = f"{root}_epoch_{epoch:06d}{ext}"

            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                param_mean=param_mean,
                param_std=param_std,
                feature_mean=feature_mean,
                feature_std=feature_std,
                feature_names=feature_names,
                params_raw_train=train_params_raw,
                features_raw_train=train_features_raw,
                params_raw_val=val_params_raw,
                features_raw_val=val_features_raw,
                best_val_mse=best_val_mse,
            )

    return model, optimizer, history


@torch.no_grad()
def sample_conditional_flow(
    model: ConditionalProspectVelocityMLP,
    conditions_norm: np.ndarray,
    cfg: TrainConfig,
) -> np.ndarray:
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


def prospect_reflectance_from_params(
    params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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


def print_stats(name: str, x: np.ndarray) -> None:
    print(
        f"{name} | shape={x.shape} | "
        f"min={np.nanmin(x):.6f} | "
        f"max={np.nanmax(x):.6f} | "
        f"mean={np.nanmean(x):.6f} | "
        f"std={np.nanstd(x):.6f}"
    )


def print_parameter_table(params: np.ndarray, title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

    for idx, row in enumerate(params):
        values = ", ".join(
            f"{name}={value:.6g}"
            for name, value in zip(PARAMETER_NAMES, row)
        )
        print(f"{idx:03d}: {values}")


def plot_training_history(history: Dict[str, list]) -> None:
    plt.figure()
    plt.plot(history["step"], history["train_mse"], label="Train MSE")
    valid_idx = ~np.isnan(np.asarray(history["val_mse"]))
    plt.plot(
        np.asarray(history["step"])[valid_idx],
        np.asarray(history["val_mse"])[valid_idx],
        label="Val MSE",
    )
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("Conditional flow MSE")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["step"], history["train_alignment"], label="Train alignment")
    plt.plot(
        np.asarray(history["step"])[valid_idx],
        np.asarray(history["val_alignment"])[valid_idx],
        label="Val alignment",
    )
    plt.xlabel("Step")
    plt.ylabel("Cosine similarity")
    plt.title("Velocity alignment")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_generated_spectra_by_stage(
    wavelengths: np.ndarray,
    measured_spectra: np.ndarray,
    generated_params: np.ndarray,
    selected_indices: np.ndarray,
    selected_stage_names: List[str],
) -> None:
    plt.figure(figsize=(10, 6))

    for idx, stage_name in enumerate(selected_stage_names):
        ref_index = selected_indices[idx]

        if ref_index < measured_spectra.shape[0]:
            plt.plot(
                wavelengths,
                measured_spectra[ref_index],
                linestyle="--",
                alpha=0.6,
                label=f"Measured ref: {stage_name}",
            )

        wl_gen, rho_gen = prospect_reflectance_from_params(generated_params[idx])
        rho_interp = np.interp(wavelengths, wl_gen, rho_gen)

        plt.plot(
            wavelengths,
            rho_interp,
            linewidth=2,
            label=f"Generated: {stage_name}",
        )

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Reflectance")
    plt.title("Generated spectra conditioned on validation patch statistics by stage")
    plt.legend()
    plt.grid(True)
    plt.show()


def run(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    wavelengths = make_wavelengths(cfg)

    train_spectra, train_features, train_params, train_stages, feature_names = (
        load_or_build_arrays_for_split(
            csv_path=cfg.train_csv_path,
            cache_path=cfg.train_cache_path,
            cfg=cfg,
            split_name="train",
            n_fit_samples=cfg.n_fit_samples,
        )
    )



    val_spectra, val_features, val_params, val_stages, val_feature_names = (
        load_or_build_arrays_for_split(
            csv_path=cfg.val_csv_path,
            cache_path=cfg.val_cache_path,
            cfg=cfg,
            split_name="val",
            n_fit_samples=cfg.n_val_fit_samples,
        )
    )

    if feature_names != val_feature_names:
        raise ValueError("Train and validation feature names do not match.")

    print_stage_distribution(train_stages, "Loaded train stage distribution")
    print_stage_distribution(val_stages, "Loaded validation stage distribution")

    print_stats("train spectra", train_spectra)
    print_stats("val spectra", val_spectra)
    print_stats("train features raw", train_features)
    print_stats("val features raw", val_features)
    print_stats("train PROSPECT params raw", train_params)
    print_stats("val PROSPECT params raw", val_params)

    param_lower, param_upper = make_data_driven_param_bounds(train_params)

    train_params_norm, param_mean, param_std = normalize_array(train_params)
    train_features_norm, feature_mean, feature_std = normalize_array(train_features)

    val_params_norm = apply_normalization(val_params, param_mean, param_std)
    val_features_norm = apply_normalization(val_features, feature_mean, feature_std)

    print_stats("train params norm", train_params_norm)
    print_stats("val params norm", val_params_norm)
    print_stats("train features norm", train_features_norm)
    print_stats("val features norm", val_features_norm)

    model, optimizer, history = train_conditional_flow(
        train_params_norm=train_params_norm,
        train_features_norm=train_features_norm,
        val_params_norm=val_params_norm,
        val_features_norm=val_features_norm,
        cfg=cfg,
        param_mean=param_mean,
        param_std=param_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=feature_names,
        train_params_raw=train_params,
        train_features_raw=train_features,
        val_params_raw=val_params,
        val_features_raw=val_features,
    )

    save_checkpoint(
        path=cfg.save_path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        epoch=cfg.epochs,
        global_step=history["step"][-1],
        param_mean=param_mean,
        param_std=param_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=feature_names,
        params_raw_train=train_params,
        features_raw_train=train_features,
        params_raw_val=val_params,
        features_raw_val=val_features,
        best_val_mse=float(np.nanmin(history["val_mse"])),
    )

    stage_order = [normalize_stage_name(s) for s in cfg.stage_order.split(",")]

    conditions_norm, selected_stage_names, selected_indices = select_stage_conditions(
        features_norm=val_features_norm,
        stages=val_stages,
        stage_order=stage_order,
        mode=cfg.sample_mode,
        seed=cfg.seed,
    )

    generated_params_norm = sample_conditional_flow(
        model=model,
        conditions_norm=conditions_norm,
        cfg=cfg,
    )

    generated_params = denormalize_array(
        x_norm=generated_params_norm,
        mean=param_mean,
        std=param_std,
    )

    generated_params = sanitize_prospect_params_data_driven(
        generated_params,
        lower=param_lower,
        upper=param_upper,
    )

    print_parameter_table(
        params=generated_params,
        title="Generated PROSPECT parameters from validation patch statistics",
    )

    print("\nGenerated sample stage order:")
    for idx, stage_name in enumerate(selected_stage_names):
        print(f"{idx}: {stage_name} | validation reference index={selected_indices[idx]}")

    if cfg.plot:
        plot_training_history(history)
        plot_generated_spectra_by_stage(
            wavelengths=wavelengths,
            measured_spectra=val_spectra,
            generated_params=generated_params,
            selected_indices=selected_indices,
            selected_stage_names=selected_stage_names,
        )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_csv_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val_csv_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/usr3/Expansion/Data/EstradaDataset/Avocado/Multispectral Images/",
    )
    parser.add_argument("--species", type=str, default="Avocado")
    parser.add_argument("--stage", type=str, default="all")

    parser.add_argument("--patch_h", type=int, default=64)
    parser.add_argument("--patch_w", type=int, default=64)
    parser.add_argument("--stride_h", type=int, default=8)
    parser.add_argument("--stride_w", type=int, default=8)
    parser.add_argument("--black_thr", type=float, default=0.0)

    parser.add_argument("--wavelength_min", type=float, default=400.0)
    parser.add_argument("--wavelength_max", type=float, default=2500.0)
    parser.add_argument("--wavelength_count", type=int, default=2101)

    parser.add_argument(
        "--train_cache_path",
        type=str,
        default="cache/train_conditional_prospect_data.npz",
    )
    parser.add_argument(
        "--val_cache_path",
        type=str,
        default="cache/val_conditional_prospect_data.npz",
    )
    parser.add_argument("--force_recompute_cache", action="store_true")
    parser.add_argument("--n_fit_samples", type=int, default=-1)
    parser.add_argument("--n_val_fit_samples", type=int, default=-1)

    parser.add_argument("--pixel_scale", type=float, default=255.0)
    parser.add_argument(
        "--include_patch_count",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--condition_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--val_repeats", type=int, default=3)

    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/conditional_prospect_flow_final.pt",
    )
    parser.add_argument(
        "--best_save_path",
        type=str,
        default="checkpoints/conditional_prospect_flow_best.pt",
    )

    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--sample_mode", type=str, default="mean", choices=["mean", "random"])
    parser.add_argument("--stage_order", type=str, default="fresh,stage1,stage2,stage3,dry")

    parser.add_argument("--early_stop_patience", type=int, default=1000)
    
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
        help="Minimum validation MSE improvement required to reset early stopping.",
    )


    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/conditional_flow_training_log.txt",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
