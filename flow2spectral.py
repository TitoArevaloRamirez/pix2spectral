# train_prospect_parameter_flow_optimized.py
#
# Optimized flow-matching training over inverted PROSPECT parameters.
#
# Pipeline:
#   measured spectra
#       -> invert spectra to PROSPECT parameters
#       -> normalize parameter vectors
#       -> train flow_matching CondOTProbPath in parameter space
#       -> sample with flow_matching ODESolver
#       -> denormalize parameters
#       -> reconstruct spectra with prospect.prospectd()
#
# PROSPECT parameter vector:
#   [N_leaf, Cab, Car, Cbrown, Cw, Cm, Ant]
#
# Example:
#   python train_prospect_parameter_flow_optimized.py \
#       --csv_path ~/Code/pix2spectral/Data/Dataset_with_images.csv \
#       --species Avocado \
#       --stage fresh \
#       --params_cache cache/avocado_fresh_prospect_params.npz \
#       --batch_size 104 \
#       --epochs 5000 \
#       --lr 1e-3 \
#       --hidden 128 \
#       --depth 3 \
#       --n_generate 8 \
#       --save_path checkpoints/fm_prospect_params.pt

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

from dataset import SpectralOnlyCSVDataset
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
class TrainConfig:
    csv_path: str
    species: str | None = None
    stage: str | None = "all"

    wavelength_min: float = 400.0
    wavelength_max: float = 2500.0
    wavelength_count: int = 2101

    params_cache: str = "cache/prospect_params.npz"
    force_recompute_params: bool = False
    n_fit_samples: int = -1

    batch_size: int = 104
    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    depth: int = 3
    time_dim: int = 64
    dropout: float = 0.0

    grad_clip: float = 1.0
    log_every: int = 100
    checkpoint_every: int = 1000
    save_path: str = "checkpoints/fm_prospect_params.pt"

    n_generate: int = 8
    sampling_steps: int = 100
    plot: bool = True

    num_workers: int = 0
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
# Model
# ---------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        if not isinstance(dim, int):
            raise TypeError(f"time_dim must be int, got {type(dim)}")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Input:
            t: [B] or [B, 1] or scalar

        Output:
            emb: [B, dim]
        """
        if t.ndim == 0:
            t = t[None]

        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B], got {t.shape}")

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


class ProspectParameterVelocityMLP(nn.Module):
    """
    Velocity model v_theta(x_t, t) for normalized PROSPECT parameters.

    Input:
        x_t: [B, 7]
        t:   [B] or [B, 1]

    Output:
        velocity: [B, 7]
    """

    def __init__(
        self,
        param_dim: int = 7,
        hidden: int = 128,
        depth: int = 3,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.param_dim = param_dim
        self.hidden = hidden
        self.depth = depth
        self.time_dim = time_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        layers = []
        dim = param_dim + hidden

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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be [B, D], got {x.shape}")

        if t.ndim == 0:
            t = t.expand(x.shape[0])

        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {t.shape}")

        if t.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch mismatch: x has {x.shape[0]}, t has {t.shape[0]}"
            )

        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)

        return self.net(h)


# ---------------------------------------------------------------------
# Data and PROSPECT inversion
# ---------------------------------------------------------------------

def make_wavelengths(cfg: TrainConfig) -> np.ndarray:
    return np.linspace(
        cfg.wavelength_min,
        cfg.wavelength_max,
        cfg.wavelength_count,
        dtype=np.float64,
    )


def load_spectral_dataset(cfg: TrainConfig) -> np.ndarray:
    dataset = SpectralOnlyCSVDataset(
        csv_path=os.path.expanduser(cfg.csv_path),
        species=cfg.species,
        stage=cfg.stage,
    )

    if not hasattr(dataset, "spectral_np"):
        raise RuntimeError("SpectralOnlyCSVDataset must expose dataset.spectral_np.")

    spectra = dataset.spectral_np.astype(np.float32)

    print(f"Dataset size: {len(dataset)}")
    print(f"Spectra shape: {spectra.shape}")
    print(
        "Spectra stats | "
        f"min={spectra.min():.6f}, "
        f"max={spectra.max():.6f}, "
        f"mean={spectra.mean():.6f}, "
        f"std={spectra.std():.6f}"
    )

    return spectra


def invert_spectra_to_params(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    n_fit_samples: int = -1,
) -> np.ndarray:
    """
    Invert spectra to PROSPECT parameters.

    Returns:
        params_raw: [N, 7]
    """
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be [N, M], got {spectra.shape}")

    if spectra.shape[1] != wavelengths.shape[0]:
        raise ValueError(
            f"spectra length {spectra.shape[1]} does not match "
            f"wavelength count {wavelengths.shape[0]}"
        )

    if n_fit_samples is None or n_fit_samples < 0:
        n_fit_samples = spectra.shape[0]

    n_fit_samples = min(n_fit_samples, spectra.shape[0])

    params = []

    for i in range(n_fit_samples):
        rho = spectra[i].astype(np.float64)

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
            f"inverted {i + 1:04d}/{n_fit_samples:04d} | "
            f"success={result.success} | "
            f"cost={float(result.fun):.8g} | "
            f"params={np.asarray(row)}"
        )

    return np.asarray(params, dtype=np.float32)


def load_or_compute_params(
    cfg: TrainConfig,
    spectra: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    cache_path = os.path.expanduser(cfg.params_cache)

    if os.path.exists(cache_path) and not cfg.force_recompute_params:
        data = np.load(cache_path)
        params_raw = data["params_raw"].astype(np.float32)
        print(f"Loaded cached PROSPECT parameters from {cache_path}")
        print(f"Cached params shape: {params_raw.shape}")
        return params_raw

    params_raw = invert_spectra_to_params(
        spectra=spectra,
        wavelengths=wavelengths,
        n_fit_samples=cfg.n_fit_samples,
    )

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(
        cache_path,
        params_raw=params_raw,
        parameter_names=np.asarray(PARAMETER_NAMES),
        wavelengths=wavelengths,
    )

    print(f"Saved PROSPECT parameter cache to {cache_path}")
    return params_raw


def normalize_params(params_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = params_raw.mean(axis=0, keepdims=True).astype(np.float32)
    std = params_raw.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    params_norm = (params_raw - mean) / std

    return params_norm.astype(np.float32), mean, std


def denormalize_params(
    params_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return params_norm * std + mean


def sanitize_params(params: np.ndarray) -> np.ndarray:
    """
    Clamp generated PROSPECT parameters to safe physical ranges.
    """
    p = params.copy()

    p[:, 0] = np.clip(p[:, 0], 1.0, 5.0)       # N_leaf
    p[:, 1] = np.clip(p[:, 1], 0.0, 150.0)     # Cab
    p[:, 2] = np.clip(p[:, 2], 0.0, 50.0)      # Car
    p[:, 3] = np.clip(p[:, 3], 0.0, 2.0)       # Cbrown
    p[:, 4] = np.clip(p[:, 4], 1e-6, 0.20)     # Cw
    p[:, 5] = np.clip(p[:, 5], 1e-6, 0.10)     # Cm
    p[:, 6] = np.clip(p[:, 6], 0.0, 50.0)      # Ant

    return p


def reflectance_from_params(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one [7] PROSPECT parameter vector to reflectance.
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


# ---------------------------------------------------------------------
# Training with flow_matching library
# ---------------------------------------------------------------------

def train_flow_matching_model(
    params_norm: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[ProspectParameterVelocityMLP, torch.optim.Optimizer, dict]:
    """
    Train with flow_matching.path.CondOTProbPath.
    """
    device = torch.device(cfg.device)

    x_data = torch.from_numpy(params_norm).float()
    dataset = TensorDataset(x_data)

    loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    param_dim = params_norm.shape[1]

    model = ProspectParameterVelocityMLP(
        param_dim=param_dim,
        hidden=cfg.hidden,
        depth=cfg.depth,
        time_dim=cfg.time_dim,
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
        "step": [],
        "loss": [],
        "rmse": [],
        "velocity_alignment": [],
    }

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        epoch_losses = []
        epoch_rmses = []
        epoch_alignments = []

        for (x_1_cpu,) in loader:
            x_1 = x_1_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=device)

            path_sample = path.sample(
                x_0=x_0,
                x_1=x_1,
                t=t,
            )

            pred_velocity = model(path_sample.x_t, path_sample.t)

            loss = mse(pred_velocity, path_sample.dx_t)
            rmse = torch.sqrt(loss + 1e-8)

            with torch.no_grad():
                velocity_alignment = F.cosine_similarity(
                    pred_velocity.flatten(start_dim=1),
                    path_sample.dx_t.flatten(start_dim=1),
                    dim=1,
                ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.grad_clip,
                )

            optimizer.step()

            global_step += 1

            epoch_losses.append(float(loss.item()))
            epoch_rmses.append(float(rmse.item()))
            epoch_alignments.append(float(velocity_alignment.item()))

        mean_loss = float(np.mean(epoch_losses))
        mean_rmse = float(np.mean(epoch_rmses))
        mean_alignment = float(np.mean(epoch_alignments))

        history["step"].append(global_step)
        history["loss"].append(mean_loss)
        history["rmse"].append(mean_rmse)
        history["velocity_alignment"].append(mean_alignment)

        if epoch == 1 or epoch % cfg.log_every == 0:
            print(
                f"epoch {epoch:06d} | "
                f"step {global_step:07d} | "
                f"mse {mean_loss:.8f} | "
                f"rmse {mean_rmse:.8f} | "
                f"alignment {mean_alignment:.6f}"
            )

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
            root, ext = os.path.splitext(cfg.save_path)
            checkpoint_path = f"{root}_epoch_{epoch:06d}{ext}"
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                param_mean=None,
                param_std=None,
                params_raw=None,
                params_norm=None,
            )

    return model, optimizer, history


# ---------------------------------------------------------------------
# Sampling with flow_matching ODESolver
# ---------------------------------------------------------------------

@torch.no_grad()
def sample_with_ode_solver(
    model: ProspectParameterVelocityMLP,
    n_samples: int,
    sampling_steps: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate normalized PROSPECT parameter samples using flow_matching.solver.ODESolver.
    """
    model.eval()

    solver = ODESolver(velocity_model=model)

    x_init = torch.randn(
        n_samples,
        model.param_dim,
        device=device,
        dtype=torch.float32,
    )

    time_grid = torch.linspace(
        0.0,
        1.0,
        sampling_steps + 1,
        device=device,
        dtype=torch.float32,
    )

    generated = solver.sample(
        x_init=x_init,
        step_size=1.0 / sampling_steps,
        method="euler",
        time_grid=time_grid,
    )

    return generated.detach().cpu().numpy()


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: ProspectParameterVelocityMLP,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    param_mean: np.ndarray | None,
    param_std: np.ndarray | None,
    params_raw: np.ndarray | None,
    params_norm: np.ndarray | None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "global_step": global_step,
        "param_dim": model.param_dim,
        "hidden": model.hidden,
        "depth": model.depth,
        "time_dim": model.time_dim,
        "parameter_names": PARAMETER_NAMES,
        "param_mean": param_mean,
        "param_std": param_std,
        "params_raw": params_raw,
        "params_norm": params_norm,
    }

    torch.save(payload, path)
    print(f"saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    device: torch.device,
) -> Tuple[ProspectParameterVelocityMLP, dict]:
    checkpoint = torch.load(path, map_location=device)

    model = ProspectParameterVelocityMLP(
        param_dim=int(checkpoint["param_dim"]),
        hidden=int(checkpoint["hidden"]),
        depth=int(checkpoint["depth"]),
        time_dim=int(checkpoint["time_dim"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------------------
# Plotting / reporting
# ---------------------------------------------------------------------

def print_parameter_table(params: np.ndarray, title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

    for i, row in enumerate(params):
        values = ", ".join(
            f"{name}={value:.6g}"
            for name, value in zip(PARAMETER_NAMES, row)
        )
        print(f"{i:03d}: {values}")


def plot_training_history(history: dict) -> None:
    plt.figure()
    plt.plot(history["step"], history["loss"], label="MSE")
    plt.plot(history["step"], history["rmse"], label="RMSE")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Flow-matching training")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["step"], history["velocity_alignment"])
    plt.xlabel("Step")
    plt.ylabel("Cosine similarity")
    plt.title("Velocity alignment")
    plt.grid(True)
    plt.show()


def plot_generated_spectra(
    wavelengths: np.ndarray,
    measured_spectra: np.ndarray,
    param_mean: np.ndarray,
    generated_params: np.ndarray,
) -> None:
    plt.figure()

    if measured_spectra is not None and len(measured_spectra) > 0:
        plt.plot(
            wavelengths,
            measured_spectra[0],
            label="Measured example",
            linewidth=2,
        )

    mean_params = sanitize_params(param_mean.copy())[0]
    wl_mean, rho_mean = reflectance_from_params(mean_params)
    rho_mean_interp = np.interp(wavelengths, wl_mean, rho_mean)

    plt.plot(
        wavelengths,
        rho_mean_interp,
        label="PROSPECT from mean params",
        linewidth=2,
    )

    for i in range(generated_params.shape[0]):
        wl_gen, rho_gen = reflectance_from_params(generated_params[i])
        rho_gen_interp = np.interp(wavelengths, wl_gen, rho_gen)

        plt.plot(
            wavelengths,
            rho_gen_interp,
            label=f"Flow sample {i + 1}",
            alpha=0.8,
        )

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Reflectance")
    plt.title("Generated spectra from flow over PROSPECT parameters")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    wavelengths = make_wavelengths(cfg)
    spectra = load_spectral_dataset(cfg)

    params_raw = load_or_compute_params(
        cfg=cfg,
        spectra=spectra,
        wavelengths=wavelengths,
    )

    print_parameter_table(params_raw, "Inverted PROSPECT parameters")

    params_norm, param_mean, param_std = normalize_params(params_raw)

    print("\nParameter mean:")
    print(dict(zip(PARAMETER_NAMES, param_mean.squeeze().tolist())))

    print("\nParameter std:")
    print(dict(zip(PARAMETER_NAMES, param_std.squeeze().tolist())))

    model, optimizer, history = train_flow_matching_model(
        params_norm=params_norm,
        cfg=cfg,
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
        params_raw=params_raw,
        params_norm=params_norm,
    )

    generated_norm = sample_with_ode_solver(
        model=model,
        n_samples=cfg.n_generate,
        sampling_steps=cfg.sampling_steps,
        device=device,
    )

    generated_params = denormalize_params(
        params_norm=generated_norm,
        mean=param_mean,
        std=param_std,
    )

    generated_params = sanitize_params(generated_params)

    print_parameter_table(generated_params, "Generated PROSPECT parameters")

    if cfg.plot:
        plot_training_history(history)
        plot_generated_spectra(
            wavelengths=wavelengths,
            measured_spectra=spectra,
            param_mean=param_mean,
            generated_params=generated_params,
        )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="~/Code/pix2spectral/Data/Dataset_with_images.csv",
    )
    parser.add_argument("--species", type=str, default="Avocado")
    parser.add_argument("--stage", type=str, default="fresh")

    parser.add_argument("--wavelength_min", type=float, default=400.0)
    parser.add_argument("--wavelength_max", type=float, default=2500.0)
    parser.add_argument("--wavelength_count", type=int, default=2101)

    parser.add_argument("--params_cache", type=str, default="cache/prospect_params.npz")
    parser.add_argument("--force_recompute_params", action="store_true")
    parser.add_argument("--n_fit_samples", type=int, default=-1)

    parser.add_argument("--batch_size", type=int, default=104)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="checkpoints/fm_prospect_params.pt")

    parser.add_argument("--n_generate", type=int, default=8)
    parser.add_argument("--sampling_steps", type=int, default=100)

    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--num_workers", type=int, default=0)

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
