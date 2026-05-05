#!/usr/bin/env python3
"""
train_flow_matching_spectral.py

Conditional Flow Matching training script for multispectral patch bags -> spectral signatures.

Expected local files:
    dataset.py      # your uploaded dataset file
    conditioner.py  # MultiSpectralConditioner from the previous step

Install:
    pip install flow-matching torch pandas pillow numpy

Example:
    python train_flow_matching_spectral.py \
        --csv_path /path/to/Dataset_with_images.csv \
        --root_dir "/path/to/Multispectral Images" \
        --species Avocado \
        --stage all \
        --epochs 100 \
        --batch_size 8 \
        --spectrum_dim 151 \
        --save_path checkpoints/fm_spectral.pt

Notes:
    - The flow is trained in spectral-signature space.
    - The multispectral patch bags are used only as conditioning information.
    - Variable numbers of patches per band are handled by MultiSpectralConditioner.
"""

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from dataset import MultiSpectralCSVPatchDataset, patch_collate_fn
from conditioner import MultiSpectralConditioner


class SpectralVelocityMLP(nn.Module):
    """
    Conditional velocity field v_theta(t, x_t, c).

    Args:
        spectrum_dim: dimensionality L of the spectral signature.
        condition_dim: dimensionality of conditioner output.
        hidden_dim: MLP width.
        depth: number of hidden layers.

    Input:
        x_t: Tensor [B, L]
        t: Tensor [B] or scalar Tensor
        condition: Tensor [B, condition_dim]

    Output:
        velocity: Tensor [B, L]
    """

    def __init__(
        self,
        spectrum_dim: int,
        condition_dim: int = 256,
        hidden_dim: int = 512,
        depth: int = 4,
    ):
        super().__init__()

        in_dim = spectrum_dim + condition_dim + 1
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, spectrum_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        if t.ndim == 1:
            t = t[:, None]

        if condition.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"condition batch size {condition.shape[0]} does not match x_t batch size {x_t.shape[0]}"
            )

        inp = torch.cat([x_t, t.to(x_t.dtype), condition], dim=-1)
        return self.net(inp)


class ConditionalVelocityWrapper(ModelWrapper):
    """
    Adapter for flow_matching.solver.ODESolver.

    ODESolver calls:
        velocity_model(x=x, t=t, **model_extras)

    We forward condition as:
        solver.sample(..., condition=condition)
    """

    def __init__(self, velocity_model: nn.Module):
        super().__init__(velocity_model)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        condition = extras["condition"]
        return self.model(x_t=x, t=t, condition=condition)


@dataclass
class TrainConfig:
    csv_path: str
    root_dir: str | None = None
    species: str | None = None
    stage: str | None = "all"

    patch_h: int = 32
    patch_w: int = 32
    stride_h: int | None = 16
    stride_w: int | None = 16
    black_thr: float = 0.0

    spectrum_dim: int = 151
    batch_size: int = 8
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0

    emb_dim: int = 128
    condition_dim: int = 256
    hidden_dim: int = 512
    depth: int = 4
    pooling: str = "attention"

    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 5
    plot_every: int = 5
    eval_batch_size: int = 4
    sample_steps: int = 100
    n_plot_samples: int = 4
    save_path: str = "checkpoints/fm_spectral.pt"
    plot_dir: str = "training_plots"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(cfg: TrainConfig) -> DataLoader:
    dataset = MultiSpectralCSVPatchDataset(
        csv_path=os.path.expanduser(cfg.csv_path),
        root_dir=cfg.root_dir,
        species=cfg.species,
        stage=cfg.stage,
        patch_h=cfg.patch_h,
        patch_w=cfg.patch_w,
        stride_h=cfg.stride_h,
        stride_w=cfg.stride_w,
        black_thr=cfg.black_thr,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=patch_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def train(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device)
    print(device)

    loader = build_dataloader(cfg)
    fixed_batch_bands, fixed_spectra = get_fixed_eval_batch(loader, cfg, device)
    history = {"epoch": [], "train_loss": [], "eval_mse": []}

    conditioner = MultiSpectralConditioner(
        emb_dim=cfg.emb_dim,
        condition_dim=cfg.condition_dim,
        pooling=cfg.pooling,
    ).to(device)

    velocity_model = SpectralVelocityMLP(
        spectrum_dim=cfg.spectrum_dim,
        condition_dim=cfg.condition_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
    ).to(device)

    path = CondOTProbPath()

    params = list(conditioner.parameters()) + list(velocity_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        conditioner.train()
        velocity_model.train()

        running_loss = 0.0
        running_batches = 0

        for batch_bands, spectra in loader:
            spectra = spectra.to(device, non_blocking=True).float()

            if spectra.shape[-1] != cfg.spectrum_dim:
                raise ValueError(
                    f"Expected spectra dim {cfg.spectrum_dim}, got {spectra.shape[-1]}. "
                    "Set --spectrum_dim to match your CSV spectral vector length."
                )

            condition = conditioner(batch_bands)  # [B, condition_dim]

            x_1 = spectra  # target spectrum [B, L]
            x_0 = torch.randn_like(x_1)  # source noise [B, L]
            t = torch.rand(x_1.shape[0], device=device)  # official API expects [B]

            path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

            pred_velocity = velocity_model(
                x_t=path_sample.x_t,
                t=path_sample.t,
                condition=condition,
            )

            loss = F.mse_loss(pred_velocity, path_sample.dx_t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)

            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            running_batches += 1

            if global_step % cfg.log_every == 0:
                print(
                    f"epoch={epoch:04d} step={global_step:06d} loss={loss.item():.6f}"
                )

        epoch_loss = running_loss / max(1, running_batches)
        eval_mse = evaluate_fixed_batch(
            batch_bands=fixed_batch_bands,
            spectra=fixed_spectra,
            conditioner=conditioner,
            velocity_model=velocity_model,
            spectrum_dim=cfg.spectrum_dim,
            n_steps=cfg.sample_steps,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["eval_mse"].append(eval_mse)

        print(
            f"[epoch {epoch:04d}] mean_loss={epoch_loss:.6f} fixed_batch_mse={eval_mse:.6f}"
        )

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                conditioner=conditioner,
                velocity_model=velocity_model,
                optimizer=optimizer,
                history=history,
                path=checkpoint_path_for_epoch(cfg.save_path, epoch),
            )

            save_checkpoint(
                cfg=cfg,
                epoch=epoch,
                conditioner=conditioner,
                velocity_model=velocity_model,
                optimizer=optimizer,
                history=history,
                path=cfg.save_path,
            )

        if epoch % cfg.plot_every == 0 or epoch == 1 or epoch == cfg.epochs:
            plot_training_curves(history, cfg.plot_dir)
            plot_fixed_batch_reconstruction(
                batch_bands=fixed_batch_bands,
                spectra=fixed_spectra,
                conditioner=conditioner,
                velocity_model=velocity_model,
                spectrum_dim=cfg.spectrum_dim,
                out_path=os.path.join(cfg.plot_dir, f"spectra_epoch_{epoch:04d}.png"),
                n_steps=cfg.sample_steps,
                n_plot_samples=cfg.n_plot_samples,
            )


@torch.no_grad()
def reconstruct_spectrum(
    batch_bands: dict,
    conditioner: MultiSpectralConditioner,
    velocity_model: SpectralVelocityMLP,
    spectrum_dim: int,
    n_steps: int = 100,
    method: str = "euler",
) -> torch.Tensor:
    """
    Generate/reconstruct spectra conditioned on multispectral patch bags.

    Args:
        batch_bands: output from patch_collate_fn.
        conditioner: trained conditioner.
        velocity_model: trained conditional velocity model.
        spectrum_dim: spectral dimensionality L.
        n_steps: number of solver steps.
        method: ODE method supported by torchdiffeq, e.g. "euler", "midpoint", "dopri5".

    Returns:
        generated spectra: Tensor [B, L]
    """
    device = next(velocity_model.parameters()).device
    conditioner.eval()
    velocity_model.eval()

    condition = conditioner(batch_bands)
    batch_size = condition.shape[0]
    x_init = torch.randn(batch_size, spectrum_dim, device=device)

    wrapper = ConditionalVelocityWrapper(velocity_model)
    solver = ODESolver(velocity_model=wrapper)

    time_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    if method == "euler":
        step_size = 1.0 / n_steps
    else:
        step_size = None

    x_hat = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=method,
        time_grid=time_grid,
        return_intermediates=False,
        condition=condition,
    )

    return x_hat


def move_batch_bands_to_cpu(batch_bands: dict) -> dict:
    """Keep a fixed batch detached from the training loop for repeated plotting."""
    return {
        band: [x.detach().cpu() for x in tensors]
        for band, tensors in batch_bands.items()
    }


def get_fixed_eval_batch(loader: DataLoader, cfg: TrainConfig, device: torch.device):
    """
    Takes one fixed batch at the beginning of training.
    This makes plots comparable across epochs.
    """
    batch_bands, spectra = next(iter(loader))

    # Optionally keep only a few samples to make plotting/sampling cheap.
    keep = min(cfg.eval_batch_size, spectra.shape[0])
    batch_bands = {band: tensors[:keep] for band, tensors in batch_bands.items()}
    spectra = spectra[:keep].to(device).float()

    return move_batch_bands_to_cpu(batch_bands), spectra


@torch.no_grad()
def evaluate_fixed_batch(
    batch_bands: dict,
    spectra: torch.Tensor,
    conditioner: MultiSpectralConditioner,
    velocity_model: SpectralVelocityMLP,
    spectrum_dim: int,
    n_steps: int = 100,
) -> float:
    pred = reconstruct_spectrum(
        batch_bands=batch_bands,
        conditioner=conditioner,
        velocity_model=velocity_model,
        spectrum_dim=spectrum_dim,
        n_steps=n_steps,
        method="euler",
    )
    return F.mse_loss(pred, spectra).item()


def plot_training_curves(history: dict, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    epochs = history["epoch"]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], label="train flow loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["eval_mse"], label="fixed-batch reconstruction MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Fixed-batch reconstruction error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fixed_batch_mse.png"), dpi=150)
    plt.close()


@torch.no_grad()
def plot_fixed_batch_reconstruction(
    batch_bands: dict,
    spectra: torch.Tensor,
    conditioner: MultiSpectralConditioner,
    velocity_model: SpectralVelocityMLP,
    spectrum_dim: int,
    out_path: str,
    n_steps: int = 100,
    n_plot_samples: int = 4,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pred = reconstruct_spectrum(
        batch_bands=batch_bands,
        conditioner=conditioner,
        velocity_model=velocity_model,
        spectrum_dim=spectrum_dim,
        n_steps=n_steps,
        method="euler",
    )

    y_true = spectra.detach().cpu()
    y_pred = pred.detach().cpu()

    n = min(n_plot_samples, y_true.shape[0])
    x_axis = torch.arange(spectrum_dim).numpy()

    plt.figure(figsize=(9, 5))
    for i in range(n):
        plt.plot(x_axis, y_true[i].numpy(), linestyle="-", label=f"true {i}")
        plt.plot(x_axis, y_pred[i].numpy(), linestyle="--", label=f"pred {i}")

    plt.xlabel("Spectral index")
    plt.ylabel("Spectral value")
    plt.title("Predicted vs ground-truth spectra on fixed batch")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def checkpoint_path_for_epoch(save_path: str, epoch: int) -> str:
    base = Path(save_path)
    return str(base.with_name(f"{base.stem}_epoch_{epoch:04d}{base.suffix}"))


def save_checkpoint(
    cfg: TrainConfig,
    epoch: int,
    conditioner: MultiSpectralConditioner,
    velocity_model: SpectralVelocityMLP,
    optimizer: torch.optim.Optimizer,
    history: dict,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "config": asdict(cfg),
            "conditioner": conditioner.state_dict(),
            "velocity_model": velocity_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        },
        path,
    )
    print(f"saved checkpoint: {path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="~/Code/pix2spectral/Data/Dataset_with_images.csv",
    )
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

    parser.add_argument("--spectrum_dim", type=int, default=2101)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--pooling", type=str, default="attention", choices=["attention", "mean"]
    )

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--n_plot_samples", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="checkpoints/fm_spectral.pt")
    parser.add_argument("--plot_dir", type=str, default="training_plots")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
