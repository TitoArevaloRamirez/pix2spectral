# train_leaf_flow_matching.py
#
# Continuous flow matching for 1D leaf spectral signatures.
#
# Uses:
#   dataset.py -> MultiSpectralCSVPatchDataset, patch_collate_fn
#
# Current model:
#   v_theta(x_t, t)
#
# Important:
#   The multispectral image patches are loaded by the dataloader,
#   but are NOT used by this model yet. The model trains only on spec: [B, L].
#
# Metrics:
#   loss = MSE(predicted velocity, target velocity)
#   velocity_alignment = cosine similarity between predicted and target velocity
#
# Example:
#   python train_leaf_flow_matching.py \
#       --csv_path ~/Code/pix2spectral/Data/Dataset_with_images.csv \
#       --root_dir "/media/usr3/Expansion/Data/EstradaDataset/Avocado/Multispectral Images/" \
#       --species Avocado \
#       --stage fresh \
#       --batch_size 30 \
#       --epochs 3000 \
#       --hidden 128 \
#       --depth 2 \
#       --checkpoint_every 10 \
#       --metrics_path logs/fm_spectral_metrics.csv \
#       --save_path checkpoints/fm_spectral.pt

import argparse
import csv
import os
from dataclasses import dataclass, asdict
from torch import nn, Tensor

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flow_matching.path import CondOTProbPath


from utils import invert_prospect_parameters, invert_prospectd_interpolated

from pypro4sail import prospect

# from dataset import MultiSpectralCSVPatchDataset, patch_collate_fn
from dataset import SpectralOnlyCSVDataset


# -----------------------------
# Configuration
# -----------------------------


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

    batch_size: int = 30
    epochs: int = 3000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0

    hidden: int = 128
    depth: int = 2
    time_dim: int = 64
    dropout: float = 0.0

    grad_clip: float = 1.0
    log_every: int = 10

    checkpoint_every: int = 10
    save_path: str = "checkpoints/fm_spectral.pt"

    metrics_path: str = "logs/fm_spectral_metrics.csv"
    metrics_every: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CSVMetricLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        file_exists = os.path.exists(path)

        self.file = open(path, mode="a", newline="")
        self.writer = csv.writer(self.file)

        if not file_exists:
            self.writer.writerow(
                [
                    "epoch",
                    "step",
                    "loss",
                    "velocity_alignment",
                ]
            )
            self.file.flush()

    def log(
        self,
        epoch: int,
        step: int,
        loss: float,
        velocity_alignment: float,
    ) -> None:
        self.writer.writerow(
            [
                epoch,
                step,
                loss,
                velocity_alignment,
            ]
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def save_model(
    path,
    model,
    optimizer,
    step,
    spectrum_length,
    hidden,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "spectrum_length": spectrum_length,
            "hidden": hidden,
        },
        path,
    )

    print(f"Saved model to {path}")


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    ema,
    optimizer: torch.optim.Optimizer,
    spectrum_length: int,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "spectrum_length": spectrum_length,
            "config": asdict(cfg),
            "epoch": epoch,
            "global_step": global_step,
        },
        save_path,
    )


def build_dataset(cfg):
    return SpectralOnlyCSVDataset(
        csv_path=os.path.expanduser(cfg.csv_path),
        species=cfg.species,
        stage=cfg.stage,
    )


def build_dataloader(dataset, cfg):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # def build_dataset(cfg: TrainConfig) -> MultiSpectralCSVPatchDataset:
    #    return MultiSpectralCSVPatchDataset(
    #        csv_path=os.path.expanduser(cfg.csv_path),
    #        root_dir=cfg.root_dir,
    #        species=cfg.species,
    #        stage=cfg.stage,
    #        patch_h=cfg.patch_h,
    #        patch_w=cfg.patch_w,
    #        stride_h=cfg.stride_h,
    #        stride_w=cfg.stride_w,
    #        black_thr=cfg.black_thr,
    #    )
    #
    #
    # def build_dataloader(
    #    dataset: MultiSpectralCSVPatchDataset,
    #    cfg: TrainConfig,
    # ) -> DataLoader:
    #    return DataLoader(
    #        dataset,
    #        batch_size=cfg.batch_size,
    #        shuffle=True,
    #        num_workers=cfg.num_workers,
    #        collate_fn=patch_collate_fn,
    #        pin_memory=torch.cuda.is_available(),
    #        drop_last=False,
    #    )


# -----------------------------
# Model
# -----------------------------
#
class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

        return x_t + (t_end - t_start) * self(
            t=t_start + (t_end - t_start) / 2,
            x_t=x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2,
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        if not isinstance(dim, int):
            raise TypeError(f"time_dim must be int, got {type(dim)}")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Input:
            t: [B] or scalar

        Output:
            emb: [B, dim]
        """
        if t.ndim == 0:
            t = t[None]

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


class SpectralVelocityMLP(nn.Module):
    """
    Velocity model v_theta(x_t, t).

    Input:
        x_t: [B, L]
        t:   [B]

    Output:
        velocity: [B, L]
    """

    def __init__(
        self,
        spectrum_length: int,
        hidden: int = 128,
        depth: int = 2,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        layers = []
        dim = spectrum_length + hidden

        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(dim, hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden

        layers.append(nn.Linear(hidden, spectrum_length))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)

        return self.net(h)


# -----------------------------
# EMA helper
# -----------------------------


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            self.shadow[name].mul_(self.decay).add_(
                p.detach(),
                alpha=1.0 - self.decay,
            )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[name])


# -----------------------------
# Training
# -----------------------------


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    dataset = build_dataset(cfg)
    loader = build_dataloader(dataset, cfg)

    if not hasattr(dataset, "spectral_np"):
        raise RuntimeError(
            "Dataset must expose dataset.spectral_np to infer spectrum length."
        )

    #spectrum_length = dataset.spectral_np.shape[1]
    spectrum_length = 7

    print(f"Dataset size: {len(dataset)}")
    print(f"Spectrum length: {spectrum_length}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Steps per epoch: {len(loader)}")

    # Useful scale check
    spectra = dataset.spectral_np
    print(
        "Spectrum stats | "
        f"min={spectra.min():.6f}, "
        f"max={spectra.max():.6f}, "
        f"mean={spectra.mean():.6f}, "
        f"std={spectra.std():.6f}"
    )

    flow = Flow(dim=spectrum_length, h=64).to(device)

    # model = SpectralVelocityMLP(
    #    spectrum_length=spectrum_length,
    #    hidden=cfg.hidden,
    #    depth=cfg.depth,
    #    time_dim=cfg.time_dim,
    #    dropout=cfg.dropout,
    # ).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=1e-2,  # cfg.lr,
    )
    loss_fn = nn.MSELoss()


    wls = np.linspace(400, 2500, 2101)
    params = []
    rho = []
    for rho in dataset.spectral_np[0:10,:]: 
        best_params, result = invert_prospect_parameters(
            rho_leaf=rho,  # shape [M]
            wls=wls,  # shape [M]
            #options={
            #        "maxiter": 5000,
            #        "maxfun": 15000,
            #        "ftol": 1e-16,
            #        "gtol": 1e-10,
            #    },
        )
        params.append([best_params["N_leaf"],
            best_params["Cab"],
            best_params["Car"],
            best_params["Cbrown"],
            best_params["Cw"],
            best_params["Cm"],
            best_params["Ant"]])

    params = np.asarray(params)
    p_mean = params.mean(axis=0, keepdims=True).astype(np.float32)
    p_std = params.std(axis=0, keepdims=True).astype(np.float32) + 1e-6

    params = (params - p_mean)/(p_std)


    print(p_mean)
    print(p_std)

        # self.mean = self.spectral_np.mean(axis=0, keepdims=True).astype(np.float32)
        # self.std = self.spectral_np.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        # self.spectral_np = (self.spectral_np - self.mean) / self.std

    for i in range(10000):
        # wavelength = np.linspace(350, 2500, 2101)

        x_1 = Tensor(np.asarray(params)).to(
            device=device, dtype=torch.float32
        )
        # print(x_1.shape)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], 1, device=device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        optimizer.zero_grad()
        loss_fn(flow(t=t, x_t=x_t), dx_t).backward()
        optimizer.step()
        print(i)

    save_model(
        path="toy_spectral_flow_final.pt",
        model=flow,
        optimizer=loss_fn,
        step=500,
        spectrum_length=spectrum_length,
        hidden=64,
    )

    checkpoint = torch.load("toy_spectral_flow_final.pt", map_location=device)

    flow = Flow(dim=checkpoint["spectrum_length"], h=checkpoint["hidden"]).to(device)
    flow.load_state_dict(checkpoint["model_state_dict"])
    flow.eval()

    x = torch.randn((1, 7), device=device)
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)

    axes[0].plot(x.cpu().detach()[0, :], marker=".")
    axes[0].set_title(f"t = {time_steps[0]:.2f}")

    for i in range(n_steps):
        x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
        axes[i + 1].plot(x.cpu().detach()[0, :], marker=".")
        axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")

    plt.tight_layout()
    plt.show()

    p_hat = x.cpu().detach().numpy()

    p_hat =  (p_hat*p_std) + p_mean

    print(p_hat)
    p_hat[p_hat < 0] = 0
    
    wl_model, rho_flow, tau_model = prospect.prospectd(
        p_hat[0, 0],
        p_hat[0,1],
        p_hat[0,2],
        p_hat[0,3],
        p_hat[0,4],
        p_hat[0,5],
        p_hat[0,6],
    )
    wl_model, rho_fit, tau_model = prospect.prospectd(
        p_mean[0, 0],
        p_mean[0,1],
        p_mean[0,2],
        p_mean[0,3],
        p_mean[0,4],
        p_mean[0,5],
        p_mean[0,6],
    )
    plt.figure()
    plt.plot(wls, rho, label="Measured")
    plt.plot(wls, rho_fit, label="PROSPECT-D fitted")
    plt.plot(wls, rho_flow, label="Flow matching")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True)
    plt.show()


    # path = CondOTProbPath()
    # ema = EMA(model, decay=0.999)
    # mse = nn.MSELoss()

    # metric_logger = CSVMetricLogger(cfg.metrics_path)

    # global_step = 0

    # try:
    #    for epoch in range(1, cfg.epochs + 1):
    #        model.train()
    #        epoch_losses = []
    #        epoch_alignments = []

    #        fixed_spec = None

    #        # for _, spec in loader:
    #        for spec in loader:
    #            if fixed_spec is None:
    #                fixed_spec = spec[:1].repeat(cfg.batch_size, 1)
    #                # print(spec.shape)
    #                # print(spec[:1].shape)
    #                # print(spec[:1].numpy().shape)
    #                # plt.figure()
    #                # plt.plot(spec[:1].numpy()[0, :])
    #                # plt.show()
    #                # os.exit()

    #            # x_1 = fixed_spec.to(device=device, dtype=torch.float32)
    #            x_1 = fixed_spec.to(
    #                device=device, dtype=torch.float32, non_blocking=True
    #            )

    #            # for _, spec in loader:
    #            #    x_1 = spec.to(device=device, dtype=torch.float32)

    #            if x_1.ndim != 2:
    #                raise RuntimeError(f"Expected spec shape [B, L], got {x_1.shape}")

    #            x_0 = torch.randn_like(x_1)
    #            # t = torch.rand(x_1.shape[0], device=device)
    #            t = torch.rand(x_1.shape[0], device=device) * 0.95

    #            path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

    #            pred_velocity = model(path_sample.x_t, path_sample.t)
    #            loss = mse(pred_velocity, path_sample.dx_t)

    #            with torch.no_grad():
    #                velocity_alignment = F.cosine_similarity(
    #                    pred_velocity.flatten(start_dim=1),
    #                    path_sample.dx_t.flatten(start_dim=1),
    #                    dim=1,
    #                ).mean()

    #            optimizer.zero_grad(set_to_none=True)
    #            loss.backward()

    #            if cfg.grad_clip is not None and cfg.grad_clip > 0:
    #                torch.nn.utils.clip_grad_norm_(
    #                    model.parameters(),
    #                    max_norm=cfg.grad_clip,
    #                )

    #            optimizer.step()
    #            ema.update(model)

    #            global_step += 1

    #            loss_value = float(loss.item())
    #            alignment_value = float(velocity_alignment.item())

    #            epoch_losses.append(loss_value)
    #            epoch_alignments.append(alignment_value)

    #            if cfg.metrics_every > 0 and global_step % cfg.metrics_every == 0:
    #                metric_logger.log(
    #                    epoch=epoch,
    #                    step=global_step,
    #                    loss=loss_value,
    #                    velocity_alignment=alignment_value,
    #                )

    #        mean_loss = float(np.mean(epoch_losses))
    #        mean_alignment = float(np.mean(epoch_alignments))

    #        if epoch == 1 or epoch % cfg.log_every == 0:
    #            print(
    #                f"epoch {epoch:05d} | "
    #                f"step {global_step:07d} | "
    #                f"loss {mean_loss:.6f} | "
    #                f"velocity_alignment {mean_alignment:.6f}"
    #            )

    #        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
    #            root, ext = os.path.splitext(cfg.save_path)
    #            checkpoint_path = f"{root}_epoch_{epoch:05d}{ext}"

    #            save_checkpoint(
    #                save_path=checkpoint_path,
    #                model=model,
    #                ema=ema,
    #                optimizer=optimizer,
    #                spectrum_length=spectrum_length,
    #                cfg=cfg,
    #                epoch=epoch,
    #                global_step=global_step,
    #            )

    #            print(f"saved checkpoint to {checkpoint_path}")

    #    save_checkpoint(
    #        save_path=cfg.save_path,
    #        model=model,
    #        ema=ema,
    #        optimizer=optimizer,
    #        spectrum_length=spectrum_length,
    #        cfg=cfg,
    #        epoch=cfg.epochs,
    #        global_step=global_step,
    #    )

    #    print(f"saved final checkpoint to {cfg.save_path}")
    #    print(f"saved metrics to {cfg.metrics_path}")

    # finally:
    #    metric_logger.close()


# -----------------------------
# CLI
# -----------------------------


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
    parser.add_argument("--stage", type=str, default="dry")

    parser.add_argument("--patch_h", type=int, default=32)
    parser.add_argument("--patch_w", type=int, default=32)
    parser.add_argument("--stride_h", type=int, default=16)
    parser.add_argument("--stride_w", type=int, default=16)
    parser.add_argument("--black_thr", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=104)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="checkpoints/fm_spectral.pt")

    parser.add_argument(
        "--metrics_path", type=str, default="logs/fm_spectral_metrics.csv"
    )
    parser.add_argument("--metrics_every", type=int, default=10)

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
    train(cfg)
