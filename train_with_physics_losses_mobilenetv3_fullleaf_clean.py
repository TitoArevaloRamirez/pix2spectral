import os
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import config_mobilenetv3_fullleaf_clean as config
except ImportError:
    import config_mobilenetv3_fullleaf as config

from dataset_mobilenetv3_fullleaf_clean import (
    FullLeafMultispectralCSVDataset,
    fullleaf_collate_fn,
    save_normalization_stats,
    load_normalization_stats,
)
from generator_model_mobilenetv3_fullleaf_clean import FullLeafMobileNetV3ProspectGenerator
from discriminator_model_mobilenetv3_fullleaf_clean import (
    FullLeafSpectralDiscriminator1D,
    ConditionalFullLeafSpectralDiscriminator1D,
)
import torch.nn.functional as F

from physics_losses import (
    PhysicsInformedLoss,
    AdversarialLoss,
    SpectralAngleMapper,
    create_wavelength_weights,
)

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt



# Metric update: each epoch logs/prints RMSE, MAE, Bias, MRE, SAM_deg, and R2 for train and validation.

# ============================================================
# Utilities
# ============================================================


def expand_path(path):
    if path is None:
        return None
    return os.path.expanduser(str(path))


def ensure_dir(path):
    os.makedirs(expand_path(path), exist_ok=True)


def ensure_parent_dir(path):
    parent = os.path.dirname(expand_path(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_checkpoint(
    model,
    optimizer,
    filename,
    epoch=None,
    metrics=None,
    best_metric=None,
    scaler=None,
):
    filename = expand_path(filename)
    ensure_parent_dir(filename)

    ckpt = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "best_metric": best_metric,
    }

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    torch.save(ckpt, filename)


def load_checkpoint(
    filename,
    model,
    optimizer=None,
    lr=None,
    device="cpu",
    scaler=None,
    strict=True,
):
    filename = expand_path(filename)
    ckpt = torch.load(filename, map_location=device)

    model.load_state_dict(ckpt["state_dict"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

        if lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt


def move_fullleaf_band_images_to_device(fullleaf_band_images, device, non_blocking=False):
    out = {}
    for band, lst in fullleaf_band_images.items():
        out[band] = [t.to(device, non_blocking=non_blocking).float() for t in lst]
    return out


def permute_fullleaf_band_images(fullleaf_band_images, perm):
    """Reorder full-leaf image conditions across the batch for mismatch training."""
    perm_list = perm.detach().cpu().tolist()
    out = {}
    for band, lst in fullleaf_band_images.items():
        out[band] = [lst[i] for i in perm_list]
    return out

def unpack_fullleaf_batch(batch):
    """Return full-leaf images, spectra, and optional stage indices from a loader batch."""
    if len(batch) == 2:
        fullleaf_band_images, spectrum = batch
        stage_index = None
    elif len(batch) >= 3:
        fullleaf_band_images, spectrum, stage_index = batch[:3]
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return fullleaf_band_images, spectrum, stage_index


def make_mismatch_permutation(batch_size, device):
    """Return a non-identity permutation when batch_size > 1."""
    if int(batch_size) <= 1:
        return None
    perm = torch.randperm(int(batch_size), device=device)
    if torch.all(perm == torch.arange(int(batch_size), device=device)):
        perm = torch.roll(perm, shifts=1)
    return perm


def discriminator_fake_target_loss(logits, adv_loss_fn):
    """
    Fake-label loss for the mismatched-pair term.

    The training script uses LSGAN by default. If the AdversarialLoss object
    exposes a criterion, use it. Otherwise fall back to the LSGAN fake target:
        mean(logits^2)
    """
    loss_type = str(getattr(adv_loss_fn, "loss_type", "lsgan")).lower()
    if loss_type in ["wgan", "wasserstein"]:
        return torch.mean(logits)

    target = torch.zeros_like(logits)
    criterion = getattr(adv_loss_fn, "criterion", None)
    if criterion is not None:
        return criterion(logits, target)
    return torch.mean((logits - target) ** 2)


def assert_finite_tensor(name, tensor, batch_idx=None):
    if not torch.isfinite(tensor).all():
        bad = ~torch.isfinite(tensor)
        num_bad = int(bad.sum().detach().cpu())
        total = tensor.numel()

        msg = f"Non-finite tensor detected: {name}. Bad values: {num_bad}/{total}."
        if batch_idx is not None:
            msg += f" Batch index: {batch_idx}."

        finite_values = tensor[torch.isfinite(tensor)]
        if finite_values.numel() > 0:
            msg += (
                f" Finite min={float(finite_values.min().detach().cpu()):.6g}, "
                f"max={float(finite_values.max().detach().cpu()):.6g}, "
                f"mean={float(finite_values.mean().detach().cpu()):.6g}."
            )

        raise FloatingPointError(msg)


def plot_and_save_spectra(y_real, y_fake, epoch, out_dir="evaluation", title_prefix=""):
    ensure_dir(out_dir)

    r = y_real[0].detach().cpu().numpy().reshape(-1)
    f = y_fake[0].detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(12, 6))
    plt.plot(r, label="Real", linewidth=2, alpha=0.8)
    plt.plot(f, label="Generated", linewidth=2, alpha=0.8)
    plt.xlabel("Wavelength index")
    plt.ylabel("Reflectance")
    t = f"Epoch {epoch}"
    if title_prefix:
        t = f"{title_prefix} {t}"
    plt.title(t)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(expand_path(out_dir), f"spectra_epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_parameters(params_batch, epoch, out_dir="evaluation"):
    ensure_dir(out_dir)

    param_names = ["Nleaf", "Cab", "Car", "Cbrown", "Cw", "Cm", "Ant"]
    params_np = params_batch.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        axes[i].hist(params_np[:, i], bins=20, alpha=0.7, edgecolor="black")
        axes[i].set_title(f"{name} Distribution")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(alpha=0.3)

    fig.delaxes(axes[7])
    plt.tight_layout()

    path = os.path.join(expand_path(out_dir), f"parameters_epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def log_metrics(epoch, metrics, log_file="training_log.json"):
    log_file = expand_path(log_file)
    ensure_parent_dir(log_file)

    entry = {
        "epoch": int(epoch),
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


class EarlyStopping:
    def __init__(self, mode="min", patience=50, min_delta=0.0, min_epochs=0):
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.best = None
        self.best_epoch = None
        self.num_bad_epochs = 0

    def is_improvement(self, current):
        current = float(current)

        if self.best is None:
            return True

        if self.mode == "min":
            return current < self.best - self.min_delta

        return current > self.best + self.min_delta

    def step(self, current, epoch):
        current = float(current)

        if self.is_improvement(current):
            self.best = current
            self.best_epoch = int(epoch)
            self.num_bad_epochs = 0
            return True, False

        self.num_bad_epochs += 1
        should_stop = epoch >= self.min_epochs and self.num_bad_epochs >= self.patience
        return False, should_stop


def get_metric_or_raise(metrics, metric_name):
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(
            f"Metric '{metric_name}' was not found. Available metrics are: {available}"
        )

    value = metrics[metric_name]

    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"Metric '{metric_name}' is not finite: {value}")

    return float(value)


def reset_spectral_metric_accumulator():
    return {
        "n_values": 0,
        "sum_sq_error": 0.0,
        "sum_abs_error": 0.0,
        "sum_error": 0.0,
        "sum_abs_relative_error": 0.0,
        "sum_y_true": 0.0,
        "sum_y_true_sq": 0.0,
        "sam_rad_sum": 0.0,
        "n_spectra": 0,
    }


def update_spectral_metric_accumulator(acc, y_pred, y_true, eps=1e-8):
    """
    Accumulate spectral prediction metrics over batches.

    Metrics are computed globally over all spectrum values:
        RMSE, MAE, Bias, MRE, R2

    SAM is computed per spectrum and averaged.

    MRE is reported as percent:
        mean(abs(y_pred - y_true) / max(abs(y_true), eps)) * 100
    """
    with torch.no_grad():
        y_pred = y_pred.detach().float()
        y_true = y_true.detach().float()

        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Metric shape mismatch: y_pred={tuple(y_pred.shape)}, "
                f"y_true={tuple(y_true.shape)}"
            )

        err = y_pred - y_true
        abs_err = torch.abs(err)
        sq_err = err.pow(2)
        denom = torch.clamp(torch.abs(y_true), min=float(eps))
        abs_rel_err = abs_err / denom

        acc["n_values"] += int(y_true.numel())
        acc["sum_sq_error"] += float(torch.sum(sq_err).detach().cpu())
        acc["sum_abs_error"] += float(torch.sum(abs_err).detach().cpu())
        acc["sum_error"] += float(torch.sum(err).detach().cpu())
        acc["sum_abs_relative_error"] += float(torch.sum(abs_rel_err).detach().cpu())
        acc["sum_y_true"] += float(torch.sum(y_true).detach().cpu())
        acc["sum_y_true_sq"] += float(torch.sum(y_true.pow(2)).detach().cpu())

        yp = y_pred.reshape(y_pred.shape[0], -1)
        yt = y_true.reshape(y_true.shape[0], -1)

        dot = torch.sum(yp * yt, dim=1)
        norm_p = torch.linalg.norm(yp, dim=1)
        norm_t = torch.linalg.norm(yt, dim=1)
        cos_sim = dot / torch.clamp(norm_p * norm_t, min=float(eps))
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        sam_rad = torch.acos(cos_sim)

        acc["sam_rad_sum"] += float(torch.sum(sam_rad).detach().cpu())
        acc["n_spectra"] += int(y_pred.shape[0])


def finalize_spectral_metric_accumulator(acc, prefix):
    n = int(acc["n_values"])
    n_spectra = int(acc["n_spectra"])

    if n == 0:
        nan = float("nan")
        return {
            f"{prefix}_rmse": nan,
            f"{prefix}_mae": nan,
            f"{prefix}_bias": nan,
            f"{prefix}_mre": nan,
            f"{prefix}_sam_rad": nan,
            f"{prefix}_sam_deg": nan,
            f"{prefix}_r2": nan,
        }

    mse = acc["sum_sq_error"] / n
    rmse = float(np.sqrt(max(mse, 0.0)))
    mae = acc["sum_abs_error"] / n
    bias = acc["sum_error"] / n
    mre = 100.0 * acc["sum_abs_relative_error"] / n

    sse = acc["sum_sq_error"]
    sst = acc["sum_y_true_sq"] - (acc["sum_y_true"] ** 2) / max(n, 1)
    if sst > 1e-12:
        r2 = 1.0 - sse / sst
    else:
        r2 = float("nan")

    sam_rad = acc["sam_rad_sum"] / max(n_spectra, 1)
    sam_deg = sam_rad * 180.0 / np.pi

    return {
        f"{prefix}_rmse": float(rmse),
        f"{prefix}_mae": float(mae),
        f"{prefix}_bias": float(bias),
        f"{prefix}_mre": float(mre),
        f"{prefix}_sam_rad": float(sam_rad),
        f"{prefix}_sam_deg": float(sam_deg),
        f"{prefix}_r2": float(r2),
    }


def print_spectral_metric_block(title, metrics, prefix):
    print(title)
    print(f"  RMSE:          {metrics[f'{prefix}_rmse']:.6f}")
    print(f"  MAE:           {metrics[f'{prefix}_mae']:.6f}")
    print(f"  Bias:          {metrics[f'{prefix}_bias']:.6f}")
    print(f"  MRE:           {metrics[f'{prefix}_mre']:.3f}%")
    print(f"  SAM:           {metrics[f'{prefix}_sam_deg']:.3f} deg")
    print(f"  R2:            {metrics[f'{prefix}_r2']:.6f}")




def compute_stage_aux_loss_and_acc(stage_logits, stage_idx, device):
    """
    Compute auxiliary dehydration-stage classification loss.

    Valid stage labels must be in [0, C-1]. Samples with label < 0 are ignored.
    This avoids disabling the whole batch if one row has an unknown stage label.

    Returns:
        loss scalar tensor
        acc scalar tensor
        n_valid integer
    """
    zero = torch.zeros((), device=device)

    if stage_logits is None or stage_idx is None:
        return zero, zero, 0

    stage_idx = stage_idx.to(device=device, dtype=torch.long)
    valid = stage_idx >= 0
    if valid.sum().item() == 0:
        return zero, zero, 0

    logits_valid = stage_logits.float()[valid]
    idx_valid = stage_idx[valid]

    n_classes = logits_valid.shape[1]
    valid2 = idx_valid < n_classes
    if valid2.sum().item() == 0:
        return zero, zero, 0

    logits_valid = logits_valid[valid2]
    idx_valid = idx_valid[valid2]

    loss = F.cross_entropy(logits_valid, idx_valid)
    acc = (logits_valid.argmax(dim=1) == idx_valid).float().mean()
    return loss, acc, int(idx_valid.numel())

# ============================================================
# Validation
# ============================================================


def validate(gen, val_loader, device, use_amp, physics_loss_fn, sam_fn):
    """
    Validation with numerical safety.

    Generator forward is full precision here. This is safer when the generator
    contains physics-based operations or when val_physics_total was becoming NaN.
    """
    gen.eval()

    metrics = {
        "val_l1": 0.0,
        "val_rmse": 0.0,
        "val_sam_rad": 0.0,
        "val_sam_deg": 0.0,
        "val_physics_total": 0.0,
        "val_spectral_l1": 0.0,
        "val_weighted_l1": 0.0,
        "val_param_penalty": 0.0,
        "val_smoothness": 0.0,
        "val_derivative": 0.0,
        "val_stage_aux_loss": 0.0,
        "val_stage_aux_acc": 0.0,
    }

    n_batches = 0
    val_metric_acc = reset_spectral_metric_accumulator()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            fullleaf_band_images, y_real, stage_idx = unpack_fullleaf_batch(batch)
            fullleaf_band_images = move_fullleaf_band_images_to_device(fullleaf_band_images, device)
            y_real = y_real.to(device).float()
            if stage_idx is not None:
                stage_idx = stage_idx.to(device).long()

            assert_finite_tensor("y_real", y_real, batch_idx=batch_idx)

            # Use full precision for validation and physics metrics.
            with torch.amp.autocast("cuda", enabled=False):
                y_fake, p_params = gen.forward_condition_batch(fullleaf_band_images)
                y_fake = y_fake.float()
                p_params = p_params.float()

                stage_logits = getattr(gen, "last_stage_logits", None)
                stage_loss, stage_acc, _ = compute_stage_aux_loss_and_acc(
                    stage_logits=stage_logits,
                    stage_idx=stage_idx,
                    device=device,
                )

                assert_finite_tensor("y_fake", y_fake, batch_idx=batch_idx)
                assert_finite_tensor("p_params", p_params, batch_idx=batch_idx)

                l1 = torch.mean(torch.abs(y_fake - y_real))
                rmse = torch.sqrt(torch.mean((y_fake - y_real) ** 2) + 1e-12)
                sam = sam_fn(y_fake, y_real)
                update_spectral_metric_accumulator(val_metric_acc, y_fake, y_real)

                _, physics_losses = physics_loss_fn(y_fake, y_real, p_params)

            metrics["val_l1"] += float(l1.detach().cpu())
            metrics["val_rmse"] += float(rmse.detach().cpu())
            metrics["val_sam_rad"] += float(sam.detach().cpu())
            metrics["val_physics_total"] += float(
                physics_losses["total"].detach().cpu()
            )
            metrics["val_spectral_l1"] += float(
                physics_losses["spectral_l1"].detach().cpu()
            )
            metrics["val_weighted_l1"] += float(
                physics_losses["weighted_l1"].detach().cpu()
            )
            metrics["val_param_penalty"] += float(
                physics_losses["param_penalty"].detach().cpu()
            )
            metrics["val_smoothness"] += float(
                physics_losses["smoothness"].detach().cpu()
            )
            metrics["val_derivative"] += float(
                physics_losses["derivative"].detach().cpu()
            )
            metrics["val_stage_aux_loss"] += float(stage_loss.detach().cpu())
            metrics["val_stage_aux_acc"] += float(stage_acc.detach().cpu())

            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Validation loader produced zero batches.")

    for key in metrics:
        metrics[key] /= n_batches

    metrics["val_sam_deg"] = metrics["val_sam_rad"] * 180.0 / np.pi
    metrics.update(finalize_spectral_metric_accumulator(val_metric_acc, "val"))
    return metrics


# ============================================================
# Training step
# ============================================================


def train_one_epoch(
    disc,
    gen,
    loader,
    opt_disc,
    opt_gen,
    adv_loss_fn,
    physics_loss_fn,
    device,
    epoch,
    use_amp,
    scaler_d,
    scaler_g,
    non_blocking,
    l1_lambda=100.0,
    use_conditional_discriminator=False,
    lambda_mismatch=0.0,
    lambda_stage_aux=0.0,
):
    disc.train()
    gen.train()

    loop = tqdm(loader, leave=True)
    t0 = time.time()

    epoch_metrics = {
        "d_loss": 0.0,
        "d_mismatch": 0.0,
        "g_loss": 0.0,
        "g_adv": 0.0,
        "g_physics": 0.0,
        "g_spectral_l1": 0.0,
        "g_weighted_l1": 0.0,
        "g_param_penalty": 0.0,
        "g_smoothness": 0.0,
        "g_derivative": 0.0,
        "g_stage_aux_loss": 0.0,
        "g_stage_aux_acc": 0.0,
    }
    n_batches = 0
    train_metric_acc = reset_spectral_metric_accumulator()

    for idx, batch in enumerate(loop):
        fullleaf_band_images, y_real, stage_idx = unpack_fullleaf_batch(batch)
        fullleaf_band_images = move_fullleaf_band_images_to_device(
            fullleaf_band_images, device, non_blocking=non_blocking
        )
        y_real = y_real.to(device, non_blocking=non_blocking).float()
        if stage_idx is not None:
            stage_idx = stage_idx.to(device, non_blocking=non_blocking).long()

        # Forward generator.
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_fake, p_params = gen.forward_condition_batch(fullleaf_band_images)

        # Train discriminator.
        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_conditional_discriminator:
                D_real = disc(y_real, fullleaf_band_images)
                D_fake = disc(y_fake.detach(), fullleaf_band_images)
            else:
                D_real = disc(y_real)
                D_fake = disc(y_fake.detach())

            D_loss = adv_loss_fn.discriminator_loss(D_real, D_fake)
            D_mismatch_loss = torch.zeros((), device=device, dtype=D_loss.dtype)

            if use_conditional_discriminator and float(lambda_mismatch) > 0.0:
                perm = make_mismatch_permutation(y_real.shape[0], y_real.device)
                if perm is not None:
                    mismatched_bands = permute_fullleaf_band_images(fullleaf_band_images, perm)
                    D_mismatch = disc(y_real, mismatched_bands)
                    D_mismatch_loss = discriminator_fake_target_loss(
                        D_mismatch,
                        adv_loss_fn,
                    )
                    D_loss = D_loss + float(lambda_mismatch) * D_mismatch_loss

        if not torch.isfinite(D_loss):
            raise FloatingPointError(f"Non-finite D_loss detected: {D_loss.item()}")

        opt_disc.zero_grad(set_to_none=True)
        if use_amp:
            scaler_d.scale(D_loss).backward()
            scaler_d.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            scaler_d.step(opt_disc)
            scaler_d.update()
        else:
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            opt_disc.step()

        # Train generator. Keep physics loss in full precision.
        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_conditional_discriminator:
                D_fake_for_G = disc(y_fake, fullleaf_band_images)
            else:
                D_fake_for_G = disc(y_fake)
            G_adv = adv_loss_fn.generator_loss(D_fake_for_G)

        with torch.amp.autocast("cuda", enabled=False):
            y_fake_fp32 = y_fake.float()
            y_real_fp32 = y_real.float()
            p_params_fp32 = p_params.float()
            G_physics, physics_components = physics_loss_fn(
                y_fake_fp32,
                y_real_fp32,
                p_params_fp32,
            )
            stage_logits = getattr(gen, "last_stage_logits", None)
            stage_aux_loss, stage_aux_acc, _ = compute_stage_aux_loss_and_acc(
                stage_logits=stage_logits,
                stage_idx=stage_idx,
                device=device,
            )

            G_loss = (
                G_adv.float()
                + float(l1_lambda) * G_physics
                + float(lambda_stage_aux) * stage_aux_loss
            )
            update_spectral_metric_accumulator(
                train_metric_acc,
                y_fake_fp32.detach(),
                y_real_fp32.detach(),
            )

        if not torch.isfinite(G_loss):
            print("Non-finite G_loss detected.")
            print(f"  G_adv: {float(G_adv.detach().float().cpu())}")
            print(f"  G_physics: {float(G_physics.detach().float().cpu())}")
            for k, v in physics_components.items():
                print(f"  {k}: {float(v.detach().float().cpu())}")
            raise FloatingPointError("Stopping because G_loss is NaN or Inf.")

        opt_gen.zero_grad(set_to_none=True)
        if use_amp:
            scaler_g.scale(G_loss).backward()
            scaler_g.unscale_(opt_gen)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            scaler_g.step(opt_gen)
            scaler_g.update()
        else:
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            opt_gen.step()

        epoch_metrics["d_loss"] += float(D_loss.detach().cpu())
        epoch_metrics["d_mismatch"] += float(D_mismatch_loss.detach().cpu())
        epoch_metrics["g_loss"] += float(G_loss.detach().cpu())
        epoch_metrics["g_adv"] += float(G_adv.detach().cpu())
        epoch_metrics["g_physics"] += float(G_physics.detach().cpu())
        epoch_metrics["g_spectral_l1"] += float(
            physics_components["spectral_l1"].detach().cpu()
        )
        epoch_metrics["g_weighted_l1"] += float(
            physics_components["weighted_l1"].detach().cpu()
        )
        epoch_metrics["g_param_penalty"] += float(
            physics_components["param_penalty"].detach().cpu()
        )
        epoch_metrics["g_smoothness"] += float(
            physics_components["smoothness"].detach().cpu()
        )
        epoch_metrics["g_derivative"] += float(
            physics_components["derivative"].detach().cpu()
        )
        epoch_metrics["g_stage_aux_loss"] += float(stage_aux_loss.detach().cpu())
        epoch_metrics["g_stage_aux_acc"] += float(stage_aux_acc.detach().cpu())
        n_batches += 1

        if idx % 5 == 0:
            loop.set_postfix(
                D=f"{float(D_loss.detach().cpu()):.3f}",
                G=f"{float(G_loss.detach().cpu()):.3f}",
                adv=f"{float(G_adv.detach().cpu()):.3f}",
                phy=f"{float(G_physics.detach().cpu()):.4f}",
            )

    if n_batches == 0:
        raise RuntimeError("Training loader produced zero batches.")

    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches

    epoch_metrics.update(finalize_spectral_metric_accumulator(train_metric_acc, "train"))

    dt = time.time() - t0
    return dt, epoch_metrics


# ============================================================
# Main
# ============================================================


def main():
    if hasattr(config, "RANDOM_SEED"):
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)

    requested = config.DEVICE
    if "cuda" in str(requested) and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested = "cpu"
    device = torch.device(requested)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    pin_memory = device.type == "cuda"
    non_blocking = device.type == "cuda"

    ensure_dir(getattr(config, "RESULTS_DIR", "."))
    ensure_dir(getattr(config, "OUTDIR_PLOT", "evaluation"))

    # ============================================================
    # Initialize models
    # ============================================================
    use_conditional_discriminator = getattr(
        config,
        "USE_CONDITIONAL_DISCRIMINATOR",
        False,
    )

    discriminator_common_kwargs = dict(
        in_channels=1,
        features=config.DISCRIMINATOR_FEATURES,
        use_bn=False,
        wavelength_min=config.WAVELENGTH_MIN,
        wavelength_max=config.WAVELENGTH_MAX,
        wavelength_count=config.WAVELENGTH_COUNT,
        spectral_segments=config.SPECTRAL_SEGMENTS,
        mode=config.DISCRIMINATOR_MODE,
        use_wavelength_channel=config.USE_WAVELENGTH_CHANNEL,
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        bands=["blue", "green", "red", "nir", "red_edge"],
    )

    if use_conditional_discriminator:
        disc = ConditionalFullLeafSpectralDiscriminator1D(
            **discriminator_common_kwargs,
            condition_dim=getattr(config, "CONDITION_DIM", config.DISCRIMINATOR_FEATURES[-1]),
            condition_embed_dim=getattr(config, "CONDITION_EMBED_DIM", config.EMBED_DIM),
            condition_base_features=getattr(config, "CONDITION_BASE_FEATURES", config.BASE_FEATURES),
            mobilenet_pretrained=getattr(config, "MOBILENET_PRETRAINED", True),
            mobilenet_freeze_all_except_last=getattr(config, "MOBILENET_FREEZE_ALL_EXCEPT_LAST", True),
            mobilenet_token_dim=getattr(config, "MOBILENET_TOKEN_DIM", 128),
            mobilenet_attention_layers=getattr(config, "MOBILENET_ATTENTION_LAYERS", 1),
            mobilenet_attention_heads=getattr(config, "MOBILENET_ATTENTION_HEADS", 4),
            mobilenet_dropout=getattr(config, "MOBILENET_DROPOUT", 0.25),
            mobilenet_adapter_hidden_channels=getattr(config, "MOBILENET_ADAPTER_HIDDEN_CHANNELS", 8),
            num_stage_classes=getattr(config, "NUM_STAGE_CLASSES", 5),
        ).to(device)
    else:
        disc = FullLeafSpectralDiscriminator1D(
            **discriminator_common_kwargs,
        ).to(device)

    gen = FullLeafMobileNetV3ProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=config.BASE_FEATURES,
        embed_dim=config.EMBED_DIM,
        mins=config.PROSPECT_PARAM_MINS,
        maxs=config.PROSPECT_PARAM_MAXS,
        wavelength_min=config.WAVELENGTH_MIN,
        wavelength_max=config.WAVELENGTH_MAX,
        wavelength_count=config.WAVELENGTH_COUNT,
        spectral_segments=config.SPECTRAL_SEGMENTS,
        use_segmented_prospect=config.USE_SEGMENTED_PROSPECT,
        use_segment_residual=config.USE_SEGMENT_RESIDUAL,
        segment_residual_scale=config.SEGMENT_RESIDUAL_SCALE,
        mobilenet_pretrained=getattr(config, "MOBILENET_PRETRAINED", True),
        mobilenet_freeze_all_except_last=getattr(config, "MOBILENET_FREEZE_ALL_EXCEPT_LAST", True),
        mobilenet_token_dim=getattr(config, "MOBILENET_TOKEN_DIM", 128),
        mobilenet_attention_layers=getattr(config, "MOBILENET_ATTENTION_LAYERS", 1),
        mobilenet_attention_heads=getattr(config, "MOBILENET_ATTENTION_HEADS", 4),
        mobilenet_dropout=getattr(config, "MOBILENET_DROPOUT", 0.25),
        mobilenet_adapter_hidden_channels=getattr(config, "MOBILENET_ADAPTER_HIDDEN_CHANNELS", 8),
        use_stage_classifier=getattr(config, "USE_STAGE_AUXILIARY_LOSS", True),
        num_stage_classes=getattr(config, "NUM_STAGE_CLASSES", 5),
    ).to(device)

    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # ============================================================
    # Losses
    # ============================================================
    wavelength_weights = create_wavelength_weights(
        num_wavelengths=2101,
        start_wl=400,
        end_wl=2500,
    )

    physics_loss_fn = PhysicsInformedLoss(
        param_bounds=gen.physics.bounds,
        wavelength_weights=wavelength_weights,
        lambda_spectral=1.0,
        lambda_weighted=0.5,
        lambda_param_penalty=0.1,
        lambda_smoothness=0.01,
        lambda_derivative=0.01,
        lambda_segment_continuity=getattr(config, "LAMBDA_SEGMENT_CONTINUITY", 0.1),
        boundary_indices=getattr(gen, "boundary_indices", None),
        continuity_width=2,
    ).to(device)

    adv_loss_fn = AdversarialLoss(loss_type="lsgan")
    sam_fn = SpectralAngleMapper()

    # ============================================================
    # Load checkpoints
    # ============================================================
    start_epoch = 0

    if config.LOAD_MODEL:
        resume_from_best = getattr(config, "RESUME_FROM_BEST", False)

        gen_ckpt_path = (
            getattr(config, "BEST_CHECKPOINT_GEN", config.CHECKPOINT_GEN)
            if resume_from_best
            else config.CHECKPOINT_GEN
        )
        disc_ckpt_path = (
            getattr(config, "BEST_CHECKPOINT_DISC", config.CHECKPOINT_DISC)
            if resume_from_best
            else config.CHECKPOINT_DISC
        )

        if os.path.isfile(expand_path(gen_ckpt_path)):
            ckpt = load_checkpoint(
                gen_ckpt_path,
                gen,
                opt_gen,
                config.LEARNING_RATE,
                device=str(device),
                scaler=scaler_g,
            )
            if ckpt.get("epoch") is not None and not resume_from_best:
                start_epoch = int(ckpt["epoch"]) + 1
            print(f"Loaded generator checkpoint: {expand_path(gen_ckpt_path)}")

        if os.path.isfile(expand_path(disc_ckpt_path)):
            load_checkpoint(
                disc_ckpt_path,
                disc,
                opt_disc,
                config.LEARNING_RATE,
                device=str(device),
                scaler=scaler_d,
            )
            print(f"Loaded discriminator checkpoint: {expand_path(disc_ckpt_path)}")

        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}")

    # ============================================================
    # Prepare datasets and normalization stats
    # ============================================================
    normalization_scope = getattr(config, "IMAGE_NORMALIZATION_SCOPE", "none")
    normalization_method = getattr(config, "IMAGE_NORMALIZATION_METHOD", "none")
    normalization_output_clip = getattr(config, "IMAGE_NORMALIZATION_OUTPUT_CLIP", None)
    compute_normalization_stats = getattr(config, "COMPUTE_NORMALIZATION_STATS", False)
    recompute_normalization_stats = getattr(
        config, "RECOMPUTE_NORMALIZATION_STATS", True
    )
    normalization_stats_path = getattr(config, "NORMALIZATION_STATS_PATH", None)

    normalization_stats = None

    can_load_stats = (
        normalization_scope != "none"
        and normalization_stats_path is not None
        and os.path.isfile(expand_path(normalization_stats_path))
        and not recompute_normalization_stats
    )

    if can_load_stats:
        normalization_stats = load_normalization_stats(normalization_stats_path)
        compute_stats_for_train_dataset = False
        print(f"Loaded normalization stats: {expand_path(normalization_stats_path)}")
    else:
        compute_stats_for_train_dataset = compute_normalization_stats

    train_dataset = FullLeafMultispectralCSVDataset(
        csv_path=config.TRAIN_CSV,
        image_root_dir=config.TRAIN_IMG_DIR,
        species_filter=config.SPECIES_FILTER,
        stage_filter=config.STAGE_FILTER,
        spectral_drop_first_n=getattr(config, "SPECTRAL_DROP_FIRST_N", 50),
        image_size=getattr(config, "FULL_IMAGE_SIZE", None),
        return_stage_index=getattr(config, "RETURN_STAGE_INDEX", True),
        cache_images=getattr(config, "CACHE_FULL_IMAGES", False),
    )

    normalization_stats = None

    if (
        normalization_scope != "none"
        and normalization_stats_path is not None
        and normalization_stats is not None
    ):
        save_normalization_stats(normalization_stats, normalization_stats_path)
        print(f"Saved normalization stats: {expand_path(normalization_stats_path)}")

    persistent_workers = config.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=fullleaf_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_dataset = FullLeafMultispectralCSVDataset(
        csv_path=config.VAL_CSV,
        image_root_dir=config.TRAIN_IMG_DIR if config.VAL_IMG_DIR is None else config.VAL_IMG_DIR,
        species_filter=config.SPECIES_FILTER,
        stage_filter=config.STAGE_FILTER,
        spectral_drop_first_n=getattr(config, "SPECTRAL_DROP_FIRST_N", 50),
        image_size=getattr(config, "FULL_IMAGE_SIZE", None),
        return_stage_index=getattr(config, "RETURN_STAGE_INDEX", True),
        cache_images=getattr(config, "CACHE_FULL_IMAGES", False),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=fullleaf_collate_fn,
        pin_memory=pin_memory,
    )

    log_file = getattr(config, "LOG_FILE", "training_log.json")
    if os.path.exists(expand_path(log_file)):
        os.remove(expand_path(log_file))

    # ============================================================
    # Print configuration
    # ============================================================
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION WITH PHYSICS-INFORMED LOSSES")
    print("=" * 70)
    print(f"Device: {device}")
    print("Dataset: FullLeafMultispectralCSVDataset (full images only)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"L1 Lambda: {config.L1_LAMBDA}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("Image normalization:")
    print(f"  Scope:  {normalization_scope}")
    print(f"  Method: {normalization_method}")
    print(f"  Clip:   {normalization_output_clip}")
    print("Discriminator:")
    print(f"  Conditional cGAN:     {use_conditional_discriminator}")
    print(f"  Mode:                 {config.DISCRIMINATOR_MODE}")
    print(f"  Mismatch lambda:      {getattr(config, 'LAMBDA_MISMATCH', 0.0)}")
    print("Stage auxiliary classifier:")
    print(f"  Enabled:       {getattr(config, 'USE_STAGE_AUXILIARY_LOSS', False)}")
    print(f"  Loss weight:   {getattr(config, 'STAGE_AUX_WEIGHT', 0.0)}")
    print(f"  Stage input:   {getattr(config, 'USE_STAGE_AS_CONDITION', False)}")
    print("Physics Loss Components:")
    print(f"  lambda_spectral:       {physics_loss_fn.lambda_spectral}")
    print(f"  lambda_weighted:       {physics_loss_fn.lambda_weighted}")
    print(f"  lambda_param_penalty:  {physics_loss_fn.lambda_param_penalty}")
    print(f"  lambda_smoothness:     {physics_loss_fn.lambda_smoothness}")
    print(f"  lambda_derivative:     {physics_loss_fn.lambda_derivative}")
    print("=" * 70 + "\n")

    # ============================================================
    # Best-model selection and early stopping
    # ============================================================
    monitor_metric = getattr(config, "BEST_MODEL_METRIC", "val_l1")
    monitor_mode = getattr(config, "BEST_MODEL_MODE", "min")

    early_stopper = EarlyStopping(
        mode=monitor_mode,
        patience=getattr(config, "EARLY_STOP_PATIENCE", 80),
        min_delta=getattr(config, "EARLY_STOP_MIN_DELTA", 1e-6),
        min_epochs=getattr(config, "EARLY_STOP_MIN_EPOCHS", 50),
    )

    early_stop_enabled = getattr(config, "EARLY_STOP_ENABLED", True)

    best_gen_path = getattr(config, "BEST_CHECKPOINT_GEN", "gen_best.pth.tar")
    best_disc_path = getattr(config, "BEST_CHECKPOINT_DISC", "disc_best.pth.tar")
    final_gen_path = getattr(config, "FINAL_CHECKPOINT_GEN", "gen_final_best.pth.tar")

    print("Best-model selection:")
    print(f"  Monitor metric: {monitor_metric}")
    print(f"  Monitor mode:   {monitor_mode}")
    print(f"  Patience:       {early_stopper.patience}")
    print(f"  Min delta:      {early_stopper.min_delta}")
    print(f"  Min epochs:     {early_stopper.min_epochs}")
    print("=" * 70 + "\n")

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        dt, train_metrics = train_one_epoch(
            disc=disc,
            gen=gen,
            loader=train_loader,
            opt_disc=opt_disc,
            opt_gen=opt_gen,
            adv_loss_fn=adv_loss_fn,
            physics_loss_fn=physics_loss_fn,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            scaler_d=scaler_d,
            scaler_g=scaler_g,
            non_blocking=non_blocking,
            l1_lambda=config.L1_LAMBDA,
            use_conditional_discriminator=use_conditional_discriminator,
            lambda_mismatch=getattr(config, "LAMBDA_MISMATCH", 0.0),
            lambda_stage_aux=(
                getattr(config, "STAGE_AUX_WEIGHT", 0.0)
                if getattr(config, "USE_STAGE_AUXILIARY_LOSS", False)
                else 0.0
            ),
        )

        val_metrics = validate(
            gen,
            val_loader,
            device,
            use_amp,
            physics_loss_fn,
            sam_fn,
        )

        all_metrics = {**train_metrics, **val_metrics}

        current_monitor_value = get_metric_or_raise(all_metrics, monitor_metric)
        improved, should_stop = early_stopper.step(current_monitor_value, epoch)

        all_metrics["monitor_metric"] = monitor_metric
        all_metrics["monitor_value"] = current_monitor_value
        all_metrics["best_monitor_value"] = early_stopper.best
        all_metrics["best_epoch"] = early_stopper.best_epoch
        all_metrics["epochs_without_improvement"] = early_stopper.num_bad_epochs
        all_metrics["is_best"] = bool(improved)

        log_metrics(epoch, all_metrics, log_file)

        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch} ({dt:.1f}s)")
        print(f"{'=' * 70}")
        print("Train:")
        print(f"  D_loss:       {train_metrics['d_loss']:.4f}")
        print(f"  D_mismatch:   {train_metrics.get('d_mismatch', 0.0):.4f}")
        print(f"  G_loss:       {train_metrics['g_loss']:.4f}")
        print(f"    adv:        {train_metrics['g_adv']:.4f}")
        print(f"    physics:    {train_metrics['g_physics']:.4f}")
        print(f"    spectral_l1:{train_metrics['g_spectral_l1']:.6f}")
        print(f"    weighted_l1:{train_metrics['g_weighted_l1']:.6f}")
        print(f"    param_pen:  {train_metrics['g_param_penalty']:.6f}")
        print(f"    smoothness: {train_metrics['g_smoothness']:.6f}")
        print(f"    derivative: {train_metrics['g_derivative']:.6f}")
        print(f"    stage_aux_loss: {train_metrics.get('g_stage_aux_loss', 0.0):.6f}")
        print(f"    stage_aux_acc:  {train_metrics.get('g_stage_aux_acc', 0.0):.4f}")
        print_spectral_metric_block("Train spectral metrics:", train_metrics, "train")

        print("Validation:")
        print(f"  L1:            {val_metrics['val_l1']:.6f}")
        print(f"  RMSE:          {val_metrics['val_rmse']:.6f}")
        print(
            f"  SAM:           {val_metrics['val_sam_deg']:.2f} deg ({val_metrics['val_sam_rad']:.4f} rad)"
        )
        print(f"  Physics total: {val_metrics['val_physics_total']:.6f}")
        print(f"  Param penalty: {val_metrics['val_param_penalty']:.6f}")
        print_spectral_metric_block("Validation spectral metrics:", val_metrics, "val")
        print(f"  Stage aux loss: {val_metrics.get('val_stage_aux_loss', 0.0):.6f}")
        print(f"  Stage aux acc:  {val_metrics.get('val_stage_aux_acc', 0.0):.4f}")

        print("Best-model tracking:")
        print(f"  Current {monitor_metric}: {current_monitor_value:.8f}")
        print(f"  Best    {monitor_metric}: {early_stopper.best:.8f}")
        print(f"  Best epoch:             {early_stopper.best_epoch}")
        print(f"  No improvement epochs:  {early_stopper.num_bad_epochs}")

        if config.SAVE_MODEL and improved:
            save_checkpoint(
                gen,
                opt_gen,
                filename=best_gen_path,
                epoch=epoch,
                metrics=all_metrics,
                best_metric=early_stopper.best,
                scaler=scaler_g,
            )
            save_checkpoint(
                disc,
                opt_disc,
                filename=best_disc_path,
                epoch=epoch,
                metrics=all_metrics,
                best_metric=early_stopper.best,
                scaler=scaler_d,
            )
            print("  New best model saved:")
            print(f"    Generator:     {expand_path(best_gen_path)}")
            print(f"    Discriminator: {expand_path(best_disc_path)}")

        should_plot = (
            epoch % getattr(config, "PLOT_INTERVAL", 1) == 0
            or improved
            or epoch == config.NUM_EPOCHS - 1
        )

        if should_plot:
            gen.eval()
            with torch.no_grad():
                for batch in val_loader:
                    fullleaf_band_images, y_real, _stage_idx = unpack_fullleaf_batch(batch)
                    fullleaf_band_images = move_fullleaf_band_images_to_device(
                        fullleaf_band_images,
                        device,
                        non_blocking=non_blocking,
                    )
                    y_real = y_real.to(device, non_blocking=non_blocking).float()

                    with torch.amp.autocast("cuda", enabled=False):
                        y_fake, p_params = gen.forward_condition_batch(fullleaf_band_images)
                        y_fake = y_fake.float()
                        p_params = p_params.float()

                    plot_and_save_spectra(
                        y_real,
                        y_fake,
                        epoch,
                        out_dir=getattr(config, "OUTDIR_PLOT", "evaluation"),
                        title_prefix="Val",
                    )

                    if epoch % 10 == 0 and y_real.shape[0] > 1:
                        plot_parameters(
                            p_params,
                            epoch,
                            out_dir=getattr(config, "OUTDIR_PLOT", "evaluation"),
                        )
                    break

        save_interval = getattr(config, "SAVE_INTERVAL", 5)
        should_save_latest = config.SAVE_MODEL and (
            (epoch + 1) % save_interval == 0 or epoch == config.NUM_EPOCHS - 1
        )

        if should_save_latest:
            save_checkpoint(
                gen,
                opt_gen,
                filename=config.CHECKPOINT_GEN,
                epoch=epoch,
                metrics=all_metrics,
                best_metric=early_stopper.best,
                scaler=scaler_g,
            )
            save_checkpoint(
                disc,
                opt_disc,
                filename=config.CHECKPOINT_DISC,
                epoch=epoch,
                metrics=all_metrics,
                best_metric=early_stopper.best,
                scaler=scaler_d,
            )
            print("  Saved latest checkpoints")

        if early_stop_enabled and should_stop:
            print("\n" + "=" * 70)
            print("EARLY STOPPING")
            print("=" * 70)
            print(
                f"No improvement in '{monitor_metric}' for "
                f"{early_stopper.num_bad_epochs} epochs."
            )
            print(f"Best epoch: {early_stopper.best_epoch}")
            print(f"Best {monitor_metric}: {early_stopper.best:.8f}")
            print("=" * 70)
            break

    # ============================================================
    # Restore/export best generator as final model
    # ============================================================
    if config.SAVE_MODEL and os.path.isfile(expand_path(best_gen_path)):
        print("\nLoading best generator checkpoint before finishing...")
        best_ckpt = load_checkpoint(
            best_gen_path,
            gen,
            optimizer=None,
            device=str(device),
        )

        final_gen_path = expand_path(final_gen_path)
        ensure_parent_dir(final_gen_path)
        torch.save(
            {
                "state_dict": gen.state_dict(),
                "best_epoch": best_ckpt.get("epoch"),
                "best_metric": best_ckpt.get("best_metric"),
                "monitor_metric": monitor_metric,
                "metrics": best_ckpt.get("metrics"),
                "normalization_scope": normalization_scope,
                "normalization_method": normalization_method,
                "normalization_output_clip": normalization_output_clip,
                "normalization_stats_path": expand_path(normalization_stats_path)
                if normalization_stats_path is not None
                else None,
            },
            final_gen_path,
        )

        print("Final model exported from best validation checkpoint:")
        print(f"  {final_gen_path}")

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Best epoch: {early_stopper.best_epoch}")
    print(f"Best {monitor_metric}: {early_stopper.best}")


if __name__ == "__main__":
    main()
