import os
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import (
    MultiSpectralCSVPatchDataset,
    patch_collate_fn,
    save_normalization_stats,
    load_normalization_stats,
)
from generator_model import MultiSpectralPatchToProspectGenerator
from discriminator_model import SegmentedSpectralDiscriminator1D

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


def move_batch_bands_to_device(batch_bands, device, non_blocking=False):
    out = {}
    for band, lst in batch_bands.items():
        out[band] = [t.to(device, non_blocking=non_blocking).float() for t in lst]
    return out


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
    }

    n_batches = 0

    with torch.no_grad():
        for batch_idx, (batch_bands, y_real) in enumerate(val_loader):
            batch_bands = move_batch_bands_to_device(batch_bands, device)
            y_real = y_real.to(device).float()

            assert_finite_tensor("y_real", y_real, batch_idx=batch_idx)

            # Use full precision for validation and physics metrics.
            with torch.amp.autocast("cuda", enabled=False):
                y_fake, p_params = gen.forward_batch_list(batch_bands)
                y_fake = y_fake.float()
                p_params = p_params.float()

                assert_finite_tensor("y_fake", y_fake, batch_idx=batch_idx)
                assert_finite_tensor("p_params", p_params, batch_idx=batch_idx)

                l1 = torch.mean(torch.abs(y_fake - y_real))
                rmse = torch.sqrt(torch.mean((y_fake - y_real) ** 2) + 1e-12)
                sam = sam_fn(y_fake, y_real)

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

            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Validation loader produced zero batches.")

    for key in metrics:
        metrics[key] /= n_batches

    metrics["val_sam_deg"] = metrics["val_sam_rad"] * 180.0 / np.pi
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
):
    disc.train()
    gen.train()

    loop = tqdm(loader, leave=True)
    t0 = time.time()

    epoch_metrics = {
        "d_loss": 0.0,
        "g_loss": 0.0,
        "g_adv": 0.0,
        "g_physics": 0.0,
        "g_spectral_l1": 0.0,
        "g_weighted_l1": 0.0,
        "g_param_penalty": 0.0,
        "g_smoothness": 0.0,
        "g_derivative": 0.0,
    }
    n_batches = 0

    for idx, (batch_bands, y_real) in enumerate(loop):
        batch_bands = move_batch_bands_to_device(
            batch_bands, device, non_blocking=non_blocking
        )
        y_real = y_real.to(device, non_blocking=non_blocking).float()

        # Forward generator.
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_fake, p_params = gen.forward_batch_list(batch_bands)

        # Train discriminator.
        with torch.amp.autocast("cuda", enabled=use_amp):
            D_real = disc(y_real)
            D_fake = disc(y_fake.detach())
            D_loss = adv_loss_fn.discriminator_loss(D_real, D_fake)

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
            G_loss = G_adv.float() + float(l1_lambda) * G_physics

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
    disc = SegmentedSpectralDiscriminator1D(
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
    ).to(device)

    gen = MultiSpectralPatchToProspectGenerator(
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
        patch_encoder_type=config.PATCH_ENCODER_TYPE,
        pooling_type=config.POOLING_TYPE,
        band_encoder_mode=config.BAND_ENCODER_MODE,
        norm_type=config.NORM_TYPE,
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

    train_dataset = MultiSpectralCSVPatchDataset(
        csv_path=config.TRAIN_CSV,
        root_dir=config.TRAIN_IMG_DIR,
        species=config.SPECIES_FILTER,
        stage=config.STAGE_FILTER,
        patch_h=config.PATCH_H,
        patch_w=config.PATCH_W,
        stride_h=config.STRIDE_H,
        stride_w=config.STRIDE_W,
        black_thr=config.BLACK_THR,
        min_leaf_coverage=config.LEAF_COVERAGE,
        min_patches_per_band=config.MIN_PATCHES,
        max_patches_per_band=getattr(config, "MAX_PATCHES_PER_BAND", None),
        border_erode_px=getattr(config, "BORDER_ERODE_PX", 2),
        mask_method=config.MASK_METHOD,
        random_seed=config.RANDOM_SEED,
        return_debug=False,
        compute_normalization_stats=compute_stats_for_train_dataset,
        normalization_stats=normalization_stats,
        normalization_scope=normalization_scope,
        normalization_method=normalization_method,
        normalization_output_clip=normalization_output_clip,
        normalization_use_leaf_mask=getattr(
            config, "NORMALIZATION_USE_LEAF_MASK", True
        ),
        normalization_sample_pixels_per_image=getattr(
            config, "NORMALIZATION_SAMPLE_PIXELS_PER_IMAGE", 20000
        ),
        normalization_lower_percentile=getattr(
            config, "NORMALIZATION_LOWER_PERCENTILE", 1.0
        ),
        normalization_upper_percentile=getattr(
            config, "NORMALIZATION_UPPER_PERCENTILE", 99.0
        ),
        # NEW
        cache_patches=True,
        clone_cached_items=False,
    )

    normalization_stats = train_dataset.normalization_stats

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
        collate_fn=patch_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_dataset = MultiSpectralCSVPatchDataset(
        csv_path=config.VAL_CSV,
        root_dir=config.TRAIN_IMG_DIR
        if config.VAL_IMG_DIR is None
        else config.VAL_IMG_DIR,
        species=config.SPECIES_FILTER,
        stage=config.STAGE_FILTER,
        patch_h=config.PATCH_H,
        patch_w=config.PATCH_W,
        stride_h=getattr(config, "VAL_STRIDE_H", config.PATCH_H),
        stride_w=getattr(config, "VAL_STRIDE_W", config.PATCH_W),
        black_thr=config.BLACK_THR,
        min_leaf_coverage=config.LEAF_COVERAGE,
        min_patches_per_band=config.MIN_PATCHES,
        max_patches_per_band=getattr(config, "MAX_PATCHES_PER_BAND", None),
        border_erode_px=getattr(config, "BORDER_ERODE_PX", 2),
        mask_method=config.MASK_METHOD,
        random_seed=config.RANDOM_SEED + 9999,
        return_debug=False,
        compute_normalization_stats=False,
        normalization_stats=normalization_stats,
        normalization_scope=normalization_scope,
        normalization_method=normalization_method,
        normalization_output_clip=normalization_output_clip,
        cache_patches=True,
        clone_cached_items=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=patch_collate_fn,
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
        print(f"  G_loss:       {train_metrics['g_loss']:.4f}")
        print(f"    adv:        {train_metrics['g_adv']:.4f}")
        print(f"    physics:    {train_metrics['g_physics']:.4f}")
        print(f"    spectral_l1:{train_metrics['g_spectral_l1']:.6f}")
        print(f"    weighted_l1:{train_metrics['g_weighted_l1']:.6f}")
        print(f"    param_pen:  {train_metrics['g_param_penalty']:.6f}")
        print(f"    smoothness: {train_metrics['g_smoothness']:.6f}")
        print(f"    derivative: {train_metrics['g_derivative']:.6f}")

        print("Validation:")
        print(f"  L1:            {val_metrics['val_l1']:.6f}")
        print(f"  RMSE:          {val_metrics['val_rmse']:.6f}")
        print(
            f"  SAM:           {val_metrics['val_sam_deg']:.2f} deg ({val_metrics['val_sam_rad']:.4f} rad)"
        )
        print(f"  Physics total: {val_metrics['val_physics_total']:.6f}")
        print(f"  Param penalty: {val_metrics['val_param_penalty']:.6f}")

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
                for batch_bands, y_real in val_loader:
                    batch_bands = move_batch_bands_to_device(
                        batch_bands,
                        device,
                        non_blocking=non_blocking,
                    )
                    y_real = y_real.to(device, non_blocking=non_blocking).float()

                    with torch.amp.autocast("cuda", enabled=False):
                        y_fake, p_params = gen.forward_batch_list(batch_bands)
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
