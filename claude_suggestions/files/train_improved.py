import os
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MultiSpectralCSVPatchDataset, patch_collate_fn
from generator_model import MultiSpectralPatchToProspectGenerator
from discriminator_model import SpectralDiscriminator1D  # FIXED: Using standard discriminator

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def save_checkpoint(model, optimizer, filename):
    ckpt = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, filename)


def load_checkpoint(filename, model, optimizer=None, lr=None, device="cpu"):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = lr


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def move_batch_bands_to_device(batch_bands, device, non_blocking=False):
    out = {}
    for band, lst in batch_bands.items():
        out[band] = [t.to(device, non_blocking=non_blocking) for t in lst]
    return out


def plot_and_save_spectra(y_real, y_fake, epoch, out_dir="evaluation", title_prefix=""):
    ensure_dir(out_dir)

    r = y_real[0].detach().cpu().numpy().reshape(-1)
    f = y_fake[0].detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(10, 6))
    plt.plot(r, label="Real", linewidth=2)
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

    path = os.path.join(out_dir, f"spectra_epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def log_metrics(epoch, metrics, log_file='training_log.json'):
    """Log training metrics to JSON file."""
    entry = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        **metrics
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


# ============================================================
# Validation with proper metrics
# ============================================================

def validate(gen, val_loader, device, use_amp):
    """
    Compute validation metrics: L1, RMSE, and Spectral Angle Mapper (SAM).
    """
    gen.eval()
    total_l1 = 0
    total_rmse = 0
    total_sam = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch_bands, y_real in val_loader:
            batch_bands = move_batch_bands_to_device(batch_bands, device)
            y_real = y_real.to(device)
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                y_fake, p_params = gen.forward_batch_list(batch_bands)
            
            # L1 error
            total_l1 += torch.mean(torch.abs(y_fake - y_real)).item()
            
            # RMSE
            total_rmse += torch.sqrt(torch.mean((y_fake - y_real)**2)).item()
            
            # Spectral Angle Mapper (SAM) - measures angular distance
            cos_sim = torch.nn.functional.cosine_similarity(y_fake, y_real, dim=1)
            sam = torch.acos(torch.clamp(cos_sim, -1, 1))
            total_sam += torch.mean(sam).item()
            
            n_samples += 1
    
    metrics = {
        'val_l1': total_l1 / n_samples,
        'val_rmse': total_rmse / n_samples,
        'val_sam_rad': total_sam / n_samples,
        'val_sam_deg': (total_sam / n_samples) * 180 / np.pi
    }
    
    return metrics


# ============================================================
# Training step with IMPROVED discriminator logic
# ============================================================

def train_one_epoch(
    disc,
    gen,
    loader,
    opt_disc,
    opt_gen,
    gan_loss,
    l1_loss,
    device,
    epoch,
    use_amp,
    scaler_d,
    scaler_g,
    non_blocking,
):
    disc.train()
    gen.train()

    loop = tqdm(loader, leave=True)
    t0 = time.time()
    
    epoch_metrics = {
        'd_loss': 0,
        'g_loss': 0,
        'g_adv': 0,
        'g_l1': 0,
    }
    n_batches = 0

    for idx, (batch_bands, y_real) in enumerate(loop):
        batch_bands = move_batch_bands_to_device(batch_bands, device, non_blocking=non_blocking)
        y_real = y_real.to(device, non_blocking=non_blocking)  # [B, L]

        # Forward generator (physics inside generator)
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_fake, p_params = gen.forward_batch_list(batch_bands)

        # ============================================================
        # IMPROVED: Standard GAN Discriminator
        #   D(real) -> 1
        #   D(fake) -> 0
        # ============================================================
        with torch.amp.autocast("cuda", enabled=use_amp):
            D_real = disc(y_real)
            D_fake = disc(y_fake.detach())

            D_real_loss = gan_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = gan_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = 0.5 * (D_real_loss + D_fake_loss)

        opt_disc.zero_grad(set_to_none=True)
        if use_amp:
            scaler_d.scale(D_loss).backward()
            scaler_d.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)  # Gradient clipping
            scaler_d.step(opt_disc)
            scaler_d.update()
        else:
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)  # Gradient clipping
            opt_disc.step()

        # ============================================================
        # Generator: fool discriminator + spectral L1
        # ============================================================
        with torch.amp.autocast("cuda", enabled=use_amp):
            D_fake_for_G = disc(y_fake)
            G_adv = gan_loss(D_fake_for_G, torch.ones_like(D_fake_for_G))

            G_l1 = l1_loss(y_fake, y_real) * config.L1_LAMBDA
            
            # Optional: Add spectral smoothness penalty (natural leaves have smooth spectra)
            # spectral_smoothness = torch.mean(torch.abs(y_fake[:, 1:] - y_fake[:, :-1]))
            # G_loss = G_adv + G_l1 + 0.01 * spectral_smoothness
            
            G_loss = G_adv + G_l1

        opt_gen.zero_grad(set_to_none=True)
        if use_amp:
            scaler_g.scale(G_loss).backward()
            scaler_g.unscale_(opt_gen)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)  # Gradient clipping
            scaler_g.step(opt_gen)
            scaler_g.update()
        else:
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)  # Gradient clipping
            opt_gen.step()

        # Track metrics
        epoch_metrics['d_loss'] += float(D_loss.detach().cpu())
        epoch_metrics['g_loss'] += float(G_loss.detach().cpu())
        epoch_metrics['g_adv'] += float(G_adv.detach().cpu())
        epoch_metrics['g_l1'] += float(G_l1.detach().cpu())
        n_batches += 1

        if idx % 5 == 0:
            loop.set_postfix(
                d_loss=float(D_loss.detach().cpu()),
                g_loss=float(G_loss.detach().cpu()),
                g_adv=float(G_adv.detach().cpu()),
                g_l1=float(G_l1.detach().cpu()),
            )

    # Average metrics over epoch
    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches

    dt = time.time() - t0
    return dt, epoch_metrics


# ============================================================
# Main
# ============================================================

def main():
    # Set random seeds for reproducibility
    if hasattr(config, 'RANDOM_SEED'):
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

    use_amp = (device.type == "cuda")
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    pin_memory = (device.type == "cuda")
    non_blocking = (device.type == "cuda")

    # FIXED: Using standard discriminator instead of pairwise
    disc = SpectralDiscriminator1D(in_channels=1, use_bn=False).to(device)

    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=config.BASE_FEATURES,
        embed_dim=config.EMBED_DIM,
    ).to(device)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # LSGAN losses
    gan_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        if os.path.isfile(config.CHECKPOINT_GEN):
            load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE, device=str(device))
            print(f"Loaded generator checkpoint: {config.CHECKPOINT_GEN}")
        if os.path.isfile(config.CHECKPOINT_DISC):
            load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE, device=str(device))
            print(f"Loaded discriminator checkpoint: {config.CHECKPOINT_DISC}")

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
    )

    persistent_workers = (config.NUM_WORKERS > 0)

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
        root_dir=config.TRAIN_IMG_DIR if config.VAL_IMG_DIR is None else config.VAL_IMG_DIR,
        species=config.SPECIES_FILTER,
        stage=config.STAGE_FILTER,
        patch_h=config.PATCH_H,
        patch_w=config.PATCH_W,
        stride_h=config.PATCH_H,
        stride_w=config.PATCH_W,
        black_thr=config.BLACK_THR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=patch_collate_fn,
        pin_memory=pin_memory,
    )

    ensure_dir("evaluation")
    
    # Clear previous log
    log_file = 'training_log.json'
    if os.path.exists(log_file):
        os.remove(log_file)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"L1 Lambda: {config.L1_LAMBDA}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"{'='*60}\n")

    for epoch in range(config.NUM_EPOCHS):
        dt, train_metrics = train_one_epoch(
            disc=disc,
            gen=gen,
            loader=train_loader,
            opt_disc=opt_disc,
            opt_gen=opt_gen,
            gan_loss=gan_loss,
            l1_loss=l1_loss,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            scaler_d=scaler_d,
            scaler_g=scaler_g,
            non_blocking=non_blocking,
        )

        # Validation
        val_metrics = validate(gen, val_loader, device, use_amp)
        
        # Combine metrics
        all_metrics = {**train_metrics, **val_metrics}
        
        # Log metrics
        log_metrics(epoch, all_metrics, log_file)
        
        # Print summary
        print(f"\nEpoch {epoch} ({dt:.1f}s):")
        print(f"  Train - D: {train_metrics['d_loss']:.4f}, G: {train_metrics['g_loss']:.4f} "
              f"(adv: {train_metrics['g_adv']:.4f}, L1: {train_metrics['g_l1']:.4f})")
        print(f"  Val   - L1: {val_metrics['val_l1']:.4f}, RMSE: {val_metrics['val_rmse']:.4f}, "
              f"SAM: {val_metrics['val_sam_deg']:.2f}°")

        # Save validation plot
        gen.eval()
        with torch.no_grad():
            for batch_bands, y_real in val_loader:
                batch_bands = move_batch_bands_to_device(batch_bands, device, non_blocking=non_blocking)
                y_real = y_real.to(device, non_blocking=non_blocking)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    y_fake, p_params = gen.forward_batch_list(batch_bands)

                plot_and_save_spectra(y_real, y_fake, epoch, out_dir="evaluation", title_prefix="Validation")
                break

        if config.SAVE_MODEL and (epoch % 5 == 0 or epoch == config.NUM_EPOCHS - 1):
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            print(f"  Saved checkpoints")

    print("\n" + "="*60)
    print("Training finished.")
    print("="*60)


if __name__ == "__main__":
    main()
