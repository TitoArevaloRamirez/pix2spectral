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
from discriminator_model import SpectralDiscriminator1D

# Import physics-based losses
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

    path = os.path.join(out_dir, f"spectra_epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_parameters(params_batch, epoch, out_dir="evaluation"):
    """Plot parameter distributions across batch."""
    ensure_dir(out_dir)
    
    param_names = ['Nleaf', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'Ant']
    params_np = params_batch.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        axes[i].hist(params_np[:, i], bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{name} Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[7])
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"parameters_epoch_{epoch:04d}.png")
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
# Validation with comprehensive metrics
# ============================================================

def validate(gen, val_loader, device, use_amp, physics_loss_fn, sam_fn):
    """
    Compute validation metrics with physics-informed losses.
    """
    gen.eval()
    
    metrics = {
        'val_l1': 0,
        'val_rmse': 0,
        'val_sam_rad': 0,
        'val_sam_deg': 0,
        'val_physics_total': 0,
        'val_spectral_l1': 0,
        'val_weighted_l1': 0,
        'val_param_penalty': 0,
        'val_smoothness': 0,
        'val_derivative': 0,
    }
    n_samples = 0
    
    with torch.no_grad():
        for batch_bands, y_real in val_loader:
            batch_bands = move_batch_bands_to_device(batch_bands, device)
            y_real = y_real.to(device)
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                y_fake, p_params = gen.forward_batch_list(batch_bands)
                
                # Basic metrics
                metrics['val_l1'] += torch.mean(torch.abs(y_fake - y_real)).item()
                metrics['val_rmse'] += torch.sqrt(torch.mean((y_fake - y_real)**2)).item()
                
                # SAM
                sam = sam_fn(y_fake, y_real)
                metrics['val_sam_rad'] += sam.item()
                
                # Physics-informed losses
                _, physics_losses = physics_loss_fn(y_fake, y_real, p_params)
                metrics['val_physics_total'] += physics_losses['total'].item()
                metrics['val_spectral_l1'] += physics_losses['spectral_l1'].item()
                metrics['val_weighted_l1'] += physics_losses['weighted_l1'].item()
                metrics['val_param_penalty'] += physics_losses['param_penalty'].item()
                metrics['val_smoothness'] += physics_losses['smoothness'].item()
                metrics['val_derivative'] += physics_losses['derivative'].item()
            
            n_samples += 1
    
    # Average all metrics
    for key in metrics:
        metrics[key] /= n_samples
    
    metrics['val_sam_deg'] = metrics['val_sam_rad'] * 180 / np.pi
    
    return metrics


# ============================================================
# Training step with PHYSICS-INFORMED LOSSES
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
        'd_loss': 0,
        'g_loss': 0,
        'g_adv': 0,
        'g_physics': 0,
        'g_spectral_l1': 0,
        'g_weighted_l1': 0,
        'g_param_penalty': 0,
        'g_smoothness': 0,
        'g_derivative': 0,
    }
    n_batches = 0

    for idx, (batch_bands, y_real) in enumerate(loop):
        batch_bands = move_batch_bands_to_device(batch_bands, device, non_blocking=non_blocking)
        y_real = y_real.to(device, non_blocking=non_blocking)

        # ============================================================
        # Forward generator (physics inside generator)
        # ============================================================
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_fake, p_params = gen.forward_batch_list(batch_bands)

        # ============================================================
        # Train Discriminator
        # ============================================================
        with torch.amp.autocast("cuda", enabled=use_amp):
            D_real = disc(y_real)
            D_fake = disc(y_fake.detach())
            D_loss = adv_loss_fn.discriminator_loss(D_real, D_fake)

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

        # ============================================================
        # Train Generator with PHYSICS-INFORMED LOSSES
        # ============================================================
        with torch.amp.autocast("cuda", enabled=use_amp):
            # Adversarial loss (fool discriminator)
            D_fake_for_G = disc(y_fake)
            G_adv = adv_loss_fn.generator_loss(D_fake_for_G)
            
            # Physics-informed reconstruction loss
            G_physics, physics_components = physics_loss_fn(y_fake, y_real, p_params)
            
            # Total generator loss
            G_loss = G_adv + l1_lambda * G_physics

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

        # Track metrics
        epoch_metrics['d_loss'] += float(D_loss.detach().cpu())
        epoch_metrics['g_loss'] += float(G_loss.detach().cpu())
        epoch_metrics['g_adv'] += float(G_adv.detach().cpu())
        epoch_metrics['g_physics'] += float(G_physics.detach().cpu())
        epoch_metrics['g_spectral_l1'] += float(physics_components['spectral_l1'].detach().cpu())
        epoch_metrics['g_weighted_l1'] += float(physics_components['weighted_l1'].detach().cpu())
        epoch_metrics['g_param_penalty'] += float(physics_components['param_penalty'].detach().cpu())
        epoch_metrics['g_smoothness'] += float(physics_components['smoothness'].detach().cpu())
        epoch_metrics['g_derivative'] += float(physics_components['derivative'].detach().cpu())
        n_batches += 1

        if idx % 5 == 0:
            loop.set_postfix(
                D=f"{float(D_loss.detach().cpu()):.3f}",
                G=f"{float(G_loss.detach().cpu()):.3f}",
                adv=f"{float(G_adv.detach().cpu()):.3f}",
                phy=f"{float(G_physics.detach().cpu()):.4f}",
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

    # ============================================================
    # Initialize models
    # ============================================================
    disc = SpectralDiscriminator1D(in_channels=1, use_bn=False).to(device)

    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=config.BASE_FEATURES,
        embed_dim=config.EMBED_DIM,
    ).to(device)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # ============================================================
    # Initialize PHYSICS-INFORMED LOSS FUNCTIONS
    # ============================================================
    
    # Create wavelength weights (emphasize red edge, NIR, water bands)
    wavelength_weights = create_wavelength_weights(
        num_wavelengths=2101,  # PROSPECT-D output length
        start_wl=400,
        end_wl=2500
    )
    
    # Physics-informed loss with all components
    physics_loss_fn = PhysicsInformedLoss(
        param_bounds=gen.physics.bounds,
        wavelength_weights=wavelength_weights,
        lambda_spectral=1.0,        # Basic spectral L1
        lambda_weighted=0.5,         # Wavelength-weighted L1
        lambda_param_penalty=0.1,    # Keep params in bounds
        lambda_smoothness=0.01,      # Natural spectral smoothness
        lambda_derivative=0.01,      # Match spectral derivatives
    ).to(device)
    
    # Adversarial loss (LSGAN by default)
    adv_loss_fn = AdversarialLoss(loss_type='lsgan')
    
    # Spectral Angle Mapper for validation
    sam_fn = SpectralAngleMapper()

    # Load checkpoints if requested
    if config.LOAD_MODEL:
        if os.path.isfile(config.CHECKPOINT_GEN):
            load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE, device=str(device))
            print(f"✅ Loaded generator checkpoint: {config.CHECKPOINT_GEN}")
        if os.path.isfile(config.CHECKPOINT_DISC):
            load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE, device=str(device))
            print(f"✅ Loaded discriminator checkpoint: {config.CHECKPOINT_DISC}")

    # ============================================================
    # Prepare datasets
    # ============================================================
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

    # ============================================================
    # Print configuration
    # ============================================================
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION WITH PHYSICS-INFORMED LOSSES")
    print("="*70)
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"L1 Lambda: {config.L1_LAMBDA}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("\nPhysics Loss Components:")
    print(f"  λ_spectral:       {physics_loss_fn.lambda_spectral}")
    print(f"  λ_weighted:       {physics_loss_fn.lambda_weighted}")
    print(f"  λ_param_penalty:  {physics_loss_fn.lambda_param_penalty}")
    print(f"  λ_smoothness:     {physics_loss_fn.lambda_smoothness}")
    print(f"  λ_derivative:     {physics_loss_fn.lambda_derivative}")
    print("="*70 + "\n")

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(config.NUM_EPOCHS):
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

        # Validation
        val_metrics = validate(gen, val_loader, device, use_amp, physics_loss_fn, sam_fn)
        
        # Combine metrics
        all_metrics = {**train_metrics, **val_metrics}
        
        # Log metrics
        log_metrics(epoch, all_metrics, log_file)
        
        # Print detailed summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} ({dt:.1f}s)")
        print(f"{'='*70}")
        print(f"Train:")
        print(f"  D_loss:       {train_metrics['d_loss']:.4f}")
        print(f"  G_loss:       {train_metrics['g_loss']:.4f}")
        print(f"    └─ adv:     {train_metrics['g_adv']:.4f}")
        print(f"    └─ physics: {train_metrics['g_physics']:.4f}")
        print(f"       ├─ spectral_l1:    {train_metrics['g_spectral_l1']:.6f}")
        print(f"       ├─ weighted_l1:    {train_metrics['g_weighted_l1']:.6f}")
        print(f"       ├─ param_penalty:  {train_metrics['g_param_penalty']:.6f}")
        print(f"       ├─ smoothness:     {train_metrics['g_smoothness']:.6f}")
        print(f"       └─ derivative:     {train_metrics['g_derivative']:.6f}")
        print(f"Validation:")
        print(f"  L1:   {val_metrics['val_l1']:.6f}")
        print(f"  RMSE: {val_metrics['val_rmse']:.6f}")
        print(f"  SAM:  {val_metrics['val_sam_deg']:.2f}° ({val_metrics['val_sam_rad']:.4f} rad)")
        print(f"  Param penalty: {val_metrics['val_param_penalty']:.6f}")

        # Save validation visualizations
        gen.eval()
        with torch.no_grad():
            for batch_bands, y_real in val_loader:
                batch_bands = move_batch_bands_to_device(batch_bands, device, non_blocking=non_blocking)
                y_real = y_real.to(device, non_blocking=non_blocking)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    y_fake, p_params = gen.forward_batch_list(batch_bands)

                # Plot spectra
                plot_and_save_spectra(y_real, y_fake, epoch, out_dir="evaluation", title_prefix="Val")
                
                # Plot parameter distributions (every 10 epochs)
                if epoch % 10 == 0 and y_real.shape[0] > 1:
                    plot_parameters(p_params, epoch, out_dir="evaluation")
                
                break

        # Save checkpoints
        if config.SAVE_MODEL and (epoch % 5 == 0 or epoch == config.NUM_EPOCHS - 1):
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            print(f"  ✅ Saved checkpoints")

    print("\n" + "="*70)
    print("✅ TRAINING FINISHED!")
    print("="*70)


if __name__ == "__main__":
    main()
