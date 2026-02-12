import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MultiSpectralCSVPatchDataset, patch_collate_fn
from generator_model import MultiSpectralPatchToProspectGenerator
from discriminator_model import SpectralPatchDiscriminator1D


import matplotlib.pyplot as plt


# ============================================================
# Small utilities (no external utils.py needed)
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


def move_batch_bands_to_device(batch_bands, device):
    # batch_bands: dict band -> list of tensors [N_i, 1, ph, pw]
    out = {}
    for band, lst in batch_bands.items():
        out[band] = [t.to(device) for t in lst]
    return out


def plot_and_save_spectra(y_real, y_fake, epoch, out_dir="evaluation", title_prefix=""):
    """
    Saves a PNG plot for the first sample in the batch:
      - real spectrum
      - fake spectrum
    """
    ensure_dir(out_dir)

    r = y_real[0].detach().cpu().numpy().reshape(-1)
    f = y_fake[0].detach().cpu().numpy().reshape(-1)

    plt.figure()
    plt.plot(r, label="real")
    plt.plot(f, label="fake")
    plt.xlabel("Wavelength index")
    plt.ylabel("Reflectance")
    t = "Epoch " + str(epoch)
    if title_prefix:
        t = title_prefix + " " + t
    plt.title(t)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "spectra_epoch_" + str(epoch).zfill(4) + ".png")
    plt.savefig(path, dpi=150)
    plt.close()


def save_some_spectra(y_real, y_fake, p_params, epoch, out_dir="evaluation"):
    """
    Saves a tiny snapshot for debugging:
      - real spectrum
      - fake spectrum
      - predicted PROSPECT params
    """
    ensure_dir(out_dir)
    # save first item in batch
    r = y_real[0].detach().cpu()
    f = y_fake[0].detach().cpu()
    p = p_params[0].detach().cpu()
    path = os.path.join(out_dir, "epoch_" + str(epoch).zfill(4) + ".pt")
    torch.save({"real": r, "fake": f, "pParams": p}, path)


def get_autocast_and_scaler(device):
    """
    CPU training:
      - autocast is optional and often not faster unless using bfloat16 kernels.
      - GradScaler is usually for CUDA; PyTorch has torch.cpu.amp.GradScaler,
        but you can also just disable AMP on CPU.

    We keep a safe default:
      - enable_autocast = False
      - scaler = None
    """
    enable_autocast = False
    scaler = None

    # If you want to experiment with CPU autocast (bfloat16), set this True:
    # enable_autocast = True
    # scaler = torch.cpu.amp.GradScaler()

    return enable_autocast, scaler


# ============================================================
# Training step
# ============================================================

def train_one_epoch(disc, gen, loader, opt_disc, opt_gen, bce, l1_loss, device, epoch):
    disc.train()
    gen.train()

    loop = tqdm(loader, leave=True)
    t0 = time.time()

    for idx, (batch_bands, y_real) in enumerate(loop):
        # Move to device
        batch_bands = move_batch_bands_to_device(batch_bands, device)
        y_real = y_real.to(device)  # [B, L]

        # ------------------------------------------------------------
        # Forward generator
        # gen.forward_batch_list expects dict band -> list[tensor]
        # returns y_fake [B,L], pParams [B,7]
        # ------------------------------------------------------------
        y_fake, p_params = gen.forward_batch_list(batch_bands)

        # ============================================================
        # (1) Train Discriminator
        # We use a pix2pix-like pairwise discriminator, but our "condition"
        # is the real spectrum itself:
        #   D(real, real) should be 1
        #   D(real, fake) should be 0
        #
        # This makes D learn "does candidate match the real spectrum distribution
        # and shape relative to the reference". It is closer to a learned
        # similarity / conditional realism check than classic pix2pix.
        #
        # If you prefer unconditional GAN, switch to SpectralDiscriminator1D
        # and feed D(real) vs D(fake).
        # ============================================================

        D_real = disc(y_real, y_real)
        D_fake = disc(y_real, y_fake.detach())

        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = 0.5 * (D_real_loss + D_fake_loss)

        opt_disc.zero_grad(set_to_none=True)
        D_loss.backward()
        opt_disc.step()

        # ============================================================
        # (2) Train Generator
        # Loss = adversarial + L1(spectrum)
        # L1 is the main supervision to match measured spectra.
        # GAN term pushes output spectra to look "realistic" across wavelengths
        # (smoothness / plausible shapes), depending on D capacity.
        # ============================================================

        D_fake_for_G = disc(y_real, y_fake)
        G_adv = bce(D_fake_for_G, torch.ones_like(D_fake_for_G))
        G_l1 = l1_loss(y_fake, y_real) * config.L1_LAMBDA
        G_loss = G_adv + G_l1

        opt_gen.zero_grad(set_to_none=True)
        G_loss.backward()
        opt_gen.step()

        if idx % 5 == 0:
            loop.set_postfix(
                d_loss=float(D_loss.detach().cpu()),
                g_loss=float(G_loss.detach().cpu()),
                g_adv=float(G_adv.detach().cpu()),
                g_l1=float(G_l1.detach().cpu()),
            )

    dt = time.time() - t0
    return dt


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device(config.DEVICE)

    # ----------------------------
    # Models
    # ----------------------------
    disc = SpectralPatchDiscriminator1D(in_channels=1, use_bn=False).to(device)

    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=config.BASE_FEATURES,
        embed_dim=config.EMBED_DIM,
    ).to(device)

    # ----------------------------
    # Optimizers and losses
    # ----------------------------
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # ----------------------------
    # Checkpoints
    # ----------------------------
    if config.LOAD_MODEL:
        if os.path.isfile(config.CHECKPOINT_GEN):
            load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE, device=str(device))
        if os.path.isfile(config.CHECKPOINT_DISC):
            load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE, device=str(device))

    # ----------------------------
    # Datasets / Loaders (CSV-based)
    # ----------------------------
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=patch_collate_fn,
        pin_memory=False,
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
        batch_size=min(2, config.BATCH_SIZE),
        shuffle=False,
        num_workers=0,
        collate_fn=patch_collate_fn,
    )

    # ----------------------------
    # Train
    # ----------------------------
    ensure_dir("evaluation")

    for epoch in range(config.NUM_EPOCHS):
        dt = train_one_epoch(disc, gen, train_loader, opt_disc, opt_gen, bce, l1_loss, device, epoch)

        # quick val snapshot
        gen.eval()
        with torch.no_grad():
            for batch_bands, y_real in val_loader:
                batch_bands = move_batch_bands_to_device(batch_bands, device)
                y_real = y_real.to(device)
                y_fake, p_params = gen.forward_batch_list(batch_bands)
                plot_and_save_spectra(y_real, y_fake, epoch, out_dir="evaluation", title_prefix="Validation")

                break

        if config.SAVE_MODEL and (epoch % 5 == 0):
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        print("Epoch", epoch, "done in", round(dt, 2), "sec")

    print("Training finished.")


if __name__ == "__main__":
    main()
