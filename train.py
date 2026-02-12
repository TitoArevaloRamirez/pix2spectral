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

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
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


def move_batch_bands_to_device(batch_bands, device, non_blocking=False):
    out = {}
    for band, lst in batch_bands.items():
        out[band] = [t.to(device, non_blocking=non_blocking) for t in lst]
    return out


def plot_and_save_spectra(y_real, y_fake, epoch, out_dir="evaluation", title_prefix=""):
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


# ============================================================
# Training step
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

    for idx, (batch_bands, y_real) in enumerate(loop):
        batch_bands = move_batch_bands_to_device(batch_bands, device, non_blocking=non_blocking)
        y_real = y_real.to(device, non_blocking=non_blocking)  # [B, L]

        # Forward generator (physics inside generator)
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_fake, p_params = gen.forward_batch_list(batch_bands)

        # ============================================================
        # LSGAN Discriminator
        #   real score -> 1
        #   fake score -> 0
        # ============================================================
        with torch.amp.autocast("cuda", enabled=use_amp):
            D_real = disc(y_real, y_real)
            D_fake = disc(y_real, y_fake.detach())

            D_real_loss = gan_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = gan_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = 0.5 * (D_real_loss + D_fake_loss)

        opt_disc.zero_grad(set_to_none=True)
        if use_amp:
            scaler_d.scale(D_loss).backward()
            scaler_d.step(opt_disc)
            scaler_d.update()
        else:
            D_loss.backward()
            opt_disc.step()

        # ============================================================
        # LSGAN Generator + spectral L1
        #   wants fake score -> 1
        # ============================================================
        with torch.amp.autocast("cuda", enabled=use_amp):
            D_fake_for_G = disc(y_real, y_fake)
            G_adv = gan_loss(D_fake_for_G, torch.ones_like(D_fake_for_G))

            G_l1 = l1_loss(y_fake, y_real) * config.L1_LAMBDA
            G_loss = G_adv + G_l1

        opt_gen.zero_grad(set_to_none=True)
        if use_amp:
            scaler_g.scale(G_loss).backward()
            scaler_g.step(opt_gen)
            scaler_g.update()
        else:
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

    disc = SpectralPatchDiscriminator1D(in_channels=1, use_bn=False).to(device)

    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=config.BASE_FEATURES,
        embed_dim=config.EMBED_DIM,
    ).to(device)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # ------------------------------------------------------------
    # LSGAN losses
    # ------------------------------------------------------------
    gan_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        if os.path.isfile(config.CHECKPOINT_GEN):
            load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE, device=str(device))
        if os.path.isfile(config.CHECKPOINT_DISC):
            load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE, device=str(device))

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

    for epoch in range(config.NUM_EPOCHS):
        dt = train_one_epoch(
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

        gen.eval()
        with torch.no_grad():
            for batch_bands, y_real in val_loader:
                batch_bands = move_batch_bands_to_device(batch_bands, device, non_blocking=non_blocking)
                y_real = y_real.to(device, non_blocking=non_blocking)

                with torch.amp.autocast("cuda", enabled=use_amp):
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
