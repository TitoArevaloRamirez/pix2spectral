"""
Improved discriminator model for pix2spectral.

Main additions relative to the original discriminator_model.py:
  - make_wavelengths(cfg) helper compatible with dataclass-style and module-style config
  - optional wavelength-coordinate channel
  - optional spectral normalization on Conv1d layers
  - segmented discriminator using the same spectral intervals as the segmented generator
  - global + segmented discriminator mode
  - backward-compatible SpectralDiscriminator1D and SpectralPatchDiscriminator1D classes
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_SPECTRAL_SEGMENTS = [
    (400.0, 900.0),
    (900.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 2500.0),
]


# ============================================================
# Wavelength utilities
# ============================================================


def _cfg_get(cfg, lower_name: str, upper_name: str, default):
    """Read either cfg.lower_name or cfg.UPPER_NAME from a config object/module."""
    if hasattr(cfg, lower_name):
        return getattr(cfg, lower_name)
    if hasattr(cfg, upper_name):
        return getattr(cfg, upper_name)
    return default


def make_wavelengths(cfg) -> np.ndarray:
    """
    Create wavelength grid from a config object/module.

    Supports both dataclass-style fields:
        cfg.wavelength_min
        cfg.wavelength_max
        cfg.wavelength_count

    and module-style constants:
        cfg.WAVELENGTH_MIN
        cfg.WAVELENGTH_MAX
        cfg.WAVELENGTH_COUNT
    """
    return np.linspace(
        _cfg_get(cfg, "wavelength_min", "WAVELENGTH_MIN", 400.0),
        _cfg_get(cfg, "wavelength_max", "WAVELENGTH_MAX", 2500.0),
        int(_cfg_get(cfg, "wavelength_count", "WAVELENGTH_COUNT", 2101)),
        dtype=np.float64,
    )


def make_wavelengths_from_values(
    wavelength_min: float = 400.0,
    wavelength_max: float = 2500.0,
    wavelength_count: int = 2101,
) -> np.ndarray:
    return np.linspace(
        float(wavelength_min),
        float(wavelength_max),
        int(wavelength_count),
        dtype=np.float64,
    )


def spectral_segment_indices(
    wavelengths: np.ndarray,
    spectral_segments: Sequence[Tuple[float, float]],
) -> List[np.ndarray]:
    """
    Convert spectral intervals into index arrays.

    Intervals are [lo, hi) except the last interval, which is [lo, hi].
    This avoids duplicate boundary wavelengths.
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    indices: List[np.ndarray] = []

    for i, (lo, hi) in enumerate(spectral_segments):
        lo = float(lo)
        hi = float(hi)

        if i == len(spectral_segments) - 1:
            mask = (wavelengths >= lo) & (wavelengths <= hi)
        else:
            mask = (wavelengths >= lo) & (wavelengths < hi)

        idx = np.where(mask)[0].astype(np.int64)
        if idx.size == 0:
            raise ValueError(f"Segment ({lo}, {hi}) produced zero indices.")
        indices.append(idx)

    all_idx = np.concatenate(indices, axis=0)
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("Segments produced duplicate wavelength indices.")

    return indices


# ============================================================
# Building blocks
# ============================================================


def maybe_spectral_norm(module: nn.Module, enabled: bool = False) -> nn.Module:
    if enabled:
        return nn.utils.spectral_norm(module)
    return module


class CNNBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        use_bn=False,
        use_spectral_norm=False,
    ):
        super().__init__()

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=stride,
            padding=1,
            padding_mode="reflect",
            bias=not use_bn,
        )

        layers = [maybe_spectral_norm(conv, enabled=use_spectral_norm)]

        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def _as_b1l(s: torch.Tensor) -> torch.Tensor:
    """Accept [B,L] or [B,1,L] and return [B,1,L]."""
    if s.dim() == 2:
        return s.unsqueeze(1)
    if s.dim() == 3:
        return s
    raise ValueError(f"Expected spectrum [B,L] or [B,1,L], got {tuple(s.shape)}")


def _wavelength_channel(
    wavelengths: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create normalized wavelength channel [B,1,L] in [-1, 1].
    """
    wl = wavelengths.to(device=device, dtype=dtype).flatten()
    if wl.numel() < 2:
        coord = torch.zeros_like(wl)
    else:
        coord = 2.0 * (wl - wl.min()) / (wl.max() - wl.min() + 1e-12) - 1.0
    return coord.view(1, 1, -1).expand(batch_size, 1, -1)


# ============================================================
# Backward-compatible discriminators with optional wavelength channel
# ============================================================


class SpectralPatchDiscriminator1D(nn.Module):
    """
    Pairwise 1D PatchGAN-style discriminator over wavelength.

    Inputs:
        x : real/reference spectrum [B,L] or [B,1,L]
        y : generated spectrum      [B,L] or [B,1,L]

    Output:
        patch logits [B,1,L']

    Optional wavelength channel:
        If use_wavelength_channel=True, a normalized wavelength coordinate is
        appended as one extra channel.
    """

    def __init__(
        self,
        in_channels=1,
        features=(64, 128, 256, 512),
        use_bn=False,
        use_wavelength_channel=False,
        wavelengths: Optional[np.ndarray] = None,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        use_spectral_norm=False,
    ):
        super().__init__()

        self.use_wavelength_channel = bool(use_wavelength_channel)

        if wavelengths is None:
            wavelengths = make_wavelengths_from_values(
                wavelength_min,
                wavelength_max,
                wavelength_count,
            )

        self.register_buffer(
            "wavelengths",
            torch.as_tensor(wavelengths, dtype=torch.float32),
            persistent=False,
        )

        input_channels = int(in_channels) * 2
        if self.use_wavelength_channel:
            input_channels += 1

        conv0 = nn.Conv1d(
            input_channels,
            features[0],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )

        self.initial = nn.Sequential(
            maybe_spectral_norm(conv0, enabled=use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        c_in = features[0]

        for f in features[1:]:
            layers.append(
                CNNBlock1D(
                    c_in,
                    f,
                    stride=1 if f == features[-1] else 2,
                    use_bn=use_bn,
                    use_spectral_norm=use_spectral_norm,
                )
            )
            c_in = f

        conv_last = nn.Conv1d(
            c_in,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )
        layers.append(maybe_spectral_norm(conv_last, enabled=use_spectral_norm))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = _as_b1l(x)
        y = _as_b1l(y)

        z = torch.cat([x, y], dim=1)

        if self.use_wavelength_channel:
            wl = _wavelength_channel(
                self.wavelengths,
                batch_size=z.shape[0],
                device=z.device,
                dtype=z.dtype,
            )
            if wl.shape[-1] != z.shape[-1]:
                raise ValueError(
                    f"Wavelength length {wl.shape[-1]} does not match spectrum "
                    f"length {z.shape[-1]}."
                )
            z = torch.cat([z, wl], dim=1)

        z = self.initial(z)
        return self.model(z)


class SpectralDiscriminator1D(nn.Module):
    """
    Single-input 1D PatchGAN discriminator.

    Input:
        s: spectrum [B,L] or [B,1,L]

    Output:
        patch logits [B,1,L']

    Optional wavelength channel:
        If use_wavelength_channel=True, the discriminator receives:
            [reflectance, normalized_wavelength]
    """

    def __init__(
        self,
        in_channels=1,
        features=(64, 128, 256, 512),
        use_bn=False,
        use_wavelength_channel=False,
        wavelengths: Optional[np.ndarray] = None,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        use_spectral_norm=False,
    ):
        super().__init__()

        self.use_wavelength_channel = bool(use_wavelength_channel)

        if wavelengths is None:
            wavelengths = make_wavelengths_from_values(
                wavelength_min,
                wavelength_max,
                wavelength_count,
            )

        self.register_buffer(
            "wavelengths",
            torch.as_tensor(wavelengths, dtype=torch.float32),
            persistent=False,
        )

        input_channels = int(in_channels)
        if self.use_wavelength_channel:
            input_channels += 1

        conv0 = nn.Conv1d(
            input_channels,
            features[0],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )

        self.initial = nn.Sequential(
            maybe_spectral_norm(conv0, enabled=use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        c_in = features[0]

        for f in features[1:]:
            layers.append(
                CNNBlock1D(
                    c_in,
                    f,
                    stride=1 if f == features[-1] else 2,
                    use_bn=use_bn,
                    use_spectral_norm=use_spectral_norm,
                )
            )
            c_in = f

        conv_last = nn.Conv1d(
            c_in,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )
        layers.append(maybe_spectral_norm(conv_last, enabled=use_spectral_norm))
        self.model = nn.Sequential(*layers)

    def forward(self, s):
        s = _as_b1l(s)

        if self.use_wavelength_channel:
            wl = _wavelength_channel(
                self.wavelengths,
                batch_size=s.shape[0],
                device=s.device,
                dtype=s.dtype,
            )
            if wl.shape[-1] != s.shape[-1]:
                raise ValueError(
                    f"Wavelength length {wl.shape[-1]} does not match spectrum "
                    f"length {s.shape[-1]}."
                )
            s = torch.cat([s, wl], dim=1)

        s = self.initial(s)
        return self.model(s)


# ============================================================
# Segmented discriminator
# ============================================================


class SegmentedSpectralDiscriminator1D(nn.Module):
    """
    Global + segmented spectral discriminator.

    Modes:
        "global":
            one discriminator over full spectrum

        "segmented":
            one local discriminator per spectral interval

        "global_plus_segmented":
            both full-spectrum and local interval discriminators

    The forward() method returns a single logits tensor [B,1,K] by
    concatenating all selected discriminator outputs along the wavelength/logit
    dimension. This keeps compatibility with existing adversarial losses that
    expect tensor logits.
    """

    def __init__(
        self,
        in_channels=1,
        features=(64, 128, 256, 512),
        use_bn=False,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        wavelengths: Optional[np.ndarray] = None,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        mode: str = "global_plus_segmented",
        use_wavelength_channel: bool = True,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        mode = str(mode).lower()
        valid_modes = ["global", "segmented", "global_plus_segmented"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        self.mode = mode
        self.use_global = mode in ["global", "global_plus_segmented"]
        self.use_segmented = mode in ["segmented", "global_plus_segmented"]

        if wavelengths is None:
            wavelengths = make_wavelengths_from_values(
                wavelength_min,
                wavelength_max,
                wavelength_count,
            )
        wavelengths = np.asarray(wavelengths, dtype=np.float64)

        if spectral_segments is None:
            spectral_segments = DEFAULT_SPECTRAL_SEGMENTS

        self.spectral_segments = [
            (float(lo), float(hi)) for lo, hi in spectral_segments
        ]

        self.register_buffer(
            "wavelengths",
            torch.as_tensor(wavelengths, dtype=torch.float32),
            persistent=False,
        )

        idx_np = spectral_segment_indices(wavelengths, self.spectral_segments)
        self._segment_buffer_names: List[str] = []

        for i, idx in enumerate(idx_np):
            name = f"_segment_idx_{i}"
            self.register_buffer(
                name,
                torch.as_tensor(idx, dtype=torch.long),
                persistent=False,
            )
            self._segment_buffer_names.append(name)

        self.global_disc: Optional[SpectralDiscriminator1D] = None
        if self.use_global:
            self.global_disc = SpectralDiscriminator1D(
                in_channels=in_channels,
                features=features,
                use_bn=use_bn,
                use_wavelength_channel=use_wavelength_channel,
                wavelengths=wavelengths,
                use_spectral_norm=use_spectral_norm,
            )

        self.segment_discs = nn.ModuleList()
        if self.use_segmented:
            for idx in idx_np:
                seg_wl = wavelengths[idx]
                self.segment_discs.append(
                    SpectralDiscriminator1D(
                        in_channels=in_channels,
                        features=features,
                        use_bn=use_bn,
                        use_wavelength_channel=use_wavelength_channel,
                        wavelengths=seg_wl,
                        use_spectral_norm=use_spectral_norm,
                    )
                )

    def segment_indices(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self._segment_buffer_names]

    def forward(self, s):
        s = _as_b1l(s)
        logits = []

        if self.global_disc is not None:
            logits.append(self.global_disc(s))

        if self.use_segmented:
            for disc, idx in zip(self.segment_discs, self.segment_indices()):
                idx = idx.to(s.device)
                s_seg = s.index_select(dim=-1, index=idx)
                logits.append(disc(s_seg))

        if len(logits) == 0:
            raise RuntimeError("No discriminator branches are enabled.")

        return torch.cat(logits, dim=-1)


# ============================================================
# Example usage
# ============================================================


if __name__ == "__main__":
    B = 2
    L = 2101

    real = torch.randn(B, L)
    fake = torch.randn(B, L)

    D_single = SpectralDiscriminator1D(
        in_channels=1,
        use_bn=False,
        use_wavelength_channel=True,
        use_spectral_norm=True,
    )
    logits_real = D_single(real)
    logits_fake = D_single(fake)
    print("single logits:", logits_real.shape, logits_fake.shape)

    D_segmented = SegmentedSpectralDiscriminator1D(
        wavelength_min=400.0,
        wavelength_max=2500.0,
        wavelength_count=2101,
        spectral_segments=DEFAULT_SPECTRAL_SEGMENTS,
        mode="global_plus_segmented",
        use_wavelength_channel=True,
        use_spectral_norm=True,
    )
    logits_real = D_segmented(real)
    logits_fake = D_segmented(fake)
    print("segmented logits:", logits_real.shape, logits_fake.shape)

    D_pair = SpectralPatchDiscriminator1D(
        in_channels=1,
        use_bn=False,
        use_wavelength_channel=True,
        use_spectral_norm=True,
    )
    logits_pair = D_pair(real, fake)
    print("pair logits:", logits_pair.shape)
