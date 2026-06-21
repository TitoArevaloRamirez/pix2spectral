"""
Conditional and unconditional discriminator models for pix2spectral.

This file keeps the original unconditional spectral discriminator interface and
adds a true image-conditioned cGAN discriminator:

    D(y, x)

where:
    y = real or generated spectrum, shape [B,L] or [B,1,L]
    x = multispectral image condition, batch_bands dict of patch lists

The conditional discriminator uses a projection-discriminator style objective:

    D(y, x) = D_uncond(y) + <phi(y), psi(x)>

This lets the discriminator judge whether a spectrum is both realistic and
compatible with the multispectral image patches that condition the generator.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from generator_model import build_patch_encoder, build_pool


DEFAULT_SPECTRAL_SEGMENTS = [
    (400.0, 700.0),
    (700.0, 800.0),
    (800.0, 1400.0),
    (1400.0, 2500.0),
]

DEFAULT_BANDS = ["blue", "green", "red", "nir", "red_edge"]


# ============================================================
# Wavelength utilities
# ============================================================


def _cfg_get(cfg, lower_name: str, upper_name: str, default):
    if hasattr(cfg, lower_name):
        return getattr(cfg, lower_name)
    if hasattr(cfg, upper_name):
        return getattr(cfg, upper_name)
    return default


def make_wavelengths(cfg) -> np.ndarray:
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
    This avoids duplicated boundary wavelengths.
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
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_bn: bool = False,
        use_spectral_norm: bool = False,
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
        layers: List[nn.Module] = [maybe_spectral_norm(conv, enabled=use_spectral_norm)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _as_b1l(s: torch.Tensor) -> torch.Tensor:
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
    wl = wavelengths.to(device=device, dtype=dtype).flatten()
    if wl.numel() < 2:
        coord = torch.zeros_like(wl)
    else:
        coord = 2.0 * (wl - wl.min()) / (wl.max() - wl.min() + 1e-12) - 1.0
    return coord.view(1, 1, -1).expand(batch_size, 1, -1)


# ============================================================
# Spectrum-only discriminators
# ============================================================


class SpectralDiscriminator1D(nn.Module):
    """
    Unconditional single-input 1D PatchGAN discriminator.

    Input:
        s: spectrum [B,L] or [B,1,L]

    Output:
        patch logits [B,1,K]
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        use_bn: bool = False,
        use_wavelength_channel: bool = False,
        wavelengths: Optional[np.ndarray] = None,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        self.use_wavelength_channel = bool(use_wavelength_channel)
        self.features = tuple(int(f) for f in features)

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

        input_channels = int(in_channels) + (1 if self.use_wavelength_channel else 0)

        conv0 = nn.Conv1d(
            input_channels,
            self.features[0],
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

        blocks: List[nn.Module] = []
        c_in = self.features[0]
        for f in self.features[1:]:
            blocks.append(
                CNNBlock1D(
                    c_in,
                    f,
                    stride=1 if f == self.features[-1] else 2,
                    use_bn=use_bn,
                    use_spectral_norm=use_spectral_norm,
                )
            )
            c_in = f
        self.feature_extractor = nn.Sequential(*blocks)

        conv_last = nn.Conv1d(
            c_in,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )
        self.final = maybe_spectral_norm(conv_last, enabled=use_spectral_norm)

    def _prepare_input(self, s: torch.Tensor) -> torch.Tensor:
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
                    f"Wavelength length {wl.shape[-1]} does not match spectrum length {s.shape[-1]}."
                )
            s = torch.cat([s, wl], dim=1)
        return s

    def forward_features(self, s: torch.Tensor) -> torch.Tensor:
        z = self._prepare_input(s)
        z = self.initial(z)
        z = self.feature_extractor(z)
        return z

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(s)
        return self.final(h)


class SpectralPatchDiscriminator1D(nn.Module):
    """
    Pairwise 1D PatchGAN-style discriminator over wavelength.

    Kept for backward compatibility. This is not the recommended cGAN
    discriminator for the current paired image-to-spectrum setup.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        use_bn: bool = False,
        use_wavelength_channel: bool = False,
        wavelengths: Optional[np.ndarray] = None,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        use_spectral_norm: bool = False,
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

        input_channels = int(in_channels) * 2 + (1 if self.use_wavelength_channel else 0)
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

        layers: List[nn.Module] = []
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
                    f"Wavelength length {wl.shape[-1]} does not match spectrum length {z.shape[-1]}."
                )
            z = torch.cat([z, wl], dim=1)
        return self.model(self.initial(z))


class SegmentedSpectralDiscriminator1D(nn.Module):
    """
    Unconditional global/segmented spectral discriminator.

    forward(s) returns a single logits tensor [B,1,K] by concatenating all
    selected discriminator outputs along the last dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        use_bn: bool = False,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        wavelengths: Optional[np.ndarray] = None,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        mode: str = "global_plus_segmented",
        use_wavelength_channel: bool = True,
        use_spectral_norm: bool = True,
        bands: Optional[Sequence[str]] = None,  # accepted for API compatibility
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
        self.spectral_segments = [(float(lo), float(hi)) for lo, hi in spectral_segments]

        self.register_buffer(
            "wavelengths",
            torch.as_tensor(wavelengths, dtype=torch.float32),
            persistent=False,
        )

        idx_np = spectral_segment_indices(wavelengths, self.spectral_segments)
        self._segment_buffer_names: List[str] = []
        for i, idx in enumerate(idx_np):
            name = f"_segment_idx_{i}"
            self.register_buffer(name, torch.as_tensor(idx, dtype=torch.long), persistent=False)
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

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = _as_b1l(s)
        logits: List[torch.Tensor] = []

        if self.global_disc is not None:
            logits.append(self.global_disc(s))

        if self.use_segmented:
            for disc, idx in zip(self.segment_discs, self.segment_indices()):
                idx = idx.to(s.device)
                s_seg = s.index_select(dim=-1, index=idx)
                logits.append(disc(s_seg))

        if not logits:
            raise RuntimeError("No discriminator branches are enabled.")
        return torch.cat(logits, dim=-1)


# ============================================================
# Conditional image encoders and cGAN discriminators
# ============================================================


class MultiBandPatchConditionEncoder(nn.Module):
    """
    Encode multispectral patch lists into one condition vector per sample.

    Input:
        batch_bands: dict mapping band name to list of tensors.
        Each tensor is [N_patches, 1, H, W] for one sample.

    Output:
        condition embedding [B, condition_dim]
    """

    def __init__(
        self,
        bands: Optional[Sequence[str]] = None,
        base_features: int = 8,
        embed_dim: int = 64,
        condition_dim: int = 512,
        patch_encoder_type: str = "cnn",
        pooling_type: str = "attention_stats",
        band_encoder_mode: str = "separate",
        norm_type: str = "group",
    ):
        super().__init__()
        if bands is None:
            bands = DEFAULT_BANDS
        self.bands = list(bands)
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(condition_dim)
        self.band_encoder_mode = str(band_encoder_mode).lower()

        if self.band_encoder_mode == "shared":
            self.patch_encoder = build_patch_encoder(
                encoder_type=patch_encoder_type,
                in_channels=1,
                base_features=base_features,
                embed_dim=embed_dim,
                norm_type=norm_type,
            )
            self.patch_encoders = None
        elif self.band_encoder_mode in ["separate", "per_band", "independent"]:
            self.patch_encoder = None
            self.patch_encoders = nn.ModuleDict(
                {
                    b: build_patch_encoder(
                        encoder_type=patch_encoder_type,
                        in_channels=1,
                        base_features=base_features,
                        embed_dim=embed_dim,
                        norm_type=norm_type,
                    )
                    for b in self.bands
                }
            )
        else:
            raise ValueError(
                f"Unknown band_encoder_mode={band_encoder_mode}. Expected 'shared' or 'separate'."
            )

        self.pool = nn.ModuleDict({b: build_pool(pooling_type, embed_dim) for b in self.bands})
        fused_dim = int(embed_dim) * len(self.bands)
        self.project = nn.Sequential(
            nn.Linear(fused_dim, max(condition_dim, fused_dim)),
            nn.LayerNorm(max(condition_dim, fused_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max(condition_dim, fused_dim), condition_dim),
            nn.LayerNorm(condition_dim),
        )

    def _encoder_for_band(self, band: str) -> nn.Module:
        if self.band_encoder_mode == "shared":
            assert self.patch_encoder is not None
            return self.patch_encoder
        assert self.patch_encoders is not None
        return self.patch_encoders[band]

    def forward(self, batch_bands: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        if not isinstance(batch_bands, dict):
            raise TypeError("batch_bands must be a dict mapping band name to list of patch tensors.")
        missing = [b for b in self.bands if b not in batch_bands]
        if missing:
            raise KeyError(f"Missing bands in batch_bands: {missing}")

        B = len(batch_bands[self.bands[0]])
        band_descriptors: List[torch.Tensor] = []

        for b in self.bands:
            patch_list = batch_bands[b]
            if len(patch_list) != B:
                raise ValueError(f"Band {b} has batch length {len(patch_list)}, expected {B}.")

            lengths = [int(p.shape[0]) for p in patch_list]
            total = sum(lengths)
            if total == 0:
                ref = next(self.parameters())
                pooled = torch.zeros(B, self.embed_dim, device=ref.device, dtype=ref.dtype)
                band_descriptors.append(pooled)
                continue

            all_patches = torch.cat(patch_list, dim=0)
            all_embeddings = self._encoder_for_band(b)(all_patches)
            split_embeddings = torch.split(all_embeddings, lengths, dim=0)
            pooled_list = [self.pool[b](E_i) for E_i in split_embeddings]
            band_descriptors.append(torch.stack(pooled_list, dim=0))

        fused = torch.cat(band_descriptors, dim=1)
        return self.project(fused)


class ConditionalSpectralDiscriminator1D(nn.Module):
    """
    Projection-style conditional spectral discriminator.

    Inputs:
        s: spectrum [B,L] or [B,1,L]
        batch_bands: multispectral image-patch condition

    Output:
        patch logits [B,1,K]
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        use_bn: bool = False,
        use_wavelength_channel: bool = False,
        wavelengths: Optional[np.ndarray] = None,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        use_spectral_norm: bool = False,
        bands: Optional[Sequence[str]] = None,
        condition_dim: Optional[int] = None,
        condition_embed_dim: int = 64,
        condition_base_features: int = 8,
        condition_patch_encoder_type: str = "cnn",
        condition_pooling_type: str = "attention_stats",
        condition_band_encoder_mode: str = "separate",
        condition_norm_type: str = "group",
    ):
        super().__init__()
        if bands is None:
            bands = DEFAULT_BANDS
        self.bands = list(bands)
        self.features = tuple(int(f) for f in features)
        spectral_feature_dim = self.features[-1]
        if condition_dim is None:
            condition_dim = spectral_feature_dim

        self.spectral = SpectralDiscriminator1D(
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            use_wavelength_channel=use_wavelength_channel,
            wavelengths=wavelengths,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            wavelength_count=wavelength_count,
            use_spectral_norm=use_spectral_norm,
        )

        self.condition_encoder = MultiBandPatchConditionEncoder(
            bands=self.bands,
            base_features=condition_base_features,
            embed_dim=condition_embed_dim,
            condition_dim=int(condition_dim),
            patch_encoder_type=condition_patch_encoder_type,
            pooling_type=condition_pooling_type,
            band_encoder_mode=condition_band_encoder_mode,
            norm_type=condition_norm_type,
        )

        self.h_proj = nn.Linear(spectral_feature_dim, int(condition_dim), bias=False)
        self.condition_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, s: torch.Tensor, batch_bands: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        h_map = self.spectral.forward_features(s)          # [B,C,K]
        patch_logits = self.spectral.final(h_map)          # [B,1,K]

        h_global = h_map.mean(dim=-1)                      # [B,C]
        c = self.condition_encoder(batch_bands)             # [B,D]
        h_cond = self.h_proj(h_global)                      # [B,D]

        projection = torch.sum(h_cond * c, dim=1, keepdim=True)
        projection = projection / (h_cond.shape[1] ** 0.5)
        projection = self.condition_scale * projection

        return patch_logits + projection.unsqueeze(-1)


class ConditionalSegmentedSpectralDiscriminator1D(nn.Module):
    """
    Conditional global/segmented spectral discriminator.

    forward(s, batch_bands) returns concatenated logits [B,1,K].
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        use_bn: bool = False,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        wavelengths: Optional[np.ndarray] = None,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        mode: str = "global",
        use_wavelength_channel: bool = True,
        use_spectral_norm: bool = True,
        bands: Optional[Sequence[str]] = None,
        condition_dim: Optional[int] = None,
        condition_embed_dim: int = 64,
        condition_base_features: int = 8,
        condition_patch_encoder_type: str = "cnn",
        condition_pooling_type: str = "attention_stats",
        condition_band_encoder_mode: str = "separate",
        condition_norm_type: str = "group",
    ):
        super().__init__()
        mode = str(mode).lower()
        valid_modes = ["global", "segmented", "global_plus_segmented"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        if bands is None:
            bands = DEFAULT_BANDS
        self.bands = list(bands)
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
        self.spectral_segments = [(float(lo), float(hi)) for lo, hi in spectral_segments]

        self.register_buffer(
            "wavelengths",
            torch.as_tensor(wavelengths, dtype=torch.float32),
            persistent=False,
        )

        idx_np = spectral_segment_indices(wavelengths, self.spectral_segments)
        self._segment_buffer_names: List[str] = []
        for i, idx in enumerate(idx_np):
            name = f"_segment_idx_{i}"
            self.register_buffer(name, torch.as_tensor(idx, dtype=torch.long), persistent=False)
            self._segment_buffer_names.append(name)

        common_kwargs = dict(
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            use_wavelength_channel=use_wavelength_channel,
            use_spectral_norm=use_spectral_norm,
            bands=self.bands,
            condition_dim=condition_dim,
            condition_embed_dim=condition_embed_dim,
            condition_base_features=condition_base_features,
            condition_patch_encoder_type=condition_patch_encoder_type,
            condition_pooling_type=condition_pooling_type,
            condition_band_encoder_mode=condition_band_encoder_mode,
            condition_norm_type=condition_norm_type,
        )

        self.global_disc: Optional[ConditionalSpectralDiscriminator1D] = None
        if self.use_global:
            self.global_disc = ConditionalSpectralDiscriminator1D(
                wavelengths=wavelengths,
                **common_kwargs,
            )

        self.segment_discs = nn.ModuleList()
        if self.use_segmented:
            for idx in idx_np:
                seg_wl = wavelengths[idx]
                self.segment_discs.append(
                    ConditionalSpectralDiscriminator1D(
                        wavelengths=seg_wl,
                        **common_kwargs,
                    )
                )

    def segment_indices(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self._segment_buffer_names]

    def forward(self, s: torch.Tensor, batch_bands: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        s = _as_b1l(s)
        logits: List[torch.Tensor] = []

        if self.global_disc is not None:
            logits.append(self.global_disc(s, batch_bands))

        if self.use_segmented:
            for disc, idx in zip(self.segment_discs, self.segment_indices()):
                idx = idx.to(s.device)
                s_seg = s.index_select(dim=-1, index=idx)
                logits.append(disc(s_seg, batch_bands))

        if not logits:
            raise RuntimeError("No discriminator branches are enabled.")
        return torch.cat(logits, dim=-1)


# ============================================================
# Smoke test
# ============================================================


def _make_fake_batch(device, B: int = 2):
    batch = {b: [] for b in DEFAULT_BANDS}
    for _ in range(B):
        for b in DEFAULT_BANDS:
            batch[b].append(torch.randn(4, 1, 32, 32, device=device))
    return batch


if __name__ == "__main__":
    device = torch.device("cpu")
    B = 2
    L = 2101
    real = torch.randn(B, L, device=device)
    batch_bands = _make_fake_batch(device, B=B)

    D_uncond = SegmentedSpectralDiscriminator1D(mode="global")
    logits = D_uncond(real)
    print("unconditional logits:", tuple(logits.shape))

    D_cond = ConditionalSegmentedSpectralDiscriminator1D(mode="global")
    logits = D_cond(real, batch_bands)
    print("conditional logits:", tuple(logits.shape))
