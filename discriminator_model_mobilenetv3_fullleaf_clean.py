"""
Clean full-leaf conditional spectral discriminator for MobileNetV3 cGAN.

It scores spectra with a 1D convolutional network and conditions the score using
MobileNetV3 full-leaf image features through a projection term.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from generator_model_mobilenetv3_fullleaf_clean import (
    DEFAULT_BANDS,
    make_wavelengths_from_values,
    MobileNetV3FullLeafConditionEncoder,
)


def maybe_spectral_norm_1d(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


class FullLeafSpectralDiscriminator1D(nn.Module):
    """1D convolutional discriminator over spectral signatures."""
    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (64, 128, 256),
        use_bn: bool = False,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        mode: str = "global",
        use_wavelength_channel: bool = True,
        use_spectral_norm: bool = True,
        bands: Optional[Sequence[str]] = None,
        **unused,
    ):
        super().__init__()
        self.use_wavelength_channel = bool(use_wavelength_channel)
        self.wavelength_count = int(wavelength_count)
        channels_in = int(in_channels) + (1 if self.use_wavelength_channel else 0)
        wavelengths = make_wavelengths_from_values(wavelength_min, wavelength_max, wavelength_count)
        wl_norm = 2.0 * ((wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min() + 1e-8)) - 1.0
        self.register_buffer("wavelength_channel", torch.tensor(wl_norm, dtype=torch.float32).view(1, 1, -1), persistent=False)

        layers = []
        prev = channels_in
        for i, f in enumerate(features):
            conv = maybe_spectral_norm_1d(nn.Conv1d(prev, int(f), kernel_size=7, stride=2, padding=3), use_spectral_norm)
            layers.append(conv)
            if use_bn and i > 0:
                layers.append(nn.InstanceNorm1d(int(f), affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = int(f)
        self.feature_net = nn.Sequential(*layers)
        self.local_score = maybe_spectral_norm_1d(nn.Conv1d(prev, 1, kernel_size=3, padding=1), use_spectral_norm)
        self.feature_dim = prev

    def prepare_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)
        if spectrum.dim() != 3:
            raise ValueError(f"Expected spectrum [B,L] or [B,1,L], got {tuple(spectrum.shape)}")
        if self.use_wavelength_channel:
            wl = self.wavelength_channel.to(device=spectrum.device, dtype=spectrum.dtype).expand(spectrum.shape[0], -1, -1)
            spectrum = torch.cat([spectrum, wl], dim=1)
        return spectrum

    def features_from_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.feature_net(self.prepare_spectrum(spectrum))

    def pooled_features(self, spectrum: torch.Tensor) -> torch.Tensor:
        fmap = self.features_from_spectrum(spectrum)
        return torch.mean(fmap, dim=-1)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        fmap = self.features_from_spectrum(spectrum)
        return self.local_score(fmap)


class ConditionalFullLeafSpectralDiscriminator1D(nn.Module):
    """Projection discriminator D(spectrum, full-leaf-image-condition)."""
    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (64, 128, 256),
        use_bn: bool = False,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        mode: str = "global",
        use_wavelength_channel: bool = True,
        use_spectral_norm: bool = True,
        bands: Optional[Sequence[str]] = None,
        condition_dim: int = 320,
        condition_embed_dim: int = 64,
        mobilenet_pretrained: bool = True,
        mobilenet_freeze_all_except_last: bool = True,
        mobilenet_token_dim: int = 128,
        mobilenet_attention_layers: int = 1,
        mobilenet_attention_heads: int = 4,
        mobilenet_dropout: float = 0.25,
        mobilenet_adapter_hidden_channels: int = 8,
        num_stage_classes: int = 5,
        **unused,
    ):
        super().__init__()
        if bands is None:
            bands = DEFAULT_BANDS
        self.spectral_net = FullLeafSpectralDiscriminator1D(
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            wavelength_count=wavelength_count,
            spectral_segments=spectral_segments,
            mode=mode,
            use_wavelength_channel=use_wavelength_channel,
            use_spectral_norm=use_spectral_norm,
            bands=bands,
        )
        self.condition_encoder = MobileNetV3FullLeafConditionEncoder(
            bands=bands,
            condition_dim=int(condition_dim),
            token_dim=int(mobilenet_token_dim),
            adapter_hidden_channels=int(mobilenet_adapter_hidden_channels),
            attention_layers=int(mobilenet_attention_layers),
            attention_heads=int(mobilenet_attention_heads),
            dropout=float(mobilenet_dropout),
            mobilenet_pretrained=bool(mobilenet_pretrained),
            freeze_all_except_last=bool(mobilenet_freeze_all_except_last),
            use_stage_classifier=False,
            num_stage_classes=int(num_stage_classes),
        )
        self.spectrum_projection = nn.Linear(self.spectral_net.feature_dim, int(condition_embed_dim))
        self.condition_projection = nn.Linear(int(condition_dim), int(condition_embed_dim))

    def forward(self, spectrum: torch.Tensor, fullleaf_band_images: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        fmap = self.spectral_net.features_from_spectrum(spectrum)
        local = self.spectral_net.local_score(fmap)
        spectrum_feat = torch.mean(fmap, dim=-1)
        condition = self.condition_encoder(fullleaf_band_images)
        phi = self.spectrum_projection(spectrum_feat)
        psi = self.condition_projection(condition)
        projection = torch.sum(phi * psi, dim=1) / (phi.shape[1] ** 0.5)
        return local + projection.view(-1, 1, 1)
