"""
Clean MobileNetV3-Small full-leaf generator for pix2spectral.

This module consumes one complete
resized/padded image per multispectral band and builds a full-leaf condition
vector using:

    band-specific 1-channel adapters
    shared MobileNetV3-Small backbone
    per-band intensity statistics
    learned band embeddings
    band-wise self-attention

The fused full-leaf condition is used by the physics-guided PROSPECT/residual
spectral generator.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyPro4Sail import prospect_jacobian


DEFAULT_BANDS = ["blue", "green", "red", "nir", "red_edge"]
DEFAULT_SPECTRAL_SEGMENTS = [
    (400, 528),
    (528, 591),
    (591, 613),
    (613, 691),
    (691, 904),
    (904, 944),
    (944, 973),
    (973, 1066),
    (1066, 1106),
    (1106, 1154),
    (1154, 1194),
    (1194, 1266),
    (1266, 1313),
    (1313, 1377),
    (1377, 1443),
    (1443, 1653),
    (1653, 1724),
    (1724, 1759),
    (1759, 1785),
    (1785, 1816),
    (1816, 1859),
    (1859, 1926),
    (1926, 1999),
    (1999, 2057),
    (2057, 2108),
    (2108, 2141),
    (2141, 2216),
    (2216, 2304),
    (2304, 2399),
    (2399, 2480),
    (2480, 2500),
]

# Numerical safety for the NumPy/SciPy PROSPECT Jacobian.
# The analytic equations can occasionally return NaN/Inf for otherwise bounded
# parameters because of sqrt/log/division operations inside pyPro4Sail.
PROSPECT_SANITIZE_NONFINITE_DEFAULT = True
PROSPECT_RHO_MIN_DEFAULT = 0.0
PROSPECT_RHO_MAX_DEFAULT = 1.0
PROSPECT_JAC_CLIP_DEFAULT = 1.0e4


# ============================================================
# Wavelength and segment utilities
# ============================================================


def make_wavelengths_from_values(
    wavelength_min: float = 400.0,
    wavelength_max: float = 2500.0,
    wavelength_count: int = 2101,
) -> np.ndarray:
    """Create a wavelength grid in nanometers."""
    return np.linspace(
        float(wavelength_min),
        float(wavelength_max),
        int(wavelength_count),
        dtype=np.float64,
    )


def normalize_segments(
    spectral_segments: Optional[Sequence[Tuple[float, float]]],
    wavelength_min: float,
    wavelength_max: float,
) -> List[Tuple[float, float]]:
    """Validate spectral segments and return a normalized list."""
    if spectral_segments is None:
        spectral_segments = [(float(wavelength_min), float(wavelength_max))]

    out: List[Tuple[float, float]] = []
    for lo, hi in spectral_segments:
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            raise ValueError(f"Invalid spectral segment ({lo}, {hi}). hi must be > lo.")
        if lo < wavelength_min or hi > wavelength_max:
            raise ValueError(
                f"Segment ({lo}, {hi}) is outside wavelength range "
                f"[{wavelength_min}, {wavelength_max}]."
            )
        out.append((lo, hi))

    # Require sorted, non-overlapping, contiguous or near-contiguous segments.
    out = sorted(out, key=lambda x: x[0])
    for i in range(1, len(out)):
        if out[i][0] < out[i - 1][1]:
            raise ValueError(
                f"Overlapping spectral segments: {out[i - 1]} and {out[i]}."
            )

    return out


def spectral_segment_indices(
    wavelengths: np.ndarray,
    spectral_segments: Sequence[Tuple[float, float]],
) -> List[np.ndarray]:
    """
    Convert wavelength intervals to index arrays.

    Intervals are half-open [lo, hi) except the last segment, which is
    inclusive [lo, hi]. This avoids duplicate wavelengths at shared boundaries.

    For default 400:2500 with 2101 wavelengths and segments:
      [400,900), [900,1000), [1000,2000), [2000,2500]
    lengths are:
      500, 100, 1000, 501 -> total 2101.
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    indices: List[np.ndarray] = []

    for i, (lo, hi) in enumerate(spectral_segments):
        if i == len(spectral_segments) - 1:
            mask = (wavelengths >= lo) & (wavelengths <= hi)
        else:
            mask = (wavelengths >= lo) & (wavelengths < hi)

        idx = np.where(mask)[0].astype(np.int64)
        if idx.size == 0:
            raise ValueError(f"Segment ({lo}, {hi}) produced zero wavelength indices.")
        indices.append(idx)

    all_idx = np.concatenate(indices, axis=0)
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("Spectral segments produced duplicate wavelength indices.")

    if all_idx.min() < 0 or all_idx.max() >= wavelengths.size:
        raise ValueError("Spectral segment indices are outside wavelength grid.")

    return indices


def boundary_indices_from_segments(
    wavelengths: np.ndarray,
    spectral_segments: Sequence[Tuple[float, float]],
) -> List[int]:
    """Return integer indices at internal segment boundaries for continuity loss."""
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    boundaries = []
    for _, hi in spectral_segments[:-1]:
        idx = int(np.argmin(np.abs(wavelengths - float(hi))))
        if 0 < idx < len(wavelengths):
            boundaries.append(idx)
    return boundaries


# ============================================================
# Normalization factories
# ============================================================


def _best_group_count(num_channels: int, max_groups: int = 8) -> int:
    """Largest group count <= max_groups that divides num_channels."""
    max_groups = min(int(max_groups), int(num_channels))
    for g in range(max_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def norm2d(num_channels: int, norm_type: str = "group") -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type in ["group", "gn", "groupnorm"]:
        return nn.GroupNorm(_best_group_count(num_channels), num_channels)
    if norm_type in ["batch", "bn", "batchnorm"]:
        return nn.BatchNorm2d(num_channels)
    if norm_type in ["instance", "in", "instancenorm"]:
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type in ["none", "identity", ""]:
        return nn.Identity()
    raise ValueError(f"Unknown norm_type={norm_type}")


def maybe_spectral_norm(module: nn.Module, enabled: bool = False) -> nn.Module:
    if enabled:
        return nn.utils.spectral_norm(module)
    return module


# ============================================================
# PROSPECT-D parameter bounding and differentiable layer
# ============================================================


class ProspectDParameterBounds(nn.Module):
    """
    Maps raw network outputs to bounded PROSPECT-D parameters with sigmoid scaling.

    Order: (Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant)
    """

    def __init__(self, mins=None, maxs=None, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

        default_mins = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0], dtype=torch.float32
        )
        # Conservative biological defaults for stable PROSPECT-D inversion.
        # If you need wider values, pass mins/maxs explicitly from config.py.
        default_maxs = torch.tensor(
            [3.6, 120.0, 40.0, 2.0, 0.06, 0.04, 30.0], dtype=torch.float32
        )

        if mins is None:
            mins = default_mins
        if maxs is None:
            maxs = default_maxs

        mins = torch.as_tensor(mins, dtype=torch.float32)
        maxs = torch.as_tensor(maxs, dtype=torch.float32)

        if mins.shape != (7,) or maxs.shape != (7,):
            raise ValueError("mins and maxs must be shape [7]")
        if torch.any(maxs <= mins):
            raise ValueError("Each max must be > min")

        self.register_buffer("mins", mins)
        self.register_buffer("maxs", maxs)

    def forward(self, raw_params: torch.Tensor) -> torch.Tensor:
        u = torch.sigmoid(raw_params)
        u = u * (1.0 - 2.0 * self.eps) + self.eps
        return self.mins + u * (self.maxs - self.mins)


class ProspectDLayerAnalytic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params):
        """
        params: torch [B, 7]
        returns: reflectance rho torch [B, L]

        This wrapper is intentionally defensive because pyPro4Sail's analytic
        Jacobian contains sqrt/log/division operations that can occasionally
        return NaN/Inf for extreme or numerically unlucky parameter values.

        The behavior is controlled by class attributes:
            ProspectDLayerAnalytic.sanitize_nonfinite
            ProspectDLayerAnalytic.rho_min
            ProspectDLayerAnalytic.rho_max
            ProspectDLayerAnalytic.jac_clip
        """
        if params.dim() != 2 or params.shape[1] != 7:
            raise ValueError(f"params must be [B,7], got {tuple(params.shape)}")

        if not torch.isfinite(params).all():
            raise FloatingPointError("Non-finite PROSPECT parameters entering forward.")

        sanitize = bool(
            getattr(
                ProspectDLayerAnalytic,
                "sanitize_nonfinite",
                PROSPECT_SANITIZE_NONFINITE_DEFAULT,
            )
        )
        rho_min = float(
            getattr(
                ProspectDLayerAnalytic,
                "rho_min",
                PROSPECT_RHO_MIN_DEFAULT,
            )
        )
        rho_max = float(
            getattr(
                ProspectDLayerAnalytic,
                "rho_max",
                PROSPECT_RHO_MAX_DEFAULT,
            )
        )
        jac_clip = float(
            getattr(
                ProspectDLayerAnalytic,
                "jac_clip",
                PROSPECT_JAC_CLIP_DEFAULT,
            )
        )

        p_np = params.detach().cpu().numpy().astype(np.float64)
        B = p_np.shape[0]

        rho_list = []
        J_list = []
        bad_items = []

        for i in range(B):
            p = p_np[i]

            with np.errstate(all="ignore"):
                wl, rho, tau, Delta_rho, Delta_tau = prospect_jacobian.JacProspectD(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    float(p[3]),
                    float(p[4]),
                    float(p[5]),
                    float(p[6]),
                )

            rho = np.asarray(rho, dtype=np.float32).reshape(-1)
            Delta_rho = np.asarray(Delta_rho, dtype=np.float32)

            rho_finite = np.isfinite(rho).all()
            jac_finite = np.isfinite(Delta_rho).all()

            if (not rho_finite) or (not jac_finite):
                bad_items.append(
                    {
                        "item": int(i),
                        "params": [float(x) for x in p.tolist()],
                        "rho_finite": bool(rho_finite),
                        "jac_finite": bool(jac_finite),
                        "rho_bad_count": int((~np.isfinite(rho)).sum()),
                        "jac_bad_count": int((~np.isfinite(Delta_rho)).sum()),
                    }
                )

                if not sanitize:
                    raise FloatingPointError(
                        f"Non-finite PROSPECT output. Details: {bad_items[-1]}"
                    )

                # Safe fallback:
                # - Replace non-finite reflectance with clipped finite values.
                # - Replace non-finite Jacobian entries with zero gradients.
                # This prevents rare numerical PROSPECT failures from crashing
                # the whole training run. If this happens often, bounds are still
                # too wide or the parameter head needs regularization.
                rho = np.nan_to_num(
                    rho,
                    nan=0.0,
                    posinf=rho_max,
                    neginf=rho_min,
                ).astype(np.float32)
                Delta_rho = np.nan_to_num(
                    Delta_rho,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)

            # Reflectance should be physically bounded.
            rho = np.clip(rho, rho_min, rho_max).astype(np.float32)

            if Delta_rho.ndim != 2:
                raise ValueError("Delta_rho must be 2D, got " + str(Delta_rho.shape))

            if Delta_rho.shape[0] == 7:
                J = Delta_rho.T
            elif Delta_rho.shape[1] == 7:
                J = Delta_rho
            else:
                raise ValueError("Unexpected Delta_rho shape: " + str(Delta_rho.shape))

            # Clip very large analytic gradients to avoid gradient explosions.
            if jac_clip is not None and jac_clip > 0:
                J = np.clip(J, -jac_clip, jac_clip).astype(np.float32)

            rho_list.append(rho)
            J_list.append(J)

        if bad_items:
            # Print only once per forward call. This is intentionally a warning,
            # not an exception, when sanitize=True.
            print(f"[WARN] Sanitized non-finite PROSPECT output(s): {bad_items[:3]}")

        rho_np = np.stack(rho_list, axis=0).astype(np.float32)
        J_np = np.stack(J_list, axis=0).astype(np.float32)

        ctx.save_for_backward(torch.from_numpy(J_np))
        return torch.from_numpy(rho_np).to(params.device).type_as(params)

    @staticmethod
    def backward(ctx, grad_output):
        (J_t,) = ctx.saved_tensors
        J_t = J_t.to(grad_output.device).type_as(grad_output)
        grad_params = torch.einsum("bl,blk->bk", grad_output, J_t)

        # Final guard: never allow NaN/Inf gradients to propagate.
        grad_params = torch.nan_to_num(grad_params, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_params


# Class-level defaults. You can override these from training before model use:
#     ProspectDLayerAnalytic.sanitize_nonfinite = True
#     ProspectDLayerAnalytic.jac_clip = 1e3
ProspectDLayerAnalytic.sanitize_nonfinite = PROSPECT_SANITIZE_NONFINITE_DEFAULT
ProspectDLayerAnalytic.rho_min = PROSPECT_RHO_MIN_DEFAULT
ProspectDLayerAnalytic.rho_max = PROSPECT_RHO_MAX_DEFAULT
ProspectDLayerAnalytic.jac_clip = PROSPECT_JAC_CLIP_DEFAULT


def prospectd_reflectance_torch(params: torch.Tensor) -> torch.Tensor:
    return ProspectDLayerAnalytic.apply(params)


class PhysicsInformedProspectHead(nn.Module):
    """
    raw_params [B, 7] -> bounded pParams [B, 7] -> y_fake [B, L]
    """

    def __init__(self, mins=None, maxs=None):
        super().__init__()
        self.bounds = ProspectDParameterBounds(mins=mins, maxs=maxs)

    def forward(self, raw_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_params = self.bounds(raw_params)
        y_fake = prospectd_reflectance_torch(p_params)
        return y_fake, p_params


# ============================================================
# MobileNetV3 full-leaf condition encoder
# ============================================================

import warnings

try:
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    TORCHVISION_AVAILABLE = True
except Exception:
    mobilenet_v3_small = None
    MobileNet_V3_Small_Weights = None
    TORCHVISION_AVAILABLE = False


def fullleaf_band_to_bchw(band_input, device=None) -> torch.Tensor:
    """Convert a collated full-leaf band input to Tensor [B,1,H,W]."""
    if isinstance(band_input, list):
        items = []
        for x in band_input:
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.dim() == 4 and x.shape[0] == 1:
                x = x.squeeze(0)
            if x.dim() != 3:
                raise ValueError(
                    f"Expected full-leaf tensor [1,H,W], got {tuple(x.shape)}"
                )
            items.append(x)
        x = torch.stack(items, dim=0)
    else:
        x = band_input
        if x.dim() == 3:
            x = x.unsqueeze(1)
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(
            f"Expected full-leaf band batch [B,1,H,W], got {tuple(x.shape)}"
        )
    if device is not None:
        x = x.to(device)
    return x.float()


class FullLeafBandToRGBAdapter(nn.Module):
    """Small band-specific adapter from one grayscale band to 3 channels."""

    def __init__(self, hidden_channels: int = 8, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, int(hidden_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, int(hidden_channels)),
            nn.Hardswish(inplace=True),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(int(hidden_channels), 3, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mobilenet_v3_small_backbone(pretrained: bool = True) -> nn.Module:
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for MobileNetV3-Small.")
    weights = None
    if pretrained:
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        return mobilenet_v3_small(weights=weights)
    except Exception as exc:
        warnings.warn(
            f"Could not load pretrained MobileNetV3-Small weights ({exc}); using random initialization.",
            RuntimeWarning,
        )
        return mobilenet_v3_small(weights=None)


def freeze_mobilenet_v3_except_last_block(model: nn.Module) -> None:
    """Freeze all MobileNetV3 parameters except the final feature block."""
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "features") and len(model.features) > 0:
        for p in model.features[-1].parameters():
            p.requires_grad = True


def fullleaf_intensity_statistics(x: torch.Tensor) -> torch.Tensor:
    """
    Compute masked per-band intensity statistics.

    Input:
        x: [B,1,H,W]

    Output:
        [B,10] = mean, median, std, min, max, p10, p25, p75, p90, foreground_ratio
    """
    B = x.shape[0]
    flat = x.reshape(B, -1)
    stats = []
    for i in range(B):
        v = flat[i]
        mask = v != 0
        foreground_ratio = mask.float().mean()
        vals = v[mask]
        if vals.numel() < 2:
            vals = v
        stats.append(
            torch.stack(
                [
                    vals.mean(),
                    torch.quantile(vals, 0.50),
                    vals.std(unbiased=False),
                    vals.min(),
                    vals.max(),
                    torch.quantile(vals, 0.10),
                    torch.quantile(vals, 0.25),
                    torch.quantile(vals, 0.75),
                    torch.quantile(vals, 0.90),
                    foreground_ratio,
                ]
            )
        )
    return torch.stack(stats, dim=0)


class MobileNetV3FullLeafConditionEncoder(nn.Module):
    """Encode five full-leaf multispectral band images into one condition vector."""

    def __init__(
        self,
        bands: Optional[Sequence[str]] = None,
        condition_dim: int = 320,
        token_dim: int = 128,
        band_embedding_dim: int = 16,
        adapter_hidden_channels: int = 8,
        attention_layers: int = 1,
        attention_heads: int = 4,
        dropout: float = 0.25,
        mobilenet_pretrained: bool = True,
        freeze_all_except_last: bool = False,
        use_stage_classifier: bool = True,
        num_stage_classes: int = 5,
    ):
        super().__init__()
        self.bands = list(DEFAULT_BANDS if bands is None else bands)
        self.condition_dim = int(condition_dim)
        self.token_dim = int(token_dim)
        self.use_stage_classifier = bool(use_stage_classifier)

        self.band_adapters = nn.ModuleDict(
            {
                b: FullLeafBandToRGBAdapter(
                    hidden_channels=int(adapter_hidden_channels),
                    dropout=float(dropout) * 0.25,
                )
                for b in self.bands
            }
        )

        backbone = load_mobilenet_v3_small_backbone(
            pretrained=bool(mobilenet_pretrained)
        )
        if freeze_all_except_last:
            freeze_mobilenet_v3_except_last_block(backbone)
        self.mobilenet_features = backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.spatial_pool(self.mobilenet_features(dummy)).flatten(1)
            mobilenet_dim = int(out.shape[1])
        self.mobilenet_feature_dim = mobilenet_dim

        self.band_embedding = nn.Embedding(len(self.bands), int(band_embedding_dim))
        token_input_dim = mobilenet_dim + 10 + int(band_embedding_dim)
        hidden = max(int(token_dim), 128)
        self.band_token_mlp = nn.Sequential(
            nn.Linear(token_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Hardswish(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(token_dim)),
            nn.LayerNorm(int(token_dim)),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=int(token_dim),
            nhead=int(attention_heads),
            dim_feedforward=int(token_dim) * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.band_self_attention = nn.TransformerEncoder(
            layer, num_layers=int(attention_layers)
        )
        self.band_pool_logits = nn.Sequential(
            nn.LayerNorm(int(token_dim)), nn.Linear(int(token_dim), 1)
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(condition_dim)),
            nn.LayerNorm(int(condition_dim)),
        )

        self.stage_classifier = None
        if self.use_stage_classifier:
            self.stage_classifier = nn.Sequential(
                nn.LayerNorm(int(condition_dim)),
                nn.Dropout(float(dropout)),
                nn.Linear(int(condition_dim), int(num_stage_classes)),
            )

        self.last_stage_logits = None
        self.last_band_attention_weights = None

    def forward(
        self, fullleaf_band_images: Dict[str, List[torch.Tensor]]
    ) -> torch.Tensor:
        ref = next(self.parameters())
        tokens = []
        for band_index, band_name in enumerate(self.bands):
            if band_name not in fullleaf_band_images:
                raise KeyError(f"Missing band '{band_name}' in fullleaf_band_images")
            x = fullleaf_band_to_bchw(
                fullleaf_band_images[band_name], device=ref.device
            )
            x3 = self.band_adapters[band_name](x)
            fmap = self.mobilenet_features(x3)
            mobile_feat = self.spatial_pool(fmap).flatten(1)
            stats = fullleaf_intensity_statistics(x).to(
                device=mobile_feat.device, dtype=mobile_feat.dtype
            )
            band_id = torch.full(
                (x.shape[0],), band_index, device=mobile_feat.device, dtype=torch.long
            )
            emb = self.band_embedding(band_id)
            token = self.band_token_mlp(torch.cat([mobile_feat, stats, emb], dim=1))
            tokens.append(token)

        band_tokens = torch.stack(tokens, dim=1)
        contextual_tokens = self.band_self_attention(band_tokens)
        weights = torch.softmax(
            self.band_pool_logits(contextual_tokens).squeeze(-1), dim=1
        )
        fused = torch.sum(contextual_tokens * weights.unsqueeze(-1), dim=1)
        condition = self.condition_projection(fused)
        self.last_band_attention_weights = weights.detach()
        self.last_stage_logits = (
            self.stage_classifier(condition)
            if self.stage_classifier is not None
            else None
        )
        return condition


class FullLeafMobileNetV3ProspectGenerator(nn.Module):
    """Physics-guided spectral generator conditioned on full-leaf MobileNetV3 features."""

    def __init__(
        self,
        bands: Optional[Sequence[str]] = None,
        base_features: int = 8,
        embed_dim: int = 64,
        mins=None,
        maxs=None,
        wavelength_min: float = 400.0,
        wavelength_max: float = 2500.0,
        wavelength_count: int = 2101,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        use_segmented_prospect: bool = True,
        use_segment_residual: bool = True,
        segment_residual_scale: float = 0.05,
        output_clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
        mlp_hidden_dims: Sequence[int] = (256, 128, 64),
        mobilenet_pretrained: bool = True,
        mobilenet_freeze_all_except_last: bool = False,
        mobilenet_token_dim: int = 128,
        mobilenet_attention_layers: int = 1,
        mobilenet_attention_heads: int = 4,
        mobilenet_dropout: float = 0.25,
        mobilenet_adapter_hidden_channels: int = 8,
        use_stage_classifier: bool = True,
        num_stage_classes: int = 5,
        **unused,
    ):
        super().__init__()
        self.bands = list(DEFAULT_BANDS if bands is None else bands)
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(embed_dim) * len(self.bands)
        self.wavelength_min = float(wavelength_min)
        self.wavelength_max = float(wavelength_max)
        self.wavelength_count = int(wavelength_count)
        self.use_segmented_prospect = bool(use_segmented_prospect)
        self.use_segment_residual = bool(use_segment_residual)
        self.segment_residual_scale = float(segment_residual_scale)
        self.output_clamp = output_clamp

        wavelengths_np = make_wavelengths_from_values(
            self.wavelength_min, self.wavelength_max, self.wavelength_count
        )
        self.register_buffer(
            "wavelengths",
            torch.as_tensor(wavelengths_np, dtype=torch.float32),
            persistent=False,
        )

        if self.use_segmented_prospect:
            if spectral_segments is None:
                spectral_segments = DEFAULT_SPECTRAL_SEGMENTS
        else:
            spectral_segments = [(self.wavelength_min, self.wavelength_max)]
        self.spectral_segments = normalize_segments(
            spectral_segments, self.wavelength_min, self.wavelength_max
        )
        self.num_segments = len(self.spectral_segments)

        idx_np = spectral_segment_indices(wavelengths_np, self.spectral_segments)
        self._segment_buffer_names = []
        for i, idx in enumerate(idx_np):
            name = f"_segment_idx_{i}"
            self.register_buffer(
                name, torch.as_tensor(idx, dtype=torch.long), persistent=False
            )
            self._segment_buffer_names.append(name)
        boundaries = boundary_indices_from_segments(
            wavelengths_np, self.spectral_segments
        )
        self.register_buffer(
            "boundary_indices",
            torch.as_tensor(boundaries, dtype=torch.long),
            persistent=False,
        )

        self.condition_encoder = MobileNetV3FullLeafConditionEncoder(
            bands=self.bands,
            condition_dim=self.condition_dim,
            token_dim=int(mobilenet_token_dim),
            adapter_hidden_channels=int(mobilenet_adapter_hidden_channels),
            attention_layers=int(mobilenet_attention_layers),
            attention_heads=int(mobilenet_attention_heads),
            dropout=float(mobilenet_dropout),
            mobilenet_pretrained=bool(mobilenet_pretrained),
            freeze_all_except_last=bool(mobilenet_freeze_all_except_last),
            use_stage_classifier=bool(use_stage_classifier),
            num_stage_classes=int(num_stage_classes),
        )

        layers: List[nn.Module] = []
        in_dim = self.condition_dim
        for h in mlp_hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, int(h)), nn.LayerNorm(int(h)), nn.ReLU(inplace=True)]
            )
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, self.num_segments * 7))
        self.param_mlp = nn.Sequential(*layers)

        self.residual_mlp = None
        if self.use_segment_residual:
            hidden = max(128, int(embed_dim) * 2)
            self.residual_mlp = nn.Sequential(
                nn.Linear(self.condition_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.wavelength_count),
                nn.Tanh(),
            )
        self.physics = PhysicsInformedProspectHead(mins=mins, maxs=maxs)
        self.last_stage_logits = None
        self.last_band_attention_weights = None

    def segment_indices(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self._segment_buffer_names]

    def encode_fullleaf_condition(
        self, fullleaf_band_images: Dict[str, List[torch.Tensor]]
    ) -> torch.Tensor:
        condition = self.condition_encoder(fullleaf_band_images)
        self.last_stage_logits = self.condition_encoder.last_stage_logits
        self.last_band_attention_weights = (
            self.condition_encoder.last_band_attention_weights
        )
        return condition

    def assemble_segmented_spectrum(self, prospect_full: torch.Tensor) -> torch.Tensor:
        pieces = []
        for i, idx in enumerate(self.segment_indices()):
            pieces.append(prospect_full[:, i, idx.to(prospect_full.device)])
        return torch.cat(pieces, dim=1)

    def forward_from_condition(
        self, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = condition.shape[0]
        raw_params = self.param_mlp(condition).view(B, self.num_segments, 7)
        raw_flat = raw_params.reshape(B * self.num_segments, 7)
        prospect_flat, p_flat = self.physics(raw_flat)
        if prospect_flat.shape[1] != self.wavelength_count:
            raise RuntimeError(
                f"PROSPECT returned length {prospect_flat.shape[1]}, expected {self.wavelength_count}."
            )
        prospect_full = prospect_flat.view(B, self.num_segments, self.wavelength_count)
        p_params = p_flat.view(B, self.num_segments, 7)
        y_fake = self.assemble_segmented_spectrum(prospect_full)
        if self.residual_mlp is not None:
            residual = self.segment_residual_scale * self.residual_mlp(condition)
            if residual.shape[1] != y_fake.shape[1]:
                raise RuntimeError(
                    f"Residual length {residual.shape[1]} does not match spectrum length {y_fake.shape[1]}"
                )
            y_fake = y_fake + residual
        if self.output_clamp is not None:
            lo, hi = self.output_clamp
            y_fake = torch.clamp(y_fake, float(lo), float(hi))
        if not self.use_segmented_prospect and p_params.shape[1] == 1:
            p_params = p_params[:, 0, :]
        return y_fake, p_params

    def segment_boundary_loss(
        self, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        if self.boundary_indices.numel() == 0:
            return torch.zeros((), device=y.device, dtype=y.dtype)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        losses = []
        for idx in self.boundary_indices.to(y.device):
            i = int(idx.item())
            if 0 < i < y.shape[1]:
                losses.append((y[:, i] - y[:, i - 1]).pow(2))
        if not losses:
            return torch.zeros((), device=y.device, dtype=y.dtype)
        out = torch.stack(losses, dim=1)
        if reduction == "mean":
            return out.mean()
        if reduction == "sum":
            return out.sum()
        return out

    def forward_condition_batch(
        self, fullleaf_band_images: Dict[str, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        condition = self.encode_fullleaf_condition(fullleaf_band_images)
        return self.forward_from_condition(condition)

    def forward(
        self, fullleaf_band_images: Dict[str, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_condition_batch(fullleaf_band_images)


if __name__ == "__main__":
    print("FullLeafMobileNetV3ProspectGenerator module loaded.")
