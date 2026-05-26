"""
Physics-Informed Loss Functions for pix2spectral / PROSPECT-D GAN.

Updated for the improved segmented generator/discriminator.

Key compatibility additions:
  - accepts PROSPECT parameters shaped [B, 7] or [B, S, 7]
  - supports optional segment-boundary continuity loss
  - safer finite-value checks for NaN/Inf debugging
  - adversarial loss can consume tensor, list/tuple, or dict discriminator outputs
  - wavelength utilities compatible with the updated segmented models

The segmented generator returns:
    y_fake:  [B, L]
    pParams: [B, S, 7]     # S spectral segments

The old generator returns:
    y_fake:  [B, L]
    pParams: [B, 7]

Both are supported.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrOutputs = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, Sequence[torch.Tensor]],
]


# ============================================================
# General utilities
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
        float(_cfg_get(cfg, "wavelength_min", "WAVELENGTH_MIN", 400.0)),
        float(_cfg_get(cfg, "wavelength_max", "WAVELENGTH_MAX", 2500.0)),
        int(_cfg_get(cfg, "wavelength_count", "WAVELENGTH_COUNT", 2101)),
        dtype=np.float64,
    )


def spectral_segment_indices(
    wavelengths: np.ndarray,
    spectral_segments: Sequence[Tuple[float, float]],
) -> List[np.ndarray]:
    """
    Convert wavelength intervals into index arrays.

    Intervals are half-open [lo, hi) except the last interval, which is
    closed [lo, hi]. This avoids duplicate boundary wavelengths.

    Example for 400:2500 with 2101 wavelengths:
        (400, 900), (900, 1000), (1000, 2000), (2000, 2500)

    produces lengths:
        500, 100, 1000, 501
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    indices: List[np.ndarray] = []

    if len(spectral_segments) == 0:
        raise ValueError("spectral_segments must contain at least one segment.")

    for i, (lo, hi) in enumerate(spectral_segments):
        lo = float(lo)
        hi = float(hi)

        if hi <= lo:
            raise ValueError(f"Invalid segment ({lo}, {hi}); hi must be > lo.")

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

    if all_idx.min() < 0 or all_idx.max() >= len(wavelengths):
        raise ValueError("Segment indices are outside wavelength grid.")

    return indices


def boundary_indices_from_segments(
    wavelengths: np.ndarray,
    spectral_segments: Sequence[Tuple[float, float]],
) -> List[int]:
    """
    Return indices nearest to internal segment boundaries.

    For default segments, this returns indices near:
        900, 1000, 2000 nm
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    boundaries: List[int] = []

    for _, hi in spectral_segments[:-1]:
        idx = int(np.argmin(np.abs(wavelengths - float(hi))))
        if 0 < idx < len(wavelengths):
            boundaries.append(idx)

    return boundaries


def flatten_segmented_params(params: torch.Tensor) -> torch.Tensor:
    """
    Convert PROSPECT parameters to [N, 7].

    Supports:
        [B, 7]       -> [B, 7]
        [B, S, 7]    -> [B*S, 7]
        [S, 7]       -> [S, 7] for single-sample segmented output
    """
    if params is None:
        raise ValueError("params is None.")

    if params.dim() == 2 and params.shape[-1] == 7:
        return params

    if params.dim() == 3 and params.shape[-1] == 7:
        return params.reshape(-1, params.shape[-1])

    raise ValueError(
        "Expected params with shape [B,7], [B,S,7], or [S,7], "
        f"got {tuple(params.shape)}."
    )


def _as_float_tensor_like(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def assert_finite_tensor(name: str, tensor: torch.Tensor):
    """
    Raise a useful error if tensor contains NaN or Inf.
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}.")

    if torch.isfinite(tensor).all():
        return

    bad = ~torch.isfinite(tensor)
    bad_count = int(bad.sum().detach().cpu())
    total = tensor.numel()

    finite_values = tensor[torch.isfinite(tensor)]
    msg = f"Non-finite tensor detected in {name}: {bad_count}/{total} values."

    if finite_values.numel() > 0:
        msg += (
            f" Finite min={float(finite_values.min().detach().cpu()):.6g},"
            f" max={float(finite_values.max().detach().cpu()):.6g},"
            f" mean={float(finite_values.mean().detach().cpu()):.6g}."
        )

    raise FloatingPointError(msg)


def _safe_zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _ensure_2d_spectrum(y: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure spectrum is [B,L].
    Accepts [L], [B,L], or [B,1,L].
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)
    elif y.dim() == 3 and y.shape[1] == 1:
        y = y.squeeze(1)

    if y.dim() != 2:
        raise ValueError(f"{name} must be [B,L], [L], or [B,1,L], got {tuple(y.shape)}.")

    return y


# ============================================================
# Physics-informed spectral reconstruction loss
# ============================================================


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for spectral reconstruction.

    Compatible with:
        params [B, 7]       old/full-spectrum generator
        params [B, S, 7]    improved segmented generator

    Components:
        1. Spectral L1 loss
        2. Wavelength-weighted L1 loss
        3. Parameter bounds penalty
        4. Spectral smoothness penalty
        5. Derivative consistency loss
        6. Optional segment-boundary continuity loss
    """

    def __init__(
        self,
        param_bounds,
        wavelength_weights: Optional[torch.Tensor] = None,
        lambda_spectral: float = 1.0,
        lambda_weighted: float = 0.0,
        lambda_param_penalty: float = 0.1,
        lambda_smoothness: float = 0.01,
        lambda_derivative: float = 0.01,
        lambda_segment_continuity: float = 0.0,
        boundary_indices: Optional[Union[Sequence[int], torch.Tensor]] = None,
        wavelengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        continuity_width: int = 1,
        roughness_factor: float = 1.5,
        finite_check: bool = True,
        eps: float = 1e-8,
    ):
        """
        Args:
            param_bounds:
                Object from generator physics head with .mins and .maxs buffers.

            wavelength_weights:
                Optional tensor [L]. Use create_wavelength_weights(...).

            lambda_*:
                Loss component weights.

            boundary_indices:
                Optional indices at spectral segment boundaries. You can pass
                gen.boundary_indices directly.

            wavelengths + spectral_segments:
                Alternative to boundary_indices. Boundaries are inferred from
                the segment upper limits.

            continuity_width:
                Number of adjacent samples used around each segment boundary.
                1 means compare y[:, idx] with y[:, idx-1].

            roughness_factor:
                Smoothness permits fake first-derivative up to
                roughness_factor * real first-derivative.

            finite_check:
                If True, raise clear errors on NaN/Inf.

            eps:
                Numerical safety constant.
        """
        super().__init__()

        self.param_bounds = param_bounds

        if wavelength_weights is not None:
            self.register_buffer(
                "wavelength_weights",
                torch.as_tensor(wavelength_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.wavelength_weights = None

        if boundary_indices is None and wavelengths is not None and spectral_segments is not None:
            if torch.is_tensor(wavelengths):
                wl_np = wavelengths.detach().cpu().numpy()
            else:
                wl_np = np.asarray(wavelengths, dtype=np.float64)
            boundary_indices = boundary_indices_from_segments(wl_np, spectral_segments)

        if boundary_indices is not None:
            self.register_buffer(
                "boundary_indices",
                torch.as_tensor(boundary_indices, dtype=torch.long),
                persistent=False,
            )
        else:
            self.boundary_indices = None

        self.lambda_spectral = float(lambda_spectral)
        self.lambda_weighted = float(lambda_weighted)
        self.lambda_param_penalty = float(lambda_param_penalty)
        self.lambda_smoothness = float(lambda_smoothness)
        self.lambda_derivative = float(lambda_derivative)
        self.lambda_segment_continuity = float(lambda_segment_continuity)

        self.continuity_width = int(continuity_width)
        self.roughness_factor = float(roughness_factor)
        self.finite_check = bool(finite_check)
        self.eps = float(eps)

        if self.continuity_width < 1:
            raise ValueError("continuity_width must be >= 1.")

    def forward(
        self,
        y_fake: torch.Tensor,
        y_real: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics-informed loss.

        Args:
            y_fake:
                Generated spectra [B, L].

            y_real:
                Real spectra [B, L].

            params:
                PROSPECT parameters:
                    [B, 7] for full-spectrum generator, or
                    [B, S, 7] for segmented generator.

        Returns:
            total_loss, losses_dict
        """
        y_fake = _ensure_2d_spectrum(y_fake, "y_fake").float()
        y_real = _ensure_2d_spectrum(y_real, "y_real").float()
        params = params.float()

        if y_fake.shape != y_real.shape:
            raise ValueError(
                f"y_fake and y_real must have same shape. "
                f"Got {tuple(y_fake.shape)} and {tuple(y_real.shape)}."
            )

        if self.finite_check:
            assert_finite_tensor("y_fake", y_fake)
            assert_finite_tensor("y_real", y_real)
            assert_finite_tensor("params", params)

        losses: Dict[str, torch.Tensor] = {}

        abs_err = torch.abs(y_fake - y_real)

        # 1. Basic spectral L1 loss.
        spectral_l1 = torch.mean(abs_err)
        losses["spectral_l1"] = spectral_l1

        # 2. Wavelength-weighted L1 loss.
        if self.wavelength_weights is not None:
            weights = self.wavelength_weights.to(device=y_fake.device, dtype=y_fake.dtype)
            if weights.numel() != y_fake.shape[1]:
                raise ValueError(
                    f"wavelength_weights length {weights.numel()} does not match "
                    f"spectrum length {y_fake.shape[1]}."
                )
            weighted_l1 = torch.mean(abs_err * weights.view(1, -1))
        else:
            weighted_l1 = _safe_zero_like_reference(y_fake)

        losses["weighted_l1"] = weighted_l1

        # 3. Parameter constraint penalty.
        param_penalty = self._parameter_penalty(params)
        losses["param_penalty"] = param_penalty

        # 4. Spectral smoothness.
        smoothness = self._spectral_smoothness(y_fake, y_real)
        losses["smoothness"] = smoothness

        # 5. Spectral derivative consistency.
        derivative_loss = self._derivative_consistency(y_fake, y_real)
        losses["derivative"] = derivative_loss

        # 6. Segment-boundary continuity.
        segment_continuity = self._segment_continuity(y_fake)
        losses["segment_continuity"] = segment_continuity

        total_loss = (
            self.lambda_spectral * losses["spectral_l1"]
            + self.lambda_weighted * losses["weighted_l1"]
            + self.lambda_param_penalty * losses["param_penalty"]
            + self.lambda_smoothness * losses["smoothness"]
            + self.lambda_derivative * losses["derivative"]
            + self.lambda_segment_continuity * losses["segment_continuity"]
        )

        losses["total"] = total_loss

        if self.finite_check:
            for name, value in losses.items():
                assert_finite_tensor(f"losses['{name}']", value)

        return total_loss, losses

    def _parameter_penalty(self, params: torch.Tensor) -> torch.Tensor:
        """
        Soft penalty for parameters outside physical bounds.

        The improved generator already bounds parameters with sigmoid scaling,
        so this is usually zero. It is still useful as a safety check, and it
        also supports future variants with unconstrained parameter heads.
        """
        params_flat = flatten_segmented_params(params)

        mins = self.param_bounds.mins.to(device=params_flat.device, dtype=params_flat.dtype)
        maxs = self.param_bounds.maxs.to(device=params_flat.device, dtype=params_flat.dtype)

        below_min = torch.relu(mins.view(1, -1) - params_flat)
        above_max = torch.relu(params_flat - maxs.view(1, -1))

        # Normalize by parameter ranges so large-scale variables do not dominate.
        ranges = torch.clamp(maxs - mins, min=self.eps).view(1, -1)
        penalty = torch.mean((below_min + above_max) / ranges)

        return penalty

    def _spectral_smoothness(self, y_fake: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        """
        Penalize fake spectra that are much rougher than real spectra.

        This is asymmetric: it allows natural real-spectrum changes, but
        discourages artificial spikes in the generated spectrum.
        """
        if y_fake.shape[1] < 2:
            return _safe_zero_like_reference(y_fake)

        diff_fake = torch.abs(y_fake[:, 1:] - y_fake[:, :-1])
        diff_real = torch.abs(y_real[:, 1:] - y_real[:, :-1])

        return torch.mean(torch.relu(diff_fake - diff_real * self.roughness_factor))

    def _derivative_consistency(self, y_fake: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        """
        Match the first derivative of generated and real spectra.

        This helps preserve absorption slopes, red-edge shape, and water-band
        transitions.
        """
        if y_fake.shape[1] < 2:
            return _safe_zero_like_reference(y_fake)

        deriv_fake = y_fake[:, 1:] - y_fake[:, :-1]
        deriv_real = y_real[:, 1:] - y_real[:, :-1]

        return torch.mean(torch.abs(deriv_fake - deriv_real))

    def _segment_continuity(self, y_fake: torch.Tensor) -> torch.Tensor:
        """
        Penalize jumps at segmented-PROSPECT boundaries.

        This is important when the generator assembles several PROSPECT slices:
        400-900, 900-1000, 1000-2000, 2000-2500 nm.
        """
        if self.boundary_indices is None:
            return _safe_zero_like_reference(y_fake)

        if self.boundary_indices.numel() == 0:
            return _safe_zero_like_reference(y_fake)

        B, L = y_fake.shape
        losses = []

        for idx_t in self.boundary_indices.to(y_fake.device):
            idx = int(idx_t.item())

            if idx <= 0 or idx >= L:
                continue

            # Compare average on left and right windows around boundary.
            w = self.continuity_width
            left_start = max(0, idx - w)
            left_end = idx
            right_start = idx
            right_end = min(L, idx + w)

            if left_end <= left_start or right_end <= right_start:
                continue

            left_mean = y_fake[:, left_start:left_end].mean(dim=1)
            right_mean = y_fake[:, right_start:right_end].mean(dim=1)

            losses.append((right_mean - left_mean).pow(2))

        if len(losses) == 0:
            return _safe_zero_like_reference(y_fake)

        return torch.stack(losses, dim=1).mean()


def create_wavelength_weights(
    num_wavelengths: int = 2101,
    start_wl: float = 400.0,
    end_wl: float = 2500.0,
    red_edge_weight: float = 2.0,
    nir_weight: float = 1.5,
    water_weight: float = 2.0,
    base_weight: float = 1.0,
) -> torch.Tensor:
    """
    Create wavelength-specific weights emphasizing important vegetation regions.

    Important regions:
      - Red edge: 680-750 nm
      - NIR plateau: 750-1300 nm
      - Water absorption: 1400-1900 nm and 2100-2300 nm

    Returns:
        torch.Tensor [num_wavelengths]
    """
    wavelengths = np.linspace(float(start_wl), float(end_wl), int(num_wavelengths))
    weights = np.ones_like(wavelengths, dtype=np.float32) * float(base_weight)

    red_edge_mask = (wavelengths >= 680.0) & (wavelengths <= 750.0)
    nir_mask = (wavelengths >= 750.0) & (wavelengths <= 1300.0)
    water1_mask = (wavelengths >= 1400.0) & (wavelengths <= 1900.0)
    water2_mask = (wavelengths >= 2100.0) & (wavelengths <= 2300.0)

    weights[red_edge_mask] = float(red_edge_weight)
    weights[nir_mask] = float(nir_weight)
    weights[water1_mask] = float(water_weight)
    weights[water2_mask] = float(water_weight)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# Adversarial losses
# ============================================================


def _collect_logits(outputs: TensorOrOutputs) -> List[torch.Tensor]:
    """
    Normalize discriminator outputs into a flat list of tensors.

    Supports:
        tensor
        list/tuple of tensors
        dict[str, tensor]
        nested dict/list combinations
    """
    if torch.is_tensor(outputs):
        return [outputs]

    if isinstance(outputs, dict):
        out: List[torch.Tensor] = []
        for value in outputs.values():
            out.extend(_collect_logits(value))
        if len(out) == 0:
            raise ValueError("Discriminator output dict is empty.")
        return out

    if isinstance(outputs, (list, tuple)):
        out: List[torch.Tensor] = []
        for value in outputs:
            out.extend(_collect_logits(value))
        if len(out) == 0:
            raise ValueError("Discriminator output sequence is empty.")
        return out

    raise TypeError(
        "Unsupported discriminator output type. Expected tensor/list/tuple/dict, "
        f"got {type(outputs)}."
    )


class AdversarialLoss(nn.Module):
    """
    Wrapper for adversarial loss types.

    Supports:
        - 'lsgan'   : least-squares GAN
        - 'vanilla' : BCEWithLogits GAN
        - 'wgan'    : Wasserstein GAN objective

    Updated behavior:
        D_real and D_fake may be tensors, lists, tuples, or dicts. This makes
        the loss compatible with segmented/global discriminator branches.
    """

    def __init__(self, loss_type: str = "lsgan"):
        super().__init__()

        loss_type = str(loss_type).lower()
        self.loss_type = loss_type

        if loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        elif loss_type == "vanilla":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "wgan":
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss_type={loss_type}. Expected lsgan, vanilla, or wgan.")

    def discriminator_loss(self, D_real: TensorOrOutputs, D_fake: TensorOrOutputs) -> torch.Tensor:
        real_logits = _collect_logits(D_real)
        fake_logits = _collect_logits(D_fake)

        if len(real_logits) != len(fake_logits):
            # This should not happen for consistent discriminator branches.
            raise ValueError(
                f"D_real has {len(real_logits)} output branches but "
                f"D_fake has {len(fake_logits)}."
            )

        losses = []

        for real, fake in zip(real_logits, fake_logits):
            if self.loss_type == "wgan":
                losses.append(-(torch.mean(real) - torch.mean(fake)))
            else:
                real_loss = self.criterion(real, torch.ones_like(real))
                fake_loss = self.criterion(fake, torch.zeros_like(fake))
                losses.append(0.5 * (real_loss + fake_loss))

        return torch.stack(losses).mean()

    def generator_loss(self, D_fake: TensorOrOutputs) -> torch.Tensor:
        fake_logits = _collect_logits(D_fake)
        losses = []

        for fake in fake_logits:
            if self.loss_type == "wgan":
                losses.append(-torch.mean(fake))
            else:
                losses.append(self.criterion(fake, torch.ones_like(fake)))

        return torch.stack(losses).mean()


# ============================================================
# Spectral Angle Mapper
# ============================================================


class SpectralAngleMapper(nn.Module):
    """
    Spectral Angle Mapper (SAM).

    Measures angular distance between spectra. Lower is better.
    This is shape-oriented and less sensitive to absolute brightness than L1.
    """

    def __init__(self, eps: float = 1e-8, clamp_eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.clamp_eps = float(clamp_eps)

    def forward(self, y_fake: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        y_fake = _ensure_2d_spectrum(y_fake, "y_fake").float()
        y_real = _ensure_2d_spectrum(y_real, "y_real").float()

        if y_fake.shape != y_real.shape:
            raise ValueError(
                f"y_fake and y_real must have same shape. "
                f"Got {tuple(y_fake.shape)} and {tuple(y_real.shape)}."
            )

        dot = torch.sum(y_fake * y_real, dim=1)
        norm_fake = torch.linalg.norm(y_fake, dim=1)
        norm_real = torch.linalg.norm(y_real, dim=1)

        denom = torch.clamp(norm_fake * norm_real, min=self.eps)
        cos_sim = dot / denom

        # Avoid NaN gradients around exact +/-1.
        cos_sim = torch.clamp(cos_sim, -1.0 + self.clamp_eps, 1.0 - self.clamp_eps)

        return torch.mean(torch.acos(cos_sim))


# ============================================================
# Optional standalone segment-continuity loss
# ============================================================


class SegmentContinuityLoss(nn.Module):
    """
    Standalone continuity loss for segmented spectra.

    Useful if you want to add this outside PhysicsInformedLoss.
    """

    def __init__(
        self,
        boundary_indices: Optional[Union[Sequence[int], torch.Tensor]] = None,
        wavelengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        spectral_segments: Optional[Sequence[Tuple[float, float]]] = None,
        continuity_width: int = 1,
    ):
        super().__init__()

        if boundary_indices is None and wavelengths is not None and spectral_segments is not None:
            if torch.is_tensor(wavelengths):
                wl_np = wavelengths.detach().cpu().numpy()
            else:
                wl_np = np.asarray(wavelengths, dtype=np.float64)
            boundary_indices = boundary_indices_from_segments(wl_np, spectral_segments)

        if boundary_indices is None:
            boundary_indices = []

        self.register_buffer(
            "boundary_indices",
            torch.as_tensor(boundary_indices, dtype=torch.long),
            persistent=False,
        )
        self.continuity_width = int(continuity_width)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = _ensure_2d_spectrum(y, "y")
        if self.boundary_indices.numel() == 0:
            return _safe_zero_like_reference(y)

        losses = []
        B, L = y.shape

        for idx_t in self.boundary_indices.to(y.device):
            idx = int(idx_t.item())
            if idx <= 0 or idx >= L:
                continue

            w = max(1, self.continuity_width)
            left = y[:, max(0, idx - w):idx].mean(dim=1)
            right = y[:, idx:min(L, idx + w)].mean(dim=1)
            losses.append((right - left).pow(2))

        if len(losses) == 0:
            return _safe_zero_like_reference(y)

        return torch.stack(losses, dim=1).mean()


# ============================================================
# Example usage / smoke tests
# ============================================================


if __name__ == "__main__":
    print("Testing updated Physics-Informed Loss Functions\n")

    B, S, L = 4, 4, 2101

    y_real = torch.rand(B, L) * 0.5
    y_fake = torch.clamp(y_real + torch.randn(B, L) * 0.05, 0.0, 1.0)

    params_segmented = torch.rand(B, S, 7)
    params_segmented[..., 0] = 1.0 + 2.5 * params_segmented[..., 0]
    params_segmented[..., 1] = 100.0 * params_segmented[..., 1]
    params_segmented[..., 2] = 30.0 * params_segmented[..., 2]
    params_segmented[..., 3] = 2.0 * params_segmented[..., 3]
    params_segmented[..., 4] = 0.0001 + 0.05 * params_segmented[..., 4]
    params_segmented[..., 5] = 0.0001 + 0.03 * params_segmented[..., 5]
    params_segmented[..., 6] = 30.0 * params_segmented[..., 6]

    class MockBounds:
        def __init__(self):
            self.mins = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0])
            self.maxs = torch.tensor([3.5, 100.0, 30.0, 2.0, 0.05, 0.03, 30.0])

    bounds = MockBounds()

    wavelengths = make_wavelengths_from_values(400, 2500, L)
    segments = [(400, 900), (900, 1000), (1000, 2000), (2000, 2500)]

    wl_weights = create_wavelength_weights(num_wavelengths=L)

    physics_loss = PhysicsInformedLoss(
        param_bounds=bounds,
        wavelength_weights=wl_weights,
        lambda_spectral=1.0,
        lambda_weighted=0.5,
        lambda_param_penalty=0.1,
        lambda_smoothness=0.01,
        lambda_derivative=0.01,
        lambda_segment_continuity=0.1,
        wavelengths=wavelengths,
        spectral_segments=segments,
        continuity_width=2,
    )

    total_loss, losses = physics_loss(y_fake, y_real, params_segmented)

    print("=" * 60)
    print("Physics-Informed Loss with segmented params")
    print("=" * 60)
    for key, val in losses.items():
        print(f"  {key:24s}: {float(val.detach().cpu()):.6f}")

    print("\n" + "=" * 60)
    print("Adversarial loss with tensor/list/dict outputs")
    print("=" * 60)

    D_real = torch.randn(B, 1, 128)
    D_fake = torch.randn(B, 1, 128)
    D_real_dict = {"global": D_real, "segments": [torch.randn(B, 1, 32), torch.randn(B, 1, 16)]}
    D_fake_dict = {"global": D_fake, "segments": [torch.randn(B, 1, 32), torch.randn(B, 1, 16)]}

    for loss_type in ["lsgan", "vanilla", "wgan"]:
        adv = AdversarialLoss(loss_type=loss_type)
        print(f"\n{loss_type.upper()}:")
        print(f"  tensor D loss: {float(adv.discriminator_loss(D_real, D_fake)):.6f}")
        print(f"  tensor G loss: {float(adv.generator_loss(D_fake)):.6f}")
        print(f"  dict   D loss: {float(adv.discriminator_loss(D_real_dict, D_fake_dict)):.6f}")
        print(f"  dict   G loss: {float(adv.generator_loss(D_fake_dict)):.6f}")

    sam = SpectralAngleMapper()(y_fake, y_real)
    print("\n" + "=" * 60)
    print(f"SAM: {float(sam):.6f} rad ({float(sam) * 180 / np.pi:.2f} deg)")
    print("=" * 60)
    print("✅ All tests passed!")
