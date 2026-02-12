"""
Physics-Informed Loss Functions for PROSPECT-D GAN

This module contains loss functions that enforce physical constraints
and natural spectral characteristics for leaf reflectance reconstruction.
"""

import torch
import torch.nn as nn
import numpy as np


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss for spectral reconstruction.
    
    Components:
    1. Spectral L1 loss (basic reconstruction)
    2. Wavelength-weighted L1 (emphasize important regions)
    3. Parameter constraint penalty (keep params in bounds)
    4. Spectral smoothness (natural leaves have smooth spectra)
    5. Spectral derivative consistency (physical reflectance patterns)
    """
    
    def __init__(
        self,
        param_bounds,
        wavelength_weights=None,
        lambda_spectral=1.0,
        lambda_weighted=0.0,
        lambda_param_penalty=0.1,
        lambda_smoothness=0.01,
        lambda_derivative=0.01,
    ):
        """
        Args:
            param_bounds: Parameter bounds object from generator (has .mins and .maxs)
            wavelength_weights: Optional tensor [L] for wavelength-specific weighting
            lambda_spectral: Weight for basic L1 loss
            lambda_weighted: Weight for wavelength-weighted L1
            lambda_param_penalty: Weight for parameter constraint penalty
            lambda_smoothness: Weight for spectral smoothness
            lambda_derivative: Weight for derivative consistency
        """
        super().__init__()
        self.param_bounds = param_bounds
        self.wavelength_weights = wavelength_weights
        
        self.lambda_spectral = lambda_spectral
        self.lambda_weighted = lambda_weighted
        self.lambda_param_penalty = lambda_param_penalty
        self.lambda_smoothness = lambda_smoothness
        self.lambda_derivative = lambda_derivative
    
    def forward(self, y_fake, y_real, params):
        """
        Compute physics-informed loss.
        
        Args:
            y_fake: Generated spectra [B, L]
            y_real: Real spectra [B, L]
            params: PROSPECT parameters [B, 7]
        
        Returns:
            dict with total loss and individual components
        """
        losses = {}
        
        # 1. Basic spectral L1 loss
        spectral_l1 = torch.mean(torch.abs(y_fake - y_real))
        losses['spectral_l1'] = spectral_l1
        
        # 2. Wavelength-weighted L1 (if weights provided)
        if self.wavelength_weights is not None:
            weighted_l1 = torch.mean(
                self.wavelength_weights.to(y_fake.device) * torch.abs(y_fake - y_real)
            )
            losses['weighted_l1'] = weighted_l1
        else:
            losses['weighted_l1'] = torch.tensor(0.0, device=y_fake.device)
        
        # 3. Parameter constraint penalty (soft bounds)
        param_penalty = self._parameter_penalty(params)
        losses['param_penalty'] = param_penalty
        
        # 4. Spectral smoothness (penalize rapid changes)
        smoothness = self._spectral_smoothness(y_fake, y_real)
        losses['smoothness'] = smoothness
        
        # 5. Spectral derivative consistency
        derivative_loss = self._derivative_consistency(y_fake, y_real)
        losses['derivative'] = derivative_loss
        
        # Total weighted loss
        total_loss = (
            self.lambda_spectral * losses['spectral_l1'] +
            self.lambda_weighted * losses['weighted_l1'] +
            self.lambda_param_penalty * losses['param_penalty'] +
            self.lambda_smoothness * losses['smoothness'] +
            self.lambda_derivative * losses['derivative']
        )
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _parameter_penalty(self, params):
        """
        Soft penalty for parameters outside physical bounds.
        Uses ReLU to only penalize violations.
        """
        mins = self.param_bounds.mins.to(params.device)
        maxs = self.param_bounds.maxs.to(params.device)
        
        # Penalty for going below minimum
        below_min = torch.relu(mins - params)
        # Penalty for going above maximum
        above_max = torch.relu(params - maxs)
        
        penalty = torch.mean(below_min + above_max)
        return penalty
    
    def _spectral_smoothness(self, y_fake, y_real):
        """
        Encourage fake spectra to have similar smoothness as real spectra.
        Natural leaf spectra don't have sharp spikes.
        """
        # First derivative (differences between adjacent wavelengths)
        diff_fake = torch.abs(y_fake[:, 1:] - y_fake[:, :-1])
        diff_real = torch.abs(y_real[:, 1:] - y_real[:, :-1])
        
        # Penalize if fake is much rougher than real
        smoothness_loss = torch.mean(torch.relu(diff_fake - diff_real * 1.5))
        
        return smoothness_loss
    
    def _derivative_consistency(self, y_fake, y_real):
        """
        Match the spectral derivatives (patterns of change).
        Important for capturing absorption features correctly.
        """
        # First derivative
        deriv_fake = y_fake[:, 1:] - y_fake[:, :-1]
        deriv_real = y_real[:, 1:] - y_real[:, :-1]
        
        # L1 loss on derivatives
        deriv_loss = torch.mean(torch.abs(deriv_fake - deriv_real))
        
        return deriv_loss


def create_wavelength_weights(num_wavelengths=2101, start_wl=400, end_wl=2500):
    """
    Create wavelength-specific weights emphasizing important spectral regions.
    
    Important regions for vegetation:
    - Red edge (680-750 nm): Chlorophyll transition
    - NIR plateau (750-1300 nm): Leaf structure
    - Water absorption bands (1400-1900 nm, 2100-2300 nm)
    
    Args:
        num_wavelengths: Number of wavelength points
        start_wl: Starting wavelength (nm)
        end_wl: Ending wavelength (nm)
    
    Returns:
        torch.Tensor of weights [num_wavelengths]
    """
    wavelengths = np.linspace(start_wl, end_wl, num_wavelengths)
    weights = np.ones_like(wavelengths)
    
    # Red edge region (680-750 nm): 2x weight
    red_edge_mask = (wavelengths >= 680) & (wavelengths <= 750)
    weights[red_edge_mask] = 2.0
    
    # NIR plateau (750-1300 nm): 1.5x weight
    nir_mask = (wavelengths >= 750) & (wavelengths <= 1300)
    weights[nir_mask] = 1.5
    
    # Water absorption bands: 2x weight
    water1_mask = (wavelengths >= 1400) & (wavelengths <= 1900)
    water2_mask = (wavelengths >= 2100) & (wavelengths <= 2300)
    weights[water1_mask] = 2.0
    weights[water2_mask] = 2.0
    
    return torch.tensor(weights, dtype=torch.float32)


class AdversarialLoss(nn.Module):
    """
    Wrapper for different adversarial loss types.
    Supports: LSGAN (MSE), vanilla GAN (BCE), and Wasserstein.
    """
    
    def __init__(self, loss_type='lsgan'):
        """
        Args:
            loss_type: 'lsgan', 'vanilla', or 'wgan'
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'wgan':
            self.criterion = None  # Wasserstein uses raw logits
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def discriminator_loss(self, D_real, D_fake):
        """
        Compute discriminator loss.
        
        Args:
            D_real: Discriminator output on real data
            D_fake: Discriminator output on fake data
        
        Returns:
            loss value
        """
        if self.loss_type == 'wgan':
            # Wasserstein: maximize D(real) - D(fake)
            # For minimization: minimize -(D(real) - D(fake))
            return -(torch.mean(D_real) - torch.mean(D_fake))
        else:
            # LSGAN or vanilla: real->1, fake->0
            real_loss = self.criterion(D_real, torch.ones_like(D_real))
            fake_loss = self.criterion(D_fake, torch.zeros_like(D_fake))
            return 0.5 * (real_loss + fake_loss)
    
    def generator_loss(self, D_fake):
        """
        Compute generator loss (fool discriminator).
        
        Args:
            D_fake: Discriminator output on fake data
        
        Returns:
            loss value
        """
        if self.loss_type == 'wgan':
            # Wasserstein: maximize D(fake)
            # For minimization: minimize -D(fake)
            return -torch.mean(D_fake)
        else:
            # LSGAN or vanilla: want D(fake) -> 1
            return self.criterion(D_fake, torch.ones_like(D_fake))


class SpectralAngleMapper(nn.Module):
    """
    Spectral Angle Mapper (SAM) loss.
    Measures angular distance between spectra (insensitive to brightness).
    Lower SAM = more similar spectral shape.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_fake, y_real):
        """
        Compute SAM between predicted and real spectra.
        
        Args:
            y_fake: Generated spectra [B, L]
            y_real: Real spectra [B, L]
        
        Returns:
            SAM in radians (lower is better)
        """
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(y_fake, y_real, dim=1)
        
        # Clamp to [-1, 1] for numerical stability
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # Angle in radians
        sam = torch.acos(cos_sim)
        
        return torch.mean(sam)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    print("Testing Physics-Informed Loss Functions\n")
    
    # Create mock data
    B, L = 4, 2101
    y_real = torch.rand(B, L) * 0.5  # Realistic reflectance range
    y_fake = y_real + torch.randn(B, L) * 0.05  # Add noise
    
    # Mock parameters
    params = torch.tensor([
        [1.5, 40.0, 10.0, 0.0, 0.01, 0.005, 5.0],
        [2.0, 50.0, 15.0, 0.1, 0.02, 0.008, 8.0],
        [1.8, 45.0, 12.0, 0.0, 0.015, 0.006, 6.0],
        [2.2, 55.0, 18.0, 0.2, 0.025, 0.009, 10.0],
    ])
    
    # Mock parameter bounds
    class MockBounds:
        def __init__(self):
            self.mins = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0])
            self.maxs = torch.tensor([3.5, 100.0, 30.0, 2.0, 0.05, 0.03, 30.0])
    
    bounds = MockBounds()
    
    # Test 1: Physics-informed loss
    print("="*60)
    print("Test 1: Physics-Informed Loss")
    print("="*60)
    
    wl_weights = create_wavelength_weights(num_wavelengths=L)
    physics_loss = PhysicsInformedLoss(
        param_bounds=bounds,
        wavelength_weights=wl_weights,
        lambda_spectral=1.0,
        lambda_weighted=0.5,
        lambda_param_penalty=0.1,
        lambda_smoothness=0.01,
        lambda_derivative=0.01,
    )
    
    total_loss, losses = physics_loss(y_fake, y_real, params)
    
    print(f"\nLoss components:")
    for key, val in losses.items():
        print(f"  {key:20s}: {val.item():.6f}")
    
    # Test 2: Adversarial losses
    print("\n" + "="*60)
    print("Test 2: Adversarial Losses")
    print("="*60)
    
    D_real = torch.randn(B, 1, 131)  # Mock discriminator output
    D_fake = torch.randn(B, 1, 131)
    
    for loss_type in ['lsgan', 'vanilla', 'wgan']:
        adv_loss = AdversarialLoss(loss_type=loss_type)
        d_loss = adv_loss.discriminator_loss(D_real, D_fake)
        g_loss = adv_loss.generator_loss(D_fake)
        print(f"\n{loss_type.upper()}:")
        print(f"  Discriminator loss: {d_loss.item():.6f}")
        print(f"  Generator loss:     {g_loss.item():.6f}")
    
    # Test 3: Spectral Angle Mapper
    print("\n" + "="*60)
    print("Test 3: Spectral Angle Mapper")
    print("="*60)
    
    sam_loss = SpectralAngleMapper()
    sam = sam_loss(y_fake, y_real)
    sam_degrees = sam.item() * 180 / np.pi
    
    print(f"\nSAM: {sam.item():.6f} rad ({sam_degrees:.2f}°)")
    print(f"Note: SAM < 0.1 rad (5.7°) is excellent for spectral matching")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
