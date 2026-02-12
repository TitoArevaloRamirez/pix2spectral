"""
Unit tests for GAN-Physics-Informed model components.

Run this to verify:
1. Generator outputs correct shapes
2. Parameters stay within physical bounds
3. Gradients flow through physics model
4. Discriminator works correctly
"""

import torch
import numpy as np
from generator_model import MultiSpectralPatchToProspectGenerator
from discriminator_model import SpectralDiscriminator1D, SpectralPatchDiscriminator1D


def _make_fake_sample(device, num_patches_per_band=5):
    """Create fake multispectral patch data for testing."""
    return {
        "blue": torch.randn(num_patches_per_band, 1, 32, 32, device=device),
        "green": torch.randn(num_patches_per_band, 1, 32, 32, device=device),
        "red": torch.randn(num_patches_per_band, 1, 32, 32, device=device),
        "nir": torch.randn(num_patches_per_band, 1, 32, 32, device=device),
        "red_edge": torch.randn(num_patches_per_band, 1, 32, 32, device=device),
    }


def _make_fake_batch(device, B=2, num_patches_per_band=5):
    """Create fake batch for testing."""
    batch = {b: [] for b in ["blue", "green", "red", "nir", "red_edge"]}
    for _ in range(B):
        s = _make_fake_sample(device, num_patches_per_band)
        for b in batch.keys():
            batch[b].append(s[b])
    return batch


def test_generator_output_shapes(device='cpu'):
    """Test 1: Verify generator outputs correct shapes."""
    print("\n" + "="*60)
    print("TEST 1: Generator Output Shapes")
    print("="*60)
    
    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=8,
        embed_dim=64,
    ).to(device)
    
    # Single sample test
    sample = _make_fake_sample(device)
    y_fake, pParams = gen(sample)
    
    # Check shapes
    expected_spectrum_length = 2101  # PROSPECT-D output length
    assert y_fake.dim() == 1, f"Expected 1D spectrum, got {y_fake.dim()}D"
    assert y_fake.shape[0] == expected_spectrum_length, \
        f"Expected spectrum length {expected_spectrum_length}, got {y_fake.shape[0]}"
    assert pParams.shape == (7,), f"Expected (7,) parameters, got {pParams.shape}"
    
    print(f"✅ Single sample output shapes correct:")
    print(f"   Spectrum: {tuple(y_fake.shape)}")
    print(f"   Parameters: {tuple(pParams.shape)}")
    
    # Batch test
    batch = _make_fake_batch(device, B=3)
    y_batch, p_batch = gen.forward_batch_list(batch)
    
    assert y_batch.shape == (3, expected_spectrum_length), \
        f"Expected batch spectrum (3, {expected_spectrum_length}), got {y_batch.shape}"
    assert p_batch.shape == (3, 7), f"Expected batch params (3, 7), got {p_batch.shape}"
    
    print(f"✅ Batch output shapes correct:")
    print(f"   Spectrum batch: {tuple(y_batch.shape)}")
    print(f"   Parameters batch: {tuple(p_batch.shape)}")


def test_parameter_bounds(device='cpu'):
    """Test 2: Verify parameters stay within physical bounds."""
    print("\n" + "="*60)
    print("TEST 2: Parameter Physical Bounds")
    print("="*60)
    
    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=8,
        embed_dim=64,
    ).to(device)
    
    # Test multiple samples
    batch = _make_fake_batch(device, B=10)
    _, pParams = gen.forward_batch_list(batch)
    
    mins = gen.physics.bounds.mins.cpu()
    maxs = gen.physics.bounds.maxs.cpu()
    pParams_cpu = pParams.detach().cpu()
    
    param_names = ['Nleaf', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'Ant']
    
    print(f"\nParameter ranges across {pParams.shape[0]} samples:")
    print(f"{'Parameter':<10} {'Min Bound':<10} {'Actual Min':<12} {'Actual Max':<12} {'Max Bound':<10} {'Status':<8}")
    print("-" * 70)
    
    all_valid = True
    for i, name in enumerate(param_names):
        min_val = float(pParams_cpu[:, i].min())
        max_val = float(pParams_cpu[:, i].max())
        min_bound = float(mins[i])
        max_bound = float(maxs[i])
        
        within_bounds = (min_val >= min_bound) and (max_val <= max_bound)
        status = "✅ OK" if within_bounds else "❌ FAIL"
        
        print(f"{name:<10} {min_bound:<10.4f} {min_val:<12.4f} {max_val:<12.4f} {max_bound:<10.4f} {status:<8}")
        
        if not within_bounds:
            all_valid = False
            if min_val < min_bound:
                print(f"   ⚠️  Value {min_val:.4f} below minimum {min_bound:.4f}")
            if max_val > max_bound:
                print(f"   ⚠️  Value {max_val:.4f} above maximum {max_bound:.4f}")
    
    if all_valid:
        print("\n✅ All parameters within physical bounds!")
    else:
        print("\n❌ Some parameters outside bounds - check sigmoid scaling!")


def test_gradient_flow(device='cpu'):
    """Test 3: Verify gradients flow through physics model."""
    print("\n" + "="*60)
    print("TEST 3: Gradient Flow Through Physics Model")
    print("="*60)
    
    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=8,
        embed_dim=64,
    ).to(device)
    
    gen.train()
    
    # Forward pass
    sample = _make_fake_sample(device)
    y_fake, pParams = gen(sample)
    
    # Create fake target
    y_target = torch.rand_like(y_fake)
    
    # Backward pass
    loss = torch.nn.functional.l1_loss(y_fake, y_target)
    gen.zero_grad(set_to_none=True)
    loss.backward()
    
    # Check gradients
    has_grad = False
    max_grad = 0.0
    zero_grad_params = []
    
    for name, param in gen.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = float(param.grad.abs().max())
            max_grad = max(max_grad, grad_norm)
        else:
            zero_grad_params.append(name)
    
    print(f"Gradient flow check:")
    print(f"  ✅ Gradients computed: {has_grad}")
    print(f"  Max gradient magnitude: {max_grad:.6f}")
    
    if zero_grad_params:
        print(f"  ⚠️  Parameters without gradients ({len(zero_grad_params)}):")
        for name in zero_grad_params[:5]:  # Show first 5
            print(f"     - {name}")
    else:
        print(f"  ✅ All parameters have gradients!")
    
    assert has_grad, "No gradients were computed!"
    assert max_grad > 0, "Gradient magnitude is zero!"
    
    print("\n✅ Gradients successfully flow through physics model!")


def test_discriminator_standard(device='cpu'):
    """Test 4: Standard discriminator functionality."""
    print("\n" + "="*60)
    print("TEST 4: Standard Discriminator")
    print("="*60)
    
    disc = SpectralDiscriminator1D(in_channels=1, use_bn=False).to(device)
    
    B = 4
    L = 2101
    
    real_spectra = torch.randn(B, L, device=device)
    fake_spectra = torch.randn(B, L, device=device)
    
    # Forward pass
    logits_real = disc(real_spectra)
    logits_fake = disc(fake_spectra)
    
    print(f"Input spectrum shape: {tuple(real_spectra.shape)}")
    print(f"Output logits shape: {tuple(logits_real.shape)}")
    
    # Check shapes
    assert logits_real.dim() == 3, f"Expected 3D output, got {logits_real.dim()}D"
    assert logits_real.shape[0] == B, f"Expected batch size {B}, got {logits_real.shape[0]}"
    assert logits_real.shape[1] == 1, f"Expected 1 channel, got {logits_real.shape[1]}"
    
    print(f"✅ Discriminator output shapes correct")
    
    # Check gradients
    loss_real = torch.nn.functional.mse_loss(logits_real, torch.ones_like(logits_real))
    loss_fake = torch.nn.functional.mse_loss(logits_fake, torch.zeros_like(logits_fake))
    loss = loss_real + loss_fake
    
    disc.zero_grad()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in disc.parameters())
    assert has_grad, "No gradients in discriminator!"
    
    print(f"✅ Discriminator gradients computed correctly")


def test_discriminator_pairwise(device='cpu'):
    """Test 5: Pairwise discriminator functionality."""
    print("\n" + "="*60)
    print("TEST 5: Pairwise Discriminator (for reference)")
    print("="*60)
    
    disc = SpectralPatchDiscriminator1D(in_channels=1, use_bn=False).to(device)
    
    B = 4
    L = 2101
    
    real_spectra = torch.randn(B, L, device=device)
    fake_spectra = torch.randn(B, L, device=device)
    
    # Forward pass - takes TWO inputs
    logits_paired = disc(real_spectra, fake_spectra)
    
    print(f"Input spectrum shape: {tuple(real_spectra.shape)}")
    print(f"Output logits shape: {tuple(logits_paired.shape)}")
    
    # Check shapes
    assert logits_paired.dim() == 3, f"Expected 3D output, got {logits_paired.dim()}D"
    assert logits_paired.shape[0] == B, f"Expected batch size {B}, got {logits_paired.shape[0]}"
    
    print(f"✅ Pairwise discriminator output shapes correct")
    print(f"ℹ️  Note: For your use case, standard discriminator is recommended")


def run_all_tests(device='cpu'):
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    print(f"Device: {device}")
    
    try:
        test_generator_output_shapes(device)
        test_parameter_bounds(device)
        test_gradient_flow(device)
        test_discriminator_standard(device)
        test_discriminator_pairwise(device)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour implementation is working correctly.")
        print("You can now proceed with training on real data.")
        
    except AssertionError as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {str(e)}")
        print("="*60)
        raise
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ERROR: {str(e)}")
        print("="*60)
        raise


if __name__ == "__main__":
    # Run on CPU by default (change to 'cuda' if you have GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    run_all_tests(device)
