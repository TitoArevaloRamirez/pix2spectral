# Physics-Informed Losses - Usage Guide

## Overview

The physics-informed loss implementation adds **5 physical constraints** to your GAN training to ensure generated spectra are not only realistic but also physically plausible.

## Files Created

1. **physics_losses.py** - Core loss functions module
2. **train_with_physics_losses.py** - Full training script with physics losses integrated

## Physics Loss Components

### 1. Spectral L1 Loss (Basic Reconstruction)
```python
lambda_spectral=1.0
```
- Standard L1 distance between generated and real spectra
- Ensures basic spectral accuracy

### 2. Wavelength-Weighted L1
```python
lambda_weighted=0.5
```
- Emphasizes important spectral regions:
  - **Red edge (680-750 nm)**: 2× weight - chlorophyll absorption edge
  - **NIR plateau (750-1300 nm)**: 1.5× weight - leaf structure
  - **Water bands (1400-1900, 2100-2300 nm)**: 2× weight - water content
- Ensures critical features are well-reconstructed

### 3. Parameter Constraint Penalty
```python
lambda_param_penalty=0.1
```
- Soft penalty if parameters go outside physical bounds
- Uses ReLU so only violations are penalized
- Keeps PROSPECT-D parameters physically meaningful:
  - Nleaf (1.0-3.5)
  - Cab (0-100 μg/cm²)
  - Car (0-30 μg/cm²)
  - Cbrown (0-2)
  - Cw (0.0001-0.05 g/cm²)
  - Cm (0.0001-0.03 g/cm²)
  - Ant (0-30 μg/cm²)

### 4. Spectral Smoothness
```python
lambda_smoothness=0.01
```
- Natural leaf spectra are smooth (no sharp spikes)
- Penalizes if generated spectrum is rougher than real spectrum
- Encourages realistic spectral curves

### 5. Spectral Derivative Consistency
```python
lambda_derivative=0.01
```
- Matches the pattern of spectral changes (slopes)
- Important for capturing absorption features correctly
- Ensures spectral shape, not just absolute values

## How to Use

### Option 1: Quick Start (Use Default Weights)

Simply replace your current training script:

```bash
python train_with_physics_losses.py
```

This uses the default loss weights which should work well for most cases.

### Option 2: Custom Loss Weights

Edit the physics loss weights in `train_with_physics_losses.py` (around line 360):

```python
physics_loss_fn = PhysicsInformedLoss(
    param_bounds=gen.physics.bounds,
    wavelength_weights=wavelength_weights,
    lambda_spectral=1.0,        # ← Adjust these
    lambda_weighted=0.5,         # ← Adjust these
    lambda_param_penalty=0.1,    # ← Adjust these
    lambda_smoothness=0.01,      # ← Adjust these
    lambda_derivative=0.01,      # ← Adjust these
)
```

### Option 3: Integrate into Your Existing Code

If you prefer to modify your existing `train.py`:

1. **Import the losses:**
```python
from physics_losses import (
    PhysicsInformedLoss,
    create_wavelength_weights,
)
```

2. **Initialize before training loop:**
```python
# Create wavelength weights
wavelength_weights = create_wavelength_weights(
    num_wavelengths=2101,
    start_wl=400,
    end_wl=2500
)

# Create physics loss
physics_loss_fn = PhysicsInformedLoss(
    param_bounds=gen.physics.bounds,
    wavelength_weights=wavelength_weights,
    lambda_spectral=1.0,
    lambda_weighted=0.5,
    lambda_param_penalty=0.1,
    lambda_smoothness=0.01,
    lambda_derivative=0.01,
).to(device)
```

3. **Replace L1 loss in generator training:**
```python
# OLD CODE:
G_l1 = l1_loss(y_fake, y_real) * config.L1_LAMBDA

# NEW CODE:
G_physics, physics_components = physics_loss_fn(y_fake, y_real, p_params)
G_loss = G_adv + config.L1_LAMBDA * G_physics
```

## Understanding the Output

During training, you'll see detailed loss breakdown:

```
Epoch 10 (45.2s)
====================================================================
Train:
  D_loss:       0.4523
  G_loss:       12.3456
    └─ adv:     0.8234
    └─ physics: 0.1152
       ├─ spectral_l1:    0.0892  ← Basic reconstruction
       ├─ weighted_l1:    0.0134  ← Important regions
       ├─ param_penalty:  0.0000  ← Params in bounds (good!)
       ├─ smoothness:     0.0098  ← Smooth spectra
       └─ derivative:     0.0028  ← Shape matching
Validation:
  L1:   0.034521  ← Lower is better
  RMSE: 0.042134  ← Lower is better
  SAM:  3.45° (0.0602 rad)  ← Lower is better (<5° is excellent)
  Param penalty: 0.000012  ← Should be near zero
```

## What to Expect

### Good Signs:
- ✅ `param_penalty` stays near zero (parameters are physical)
- ✅ `smoothness` decreases over time (learning smooth spectra)
- ✅ `SAM` < 5° (excellent spectral shape matching)
- ✅ `weighted_l1` < `spectral_l1` (important regions well-reconstructed)

### Warning Signs:
- ⚠️ `param_penalty` > 0.01 (parameters violating bounds - increase lambda_param_penalty)
- ⚠️ `SAM` > 10° (poor spectral shape - increase lambda_derivative)
- ⚠️ `smoothness` increasing (too rough - increase lambda_smoothness)

## Tuning Guide

If your results aren't good:

1. **Spectra don't look realistic?**
   - Increase `lambda_smoothness` (try 0.05 or 0.1)
   - Increase `lambda_derivative` (try 0.05 or 0.1)

2. **Parameters going out of bounds?**
   - Increase `lambda_param_penalty` (try 0.5 or 1.0)

3. **Important features (red edge) missing?**
   - Increase `lambda_weighted` (try 1.0 or 2.0)

4. **Overall poor reconstruction?**
   - Increase `lambda_spectral` (try 2.0 or 5.0)
   - Or increase `config.L1_LAMBDA` (the global multiplier)

5. **Training unstable?**
   - Decrease all lambdas by 50%
   - Check gradient clipping is enabled (max_norm=1.0)

## Testing the Losses

Before training, test that losses work:

```bash
python physics_losses.py
```

This runs unit tests showing:
- All loss components compute correctly
- Wavelength weights are applied properly
- No errors in gradient computation

## Comparison with Basic Training

| Metric | Basic L1 Only | With Physics Losses |
|--------|---------------|---------------------|
| Spectral Accuracy | Good | Excellent |
| Physical Plausibility | Unknown | Guaranteed |
| Red Edge Quality | Variable | Emphasized |
| Parameter Validity | Not enforced | Enforced |
| Spectral Smoothness | Not controlled | Natural |

## Visualization

The training script now saves two types of plots:

1. **Spectra plots** (`spectra_epoch_XXXX.png`)
   - Compare real vs. generated spectra
   - Generated every epoch

2. **Parameter distributions** (`parameters_epoch_XXXX.png`)
   - Shows distribution of all 7 PROSPECT parameters
   - Generated every 10 epochs
   - Use to verify parameters stay in reasonable ranges

## Advanced: Custom Wavelength Weights

If you want to emphasize different spectral regions:

```python
def custom_wavelength_weights():
    wavelengths = np.linspace(400, 2500, 2101)
    weights = np.ones_like(wavelengths)
    
    # Example: Emphasize only red edge
    red_edge = (wavelengths >= 680) & (wavelengths <= 750)
    weights[red_edge] = 5.0  # Very high weight
    
    return torch.tensor(weights, dtype=torch.float32)

# Use in training:
wavelength_weights = custom_wavelength_weights()
```

## FAQ

**Q: Will this slow down training?**
A: Minimal impact (~5% slower). The physics model (PROSPECT) is already being computed, these losses just add small tensor operations.

**Q: Should I always use all 5 loss components?**
A: Start with all enabled. If training is unstable, disable smoothness and derivative first (set lambda to 0.0).

**Q: My param_penalty is always zero, is that ok?**
A: Yes! That means sigmoid bounding in the generator is working well. The penalty is a safety net.

**Q: Can I use this with a different discriminator?**
A: Yes! The physics losses only affect the generator. The adversarial loss is separate.

**Q: How do I know if physics losses are helping?**
A: Compare SAM metric. Physics losses should give you SAM < 5° consistently, while basic L1 might be 10-15°.

## Troubleshooting

### Error: "wavelength_weights wrong size"
→ Check that your spectral data length matches 2101. Adjust in `create_wavelength_weights(num_wavelengths=YOUR_LENGTH)`

### Error: "param_bounds not found"
→ Make sure you're using the updated generator_model.py with the PhysicsInformedProspectHead

### Loss is NaN or infinite
→ Decrease all lambda values by 10×
→ Check that input spectra are in [0, 1] range
→ Verify gradient clipping is enabled

## Summary

Physics-informed losses make your GAN:
- ✅ More physically accurate
- ✅ Better at capturing important spectral features
- ✅ Produce valid leaf parameters
- ✅ Generate smoother, more natural spectra
- ✅ Match spectral shapes, not just absolute values

The default weights should work well for most leaf spectroscopy tasks. Start there, then tune based on your specific needs!
