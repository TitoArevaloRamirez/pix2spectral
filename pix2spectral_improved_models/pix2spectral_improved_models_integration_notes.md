# pix2spectral improved generator/discriminator integration notes

## Files

- `generator_model_improved.py`
- `discriminator_model_improved.py`

They are designed to replace your existing `generator_model.py` and `discriminator_model.py`, or to be imported under the new filenames during testing.

## Recommended config additions

```python
# ============================================================
# Wavelength grid
# ============================================================
WAVELENGTH_MIN = 400.0
WAVELENGTH_MAX = 2500.0
WAVELENGTH_COUNT = 2101

# Lowercase aliases for TrainConfig-style utilities
wavelength_min = WAVELENGTH_MIN
wavelength_max = WAVELENGTH_MAX
wavelength_count = WAVELENGTH_COUNT

# ============================================================
# Spectral segmentation
# ============================================================
SPECTRAL_SEGMENTS = [
    (400.0, 900.0),
    (900.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 2500.0),
]

USE_SEGMENTED_PROSPECT = True
USE_SEGMENT_RESIDUAL = True
SEGMENT_RESIDUAL_SCALE = 0.05
LAMBDA_SEGMENT_CONTINUITY = 0.1

# ============================================================
# Generator architecture
# ============================================================
PATCH_ENCODER_TYPE = "cnn"          # "cnn" or "unet"
POOLING_TYPE = "attention_stats"    # "mean" or "attention_stats"
BAND_ENCODER_MODE = "shared"        # "shared" or "separate"
NORM_TYPE = "group"                 # "group", "batch", "instance", "none"

# ============================================================
# Discriminator architecture
# ============================================================
DISCRIMINATOR_MODE = "global_plus_segmented"  # "global", "segmented", "global_plus_segmented"
USE_WAVELENGTH_CHANNEL = True
USE_SPECTRAL_NORM = True
DISCRIMINATOR_FEATURES = (64, 128, 256, 512)
```

## Recommended training imports

```python
from generator_model_improved import MultiSpectralPatchToProspectGenerator
from discriminator_model_improved import SegmentedSpectralDiscriminator1D
```

or rename the improved files to your original filenames and keep your imports unchanged for the generator class.

## Recommended generator construction

```python
gen = MultiSpectralPatchToProspectGenerator(
    bands=["blue", "green", "red", "nir", "red_edge"],
    base_features=config.BASE_FEATURES,
    embed_dim=config.EMBED_DIM,
    mins=config.PROSPECT_PARAM_MINS,
    maxs=config.PROSPECT_PARAM_MAXS,
    wavelength_min=config.WAVELENGTH_MIN,
    wavelength_max=config.WAVELENGTH_MAX,
    wavelength_count=config.WAVELENGTH_COUNT,
    spectral_segments=config.SPECTRAL_SEGMENTS,
    use_segmented_prospect=config.USE_SEGMENTED_PROSPECT,
    use_segment_residual=config.USE_SEGMENT_RESIDUAL,
    segment_residual_scale=config.SEGMENT_RESIDUAL_SCALE,
    patch_encoder_type=config.PATCH_ENCODER_TYPE,
    pooling_type=config.POOLING_TYPE,
    band_encoder_mode=config.BAND_ENCODER_MODE,
    norm_type=config.NORM_TYPE,
).to(device)
```

## Recommended discriminator construction

```python
disc = SegmentedSpectralDiscriminator1D(
    in_channels=1,
    features=config.DISCRIMINATOR_FEATURES,
    use_bn=False,
    wavelength_min=config.WAVELENGTH_MIN,
    wavelength_max=config.WAVELENGTH_MAX,
    wavelength_count=config.WAVELENGTH_COUNT,
    spectral_segments=config.SPECTRAL_SEGMENTS,
    mode=config.DISCRIMINATOR_MODE,
    use_wavelength_channel=config.USE_WAVELENGTH_CHANNEL,
    use_spectral_norm=config.USE_SPECTRAL_NORM,
).to(device)
```

## Physics loss compatibility

The improved segmented generator returns:

```python
y_fake:  [B, L]
pParams: [B, S, 7]
```

If your `PhysicsInformedLoss` expects parameter tensors shaped `[B, 7]`, flatten the parameter tensor before the parameter-penalty part, or modify the loss to accept `[B, S, 7]`.

A simple helper is:

```python
def flatten_segmented_params(p_params):
    if p_params.dim() == 3:
        return p_params.reshape(-1, p_params.shape[-1])
    return p_params
```

## Segment continuity loss

Add this to the generator loss if segmented PROSPECT is enabled:

```python
continuity = gen.segment_boundary_loss(y_fake)
G_loss = G_adv.float() + config.L1_LAMBDA * G_physics + config.LAMBDA_SEGMENT_CONTINUITY * continuity
```

This discourages artificial discontinuities at 900, 1000, and 2000 nm.
