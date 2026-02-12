import torch
import os

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42  # For reproducible results

# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Data Paths (CSV-based multispectral)
# ============================================================
# RECOMMENDATION: Use environment variables or relative paths for portability
# Example: DATA_ROOT = os.environ.get('DATA_ROOT', '/default/path')

# CSV must include: blue, green, red, nir, red_edge, spectral, Species, Stages
TRAIN_CSV = "/home/jruben/Code/pix2spectral/Data/Dataset_with_images.csv"
VAL_CSV = "/home/jruben/Code/pix2spectral/Data/Dataset_with_images.csv"

# Root directory where the band images referenced in the CSV live
TRAIN_IMG_DIR = "/home/jruben/Data/EstradaDataset/Avocado/Multispectral_Images/"
VAL_IMG_DIR = TRAIN_IMG_DIR  # Use same directory for validation

# Optional filtering (set to None to disable)
SPECIES_FILTER = "Avocado"
STAGE_FILTER = "Fresh"  # Note: This will now correctly filter TO "Fresh" samples

# ============================================================
# Patch extraction (ROI-only patches; no transforms)
# ============================================================
# Patch size for each band patch
PATCH_H = 32
PATCH_W = 32

# Stride controls patch overlap; = patch size means non-overlapping
# Smaller stride = more patches per image (more data but slower)
STRIDE_H = 16
STRIDE_W = 16

# ROI masking threshold: pixels <= BLACK_THR are considered background (black)
# Increase (e.g. 2.0 or 5.0) if background isn't exactly 0 due to noise/compression
BLACK_THR = 0.0

# ============================================================
# Training hyperparameters
# ============================================================
LEARNING_RATE = 2e-4
BATCH_SIZE = 8              # Start smaller since PROSPECT is CPU-heavy
NUM_WORKERS = 4             # Increase if CPU is underutilized
NUM_EPOCHS = 100            # GANs typically need 100+ epochs

# Loss weights
L1_LAMBDA = 100             # Spectrum reconstruction weight (pix2pix-style)
                            # Higher = more emphasis on accurate spectral reconstruction
                            # Lower = more emphasis on adversarial realism

# ============================================================
# Model Architecture
# ============================================================
# Patch encoder embedding size and base channels (small since patches are 32x32)
EMBED_DIM = 64
BASE_FEATURES = 8

# ============================================================
# PROSPECT-D Parameter Bounds
# ============================================================
# These bounds constrain the generator's output to physically plausible leaf parameters
# Order: (Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant)
# 
# Default ranges (from PROSPECT-D literature):
#   Nleaf (leaf structure):        1.0 - 3.5   (can go up to 3.6)
#   Cab (chlorophyll a+b):         0.0 - 100.0 (μg/cm²)
#   Car (carotenoids):             0.0 - 30.0  (μg/cm²)
#   Cbrown (brown pigments):       0.0 - 2.0   (unitless)
#   Cw (water content):            0.0001 - 0.05 (g/cm² or cm)
#   Cm (dry matter):               0.0001 - 0.03 (g/cm²)
#   Ant (anthocyanins):            0.0 - 30.0  (μg/cm²)
#
# RECOMMENDATION: Analyze your dataset to see if these bounds are appropriate
# You can adjust these based on your specific leaf types

PROSPECT_PARAM_MINS = [1.0,   0.0,  0.0,  0.0,  0.0001, 0.0001, 0.0]
PROSPECT_PARAM_MAXS = [3.5, 100.0, 30.0,  2.0,  0.0500, 0.0300, 30.0]

# ============================================================
# Checkpointing
# ============================================================
LOAD_MODEL = False          # Load from checkpoint if True
SAVE_MODEL = True           # Save checkpoints during training
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

# ============================================================
# Logging and Evaluation
# ============================================================
SAVE_INTERVAL = 5           # Save checkpoints every N epochs
PLOT_INTERVAL = 1           # Save validation plots every N epochs
LOG_FILE = "training_log.json"  # JSON log file for metrics

# ============================================================
# Notes and TODOs
# ============================================================
# TODO: Verify that wavelength truncation in dataset.py ([:,50:]) is correct
#       PROSPECT-D outputs 2101 wavelengths (400-2500nm), so removing first 50
#       means you're starting at ~450nm. Document this decision.
#
# TODO: Consider adding data augmentation (horizontal/vertical flips of patches)
#
# TODO: Profile the code to identify bottlenecks:
#       - Is PROSPECT forward pass the slowest part?
#       - Are there too many/too few patches per sample?
#       - Is the dataloader saturating the GPU?
