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
# TRAIN_CSV = "/home/jruben/Code/pix2spectral/Data/Dataset_with_images.csv"
# VAL_CSV = "/home/jruben/Code/pix2spectral/Data/Dataset_with_images.csv"
TRAIN_CSV = "./Data/train_avocado.csv"
VAL_CSV = "./Data/val_avocado.csv"

OUTDIR_PLOT = "~/Results/pix2spectral/plots/"

# Root directory where the band images referenced in the CSV live
TRAIN_IMG_DIR = "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/"
VAL_IMG_DIR = TRAIN_IMG_DIR  # Use same directory for validation

# Optional filtering (set to None to disable)
SPECIES_FILTER = "Avocado"
STAGE_FILTER = "all"  # Note: This will now correctly filter TO "Fresh" samples

# ============================================================
# Patch extraction (ROI-only patches; no transforms)
# ============================================================
# Patch size for each band patch
PATCH_H = 32
PATCH_W = 32

# Stride controls patch overlap; = patch size means non-overlapping
# Smaller stride = more patches per image (more data but slower)
STRIDE_H = 4
STRIDE_W = 4

# ROI masking threshold: pixels <= BLACK_THR are considered background (black)
# Increase (e.g. 2.0 or 5.0) if background isn't exactly 0 due to noise/compression
BLACK_THR = 0.0

LEAF_COVERAGE = 0.9  # 0.90,
MIN_PATCHES = 10  # 10,
MASK_METHOD = "contour"

# ============================================================
# Training hyperparameters
# ============================================================
LEARNING_RATE = 2e-4
BATCH_SIZE = 2  # Start smaller since PROSPECT is CPU-heavy
NUM_WORKERS = 4  # Increase if CPU is underutilized
NUM_EPOCHS = 6000  # GANs typically need 100+ epochs

# Loss weights
L1_LAMBDA = 100  # Spectrum reconstruction weight (pix2pix-style)
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

PROSPECT_PARAM_MINS = [1.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0]
PROSPECT_PARAM_MAXS = [300.0, 1500.0, 100.0, 100.0, 1.0, 1.0, 30.0]

# ============================================================
# Checkpointing
# ============================================================
LOAD_MODEL = True
SAVE_MODEL = True

# These are "last epoch / resume" checkpoints.
CHECKPOINT_DISC = "~/Results/pix2spectral/disc_all_last.pth.tar"
CHECKPOINT_GEN = "~/Results/pix2spectral/gen_all_last.pth.tar"

# These are the best-validation checkpoints.
BEST_CHECKPOINT_DISC = "~/Results/pix2spectral/disc_all_best.pth.tar"
BEST_CHECKPOINT_GEN = "~/Results/pix2spectral/gen_all_best.pth.tar"

# Final exported model copied from the best-validation generator.
FINAL_CHECKPOINT_GEN = "~/Results/pix2spectral/gen_all_final_best.pth.tar"

# If True, LOAD_MODEL loads BEST_CHECKPOINT_* when available.
# If False, LOAD_MODEL loads CHECKPOINT_* for normal training resume.
RESUME_FROM_BEST = False

# ============================================================
# Best-model selection and early stopping
# ============================================================
# Recommended choices:
#   "val_l1"             -> simple spectral reconstruction error
#   "val_rmse"           -> penalizes larger errors more strongly
#   "val_physics_total"  -> full validation physics-informed loss
#   "val_sam_deg"        -> spectral angle in degrees
BEST_MODEL_METRIC = "val_physics_total"

# "min" for losses/errors, "max" for scores where higher is better.
BEST_MODEL_MODE = "min"

# Minimum improvement required to reset early-stopping patience.
# For loss metrics, improvement means:
#   new_metric < best_metric - EARLY_STOP_MIN_DELTA
EARLY_STOP_MIN_DELTA = 1e-6

# Stop if validation metric does not improve for this many epochs.
EARLY_STOP_PATIENCE = 1000

# Do not early-stop before this epoch.
EARLY_STOP_MIN_EPOCHS = 500

# Enable / disable early stopping.
EARLY_STOP_ENABLED = True

# ============================================================
# Logging and Evaluation
# ============================================================
SAVE_INTERVAL = 5
PLOT_INTERVAL = 1
LOG_FILE = "training_all_log.json"
# Optional: keep validation patch extraction explicit.
# You may set these equal to STRIDE_H/STRIDE_W for denser validation.
VAL_STRIDE_H = PATCH_H
VAL_STRIDE_W = PATCH_W
