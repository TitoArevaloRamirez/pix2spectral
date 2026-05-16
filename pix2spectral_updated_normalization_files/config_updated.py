import os
import torch

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Data Paths (CSV-based multispectral)
# ============================================================
TRAIN_CSV = "./Data/dataset_splits_70_20_10/avocado_train_minimal.csv"
VAL_CSV = "./Data/dataset_splits_70_20_10/avocado_train_minimal.csv"

# Root directory where the band images referenced in the CSV live.
TRAIN_IMG_DIR = "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/"
VAL_IMG_DIR = TRAIN_IMG_DIR

# Output folders / files. Tilde is expanded by the training script.
RESULTS_DIR = "~/Results/pix2spectral"
OUTDIR_PLOT = os.path.join(RESULTS_DIR, "plots")

# Optional filtering.
# Use STAGE_FILTER="all" or None to train with all dehydration stages.
SPECIES_FILTER = "Avocado"
STAGE_FILTER = "all"

# ============================================================
# Patch extraction
# ============================================================
PATCH_H = 32
PATCH_W = 32

# Smaller stride = more candidate patches.
STRIDE_H = 4
STRIDE_W = 4

BLACK_THR = 0.0
LEAF_COVERAGE = 0.90
MIN_PATCHES = 10
MAX_PATCHES_PER_BAND = None
MASK_METHOD = "contour"
BORDER_ERODE_PX = 2

# Validation stride can be denser or coarser than training.
VAL_STRIDE_H = PATCH_H
VAL_STRIDE_W = PATCH_W

# ============================================================
# Image normalization before patch generation
# ============================================================
# IMAGE_NORMALIZATION_SCOPE options:
#   "none"        -> no image normalization
#   "stage_band"  -> normalize each band separately within each dehydration stage
#   "global_band" -> normalize each band separately using all dehydration stages
#
# IMAGE_NORMALIZATION_METHOD options:
#   "zscore"        -> (x - mean) / std
#   "robust_zscore" -> clip to percentiles, then (x - mean) / std
#   "minmax"        -> clip to percentiles, then scale to [0, 1]
IMAGE_NORMALIZATION_SCOPE = "stage_band"
IMAGE_NORMALIZATION_METHOD = "robust_zscore"

# Recommended for z-score modes. For minmax, use (0.0, 1.0).
IMAGE_NORMALIZATION_OUTPUT_CLIP = (-5.0, 5.0)

# Compute stats from training images only and reuse them for validation/test.
COMPUTE_NORMALIZATION_STATS = True
RECOMPUTE_NORMALIZATION_STATS = True
NORMALIZATION_STATS_PATH = os.path.join(RESULTS_DIR, "image_normalization_stats.json")

# Compute statistics over detected leaf pixels rather than black background.
NORMALIZATION_USE_LEAF_MASK = True
NORMALIZATION_SAMPLE_PIXELS_PER_IMAGE = 20000
NORMALIZATION_LOWER_PERCENTILE = 1.0
NORMALIZATION_UPPER_PERCENTILE = 99.0

# ============================================================
# Training hyperparameters
# ============================================================
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_EPOCHS = 6000

# Loss weights
L1_LAMBDA = 100

# ============================================================
# Model Architecture
# ============================================================
EMBED_DIM = 64
BASE_FEATURES = 8

# ============================================================
# PROSPECT-D Parameter Bounds
# ============================================================
PROSPECT_PARAM_MINS = [1.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0]
PROSPECT_PARAM_MAXS = [300.0, 1500.0, 100.0, 100.0, 1.0, 1.0, 30.0]

# ============================================================
# Checkpointing
# ============================================================
LOAD_MODEL = True
SAVE_MODEL = True

CHECKPOINT_DISC = os.path.join(RESULTS_DIR, "disc_all_last.pth.tar")
CHECKPOINT_GEN = os.path.join(RESULTS_DIR, "gen_all_last.pth.tar")

BEST_CHECKPOINT_DISC = os.path.join(RESULTS_DIR, "disc_all_best.pth.tar")
BEST_CHECKPOINT_GEN = os.path.join(RESULTS_DIR, "gen_all_best.pth.tar")
FINAL_CHECKPOINT_GEN = os.path.join(RESULTS_DIR, "gen_all_final_best.pth.tar")

RESUME_FROM_BEST = False

# ============================================================
# Best-model selection and early stopping
# ============================================================
# Recommended while debugging numerical stability: "val_l1".
# After val_physics_total is confirmed finite, you can switch to "val_physics_total".
BEST_MODEL_METRIC = "val_l1"
BEST_MODEL_MODE = "min"
EARLY_STOP_MIN_DELTA = 1e-6
EARLY_STOP_PATIENCE = 1000
EARLY_STOP_MIN_EPOCHS = 500
EARLY_STOP_ENABLED = True

# ============================================================
# Logging and Evaluation
# ============================================================
SAVE_INTERVAL = 5
PLOT_INTERVAL = 1
LOG_FILE = os.path.join(RESULTS_DIR, "training_all_log.json")
