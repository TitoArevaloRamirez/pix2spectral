import os
import torch


# ============================================================ Environment-driven experiment naming and stage sweep support
# ============================================================
# Add this block near the top of config.py, after `import os`.


def _env_str(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ["1", "true", "yes", "y", "on"]


RESULTS_DIR = os.path.expanduser(
    _env_str(
        "PIX2SPECTRAL_RESULTS_DIR",
        "~/Results/pix2spectral_segmented/",
    )
)

EXPERIMENT_NAME = _env_str("PIX2SPECTRAL_EXPERIMENT_NAME", "pix2spectral_all")

# Then replace these existing config variables with env-aware versions:

STAGE_FILTER = _env_str("PIX2SPECTRAL_STAGE_FILTER", "all")

BATCH_SIZE = _env_int("PIX2SPECTRAL_BATCH_SIZE", 2)
NUM_WORKERS = _env_int("PIX2SPECTRAL_NUM_WORKERS", 4)
NUM_EPOCHS = _env_int("PIX2SPECTRAL_NUM_EPOCHS", 500)

LOAD_MODEL = _env_bool("PIX2SPECTRAL_LOAD_MODEL", False)
RESUME_FROM_BEST = _env_bool("PIX2SPECTRAL_RESUME_FROM_BEST", False)

# Normalization / architecture overrides, if using the improved files.
IMAGE_NORMALIZATION_SCOPE = _env_str(
    "PIX2SPECTRAL_IMAGE_NORMALIZATION_SCOPE",
    "global_band",
)
BAND_ENCODER_MODE = _env_str(
    "PIX2SPECTRAL_BAND_ENCODER_MODE",
    "separate",
)

# Stage-specific output paths.
CHECKPOINT_DISC = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_disc_last.pth.tar")
CHECKPOINT_GEN = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_gen_last.pth.tar")

BEST_CHECKPOINT_DISC = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_disc_best.pth.tar")
BEST_CHECKPOINT_GEN = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}_gen_best.pth.tar")

FINAL_CHECKPOINT_GEN = os.path.join(
    RESULTS_DIR,
    f"{EXPERIMENT_NAME}_gen_final_best.pth.tar",
)

OUTDIR_PLOT = os.path.join(RESULTS_DIR, "plots", EXPERIMENT_NAME)
LOG_FILE = os.path.join(RESULTS_DIR, "logs", f"{EXPERIMENT_NAME}_training_log.json")


# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Wavelength grid
# ============================================================
WAVELENGTH_MIN = 400.0
WAVELENGTH_MAX = 2500.0
WAVELENGTH_COUNT = 2101
wavelength_min = WAVELENGTH_MIN
wavelength_max = WAVELENGTH_MAX
wavelength_count = WAVELENGTH_COUNT

# ============================================================
# Spectral segmentation
# ============================================================
SPECTRAL_SEGMENTS = [
    (400.0, 700.0),
    (700.0, 800.0),
    (800.0, 1400.0),
    (1400.0, 2500.0),
]

USE_SEGMENTED_PROSPECT = True
USE_SEGMENT_RESIDUAL = True
SEGMENT_RESIDUAL_SCALE = 0.05
LAMBDA_SEGMENT_CONTINUITY = 0.1


# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Data Paths (CSV-based multispectral)
# ============================================================
TRAIN_CSV = "./Data/dataset_splits_70_20_10/vineyard_train.csv"
VAL_CSV = "./Data/dataset_splits_70_20_10/vineyard_val.csv"

# Root directory where the band images referenced in the CSV live.
TRAIN_IMG_DIR = "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/"
VAL_IMG_DIR = TRAIN_IMG_DIR


# STAGE_FILTER = "dry"
# Output folders / files. Tilde is expanded by the training script.
# RESULTS_DIR = os.path.join("~/Results/pix2spectral_segmented/avocado2/", STAGE_FILTER)
# OUTDIR_PLOT = os.path.join(RESULTS_DIR, "plots")

# Optional filtering.
# Use STAGE_FILTER="all" or None to train with all dehydration stages.
SPECIES_FILTER = "vineyard"

# ============================================================
# Patch extraction
# ============================================================
PATCH_H = 32
PATCH_W = 32

# Smaller stride = more candidate patches.
STRIDE_H = 16
STRIDE_W = 16

BLACK_THR = 0.0
LEAF_COVERAGE = 0.90
MIN_PATCHES = 10
MAX_PATCHES_PER_BAND = 500
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
# IMAGE_NORMALIZATION_SCOPE = "global_band"
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
# Generator architecture
# ============================================================
PATCH_ENCODER_TYPE = "cnn"  # "cnn" or "unet"
POOLING_TYPE = "attention_stats"  # "mean" or "attention_stats"
# BAND_ENCODER_MODE = "separate"  # "shared" or "separate"
NORM_TYPE = "group"  # "group", "batch", "instance", "none"

# ============================================================
# Discriminator architecture
# ============================================================
DISCRIMINATOR_MODE = "segmented"  # "global", "segmented", "global_plus_segmented"
USE_WAVELENGTH_CHANNEL = True
USE_SPECTRAL_NORM = True
DISCRIMINATOR_FEATURES = (64, 128, 256, 512)


# ============================================================
# Training hyperparameters
# ============================================================
LEARNING_RATE = 2e-4
# BATCH_SIZE = 2
# NUM_WORKERS = 0
# NUM_EPOCHS = 1500

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
PROSPECT_PARAM_MAXS = [10.0, 150.0, 40.0, 10.0, 1.0, 1.0, 30.0]

# ============================================================
# Checkpointing
# ============================================================
LOAD_MODEL = False
SAVE_MODEL = True

# CHECKPOINT_DISC = os.path.join(RESULTS_DIR, "disc_last_segmented.pth.tar")
# CHECKPOINT_GEN = os.path.join(RESULTS_DIR, "gen_last_segmented.pth.tar")
#
# BEST_CHECKPOINT_DISC = os.path.join(RESULTS_DIR, "disc_best_segmented.pth.tar")
# BEST_CHECKPOINT_GEN = os.path.join(RESULTS_DIR, "gen_best_segmented.pth.tar")
# FINAL_CHECKPOINT_GEN = os.path.join(RESULTS_DIR, "gen_final_best_segmented.pth.tar")

# RESUME_FROM_BEST = False

# ============================================================
# Best-model selection and early stopping
# ============================================================
# Recommended while debugging numerical stability: "val_l1".
# After val_physics_total is confirmed finite, you can switch to "val_physics_total".
BEST_MODEL_METRIC = "val_physics_total"
BEST_MODEL_MODE = "min"
EARLY_STOP_MIN_DELTA = 1e-6
EARLY_STOP_PATIENCE = 200
EARLY_STOP_MIN_EPOCHS = 200
EARLY_STOP_ENABLED = True

# ============================================================
# Logging and Evaluation
# ============================================================
SAVE_INTERVAL = 5
PLOT_INTERVAL = 1
# LOG_FILE = os.path.join(RESULTS_DIR, "training_log.json")
