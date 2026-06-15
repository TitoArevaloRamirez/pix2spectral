# ============================================================
# PROSPECT-D Parameter Bounds
# ============================================================
# PROSPECT_PARAM_MINS = [1.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0]
# PROSPECT_PARAM_MAXS = [10.0, 150.0, 40.0, 10.0, 1.0, 1.0, 30.0]
#
#

import os
import torch

# ============================================================
# Environment helpers
# ============================================================


def _env_str(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    value = str(value)
    if value.strip().lower() in ["none", "null"]:
        return None
    return value


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


def _expand(path):
    if path is None:
        return None
    return os.path.expanduser(str(path))


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = _env_int("PIX2SPECTRAL_RANDOM_SEED", 42)


# ============================================================
# Device
# ============================================================

DEVICE = _env_str(
    "PIX2SPECTRAL_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)


# ============================================================
# Experiment identity and output paths
# ============================================================

RESULTS_DIR = _expand(
    _env_str(
        "PIX2SPECTRAL_RESULTS_DIR",
        "~/Results/pix2spectral",
    )
)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = _env_str(
    "PIX2SPECTRAL_EXPERIMENT_NAME",
    "pix2spectral_experiment",
)

# Output files are stage/experiment-specific.
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

os.makedirs(OUTDIR_PLOT, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


# ============================================================
# Data paths and dataset filters
# ============================================================
# These are now fully environment-driven so the experiment runner can control them.
#
# Example environment variables:
#   PIX2SPECTRAL_TRAIN_CSV=~/Code/pix2spectral/Data/.../avocado_train.csv
#   PIX2SPECTRAL_VAL_CSV=~/Code/pix2spectral/Data/.../avocado_val.csv
#   PIX2SPECTRAL_TEST_CSV=~/Code/pix2spectral/Data/.../avocado_test.csv
#   PIX2SPECTRAL_IMG_DIR="/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/"
#   PIX2SPECTRAL_SPECIES_FILTER=Avocado
#   PIX2SPECTRAL_STAGE_FILTER=dry

TRAIN_CSV = _expand(
    _env_str(
        "PIX2SPECTRAL_TRAIN_CSV",
        "./Data/dataset_splits_70_20_10/avocado_train.csv",
    )
)

VAL_CSV = _expand(
    _env_str(
        "PIX2SPECTRAL_VAL_CSV",
        "./Data/dataset_splits_70_20_10/avocado_val.csv",
    )
)

TEST_CSV = _expand(
    _env_str(
        "PIX2SPECTRAL_TEST_CSV",
        "./Data/dataset_splits_70_20_10/avocado_test.csv",
    )
)

# One common root for all splits. Specific split directories can override this.
IMG_DIR = _expand(
    _env_str(
        "PIX2SPECTRAL_IMG_DIR",
        "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/",
    )
)

TRAIN_IMG_DIR = _expand(_env_str("PIX2SPECTRAL_TRAIN_IMG_DIR", IMG_DIR))
VAL_IMG_DIR = _expand(_env_str("PIX2SPECTRAL_VAL_IMG_DIR", IMG_DIR))
TEST_IMG_DIR = _expand(_env_str("PIX2SPECTRAL_TEST_IMG_DIR", IMG_DIR))

# Use None or "all" to disable species filtering.
SPECIES_FILTER = _env_str("PIX2SPECTRAL_SPECIES_FILTER", "vineyard")
if SPECIES_FILTER is not None and str(SPECIES_FILTER).strip().lower() in [
    "all",
    "any",
    "*",
    "",
]:
    SPECIES_FILTER = None

# Use "all" to train/evaluate using all dehydration stages.
STAGE_FILTER = _env_str("PIX2SPECTRAL_STAGE_FILTER", "all")


# ============================================================
# Wavelength grid
# ============================================================

WAVELENGTH_MIN = _env_float("PIX2SPECTRAL_WAVELENGTH_MIN", 400.0)
WAVELENGTH_MAX = _env_float("PIX2SPECTRAL_WAVELENGTH_MAX", 2500.0)
WAVELENGTH_COUNT = _env_int("PIX2SPECTRAL_WAVELENGTH_COUNT", 2101)

# Lowercase aliases for compatibility with make_wavelengths(cfg)-style code.
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


# ============================================================
# Patch extraction
# ============================================================

PATCH_H = _env_int("PIX2SPECTRAL_PATCH_H", 32)
PATCH_W = _env_int("PIX2SPECTRAL_PATCH_W", 32)

STRIDE_H = _env_int("PIX2SPECTRAL_STRIDE_H", 16)
STRIDE_W = _env_int("PIX2SPECTRAL_STRIDE_W", 16)

VAL_STRIDE_H = _env_int("PIX2SPECTRAL_VAL_STRIDE_H", PATCH_H)
VAL_STRIDE_W = _env_int("PIX2SPECTRAL_VAL_STRIDE_W", PATCH_W)

BLACK_THR = _env_float("PIX2SPECTRAL_BLACK_THR", 0.0)
LEAF_COVERAGE = _env_float("PIX2SPECTRAL_LEAF_COVERAGE", 0.90)
MIN_PATCHES = _env_int("PIX2SPECTRAL_MIN_PATCHES", 10)

# Keep this controlled for speed. Increase only if you really need many patches.
MAX_PATCHES_PER_BAND = _env_int("PIX2SPECTRAL_MAX_PATCHES_PER_BAND", 500)  # 500

MASK_METHOD = _env_str("PIX2SPECTRAL_MASK_METHOD", "contour")
BORDER_ERODE_PX = _env_int("PIX2SPECTRAL_BORDER_ERODE_PX", 2)

CACHE_PATCHES = _env_bool("PIX2SPECTRAL_CACHE_PATCHES", True)
CLONE_CACHED_ITEMS = _env_bool("PIX2SPECTRAL_CLONE_CACHED_ITEMS", False)


# ============================================================
# Image normalization before patch generation
# ============================================================
# IMAGE_NORMALIZATION_SCOPE:
#   "none"
#   "stage_band"
#   "global_band"
#
# IMAGE_NORMALIZATION_METHOD:
#   "zscore"
#   "robust_zscore"
#   "minmax"

IMAGE_NORMALIZATION_SCOPE = _env_str(
    "PIX2SPECTRAL_IMAGE_NORMALIZATION_SCOPE",
    "global_band",
)

IMAGE_NORMALIZATION_METHOD = _env_str(
    "PIX2SPECTRAL_IMAGE_NORMALIZATION_METHOD",
    "robust_zscore",
)

if IMAGE_NORMALIZATION_SCOPE == "none":
    IMAGE_NORMALIZATION_MODE = "none"
else:
    IMAGE_NORMALIZATION_MODE = _env_str(
        "PIX2SPECTRAL_IMAGE_NORMALIZATION_MODE",
        f"{IMAGE_NORMALIZATION_SCOPE}_{IMAGE_NORMALIZATION_METHOD}",
    )

# Recommended for z-score modes. For minmax, use (0.0, 1.0).
IMAGE_NORMALIZATION_OUTPUT_CLIP = (-5.0, 5.0)

COMPUTE_NORMALIZATION_STATS = _env_bool(
    "PIX2SPECTRAL_COMPUTE_NORMALIZATION_STATS", True
)
RECOMPUTE_NORMALIZATION_STATS = _env_bool(
    "PIX2SPECTRAL_RECOMPUTE_NORMALIZATION_STATS", True
)

NORMALIZATION_STATS_PATH = os.path.join(RESULTS_DIR, "image_normalization_stats.json")

NORMALIZATION_USE_LEAF_MASK = _env_bool(
    "PIX2SPECTRAL_NORMALIZATION_USE_LEAF_MASK", True
)
NORMALIZATION_SAMPLE_PIXELS_PER_IMAGE = _env_int(
    "PIX2SPECTRAL_NORMALIZATION_SAMPLE_PIXELS_PER_IMAGE",
    20000,
)
NORMALIZATION_LOWER_PERCENTILE = _env_float(
    "PIX2SPECTRAL_NORMALIZATION_LOWER_PERCENTILE",
    1.0,
)
NORMALIZATION_UPPER_PERCENTILE = _env_float(
    "PIX2SPECTRAL_NORMALIZATION_UPPER_PERCENTILE",
    99.0,
)


# ============================================================
# Generator architecture
# ============================================================

PATCH_ENCODER_TYPE = _env_str("PIX2SPECTRAL_PATCH_ENCODER_TYPE", "cnn")
POOLING_TYPE = _env_str("PIX2SPECTRAL_POOLING_TYPE", "attention_stats")
BAND_ENCODER_MODE = _env_str("PIX2SPECTRAL_BAND_ENCODER_MODE", "separate")
NORM_TYPE = _env_str("PIX2SPECTRAL_NORM_TYPE", "group")

EMBED_DIM = _env_int("PIX2SPECTRAL_EMBED_DIM", 64)
BASE_FEATURES = _env_int("PIX2SPECTRAL_BASE_FEATURES", 8)

# Group A controls.
GROUPA_VARIANT_ID = _env_str("PIX2SPECTRAL_GROUPA_VARIANT_ID", "")
GROUPA_VARIANT_NAME = _env_str("PIX2SPECTRAL_GROUPA_VARIANT_NAME", "")

USE_SEGMENTED_PROSPECT = _env_bool(
    "PIX2SPECTRAL_USE_SEGMENTED_PROSPECT",
    True,
)

USE_SEGMENT_RESIDUAL = _env_bool(
    "PIX2SPECTRAL_USE_SEGMENT_RESIDUAL",
    True,
)

SEGMENT_RESIDUAL_SCALE = _env_float(
    "PIX2SPECTRAL_SEGMENT_RESIDUAL_SCALE",
    0.05,
)

SEGMENT_BLEND_WIDTH = _env_int("PIX2SPECTRAL_SEGMENT_BLEND_WIDTH", 0)


# ============================================================
# Discriminator architecture
# ============================================================

DISCRIMINATOR_MODE = _env_str(
    "PIX2SPECTRAL_DISCRIMINATOR_MODE",
    "global",
)

USE_WAVELENGTH_CHANNEL = _env_bool("PIX2SPECTRAL_USE_WAVELENGTH_CHANNEL", True)
USE_SPECTRAL_NORM = _env_bool("PIX2SPECTRAL_USE_SPECTRAL_NORM", True)
DISCRIMINATOR_FEATURES = (64, 128, 256, 512)


# ============================================================
# Training hyperparameters
# ============================================================

LEARNING_RATE = _env_float("PIX2SPECTRAL_LEARNING_RATE", 2e-4)
BATCH_SIZE = _env_int("PIX2SPECTRAL_BATCH_SIZE", 2)
NUM_WORKERS = _env_int("PIX2SPECTRAL_NUM_WORKERS", 0)
NUM_EPOCHS = _env_int("PIX2SPECTRAL_NUM_EPOCHS", 300)  # 500

PERSISTENT_WORKERS = _env_bool("PIX2SPECTRAL_PERSISTENT_WORKERS", False)
PREFETCH_FACTOR = _env_int("PIX2SPECTRAL_PREFETCH_FACTOR", 1)

# Adversarial/physics tradeoff used by your training script.
L1_LAMBDA = _env_float("PIX2SPECTRAL_L1_LAMBDA", 100.0)


# ============================================================
# PROSPECT-D parameter bounds
# ============================================================
# These are safer than the previously very broad values.
# Wider bounds can trigger non-finite PROSPECT/Jacobian outputs.

PROSPECT_PARAM_MINS = [1.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0]
PROSPECT_PARAM_MAXS = [10.0, 150.0, 40.0, 10.0, 1.0, 1.0, 30.0]


# ============================================================
# Loss profile and physics-informed loss weights
# ============================================================

LOSS_PROFILE = _env_str(
    "PIX2SPECTRAL_LOSS_PROFILE",
    "L5_FULL_PHYSICS",
)


LAMBDA_SPECTRAL = _env_float("PIX2SPECTRAL_LAMBDA_SPECTRAL", 1.0)
LAMBDA_WEIGHTED = _env_float("PIX2SPECTRAL_LAMBDA_WEIGHTED", 0.5)
LAMBDA_PARAM_PENALTY = _env_float("PIX2SPECTRAL_LAMBDA_PARAM_PENALTY", 0.1)
LAMBDA_SMOOTHNESS = _env_float("PIX2SPECTRAL_LAMBDA_SMOOTHNESS", 0.01)
LAMBDA_DERIVATIVE = _env_float("PIX2SPECTRAL_LAMBDA_DERIVATIVE", 0.01)
LAMBDA_SEGMENT_CONTINUITY = _env_float("PIX2SPECTRAL_LAMBDA_SEGMENT_CONTINUITY", 0.1)

# Segment continuity only applies to segmented generator variants.
if not USE_SEGMENTED_PROSPECT:
    LAMBDA_SEGMENT_CONTINUITY = 0.0


# ============================================================
# Checkpointing
# ============================================================

LOAD_MODEL = _env_bool("PIX2SPECTRAL_LOAD_MODEL", False)
SAVE_MODEL = _env_bool("PIX2SPECTRAL_SAVE_MODEL", True)
RESUME_FROM_BEST = _env_bool("PIX2SPECTRAL_RESUME_FROM_BEST", False)


# ============================================================
# Best-model selection and early stopping
# ============================================================

BEST_MODEL_METRIC = _env_str("PIX2SPECTRAL_BEST_MODEL_METRIC", "val_l1")
BEST_MODEL_MODE = _env_str("PIX2SPECTRAL_BEST_MODEL_MODE", "min")

EARLY_STOP_MIN_DELTA = _env_float("PIX2SPECTRAL_EARLY_STOP_MIN_DELTA", 1e-6)
EARLY_STOP_PATIENCE = _env_int("PIX2SPECTRAL_EARLY_STOP_PATIENCE", 100)  # 200
EARLY_STOP_MIN_EPOCHS = _env_int("PIX2SPECTRAL_EARLY_STOP_MIN_EPOCHS", 100)  # 200
EARLY_STOP_ENABLED = _env_bool("PIX2SPECTRAL_EARLY_STOP_ENABLED", True)


# ============================================================
# Logging and evaluation
# ============================================================

SAVE_INTERVAL = _env_int("PIX2SPECTRAL_SAVE_INTERVAL", 5)
PLOT_INTERVAL = _env_int("PIX2SPECTRAL_PLOT_INTERVAL", 1)


# ============================================================
# Debug print helper
# ============================================================


def print_config_summary():
    print("=" * 80)
    print("pix2spectral config summary")
    print("=" * 80)
    print(f"EXPERIMENT_NAME:          {EXPERIMENT_NAME}")
    print(f"RESULTS_DIR:              {RESULTS_DIR}")
    print(f"TRAIN_CSV:                {TRAIN_CSV}")
    print(f"VAL_CSV:                  {VAL_CSV}")
    print(f"TEST_CSV:                 {TEST_CSV}")
    print(f"TRAIN_IMG_DIR:            {TRAIN_IMG_DIR}")
    print(f"VAL_IMG_DIR:              {VAL_IMG_DIR}")
    print(f"TEST_IMG_DIR:             {TEST_IMG_DIR}")
    print(f"SPECIES_FILTER:           {SPECIES_FILTER}")
    print(f"STAGE_FILTER:             {STAGE_FILTER}")
    print(f"DISCRIMINATOR_MODE:       {DISCRIMINATOR_MODE}")
    print(f"USE_SEGMENTED_PROSPECT:   {USE_SEGMENTED_PROSPECT}")
    print(f"USE_SEGMENT_RESIDUAL:     {USE_SEGMENT_RESIDUAL}")
    print(f"LOSS_PROFILE:             {LOSS_PROFILE}")
    print(f"BEST_MODEL_METRIC:        {BEST_MODEL_METRIC}")
    print("=" * 80)
