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
    _env_str("PIX2SPECTRAL_RESULTS_DIR", "~/Results/pix2spectral")
)

EXPERIMENT_NAME = _env_str("PIX2SPECTRAL_EXPERIMENT_NAME", "pix2spectral_all")

# Then replace these existing config variables with env-aware versions:

STAGE_FILTER = _env_str("PIX2SPECTRAL_STAGE_FILTER", "all")

BATCH_SIZE = _env_int("PIX2SPECTRAL_BATCH_SIZE", 2)
NUM_WORKERS = _env_int("PIX2SPECTRAL_NUM_WORKERS", 4)
NUM_EPOCHS = _env_int("PIX2SPECTRAL_NUM_EPOCHS", 6000)

LOAD_MODEL = _env_bool("PIX2SPECTRAL_LOAD_MODEL", False)
RESUME_FROM_BEST = _env_bool("PIX2SPECTRAL_RESUME_FROM_BEST", False)

# Normalization / architecture overrides, if using the improved files.
IMAGE_NORMALIZATION_SCOPE = _env_str(
    "PIX2SPECTRAL_IMAGE_NORMALIZATION_SCOPE",
    "stage_band",
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
