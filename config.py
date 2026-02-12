import torch

# ============================================================
# Device
# ============================================================
DEVICE = "cuda"  # CPU for testing; change to "cuda" later


# ============================================================
# Data (CSV-based multispectral)
# ============================================================
# CSV must include: blue, green, red, nir, red_edge, spectral, species, stage
TRAIN_CSV = "/home/jruben/Code/pix2spectral/Data/Dataset_with_images.csv"
VAL_CSV = "/home/jruben/Code/pix2spectral/Data/Dataset_with_images.csv"

# Root directory where the band images referenced in the CSV live
TRAIN_IMG_DIR = "/home/jruben/Data/EstradaDataset/Avocado/Multispectral_Images/"
VAL_IMG_DIR = "/home/jruben/Data/EstradaDataset/Avocado/Multispectral_Images/"

# Optional filtering (set to None to disable)
SPECIES_FILTER = "Avocado"
STAGE_FILTER = "Fresh"
# ============================================================
# Patch extraction (ROI-only patches; no transforms)
# ============================================================
# Patch size for each band patch
PATCH_H = 32
PATCH_W = 32

# Stride controls patch overlap; = patch size means non-overlapping
STRIDE_H = 16
STRIDE_W = 16

# ROI masking threshold: pixels <= BLACK_THR are considered background (black)
# Increase (e.g. 2.0 or 5.0) if background isn't exactly 0 due to noise/compression
BLACK_THR = 0.0


# ============================================================
# Training
# ============================================================
LEARNING_RATE = 2e-4
BATCH_SIZE = 8              # start smaller since PROSPECT is CPU-heavy
NUM_WORKERS = 4             # safer on CPU/mac; set >0 later if stable
NUM_EPOCHS = 50             # for CPU test; increase later

L1_LAMBDA = 100             # spectrum reconstruction weight (pix2pix-style)

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


# ============================================================
# Model sizes
# ============================================================
# Patch encoder embedding size and base channels (small since patches are 32x32)
EMBED_DIM = 64
BASE_FEATURES = 8
