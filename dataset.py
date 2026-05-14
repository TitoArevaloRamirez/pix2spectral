import os
import ast
import warnings
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


try:
    import cv2
except ImportError:
    cv2 = None


# -------------------------------------------------------------------------
# Spectral utilities
# -------------------------------------------------------------------------


def get_spectral_data(data, spectral_col="spectral", drop_first_n=50):
    """
    Parse the spectral signatures stored in the CSV.

    Expected CSV format:
        spectral column contains a Python-like list string, e.g.
        "[0.12, 0.13, 0.14, ...]"

    Args:
        data: pandas DataFrame.
        spectral_col: name of the spectral column.
        drop_first_n: remove first N spectral values, preserving behavior
                      from the original dataset.py.

    Returns:
        np.ndarray [num_samples, spectral_length]
    """
    x_data = data[spectral_col].to_list()

    parsed = []
    for i, s in enumerate(x_data):
        try:
            parsed.append(ast.literal_eval(s))
        except Exception as exc:
            raise ValueError(
                f"Could not parse spectral signature at row {i}. Value was: {s}"
            ) from exc

    np_data = np.asarray(parsed, dtype=np.float32)

    if np_data.ndim != 2:
        raise ValueError(
            f"Parsed spectral data should be 2D, got shape {np_data.shape}."
        )

    if drop_first_n is not None and drop_first_n > 0:
        if np_data.shape[1] <= drop_first_n:
            raise ValueError(
                f"Cannot drop first {drop_first_n} spectral samples because "
                f"spectrum length is only {np_data.shape[1]}."
            )
        np_data = np_data[:, drop_first_n:]

    return np_data.astype(np.float32)


# -------------------------------------------------------------------------
# Path utilities
# -------------------------------------------------------------------------


def resolve_image_path(fname, root_dir=None):
    """
    Robust path resolver for paths stored in the CSV.

    Handles:
      - absolute paths
      - relative paths
      - Windows-style paths stored in CSV
      - root_dir + basename fallback

    The original code used:
        parts = fname.split("\\")
        path = os.path.join(root_dir, parts[1])

    That is fragile because it assumes Windows separators and exactly two parts.
    """
    fname = str(fname).strip()

    if fname == "" or fname.lower() == "nan":
        raise ValueError("Empty image filename in CSV.")

    # 1. Direct absolute path.
    p = Path(fname)
    if p.is_absolute() and p.exists():
        return str(p)

    # 2. root_dir + original relative path.
    if root_dir is not None and root_dir != "":
        candidate = Path(root_dir) / fname
        if candidate.exists():
            return str(candidate)

    # 3. Windows basename fallback.
    win_name = PureWindowsPath(fname).name

    if root_dir is not None and root_dir != "":
        candidate = Path(root_dir) / win_name
        if candidate.exists():
            return str(candidate)

    # 4. POSIX basename fallback.
    posix_name = Path(fname).name

    if root_dir is not None and root_dir != "":
        candidate = Path(root_dir) / posix_name
        if candidate.exists():
            return str(candidate)

    # 5. If nothing exists, return the most plausible path for the error msg.
    if root_dir is not None and root_dir != "":
        return str(Path(root_dir) / win_name)

    return fname


# -------------------------------------------------------------------------
# Image and mask utilities
# -------------------------------------------------------------------------


def read_grayscale_image(path):
    """
    Read an image as float32 grayscale without forcing it to 8-bit.

    This is safer for multispectral bands that may be 16-bit.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)
    arr = np.asarray(img)

    if arr.ndim == 3:
        # If the file unexpectedly has multiple channels, convert to grayscale.
        # Avoid PIL.convert("L") because that can collapse 16-bit data to 8-bit.
        arr = arr[..., :3].astype(np.float32).mean(axis=2)

    arr = arr.astype(np.float32)

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


def robust_uint8(arr, lower_percentile=1.0, upper_percentile=99.0):
    """
    Convert arbitrary numeric image to uint8 using robust percentile scaling.
    Used only for mask detection, not for returned patch values.
    """
    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    lo, hi = np.percentile(valid, [lower_percentile, upper_percentile])

    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return out


def _largest_connected_component_cv2(mask):
    """
    Keep only the largest connected foreground component.
    """
    if cv2 is None:
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    if num_labels <= 1:
        return mask.astype(bool)

    # Skip label 0 because it is background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    return labels == largest_label


def _clean_binary_mask_cv2(
    mask,
    close_kernel_size=15,
    open_kernel_size=5,
    keep_largest=True,
):
    """
    Morphologically clean a binary leaf mask.
    """
    if cv2 is None:
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8) * 255

    if close_kernel_size is not None and close_kernel_size > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)

    if open_kernel_size is not None and open_kernel_size > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)

    mask = mask_u8 > 0

    if keep_largest:
        mask = _largest_connected_component_cv2(mask)

    return mask.astype(bool)


def _erode_mask_cv2(mask, erode_px):
    """
    Erode mask to avoid sampling patches too close to the leaf border.
    """
    if erode_px is None or erode_px <= 0:
        return mask.astype(bool)

    if cv2 is None:
        return mask.astype(bool)

    k_size = int(2 * erode_px + 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded = cv2.erode(mask.astype(np.uint8), k, iterations=1).astype(bool)

    # Avoid destroying very small masks.
    if eroded.sum() < max(10, 0.05 * mask.sum()):
        return mask.astype(bool)

    return eroded


def detect_leaf_mask_by_contour(
    arr,
    min_leaf_area_fraction=0.005,
    max_leaf_area_fraction=0.98,
    canny_sigma=0.33,
    close_kernel_size=21,
    open_kernel_size=5,
    fallback_black_thr=0.0,
):
    """
    Detect a filled leaf mask from a single band image.

    Primary method:
      1. Robust image normalization.
      2. Canny edge detection.
      3. Morphological closing/dilation to close the leaf boundary.
      4. External contour detection.
      5. Fill the largest plausible contour.

    Fallback:
      Adaptive/Otsu-like intensity mask, largest connected component.

    Notes:
      - This is designed to replace brittle arr > black_thr masking.
      - For best behavior, install opencv-python:
            pip install opencv-python
    """
    H, W = arr.shape
    image_area = float(H * W)
    min_area = int(min_leaf_area_fraction * image_area)
    max_area = int(max_leaf_area_fraction * image_area)

    n8 = robust_uint8(arr)

    if cv2 is not None:
        blurred = cv2.GaussianBlur(n8, (5, 5), 0)

        med = float(np.median(blurred))
        lower = int(max(0, (1.0 - canny_sigma) * med))
        upper = int(min(255, (1.0 + canny_sigma) * med))

        if upper <= lower:
            lower, upper = 30, 120

        edges = cv2.Canny(blurred, lower, upper)

        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k, iterations=2)
        edges_closed = cv2.dilate(edges_closed, close_k, iterations=1)

        contours, _ = cv2.findContours(
            edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        plausible = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                plausible.append((area, contour))

        if len(plausible) > 0:
            plausible.sort(key=lambda x: x[0], reverse=True)
            _, best_contour = plausible[0]

            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [best_contour], contourIdx=-1, color=1, thickness=-1)

            mask = _clean_binary_mask_cv2(
                mask,
                close_kernel_size=close_kernel_size,
                open_kernel_size=open_kernel_size,
                keep_largest=True,
            )

            if min_area <= int(mask.sum()) <= max_area:
                return mask.astype(bool)

        # Fallback 1: Otsu foreground. Try both polarities.
        candidates = []

        _, otsu_hi = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates.append(otsu_hi > 0)
        candidates.append(otsu_hi == 0)

        # Fallback 2: old black-threshold idea, but cleaned and largest component.
        if fallback_black_thr is not None:
            candidates.append(arr > float(fallback_black_thr))

        best_mask = None
        best_score = -np.inf

        for cand in candidates:
            cand = _clean_binary_mask_cv2(
                cand,
                close_kernel_size=close_kernel_size,
                open_kernel_size=open_kernel_size,
                keep_largest=True,
            )

            area = int(cand.sum())
            if area < min_area or area > max_area:
                continue

            # Penalize masks that occupy the image border too strongly.
            border_pixels = (
                cand[0, :].sum()
                + cand[-1, :].sum()
                + cand[:, 0].sum()
                + cand[:, -1].sum()
            )
            border_ratio = border_pixels / max(1.0, float(area))

            score = float(area) * (1.0 - min(border_ratio, 0.9))

            if score > best_score:
                best_score = score
                best_mask = cand

        if best_mask is not None:
            return best_mask.astype(bool)

    # Minimal fallback if OpenCV is unavailable or all contour methods fail.
    # This is not as robust as the contour method, but keeps the dataset usable.
    warnings.warn(
        "OpenCV contour mask failed or OpenCV is not installed. "
        "Falling back to simple cleaned intensity mask. "
        "Install opencv-python for better leaf border detection.",
        RuntimeWarning,
    )

    fallback = arr > float(fallback_black_thr)

    if fallback.sum() < min_area:
        # Try robust normalized threshold as a last fallback.
        fallback = n8 > 10

    return fallback.astype(bool)


def pad_image_and_mask_to_patch_size(arr, mask, patch_h, patch_w):
    """
    Pad image/mask if image is smaller than the requested patch size.
    """
    H, W = arr.shape

    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)

    if pad_h == 0 and pad_w == 0:
        return arr, mask

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    arr_padded = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )

    mask_padded = np.pad(
        mask.astype(bool),
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=False,
    )

    return arr_padded, mask_padded


def mask_bbox(mask):
    """
    Return bounding box of mask as top, bottom, left, right inclusive/exclusive.
    """
    ys, xs = np.where(mask)

    if ys.size == 0:
        H, W = mask.shape
        return 0, H, 0, W

    top = int(ys.min())
    bottom = int(ys.max()) + 1
    left = int(xs.min())
    right = int(xs.max()) + 1

    return top, bottom, left, right


def extract_patch(arr, top, left, patch_h, patch_w):
    """
    Extract a single patch as Tensor [1, patch_h, patch_w].
    """
    patch = arr[top : top + patch_h, left : left + patch_w]
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    return torch.from_numpy(patch).unsqueeze(0)


def read_band_image_as_leaf_patches(
    path,
    patch_h,
    patch_w,
    stride_h=None,
    stride_w=None,
    min_leaf_coverage=0.85,
    min_patches=10,
    max_patches=None,
    border_erode_px=2,
    mask_method="contour",
    fallback_black_thr=0.0,
    random_seed=None,
    random_fallback=True,
    allow_replacement=True,
    return_debug=False,
):
    """
    Read one multispectral band and return leaf patches.

    Differences from the original implementation:
      - Uses contour/border-based leaf detection instead of raw threshold ROI.
      - Keeps a patch if enough of it is inside the filled leaf mask.
      - Does NOT require every pixel in the patch to be foreground.
      - Guarantees at least min_patches by fallback sampling and replacement.
      - Can optionally cap the number of returned patches using max_patches.

    Args:
        path: path to one band image.
        patch_h, patch_w: patch size.
        stride_h, stride_w: sliding-window stride. Defaults to patch size.
        min_leaf_coverage: fraction of patch mask that must be leaf.
        min_patches: minimum number of patches to return.
        max_patches: optional maximum number of patches to return.
        border_erode_px: erode mask before sampling to avoid border patches.
        mask_method: "contour" or "threshold".
        fallback_black_thr: used only as fallback.
        random_seed: deterministic seed for fallback sampling.
        random_fallback: whether to sample additional random patches if needed.
        allow_replacement: if True, duplicate valid patches to guarantee min_patches.
        return_debug: if True, return (patches, debug_dict).

    Returns:
        patches: Tensor [N, 1, patch_h, patch_w]
        optionally debug dictionary
    """
    if stride_h is None:
        stride_h = patch_h
    if stride_w is None:
        stride_w = patch_w

    patch_h = int(patch_h)
    patch_w = int(patch_w)
    stride_h = int(stride_h)
    stride_w = int(stride_w)
    min_patches = int(min_patches)

    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_h and patch_w must be positive.")

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride_h and stride_w must be positive.")

    if min_patches < 1:
        raise ValueError("min_patches must be >= 1.")

    if max_patches is not None and int(max_patches) < min_patches:
        raise ValueError("max_patches must be None or >= min_patches.")

    rng = np.random.default_rng(random_seed)

    arr = read_grayscale_image(path)

    if mask_method == "contour":
        mask = detect_leaf_mask_by_contour(
            arr,
            fallback_black_thr=fallback_black_thr,
        )
    elif mask_method == "threshold":
        raw_mask = arr > float(fallback_black_thr)
        mask = _clean_binary_mask_cv2(raw_mask, keep_largest=True)
    else:
        raise ValueError(
            f"Unknown mask_method={mask_method}. Expected 'contour' or 'threshold'."
        )

    arr, mask = pad_image_and_mask_to_patch_size(arr, mask, patch_h, patch_w)

    H, W = arr.shape
    sampling_mask = _erode_mask_cv2(mask, border_erode_px)

    top_box, bottom_box, left_box, right_box = mask_bbox(sampling_mask)

    # Restrict sliding-window search to a loose bounding box around the leaf.
    top_start = max(0, top_box - patch_h)
    top_end = min(H - patch_h, bottom_box) + 1

    left_start = max(0, left_box - patch_w)
    left_end = min(W - patch_w, right_box) + 1

    candidates = []
    seen_positions = set()

    def maybe_add_candidate(top, left, coverage_threshold):
        top = int(np.clip(top, 0, H - patch_h))
        left = int(np.clip(left, 0, W - patch_w))

        key = (top, left)
        if key in seen_positions:
            return

        patch_mask = mask[top : top + patch_h, left : left + patch_w]
        coverage = float(patch_mask.mean())

        if coverage >= coverage_threshold:
            candidates.append((coverage, top, left))
            seen_positions.add(key)

    # 1. Deterministic sliding-window candidates.
    for top in range(top_start, max(top_start, top_end), stride_h):
        for left in range(left_start, max(left_start, left_end), stride_w):
            maybe_add_candidate(top, left, min_leaf_coverage)

    # 2. Fallback: progressively relax coverage and sample around leaf pixels.
    if len(candidates) < min_patches and random_fallback:
        ys, xs = np.where(sampling_mask)

        if ys.size == 0:
            ys, xs = np.where(mask)

        relaxed_thresholds = [
            min_leaf_coverage,
            min(0.75, min_leaf_coverage),
            min(0.60, min_leaf_coverage),
            min(0.40, min_leaf_coverage),
            0.10,
            0.00,
        ]

        for threshold in relaxed_thresholds:
            if len(candidates) >= min_patches:
                break

            if ys.size == 0:
                break

            # Oversample attempts to avoid duplicates and bad border windows.
            attempts = max(200, 50 * min_patches)

            for _ in range(attempts):
                if len(candidates) >= min_patches:
                    break

                idx = int(rng.integers(0, ys.size))
                cy = int(ys[idx])
                cx = int(xs[idx])

                top = cy - patch_h // 2
                left = cx - patch_w // 2

                maybe_add_candidate(top, left, threshold)

    # 3. Absolute fallback: one centroid/image-center patch if everything failed.
    if len(candidates) == 0:
        ys, xs = np.where(mask)

        if ys.size > 0:
            cy = int(np.round(ys.mean()))
            cx = int(np.round(xs.mean()))
        else:
            cy = H // 2
            cx = W // 2

        top = int(np.clip(cy - patch_h // 2, 0, H - patch_h))
        left = int(np.clip(cx - patch_w // 2, 0, W - patch_w))

        patch_mask = mask[top : top + patch_h, left : left + patch_w]
        coverage = float(patch_mask.mean())

        candidates.append((coverage, top, left))

        warnings.warn(
            f"No valid contour-based patches found for {path}. "
            f"Using fallback center patch with coverage={coverage:.3f}.",
            RuntimeWarning,
        )

    # Sort by mask coverage, highest first.
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Optional maximum cap.
    if max_patches is not None and len(candidates) > int(max_patches):
        max_patches = int(max_patches)
        selected_idx = rng.choice(len(candidates), size=max_patches, replace=False)
        candidates = [candidates[int(i)] for i in selected_idx]
        candidates.sort(key=lambda x: x[0], reverse=True)

    # Build patch tensors.
    patches = [
        extract_patch(arr, top, left, patch_h, patch_w) for _, top, left in candidates
    ]

    # 4. Guarantee minimum number of patches.
    duplicated_count = 0

    if len(patches) < min_patches:
        if not allow_replacement:
            raise RuntimeError(
                f"Only found {len(patches)} patches for {path}, "
                f"but min_patches={min_patches}. "
                "Set allow_replacement=True to duplicate valid patches."
            )

        while len(patches) < min_patches:
            src_idx = int(rng.integers(0, len(patches)))
            patches.append(patches[src_idx].clone())
            candidates.append(candidates[src_idx])
            duplicated_count += 1

    patches = torch.stack(patches, dim=0).float()

    debug = {
        "path": str(path),
        "image_shape": tuple(arr.shape),
        "mask_area_pixels": int(mask.sum()),
        "mask_area_fraction": float(mask.mean()),
        "num_candidates_before_guarantee": int(len(candidates) - duplicated_count),
        "num_returned_patches": int(patches.shape[0]),
        "num_duplicated_patches": int(duplicated_count),
        "min_leaf_coverage": float(min_leaf_coverage),
        "min_patches": int(min_patches),
        "max_patches": None if max_patches is None else int(max_patches),
        "mask_bbox": tuple(map(int, mask_bbox(mask))),
        "mask_method": mask_method,
    }

    if patches.shape[0] < 2:
        raise RuntimeError(
            f"Internal error: expected at least 2 patches for {path}, "
            f"got {patches.shape[0]}."
        )

    if return_debug:
        return patches, debug

    return patches


# Backward-compatible alias.
def read_band_image_as_roi_patches(
    path,
    patch_h,
    patch_w,
    stride_h=None,
    stride_w=None,
    black_thr=0.0,
):
    """
    Backward-compatible wrapper.

    WARNING:
        The old implementation used arr > black_thr and required all patch
        pixels to be foreground. This wrapper now uses the improved contour
        patch extractor.
    """
    return read_band_image_as_leaf_patches(
        path=path,
        patch_h=patch_h,
        patch_w=patch_w,
        stride_h=stride_h,
        stride_w=stride_w,
        min_leaf_coverage=0.85,
        min_patches=10,
        max_patches=None,
        mask_method="contour",
        fallback_black_thr=black_thr,
        return_debug=False,
    )


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------


class MultiSpectralCSVPatchDataset(Dataset):
    """
    CSV-driven multispectral patch dataset.

    Expected CSV columns:
        blue, green, red, nir, red_edge, spectral, Species, Stages

    Returns by default:
        band_patches:
            dict {
                "blue":     Tensor [N_blue,     1, patch_h, patch_w],
                "green":    Tensor [N_green,    1, patch_h, patch_w],
                "red":      Tensor [N_red,      1, patch_h, patch_w],
                "nir":      Tensor [N_nir,      1, patch_h, patch_w],
                "red_edge": Tensor [N_red_edge, 1, patch_h, patch_w],
            }

        spectrum:
            Tensor [spectral_length]

    If return_debug=True:
        returns:
            band_patches, spectrum, debug_dict

    Important:
        Because your multispectral channels are not aligned, this dataset
        extracts patches independently per band. The training model should
        therefore fuse per-band features after per-band encoding, unless you
        explicitly add image registration before patch extraction.
    """

    def __init__(
        self,
        csv_path,
        root_dir=None,
        species=None,
        stage=None,
        patch_h=32,
        patch_w=32,
        stride_h=None,
        stride_w=None,
        black_thr=0.0,
        min_leaf_coverage=0.85,
        min_patches_per_band=10,
        max_patches_per_band=None,
        border_erode_px=2,
        mask_method="contour",
        random_seed=123,
        return_debug=False,
        spectral_drop_first_n=50,
    ):
        self.csv_path = csv_path
        self.root_dir = root_dir if root_dir is not None else ""

        df = pd.read_csv(csv_path)

        self.band_cols = ["blue", "green", "red", "nir", "red_edge"]

        required_cols = [
            "blue",
            "green",
            "red",
            "nir",
            "red_edge",
            "spectral",
            "Species",
            "Stages",
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if len(missing) > 0:
            raise ValueError("Missing columns in CSV: " + ", ".join(missing))

        # Normalize for robust matching.
        df["Species"] = df["Species"].astype(str).str.strip().str.lower()
        df["Stages"] = df["Stages"].astype(str).str.strip().str.lower()

        if species is not None:
            species = str(species).strip().lower()
            df = df[df["Species"] == species]

        # if stage is not None:
        #    stage = str(stage).strip().lower()
        #    df = df[df["Stages"] == stage]

        if stage is not None:
            stage = str(stage).strip().lower()
            if stage not in ["all", "any", "*", ""]:
                df = df[df["Stages"] == stage]

        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                "No rows left after filtering. Check species/stage values."
            )

        self.df = df

        # Parse spectra once after filtering.
        self.spectral_np = get_spectral_data(
            self.df,
            spectral_col="spectral",
            drop_first_n=spectral_drop_first_n,
        ).astype(np.float32)

        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.stride_h = int(stride_h) if stride_h is not None else None
        self.stride_w = int(stride_w) if stride_w is not None else None

        self.black_thr = float(black_thr)
        self.min_leaf_coverage = float(min_leaf_coverage)
        self.min_patches_per_band = int(min_patches_per_band)
        self.max_patches_per_band = (
            None if max_patches_per_band is None else int(max_patches_per_band)
        )

        self.border_erode_px = int(border_erode_px)
        self.mask_method = str(mask_method)
        self.random_seed = int(random_seed)
        self.return_debug = bool(return_debug)

        if self.min_patches_per_band < 10:
            warnings.warn(
                f"min_patches_per_band={self.min_patches_per_band}. "
                "You requested at least 10 patches; consider setting "
                "min_patches_per_band=10.",
                RuntimeWarning,
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        band_patches = {}
        debug = {
            "index": int(index),
            "bands": {},
        }

        for band_idx, band_name in enumerate(self.band_cols):
            fname = row[band_name]
            path = resolve_image_path(fname, self.root_dir)

            # Deterministic per sample and per band.
            band_seed = self.random_seed + int(index) * 1009 + band_idx * 9176

            patches, band_debug = read_band_image_as_leaf_patches(
                path=path,
                patch_h=self.patch_h,
                patch_w=self.patch_w,
                stride_h=self.stride_h,
                stride_w=self.stride_w,
                min_leaf_coverage=self.min_leaf_coverage,
                min_patches=self.min_patches_per_band,
                max_patches=self.max_patches_per_band,
                border_erode_px=self.border_erode_px,
                mask_method=self.mask_method,
                fallback_black_thr=self.black_thr,
                random_seed=band_seed,
                random_fallback=True,
                allow_replacement=True,
                return_debug=True,
            )

            if patches.shape[0] < self.min_patches_per_band:
                raise RuntimeError(
                    f"Band {band_name} for sample {index} returned only "
                    f"{patches.shape[0]} patches, expected at least "
                    f"{self.min_patches_per_band}."
                )

            band_patches[band_name] = patches
            debug["bands"][band_name] = band_debug

        spec = torch.from_numpy(self.spectral_np[index]).float()

        if self.return_debug:
            return band_patches, spec, debug

        return band_patches, spec


# -------------------------------------------------------------------------
# Collate function
# -------------------------------------------------------------------------


def patch_collate_fn(batch):
    """
    Collate function for variable number of patches per sample/band.

    Input:
        batch = list of:
            (band_patches_dict, spectrum)
        or:
            (band_patches_dict, spectrum, debug)

    Output:
        batch_bands:
            dict {band_name: list of Tensor [N_i, 1, patch_h, patch_w]}

        batch_spec:
            Tensor [B, spectral_length]

        optionally:
            debug_list
    """
    band_keys = ["blue", "green", "red", "nir", "red_edge"]

    has_debug = len(batch[0]) == 3

    batch_bands = {k: [] for k in band_keys}
    specs = []
    debug_list = []

    for item in batch:
        if has_debug:
            band_dict, spec, debug = item
            debug_list.append(debug)
        else:
            band_dict, spec = item

        for k in band_keys:
            batch_bands[k].append(band_dict[k])

        specs.append(spec)

    batch_spec = torch.stack(specs, dim=0)

    if has_debug:
        return batch_bands, batch_spec, debug_list

    return batch_bands, batch_spec


# -------------------------------------------------------------------------
# Debug entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = MultiSpectralCSVPatchDataset(
        csv_path="./Data/train_avocado.csv",
        root_dir="/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/",
        species="Avocado",
        stage="dry",
        patch_h=32,
        patch_w=32,
        stride_h=4,
        stride_w=4,
        black_thr=0.0,
        min_leaf_coverage=0.90,
        min_patches_per_band=10,
        max_patches_per_band=None,
        border_erode_px=2,
        mask_method="contour",
        random_seed=123,
        return_debug=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=patch_collate_fn,
    )

    for batch in loader:
        band_dict, spec, debug = batch

        print("spec shape:", spec.shape)

        for band_name in ["blue", "green", "red", "nir", "red_edge"]:
            patches = band_dict[band_name][0]
            print(f"{band_name:8s} patches:", patches.shape)

        print("debug sample:")
        sample_debug = debug[0]
        for band_name, band_debug in sample_debug["bands"].items():
            print(
                f"  {band_name:8s} "
                f"mask_area_fraction={band_debug['mask_area_fraction']:.4f} "
                f"returned={band_debug['num_returned_patches']} "
                f"duplicated={band_debug['num_duplicated_patches']}"
            )

        break
