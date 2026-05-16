import os
import ast
import json
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
    Parse spectral signatures stored in the CSV.

    Expected CSV format:
        spectral column contains a Python-like list string, e.g.
        "[0.12, 0.13, 0.14, ...]"

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
    Robust path resolver for image paths stored in the CSV.

    Handles:
      - absolute paths
      - relative paths
      - Windows-style paths stored in CSV
      - root_dir + basename fallback
    """
    fname = str(fname).strip()

    if fname == "" or fname.lower() == "nan":
        raise ValueError("Empty image filename in CSV.")

    p = Path(os.path.expanduser(fname))
    if p.is_absolute() and p.exists():
        return str(p)

    if root_dir is not None and root_dir != "":
        root = Path(os.path.expanduser(str(root_dir)))

        candidate = root / fname
        if candidate.exists():
            return str(candidate)

        win_name = PureWindowsPath(fname).name
        candidate = root / win_name
        if candidate.exists():
            return str(candidate)

        posix_name = Path(fname).name
        candidate = root / posix_name
        if candidate.exists():
            return str(candidate)

        return str(root / win_name)

    return fname


# -------------------------------------------------------------------------
# Image and mask utilities
# -------------------------------------------------------------------------


def read_grayscale_image(path):
    """
    Read an image as float32 grayscale without forcing it to 8-bit.

    This is safer for multispectral bands that may be 16-bit.
    """
    path = os.path.expanduser(str(path))

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)
    arr = np.asarray(img)

    if arr.ndim == 3:
        # If the file unexpectedly has multiple channels, convert to grayscale.
        # Avoid PIL.convert("L") because it can collapse 16-bit data to 8-bit.
        arr = arr[..., :3].astype(np.float32).mean(axis=2)

    arr = arr.astype(np.float32)
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
    """Keep only the largest connected foreground component."""
    if cv2 is None:
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    if num_labels <= 1:
        return mask.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return labels == largest_label


def _clean_binary_mask_cv2(
    mask,
    close_kernel_size=15,
    open_kernel_size=5,
    keep_largest=True,
):
    """Morphologically clean a binary leaf mask."""
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
    """Erode mask to avoid sampling patches too close to the leaf border."""
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
        _, otsu_hi = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

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

    warnings.warn(
        "OpenCV contour mask failed or OpenCV is not installed. "
        "Falling back to simple intensity mask.",
        RuntimeWarning,
    )

    fallback = arr > float(fallback_black_thr)

    if fallback.sum() < min_area:
        fallback = n8 > 10

    return fallback.astype(bool)


# -------------------------------------------------------------------------
# Normalization utilities
# -------------------------------------------------------------------------


def _safe_std(x, eps=1e-6):
    std = float(np.std(x))
    if not np.isfinite(std) or std < eps:
        return 1.0
    return std


def _safe_percentile(x, q, default=0.0):
    if x.size == 0:
        return float(default)

    value = float(np.percentile(x, q))
    if not np.isfinite(value):
        return float(default)

    return value


def parse_normalization_mode(normalization_mode=None, normalization_scope=None, normalization_method=None):
    """
    Normalize old and new config styles into (scope, method).

    New recommended style:
        normalization_scope: "none", "stage_band", "global_band"
        normalization_method: "zscore", "robust_zscore", "minmax"

    Backward-compatible style:
        normalization_mode: "stage_band_robust_zscore", "global_band_minmax", etc.
    """
    if normalization_scope is not None:
        scope = str(normalization_scope).strip().lower()
        method = "robust_zscore" if normalization_method is None else str(normalization_method).strip().lower()
        return scope, method

    if normalization_mode is None:
        return "none", "none"

    mode = str(normalization_mode).strip().lower()

    if mode in ["", "none", "raw"]:
        return "none", "none"

    if mode.startswith("stage_band_"):
        return "stage_band", mode.replace("stage_band_", "", 1)

    if mode.startswith("global_band_"):
        return "global_band", mode.replace("global_band_", "", 1)

    # Accept method-only strings as global band by default.
    if mode in ["zscore", "robust_zscore", "minmax"]:
        return "global_band", mode

    raise ValueError(
        f"Unknown normalization_mode='{normalization_mode}'. Expected 'none', "
        "'stage_band_zscore', 'stage_band_robust_zscore', 'stage_band_minmax', "
        "'global_band_zscore', 'global_band_robust_zscore', or 'global_band_minmax'."
    )


def _normalization_group_key(stage, scope):
    if scope == "global_band":
        return "__global__"
    if scope == "stage_band":
        return str(stage).strip().lower()
    return "__none__"


def _make_stat_record(pixels, lower_percentile, upper_percentile):
    pixels = np.asarray(pixels, dtype=np.float32)
    pixels = pixels[np.isfinite(pixels)]

    if pixels.size == 0:
        return {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 1.0,
            "p_low": 0.0,
            "p_high": 1.0,
            "num_pixels": 0,
        }

    p_low = _safe_percentile(pixels, lower_percentile, default=float(np.min(pixels)))
    p_high = _safe_percentile(pixels, upper_percentile, default=float(np.max(pixels)))

    clipped = np.clip(pixels, p_low, p_high)

    return {
        "mean": float(np.mean(clipped)),
        "std": _safe_std(clipped),
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
        "p_low": float(p_low),
        "p_high": float(p_high),
        "num_pixels": int(pixels.size),
    }


def compute_band_normalization_stats(
    df,
    root_dir=None,
    band_cols=("blue", "green", "red", "nir", "red_edge"),
    stage_col="Stages",
    normalization_scope="stage_band",
    use_leaf_mask=True,
    mask_method="contour",
    fallback_black_thr=0.0,
    sample_pixels_per_image=20000,
    lower_percentile=1.0,
    upper_percentile=99.0,
    random_seed=42,
):
    """
    Compute normalization statistics from the dataframe images.

    Options:
        normalization_scope="stage_band"
            Compute independent statistics for every dehydration stage and band.
            Example groups: fresh/blue, fresh/nir, dry/blue, dry/nir.

        normalization_scope="global_band"
            Compute one statistic per band using all dehydration stages together.
            Example groups: all_stages/blue, all_stages/nir.

    Recommended usage:
        Compute on TRAINING DATA ONLY, then pass the same stats to validation/test.
    """
    scope = str(normalization_scope).strip().lower()

    if scope not in ["stage_band", "global_band"]:
        raise ValueError(
            f"normalization_scope must be 'stage_band' or 'global_band', got '{scope}'."
        )

    rng = np.random.default_rng(random_seed)

    df = df.copy()
    df[stage_col] = df[stage_col].astype(str).str.strip().str.lower()

    if scope == "global_band":
        group_items = [("__global__", df)]
    else:
        group_items = [
            (stage, df[df[stage_col] == stage].reset_index(drop=True))
            for stage in sorted(df[stage_col].dropna().unique().tolist())
        ]

    stats = {
        "version": 2,
        "scope": scope,
        "stage_col": stage_col,
        "band_cols": list(band_cols),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "use_leaf_mask": bool(use_leaf_mask),
        "groups": {},
    }

    for group_key, group_df in group_items:
        stats["groups"][group_key] = {}

        for band in band_cols:
            all_pixels = []
            num_images_used = 0

            for _, row in group_df.iterrows():
                path = resolve_image_path(row[band], root_dir)

                try:
                    arr = read_grayscale_image(path)
                except Exception as exc:
                    warnings.warn(
                        f"Skipping image while computing normalization stats: "
                        f"{path}. Error: {exc}",
                        RuntimeWarning,
                    )
                    continue

                arr = np.asarray(arr, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

                if use_leaf_mask:
                    try:
                        if mask_method == "contour":
                            mask = detect_leaf_mask_by_contour(
                                arr,
                                fallback_black_thr=fallback_black_thr,
                            )
                        elif mask_method == "threshold":
                            raw_mask = arr > float(fallback_black_thr)
                            mask = _clean_binary_mask_cv2(raw_mask, keep_largest=True)
                        else:
                            raise ValueError(f"Unknown mask_method={mask_method}")

                        pixels = arr[mask]
                    except Exception as exc:
                        warnings.warn(
                            f"Leaf mask failed while computing stats for {path}. "
                            f"Falling back to non-background pixels. Error: {exc}",
                            RuntimeWarning,
                        )
                        pixels = arr[arr > float(fallback_black_thr)]
                else:
                    pixels = arr.reshape(-1)

                pixels = pixels[np.isfinite(pixels)]

                if pixels.size == 0:
                    pixels = arr[arr > float(fallback_black_thr)]
                    pixels = pixels[np.isfinite(pixels)]

                if pixels.size == 0:
                    warnings.warn(
                        f"No valid pixels found while computing stats for {path}.",
                        RuntimeWarning,
                    )
                    continue

                if sample_pixels_per_image is not None and pixels.size > sample_pixels_per_image:
                    idx = rng.choice(
                        pixels.size,
                        size=int(sample_pixels_per_image),
                        replace=False,
                    )
                    pixels = pixels[idx]

                all_pixels.append(pixels.astype(np.float32))
                num_images_used += 1

            if len(all_pixels) == 0:
                warnings.warn(
                    f"No valid images found for group='{group_key}', band='{band}'. "
                    "Using identity normalization.",
                    RuntimeWarning,
                )
                rec = _make_stat_record(np.asarray([], dtype=np.float32), lower_percentile, upper_percentile)
                rec["num_images"] = 0
                stats["groups"][group_key][band] = rec
                continue

            pixels = np.concatenate(all_pixels, axis=0)
            rec = _make_stat_record(pixels, lower_percentile, upper_percentile)
            rec["num_images"] = int(num_images_used)
            stats["groups"][group_key][band] = rec

    return stats


# Backward-compatible name used by older drafts.
def compute_stage_band_normalization_stats(*args, **kwargs):
    kwargs["normalization_scope"] = "stage_band"
    return compute_band_normalization_stats(*args, **kwargs)


def save_normalization_stats(stats, path):
    path = Path(os.path.expanduser(str(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(path):
    path = Path(os.path.expanduser(str(path)))
    with open(path, "r") as f:
        return json.load(f)


def _get_stat_from_normalization_stats(normalization_stats, stage, band, scope):
    """
    Fetch a stat record from either the new v2 stats format or legacy nested stats.
    """
    if normalization_stats is None:
        raise ValueError("normalization_stats is None.")

    stage = str(stage).strip().lower()
    band = str(band).strip()
    scope = str(scope).strip().lower()

    # New v2 format.
    if isinstance(normalization_stats, dict) and "groups" in normalization_stats:
        group_key = _normalization_group_key(stage, scope)
        groups = normalization_stats["groups"]

        if group_key not in groups:
            available = list(groups.keys())
            raise KeyError(
                f"Normalization group '{group_key}' not found. Available groups: {available}"
            )

        if band not in groups[group_key]:
            available = list(groups[group_key].keys())
            raise KeyError(
                f"Band '{band}' not found in normalization group '{group_key}'. "
                f"Available bands: {available}"
            )

        return groups[group_key][band]

    # Legacy stage -> band format.
    if scope == "stage_band":
        if stage not in normalization_stats:
            available = list(normalization_stats.keys())
            raise KeyError(
                f"Stage '{stage}' not found in legacy normalization_stats. "
                f"Available stages: {available}"
            )
        return normalization_stats[stage][band]

    # Legacy global band format, if user manually passed band -> stats.
    if scope == "global_band":
        if band in normalization_stats:
            return normalization_stats[band]
        if "__global__" in normalization_stats and band in normalization_stats["__global__"]:
            return normalization_stats["__global__"][band]

    raise KeyError(
        f"Could not find normalization stats for scope='{scope}', stage='{stage}', band='{band}'."
    )


def normalize_band_image(
    arr,
    stage,
    band,
    normalization_stats,
    normalization_scope="stage_band",
    normalization_method="robust_zscore",
    output_clip=None,
    clip_percentiles=True,
    eps=1e-6,
):
    """
    Normalize a full band image BEFORE patch extraction.

    Scopes:
        "stage_band": normalize each band separately inside each dehydration stage.
        "global_band": normalize each band separately using all stages together.

    Methods:
        "zscore":        (arr - mean) / std
        "robust_zscore": clip(arr, p_low, p_high), then (arr - mean) / std
        "minmax":        clip(arr, p_low, p_high), then scale to [0, 1]
        "none":          return raw float32 image
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    scope = str(normalization_scope).strip().lower()
    method = str(normalization_method).strip().lower()

    if scope in ["", "none", "raw"] or method in ["", "none", "raw"]:
        return arr.astype(np.float32)

    if scope not in ["stage_band", "global_band"]:
        raise ValueError(
            f"normalization_scope must be 'none', 'stage_band', or 'global_band', got '{scope}'."
        )

    if method not in ["zscore", "robust_zscore", "minmax"]:
        raise ValueError(
            f"normalization_method must be 'zscore', 'robust_zscore', or 'minmax', got '{method}'."
        )

    s = _get_stat_from_normalization_stats(
        normalization_stats=normalization_stats,
        stage=stage,
        band=band,
        scope=scope,
    )

    mean = float(s.get("mean", 0.0))
    std = max(float(s.get("std", 1.0)), eps)
    p_low = float(s.get("p_low", s.get("min", 0.0)))
    p_high = float(s.get("p_high", s.get("max", 1.0)))

    if method == "zscore":
        arr_norm = (arr - mean) / std

    elif method == "robust_zscore":
        if clip_percentiles:
            arr = np.clip(arr, p_low, p_high)
        arr_norm = (arr - mean) / std

    elif method == "minmax":
        if clip_percentiles:
            arr = np.clip(arr, p_low, p_high)
        denom = max(p_high - p_low, eps)
        arr_norm = (arr - p_low) / denom

    if output_clip is not None:
        lo, hi = output_clip
        arr_norm = np.clip(arr_norm, lo, hi)

    return arr_norm.astype(np.float32)


# Backward-compatible name used by older drafts.
def normalize_band_image_by_stage(
    arr,
    stage,
    band,
    normalization_stats,
    mode="stage_band_zscore",
    clip_percentiles=True,
    output_clip=None,
    eps=1e-6,
):
    scope, method = parse_normalization_mode(normalization_mode=mode)
    return normalize_band_image(
        arr=arr,
        stage=stage,
        band=band,
        normalization_stats=normalization_stats,
        normalization_scope=scope,
        normalization_method=method,
        output_clip=output_clip,
        clip_percentiles=clip_percentiles,
        eps=eps,
    )


# -------------------------------------------------------------------------
# Patch extraction utilities
# -------------------------------------------------------------------------


def pad_image_and_mask_to_patch_size(arr, mask, patch_h, patch_w):
    """Pad image/mask if image is smaller than the requested patch size."""
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
    """Return bounding box of mask as top, bottom, left, right."""
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

    IMPORTANT:
        This function intentionally does not normalize each patch. The full band
        image should already be normalized before patch generation.
    """
    patch = arr[top : top + patch_h, left : left + patch_w]
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
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
    stage=None,
    band_name=None,
    normalization_stats=None,
    normalization_mode=None,
    normalization_scope="none",
    normalization_method="none",
    normalization_output_clip=None,
):
    """
    Read one multispectral band and return leaf patches.

    Important order:
      1. Read raw image.
      2. Detect leaf mask from raw image.
      3. Normalize the full image using stage/band or global/band stats.
      4. Extract patches from normalized full image.

    This avoids per-patch normalization and preserves physically meaningful
    relative intensity structure inside each normalized stage/band or band group.
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

    scope, method = parse_normalization_mode(
        normalization_mode=normalization_mode,
        normalization_scope=normalization_scope,
        normalization_method=normalization_method,
    )

    rng = np.random.default_rng(random_seed)

    # Raw image is used for mask detection.
    arr_raw = read_grayscale_image(path)

    if mask_method == "contour":
        mask = detect_leaf_mask_by_contour(
            arr_raw,
            fallback_black_thr=fallback_black_thr,
        )
    elif mask_method == "threshold":
        raw_mask = arr_raw > float(fallback_black_thr)
        mask = _clean_binary_mask_cv2(raw_mask, keep_largest=True)
    else:
        raise ValueError(
            f"Unknown mask_method={mask_method}. Expected 'contour' or 'threshold'."
        )

    # Normalize full image before patch extraction.
    if scope != "none" and method != "none":
        if stage is None:
            raise ValueError("stage must be provided when using image normalization.")
        if band_name is None:
            raise ValueError("band_name must be provided when using image normalization.")

        arr = normalize_band_image(
            arr=arr_raw,
            stage=stage,
            band=band_name,
            normalization_stats=normalization_stats,
            normalization_scope=scope,
            normalization_method=method,
            output_clip=normalization_output_clip,
        )
    else:
        arr = arr_raw.astype(np.float32)

    arr, mask = pad_image_and_mask_to_patch_size(arr, mask, patch_h, patch_w)

    H, W = arr.shape
    sampling_mask = _erode_mask_cv2(mask, border_erode_px)

    top_box, bottom_box, left_box, right_box = mask_bbox(sampling_mask)

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

    candidates.sort(key=lambda x: x[0], reverse=True)

    if max_patches is not None and len(candidates) > int(max_patches):
        max_patches = int(max_patches)
        selected_idx = rng.choice(len(candidates), size=max_patches, replace=False)
        candidates = [candidates[int(i)] for i in selected_idx]
        candidates.sort(key=lambda x: x[0], reverse=True)

    patches = [
        extract_patch(arr, top, left, patch_h, patch_w)
        for _, top, left in candidates
    ]

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
        "normalization_scope": scope,
        "normalization_method": method,
        "normalization_output_clip": normalization_output_clip,
        "stage": None if stage is None else str(stage),
        "band_name": None if band_name is None else str(band_name),
        "patch_min": float(torch.min(patches).item()),
        "patch_max": float(torch.max(patches).item()),
        "patch_mean": float(torch.mean(patches).item()),
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
        band_patches: dict of Tensor [N_band, 1, patch_h, patch_w]
        spectrum: Tensor [spectral_length]

    Normalization options:
        normalization_scope="stage_band"
            each band normalized separately for each dehydration stage.

        normalization_scope="global_band"
            each band normalized separately using all dehydration stages.

        normalization_scope="none"
            no image normalization.
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
        normalization_stats=None,
        normalization_mode=None,
        normalization_scope="none",
        normalization_method="none",
        normalization_output_clip=None,
        compute_normalization_stats=False,
        normalization_use_leaf_mask=True,
        normalization_sample_pixels_per_image=20000,
        normalization_lower_percentile=1.0,
        normalization_upper_percentile=99.0,
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

        df["Species"] = df["Species"].astype(str).str.strip().str.lower()
        df["Stages"] = df["Stages"].astype(str).str.strip().str.lower()

        if species is not None:
            species = str(species).strip().lower()
            if species not in ["all", "any", "*", ""]:
                df = df[df["Species"] == species]

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

        self.normalization_scope, self.normalization_method = parse_normalization_mode(
            normalization_mode=normalization_mode,
            normalization_scope=normalization_scope,
            normalization_method=normalization_method,
        )
        self.normalization_output_clip = normalization_output_clip
        self.normalization_use_leaf_mask = bool(normalization_use_leaf_mask)
        self.normalization_sample_pixels_per_image = normalization_sample_pixels_per_image
        self.normalization_lower_percentile = float(normalization_lower_percentile)
        self.normalization_upper_percentile = float(normalization_upper_percentile)

        if compute_normalization_stats:
            print(
                "Computing image normalization statistics "
                f"scope={self.normalization_scope}, method={self.normalization_method}..."
            )

            if self.normalization_scope == "none":
                self.normalization_stats = None
            else:
                self.normalization_stats = compute_band_normalization_stats(
                    df=self.df,
                    root_dir=self.root_dir,
                    band_cols=self.band_cols,
                    stage_col="Stages",
                    normalization_scope=self.normalization_scope,
                    use_leaf_mask=self.normalization_use_leaf_mask,
                    mask_method=mask_method,
                    fallback_black_thr=black_thr,
                    sample_pixels_per_image=self.normalization_sample_pixels_per_image,
                    lower_percentile=self.normalization_lower_percentile,
                    upper_percentile=self.normalization_upper_percentile,
                    random_seed=random_seed,
                )

            print("Done computing image normalization statistics.")
        else:
            self.normalization_stats = normalization_stats

        if (
            self.normalization_scope != "none"
            and self.normalization_method != "none"
            and self.normalization_stats is None
        ):
            raise ValueError(
                f"normalization_scope='{self.normalization_scope}' and "
                f"normalization_method='{self.normalization_method}' require "
                "normalization_stats or compute_normalization_stats=True."
            )

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
        stage_label = str(row["Stages"]).strip().lower()

        band_patches = {}
        debug = {
            "index": int(index),
            "stage": stage_label,
            "normalization_scope": self.normalization_scope,
            "normalization_method": self.normalization_method,
            "bands": {},
        }

        for band_idx, band_name in enumerate(self.band_cols):
            fname = row[band_name]
            path = resolve_image_path(fname, self.root_dir)

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
                stage=stage_label,
                band_name=band_name,
                normalization_stats=self.normalization_stats,
                normalization_scope=self.normalization_scope,
                normalization_method=self.normalization_method,
                normalization_output_clip=self.normalization_output_clip,
            )

            if patches.shape[0] < self.min_patches_per_band:
                raise RuntimeError(
                    f"Band {band_name} for sample {index} returned only "
                    f"{patches.shape[0]} patches, expected at least "
                    f"{self.min_patches_per_band}."
                )

            if not torch.isfinite(patches).all():
                raise FloatingPointError(
                    f"Non-finite patch values detected for sample={index}, "
                    f"band={band_name}, path={path}."
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
        stage="all",
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
        compute_normalization_stats=True,
        normalization_scope="stage_band",
        normalization_method="robust_zscore",
        normalization_output_clip=(-5.0, 5.0),
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
            print(f"{band_name:8s} patches:", patches.shape, patches.min().item(), patches.max().item())

        print("debug sample:")
        sample_debug = debug[0]
        for band_name, band_debug in sample_debug["bands"].items():
            print(
                f"  {band_name:8s} "
                f"norm={band_debug['normalization_scope']}/{band_debug['normalization_method']} "
                f"mask_area_fraction={band_debug['mask_area_fraction']:.4f} "
                f"returned={band_debug['num_returned_patches']} "
                f"patch_range=({band_debug['patch_min']:.3f}, {band_debug['patch_max']:.3f})"
            )
        break
