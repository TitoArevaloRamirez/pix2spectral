#!/usr/bin/env python3
"""
Resize, zero-pad, and optionally normalize multispectral band images.

This script is intended to run OUTSIDE the training pipeline.

It reads all images under a root directory and selected subfolders, resizes each
image while preserving aspect ratio, pads with zeros to a square image, and saves
the transformed image to a different output directory while preserving the
relative folder structure.

Example:
    image size 150 x 300, desired size 220
        largest dimension 300 -> resized to 220
        other dimension 150 -> resized to 110
        output canvas: 220 x 220
        image centered, remaining pixels padded with zeros

Optional band-wise normalization:
    The script can compute per-band normalization statistics across all available
    images and save already normalized images. The default is robust min-max
    normalization per band:
        x_norm = clip((x - p_low) / (p_high - p_low), 0, 1)

For normalized image saving, values are saved as uint16 TIFF by default, where:
    0     -> 0.0
    65535 -> 1.0

Band inference:
    The default inference supports filenames such as:
        leaf028d0_1.tif -> blue
        leaf028d0_2.tif -> green
        leaf028d0_3.tif -> red
        leaf028d0_4.tif -> nir
        leaf028d0_5.tif -> red_edge

It also supports names containing band strings:
    blue, green, red, nir, red_edge, rededge

python preprocess_resize_pad_normalize_images.py \
    --root-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --output-dir "/home/usr3/Data/EstradaDataset/Avocado/MultispectralImages_resized_220_norm/" \
    --image-size 220 \
    --method none \
    --save-format tiff_uint16 \
    --overwrite


python preprocess_resize_pad_normalize_images.py \
    --root-dir "/home/usr3/Data/EstradaDataset/Olive/Multispectral Images/" \
    --output-dir "/home/usr3/Data/EstradaDataset/Olive/MultispectralImages_resized_220_norm/" \
    --image-size 220 \
    --method none \
    --save-format tiff_uint16 \
    --overwrite

python preprocess_resize_pad_normalize_images.py \
    --root-dir "/home/usr3/Data/EstradaDataset/Vineyard/Multispectral Images/" \
    --output-dir "/home/usr3/Data/EstradaDataset/Vineyard/MultispectralImages_resized_220_norm/" \
    --image-size 220 \
    --method none \
    --save-format tiff_uint16 \
    --overwrite


"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


BANDS = ["blue", "green", "red", "nir", "red_edge"]
BAND_SUFFIX_MAP = {
    "1": "blue",
    "2": "green",
    "3": "red",
    "4": "nir",
    "5": "red_edge",
}
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Resize, pad, and optionally normalize multispectral images."
    )
    p.add_argument("--root-dir", required=True, help="Input image root directory.")
    p.add_argument("--output-dir", required=True, help="Output root directory.")
    p.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help=(
            "Subfolders under root-dir to process. If omitted, all images under "
            "root-dir are processed recursively."
        ),
    )
    p.add_argument(
        "--image-size", type=int, required=True, help="Final square size, e.g. 220."
    )
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--no-recursive", action="store_false", dest="recursive")
    p.add_argument(
        "--method",
        choices=["none", "minmax", "robust_minmax"],
        default="robust_minmax",
        help="Band-wise normalization method applied after resize/pad. Default: robust_minmax.",
    )
    p.add_argument("--lower-percentile", type=float, default=1.0)
    p.add_argument("--upper-percentile", type=float, default=99.0)
    p.add_argument(
        "--exclude-zero-from-stats",
        action="store_true",
        default=True,
        help="Exclude zero pixels from normalization stats. Recommended for padded/background images.",
    )
    p.add_argument(
        "--include-zero-in-stats", action="store_false", dest="exclude_zero_from_stats"
    )
    p.add_argument(
        "--save-format",
        choices=["tiff_uint16", "png_uint16", "npy_float32"],
        default="tiff_uint16",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


def expand_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()


def list_images(
    root: Path, folders: Optional[List[str]], recursive: bool
) -> List[Path]:
    paths: List[Path] = []
    search_roots = [root / f for f in folders] if folders else [root]
    for sr in search_roots:
        if not sr.exists():
            raise FileNotFoundError(f"Input folder not found: {sr}")
        iterator = sr.rglob("*") if recursive else sr.glob("*")
        for p in iterator:
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
    return sorted(paths)


def infer_band(path: Path) -> str:
    stem = path.stem.lower()

    # Explicit band strings first.
    if "red_edge" in stem or "rededge" in stem or "red-edge" in stem:
        return "red_edge"
    for b in ["blue", "green", "nir", "red"]:
        if re.search(rf"(^|[_\-.]){b}($|[_\-.])", stem):
            return b

    # Common Estrada naming: leafXXXdY_1..5
    m = re.search(r"[_\-](\d+)$", stem)
    if m:
        suffix = m.group(1)
        if suffix in BAND_SUFFIX_MAP:
            return BAND_SUFFIX_MAP[suffix]

    # Fallback: last single digit.
    m = re.search(r"(\d)$", stem)
    if m and m.group(1) in BAND_SUFFIX_MAP:
        return BAND_SUFFIX_MAP[m.group(1)]

    return "unknown"


def read_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=2)
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def resize_pad_square(arr: np.ndarray, image_size: int) -> np.ndarray:
    H, W = arr.shape
    if H <= 0 or W <= 0:
        raise ValueError("Invalid image with zero dimension.")

    scale = float(image_size) / float(max(H, W))
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))

    pil_mode = "F"
    img = Image.fromarray(arr.astype(np.float32), mode=pil_mode)
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    resized = np.asarray(img).astype(np.float32)

    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    top = (image_size - new_h) // 2
    left = (image_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def collect_stats(
    paths: List[Path],
    image_size: int,
    method: str,
    exclude_zero: bool,
    lower: float,
    upper: float,
) -> Dict[str, Dict[str, float]]:
    pixels_by_band: Dict[str, List[np.ndarray]] = {b: [] for b in BANDS}
    pixels_by_band["unknown"] = []

    for p in paths:
        band = infer_band(p)
        arr = resize_pad_square(read_image(p), image_size=image_size)
        vals = arr.reshape(-1)
        vals = vals[np.isfinite(vals)]
        if exclude_zero:
            vals = vals[vals != 0]
        if vals.size == 0:
            continue
        if vals.size > 200000:
            idx = np.linspace(0, vals.size - 1, 200000).astype(int)
            vals = vals[idx]
        pixels_by_band.setdefault(band, []).append(vals.astype(np.float32))

    stats: Dict[str, Dict[str, float]] = {}
    for band, chunks in pixels_by_band.items():
        if not chunks:
            continue
        vals = np.concatenate(chunks, axis=0)
        if method == "none":
            lo = 0.0
            hi = 1.0
        elif method == "minmax":
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        elif method == "robust_minmax":
            lo = float(np.percentile(vals, lower))
            hi = float(np.percentile(vals, upper))
        else:
            raise ValueError(method)

        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0

        stats[band] = {
            "method": method,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p_low": float(lo),
            "p_high": float(hi),
            "num_pixels": int(vals.size),
        }

    return stats


def normalize_image(
    arr: np.ndarray, band: str, stats: Dict[str, Dict[str, float]], method: str
) -> np.ndarray:
    if method == "none":
        return arr.astype(np.float32)

    if band not in stats:
        # Unknown band: keep robust per-image scaling as a safe fallback.
        vals = arr[arr != 0]
        if vals.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo, hi = np.percentile(vals, [1, 99])
    else:
        lo = float(stats[band]["p_low"])
        hi = float(stats[band]["p_high"])

    out = (arr.astype(np.float32) - lo) / (hi - lo + 1e-8)
    out = np.clip(out, 0.0, 1.0)
    # Preserve padding exactly as zero.
    out[arr == 0] = 0.0
    return out.astype(np.float32)


def save_image(arr: np.ndarray, out_path: Path, save_format: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "npy_float32":
        out_path = out_path.with_suffix(".npy")
        np.save(out_path, arr.astype(np.float32))
        return out_path

    if save_format == "tiff_uint16":
        out_path = out_path.with_suffix(".tif")
    elif save_format == "png_uint16":
        out_path = out_path.with_suffix(".png")
    else:
        raise ValueError(save_format)

    # arr16 = np.clip(arr, 0.0, 1.0)
    # arr16 = np.round(arr16 * 65535.0).astype(np.uint16)
    arr16 = arr.astype(np.uint16)
    Image.fromarray(arr16).save(out_path)
    return out_path


def main() -> int:
    args = parse_args()
    root = expand_path(args.root_dir)
    out_root = expand_path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = list_images(root, args.folders, args.recursive)
    if args.max_images is not None:
        paths = paths[: int(args.max_images)]

    if not paths:
        raise RuntimeError("No images found.")

    print(f"Found {len(paths)} images.")
    print("Computing band-wise normalization statistics...")
    stats = collect_stats(
        paths=paths,
        image_size=args.image_size,
        method=args.method,
        exclude_zero=args.exclude_zero_from_stats,
        lower=args.lower_percentile,
        upper=args.upper_percentile,
    )

    stats_path = out_root / "image_resize_pad_normalization_stats.json"
    stats_payload = {
        "root_dir": str(root),
        "output_dir": str(out_root),
        "image_size": int(args.image_size),
        "method": args.method,
        "exclude_zero_from_stats": bool(args.exclude_zero_from_stats),
        "lower_percentile": float(args.lower_percentile),
        "upper_percentile": float(args.upper_percentile),
        "band_stats": stats,
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2))

    rows = []
    for i, p in enumerate(paths):
        rel = p.relative_to(root)
        band = infer_band(p)
        out_rel = rel

        if args.save_format == "npy_float32":
            out_rel = out_rel.with_suffix(".npy")
        elif args.save_format == "tiff_uint16":
            out_rel = out_rel.with_suffix(".tif")
        elif args.save_format == "png_uint16":
            out_rel = out_rel.with_suffix(".png")

        out_path = out_root / out_rel

        if out_path.exists() and not args.overwrite:
            rows.append(
                {
                    "input_path": str(p),
                    "output_path": str(out_path),
                    "relative_path": str(rel),
                    "band": band,
                    "status": "skipped_exists",
                }
            )
            continue

        if args.dry_run:
            rows.append(
                {
                    "input_path": str(p),
                    "output_path": str(out_path),
                    "relative_path": str(rel),
                    "band": band,
                    "status": "dry_run",
                }
            )
            continue

        arr = read_image(p)
        arr_sq = resize_pad_square(arr, image_size=args.image_size)
        arr_norm = normalize_image(arr_sq, band=band, stats=stats, method=args.method)
        saved = save_image(arr_norm, out_path, args.save_format)

        rows.append(
            {
                "input_path": str(p),
                "output_path": str(saved),
                "relative_path": str(rel),
                "band": band,
                "input_height": int(arr.shape[0]),
                "input_width": int(arr.shape[1]),
                "output_height": int(args.image_size),
                "output_width": int(args.image_size),
                "status": "written",
            }
        )

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(paths)} images.")

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "image_resize_pad_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("=" * 80)
    print("Image resize/pad/normalize finished")
    print("=" * 80)
    print(f"Input root:      {root}")
    print(f"Output root:     {out_root}")
    print(f"Image size:      {args.image_size} x {args.image_size}")
    print(f"Method:          {args.method}")
    print(f"Save format:     {args.save_format}")
    print(f"Stats JSON:      {stats_path}")
    print(f"Manifest CSV:    {manifest_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
