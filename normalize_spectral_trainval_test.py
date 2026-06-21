#!/usr/bin/env python3
"""
Normalize spectral signatures in pix2spectral CSV files using train+val statistics.

Purpose
-------
Read train, validation, and test CSV files with the same row format used by
dataset.py:

    blue, green, red, nir, red_edge, spectral, Species, Stages, ...

Fit spectral normalization statistics using ONLY train + validation spectra.
Then write:

    train_val_normalized.csv
    test_normalized.csv

The output CSVs keep exactly the same columns as the original CSVs. Only the
`spectral` column is replaced by normalized spectral values.

Important safety rule
---------------------
The script never recombines rows. Every row keeps its own multispectral image
filenames, spectral signature, species/stage labels, and target values. It only
changes the values inside the spectral vector of that same row.

Compatibility with dataset.py
-----------------------------
Your dataset.py parses spectra with ast.literal_eval and, by default, removes
the first 50 spectral samples:

    np_data = np.asarray(ast.literal_eval(spectral), dtype=np.float32)
    np_data = np_data[:, 50:]

Therefore, the default behavior of this script is:

    --spectral-drop-first-n 50
    --output-mode preserve_length

This means:
    - statistics are fitted on spectral[:, 50:]
    - spectral[:, 50:] is normalized
    - spectral[:, :50] is preserved unchanged in the output CSV

Then dataset.py can keep using its default drop_first_n=50 and it will see the
normalized spectral region.

If you want the output spectral vectors to contain only the normalized retained
region, use:

    --output-mode normalized_only

and then configure dataset.py/training with spectral_drop_first_n=0.

python normalize_spectral_trainval_test.py \
    --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --val-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_val.csv \
    --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
    --output-dir ~/Code/pix2spectral/Data/dataset_splits_70_20_10/spectral_normalized \
    --method minmax \
    --spectral-drop-first-n 0 \
    --output-mode preserve_length \
    --debug

"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_BASE_COLUMNS = ["blue", "green", "red", "nir", "red_edge", "spectral"]


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize the spectral column of train/val/test CSVs using "
            "statistics fitted on train+val spectra only."
        )
    )

    parser.add_argument("--train-csv", required=True, help="Training CSV file.")
    parser.add_argument("--val-csv", required=True, help="Validation CSV file.")
    parser.add_argument("--test-csv", required=True, help="Test CSV file.")

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train_val_normalized.csv and test_normalized.csv are written.",
    )
    parser.add_argument(
        "--train-val-output-name",
        default="train_val_normalized.csv",
        help="Output filename for concatenated normalized train+val CSV.",
    )
    parser.add_argument(
        "--test-output-name",
        default="test_normalized.csv",
        help="Output filename for normalized test CSV.",
    )

    parser.add_argument(
        "--spectral-column",
        default="spectral",
        help="Name of the spectral signature column. Default: spectral.",
    )
    parser.add_argument(
        "--spectral-drop-first-n",
        type=int,
        default=50,
        help=(
            "Number of leading spectral samples ignored by dataset.py. "
            "Default: 50. Statistics are fitted to the retained region."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=["preserve_length", "normalized_only"],
        default="preserve_length",
        help=(
            "preserve_length: keep the original vector length and normalize only the retained region. "
            "normalized_only: output only the normalized retained region."
        ),
    )

    parser.add_argument(
        "--method",
        choices=["zscore", "robust_zscore", "minmax", "none"],
        default="zscore",
        help="Spectral normalization method. Default: zscore.",
    )
    parser.add_argument(
        "--clip",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional clipping applied after normalization, e.g. --clip -5 5.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small value for numerical stability. Default: 1e-8.",
    )

    parser.add_argument(
        "--check-columns",
        action="store_true",
        default=True,
        help="Require train/val/test to have exactly the same columns. Default: enabled.",
    )
    parser.add_argument(
        "--no-check-columns",
        action="store_false",
        dest="check_columns",
        help="Disable exact column check.",
    )

    parser.add_argument(
        "--json-precision",
        type=int,
        default=8,
        help="Decimal precision used when writing spectral arrays as JSON lists.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write debug plots/reports.",
    )
    parser.add_argument(
        "--debug-preview-n",
        type=int,
        default=20,
        help="Number of spectra used in preview overlay plots. Default: 20.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------------------


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def parse_spectrum_value(value: Any) -> np.ndarray:
    """
    Robust parser for the CSV spectral column.

    dataset.py uses ast.literal_eval, so this parser keeps that as the primary
    behavior but also accepts JSON-like lists and plain comma/space-separated
    arrays for robustness.
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32).reshape(-1)

    if pd.isna(value):
        raise ValueError("Missing spectral value.")

    text = str(value).strip()
    if not text:
        raise ValueError("Empty spectral string.")

    try:
        parsed = ast.literal_eval(text)
        return np.asarray(parsed, dtype=np.float32).reshape(-1)
    except Exception:
        pass

    try:
        parsed = json.loads(text)
        return np.asarray(parsed, dtype=np.float32).reshape(-1)
    except Exception:
        pass

    cleaned = text.strip().strip("[]()").replace(";", ",")
    if "," in cleaned:
        arr = np.fromstring(cleaned, sep=",", dtype=np.float32)
    else:
        arr = np.fromstring(cleaned, sep=" ", dtype=np.float32)

    if arr.size == 0:
        raise ValueError(f"Could not parse spectral string: {text[:120]}...")

    return arr.reshape(-1).astype(np.float32)


def parse_spectral_column(
    df: pd.DataFrame, spectral_col: str, source_name: str
) -> Tuple[List[np.ndarray], pd.DataFrame]:
    if spectral_col not in df.columns:
        raise ValueError(
            f"{source_name}: spectral column '{spectral_col}' was not found."
        )

    spectra: List[np.ndarray] = []
    report_rows = []

    for i, value in enumerate(df[spectral_col].values):
        try:
            spec = parse_spectrum_value(value)
            finite = np.isfinite(spec)
            spectra.append(spec)
            report_rows.append(
                {
                    "source": source_name,
                    "row": int(i),
                    "parse_ok": True,
                    "length": int(spec.size),
                    "finite_count": int(finite.sum()),
                    "nonfinite_count": int((~finite).sum()),
                    "min": float(np.nanmin(spec)) if spec.size else np.nan,
                    "max": float(np.nanmax(spec)) if spec.size else np.nan,
                    "mean": float(np.nanmean(spec)) if spec.size else np.nan,
                }
            )
        except Exception as exc:
            report_rows.append(
                {
                    "source": source_name,
                    "row": int(i),
                    "parse_ok": False,
                    "error": str(exc),
                }
            )

    report = pd.DataFrame(report_rows)

    bad = (
        report[report["parse_ok"] == False]
        if "parse_ok" in report.columns
        else pd.DataFrame()
    )
    if len(bad) > 0:
        first = bad.iloc[0].to_dict()
        raise ValueError(
            f"{source_name}: failed to parse {len(bad)} spectra. First error: {first}"
        )

    lengths = [s.size for s in spectra]
    if len(set(lengths)) != 1:
        counts = pd.Series(lengths).value_counts().to_dict()
        raise ValueError(f"{source_name}: inconsistent spectral lengths: {counts}")

    return spectra, report


def stack_retained_spectra(
    spectra: Sequence[np.ndarray],
    drop_first_n: int,
    source_name: str,
) -> np.ndarray:
    if len(spectra) == 0:
        raise ValueError(f"{source_name}: zero spectra.")

    n = int(drop_first_n)
    retained = []

    for i, spec in enumerate(spectra):
        if n < 0:
            raise ValueError("--spectral-drop-first-n must be >= 0.")
        if spec.size <= n:
            raise ValueError(
                f"{source_name}: row {i} spectrum length {spec.size} is <= "
                f"spectral_drop_first_n={n}."
            )
        retained.append(spec[n:].astype(np.float32))

    arr = np.stack(retained, axis=0).astype(np.float32)

    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"{source_name}: retained spectra contain {bad} non-finite values."
        )

    return arr


# -------------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------------


def fit_spectral_normalization_stats(
    train_val_retained: np.ndarray,
    method: str,
    epsilon: float,
) -> Dict[str, Any]:
    method = str(method).lower()
    eps = float(epsilon)

    if train_val_retained.ndim != 2:
        raise ValueError(
            f"Expected [N,L] retained spectra, got {train_val_retained.shape}"
        )

    stats: Dict[str, Any] = {
        "method": method,
        "n_fit_samples": int(train_val_retained.shape[0]),
        "spectral_length_retained": int(train_val_retained.shape[1]),
        "epsilon": eps,
    }

    if method == "none":
        stats["center"] = np.zeros(train_val_retained.shape[1], dtype=np.float32)
        stats["scale"] = np.ones(train_val_retained.shape[1], dtype=np.float32)
        return stats

    if method == "zscore":
        center = np.mean(train_val_retained, axis=0).astype(np.float32)
        scale = np.std(train_val_retained, axis=0, ddof=0).astype(np.float32)
        scale = np.where(scale < eps, 1.0, scale).astype(np.float32)
        stats["center"] = center
        stats["scale"] = scale
        return stats

    if method == "robust_zscore":
        center = np.median(train_val_retained, axis=0).astype(np.float32)
        q25 = np.percentile(train_val_retained, 25, axis=0).astype(np.float32)
        q75 = np.percentile(train_val_retained, 75, axis=0).astype(np.float32)
        # IQR / 1.349 estimates standard deviation for a normal distribution.
        scale = ((q75 - q25) / 1.349).astype(np.float32)
        scale = np.where(scale < eps, 1.0, scale).astype(np.float32)
        stats["center"] = center
        stats["scale"] = scale
        stats["q25"] = q25
        stats["q75"] = q75
        return stats

    if method == "minmax":
        vmin = np.min(train_val_retained, axis=0).astype(np.float32)
        vmax = np.max(train_val_retained, axis=0).astype(np.float32)
        scale = (vmax - vmin).astype(np.float32)
        scale = np.where(scale < eps, 1.0, scale).astype(np.float32)
        stats["center"] = vmin
        stats["scale"] = scale
        stats["min"] = vmin
        stats["max"] = vmax
        print(np.shape(vmin))
        return stats

    raise ValueError(f"Unknown normalization method: {method}")


def apply_spectral_normalization(
    retained: np.ndarray,
    stats: Dict[str, Any],
    clip: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    method = stats["method"]

    if method == "none":
        out = retained.astype(np.float32).copy()
    elif method in ["zscore", "robust_zscore"]:
        center = np.asarray(stats["center"], dtype=np.float32)
        scale = np.asarray(stats["scale"], dtype=np.float32)
        out = (retained.astype(np.float32) - center) / scale
    elif method == "minmax":
        center = np.asarray(stats["center"], dtype=np.float32)
        scale = np.asarray(stats["scale"], dtype=np.float32)
        out = (retained.astype(np.float32) - center) / scale
    else:
        raise ValueError(f"Unknown method in stats: {method}")

    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        out = np.clip(out, lo, hi)

    return out.astype(np.float32)


def format_float_list(values: np.ndarray, precision: int) -> str:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rounded = [round(float(x), int(precision)) for x in values.tolist()]
    # JSON arrays are valid Python literals, so dataset.py ast.literal_eval can parse them.
    return json.dumps(rounded, separators=(",", ":"))


def build_output_spectra(
    original_spectra: Sequence[np.ndarray],
    normalized_retained: np.ndarray,
    drop_first_n: int,
    output_mode: str,
    precision: int,
) -> List[str]:
    out_strings = []
    n = int(drop_first_n)

    if len(original_spectra) != normalized_retained.shape[0]:
        raise ValueError("Number of spectra and normalized rows does not match.")

    for i, spec in enumerate(original_spectra):
        if output_mode == "preserve_length":
            combined = spec.astype(np.float32).copy()
            combined[n:] = normalized_retained[i]
        elif output_mode == "normalized_only":
            combined = normalized_retained[i]
        else:
            raise ValueError(f"Unknown output_mode={output_mode}")

        out_strings.append(format_float_list(combined, precision=precision))

    return out_strings


def stats_to_jsonable(stats: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = [float(x) for x in v.reshape(-1).tolist()]
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# -------------------------------------------------------------------------
# Debug plots
# -------------------------------------------------------------------------


def maybe_wavelength_axis(
    length: int, wavelength_min: float = 400.0, wavelength_max: float = 2500.0
) -> np.ndarray:
    return np.linspace(float(wavelength_min), float(wavelength_max), int(length))


def plot_mean_spectra_before_after(
    before: np.ndarray,
    after: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    x = maybe_wavelength_axis(before.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, before.mean(axis=0), label="Before mean", linewidth=2)
    ax.plot(x, after.mean(axis=0), label="After mean", linewidth=2)
    ax.fill_between(
        x,
        before.mean(axis=0) - before.std(axis=0),
        before.mean(axis=0) + before.std(axis=0),
        alpha=0.18,
        label="Before +/- 1 std",
    )
    ax.fill_between(
        x,
        after.mean(axis=0) - after.std(axis=0),
        after.mean(axis=0) + after.std(axis=0),
        alpha=0.18,
        label="After +/- 1 std",
    )
    ax.set_xlabel("Wavelength approximation / spectral index")
    ax.set_ylabel("Spectral value")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_value_histograms(
    train_val_before: np.ndarray,
    train_val_after: np.ndarray,
    test_before: np.ndarray,
    test_after: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        train_val_before.reshape(-1), bins=80, alpha=0.45, label="Train+val before"
    )
    axes[0].hist(
        train_val_after.reshape(-1), bins=80, alpha=0.55, label="Train+val after"
    )
    axes[0].set_title("Train+val spectral values")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].hist(test_before.reshape(-1), bins=80, alpha=0.45, label="Test before")
    axes[1].hist(test_after.reshape(-1), bins=80, alpha=0.55, label="Test after")
    axes[1].set_title("Test spectral values")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_overlay_preview(
    arr: np.ndarray,
    title: str,
    out_path: Path,
    n_preview: int,
) -> None:
    if arr.shape[0] == 0:
        return

    n = min(int(n_preview), arr.shape[0])
    idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
    x = maybe_wavelength_axis(arr.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in idx:
        ax.plot(x, arr[i], alpha=0.55, linewidth=0.9)
    ax.set_xlabel("Wavelength approximation / spectral index")
    ax.set_ylabel("Normalized spectral value")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def check_required_columns(df: pd.DataFrame, spectral_col: str, name: str) -> None:
    required = REQUIRED_BASE_COLUMNS.copy()
    if spectral_col != "spectral":
        required = [c if c != "spectral" else spectral_col for c in required]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def main() -> int:
    args = parse_args()

    train_csv = expand_path(args.train_csv)
    val_csv = expand_path(args.val_csv)
    test_csv = expand_path(args.test_csv)
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / args.train_val_output_name
    test_out = output_dir / args.test_output_name

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    check_required_columns(train_df, args.spectral_column, "train")
    check_required_columns(val_df, args.spectral_column, "val")
    check_required_columns(test_df, args.spectral_column, "test")

    if args.check_columns:
        if list(train_df.columns) != list(val_df.columns):
            raise ValueError("Train and val CSV columns are not identical.")
        if list(train_df.columns) != list(test_df.columns):
            raise ValueError("Train and test CSV columns are not identical.")

    train_spectra, train_report = parse_spectral_column(
        train_df, args.spectral_column, "train"
    )
    val_spectra, val_report = parse_spectral_column(val_df, args.spectral_column, "val")
    test_spectra, test_report = parse_spectral_column(
        test_df, args.spectral_column, "test"
    )

    # Length consistency across all splits.
    all_lengths = [s.size for s in train_spectra + val_spectra + test_spectra]
    if len(set(all_lengths)) != 1:
        counts = pd.Series(all_lengths).value_counts().to_dict()
        raise ValueError(f"Spectral lengths differ across train/val/test: {counts}")

    train_retained = stack_retained_spectra(
        train_spectra, args.spectral_drop_first_n, "train"
    )
    val_retained = stack_retained_spectra(
        val_spectra, args.spectral_drop_first_n, "val"
    )
    test_retained = stack_retained_spectra(
        test_spectra, args.spectral_drop_first_n, "test"
    )

    train_val_retained = np.concatenate([train_retained, val_retained], axis=0)

    stats = fit_spectral_normalization_stats(
        train_val_retained=train_val_retained,
        method=args.method,
        epsilon=args.epsilon,
    )

    clip = None
    if args.clip is not None:
        clip = (float(args.clip[0]), float(args.clip[1]))

    train_norm = apply_spectral_normalization(train_retained, stats, clip=clip)
    val_norm = apply_spectral_normalization(val_retained, stats, clip=clip)
    test_norm = apply_spectral_normalization(test_retained, stats, clip=clip)

    # Concatenate train and val after normalization.
    train_val_df = pd.concat(
        [train_df.copy(), val_df.copy()], axis=0, ignore_index=True
    )
    train_val_spectra = train_spectra + val_spectra
    train_val_norm = np.concatenate([train_norm, val_norm], axis=0)

    train_val_df[args.spectral_column] = build_output_spectra(
        original_spectra=train_val_spectra,
        normalized_retained=train_val_norm,
        drop_first_n=args.spectral_drop_first_n,
        output_mode=args.output_mode,
        precision=args.json_precision,
    )

    test_norm_df = test_df.copy()
    test_norm_df[args.spectral_column] = build_output_spectra(
        original_spectra=test_spectra,
        normalized_retained=test_norm,
        drop_first_n=args.spectral_drop_first_n,
        output_mode=args.output_mode,
        precision=args.json_precision,
    )

    train_val_df.to_csv(train_out, index=False)
    test_norm_df.to_csv(test_out, index=False)

    stats_json = stats_to_jsonable(stats)
    stats_json.update(
        {
            "spectral_column": args.spectral_column,
            "spectral_drop_first_n": int(args.spectral_drop_first_n),
            "output_mode": args.output_mode,
            "clip": None if clip is None else [float(clip[0]), float(clip[1])],
            "fit_sources": ["train", "val"],
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "train_val_output_csv": str(train_out),
            "test_output_csv": str(test_out),
            "original_spectral_length": int(all_lengths[0]),
            "retained_spectral_length": int(train_val_retained.shape[1]),
            "safety_note": (
                "Normalization statistics were fitted only on train+val retained spectra. "
                "The test split was transformed with train+val statistics only."
            ),
        }
    )

    stats_path = output_dir / "spectral_normalization_stats.json"
    stats_path.write_text(json.dumps(stats_json, indent=2))

    manifest = {
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_train_val_rows": int(len(train_val_df)),
        "n_test_rows": int(len(test_df)),
        "columns": list(train_df.columns),
        "output_columns_train_val": list(train_val_df.columns),
        "output_columns_test": list(test_norm_df.columns),
        "method": args.method,
        "spectral_drop_first_n": int(args.spectral_drop_first_n),
        "output_mode": args.output_mode,
        "train_val_output_csv": str(train_out),
        "test_output_csv": str(test_out),
        "stats_json": str(stats_path),
    }
    (output_dir / "spectral_normalization_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    reports = pd.concat(
        [train_report, val_report, test_report], axis=0, ignore_index=True
    )
    reports.to_csv(output_dir / "spectral_parse_report.csv", index=False)

    if args.debug:
        plot_mean_spectra_before_after(
            before=train_val_retained,
            after=train_val_norm,
            title=f"Train+val spectra before/after {args.method}",
            out_path=output_dir / "train_val_spectral_mean_before_after.svg",
        )
        plot_mean_spectra_before_after(
            before=test_retained,
            after=test_norm,
            title=f"Test spectra before/after train+val {args.method}",
            out_path=output_dir / "test_spectral_mean_before_after.svg",
        )
        plot_value_histograms(
            train_val_before=train_val_retained,
            train_val_after=train_val_norm,
            test_before=test_retained,
            test_after=test_norm,
            out_path=output_dir / "spectral_value_histograms_before_after.svg",
        )
        plot_overlay_preview(
            arr=train_val_norm,
            title="Preview of normalized train+val spectra",
            out_path=output_dir / "train_val_normalized_overlay_preview.svg",
            n_preview=args.debug_preview_n,
        )
        plot_overlay_preview(
            arr=test_norm,
            title="Preview of normalized test spectra",
            out_path=output_dir / "test_normalized_overlay_preview.svg",
            n_preview=args.debug_preview_n,
        )

    print("=" * 80)
    print("Spectral CSV normalization finished")
    print("=" * 80)
    print(f"Method:                       {args.method}")
    print(f"Fit statistics from:          train + val")
    print(f"Spectral column:              {args.spectral_column}")
    print(f"Original spectral length:     {all_lengths[0]}")
    print(f"Dropped/ignored prefix:       {args.spectral_drop_first_n}")
    print(f"Retained normalized length:   {train_val_retained.shape[1]}")
    print(f"Output mode:                  {args.output_mode}")
    print(f"Train rows:                   {len(train_df)}")
    print(f"Val rows:                     {len(val_df)}")
    print(f"Train+val output rows:        {len(train_val_df)}")
    print(f"Test output rows:             {len(test_norm_df)}")
    print(f"Train+val output CSV:         {train_out}")
    print(f"Test output CSV:              {test_out}")
    print(f"Stats JSON:                   {stats_path}")
    if args.debug:
        print(f"Debug directory:              {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
