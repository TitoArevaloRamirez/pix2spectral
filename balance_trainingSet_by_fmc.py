#!/usr/bin/env python3
"""
Balance a pix2spectral-style CSV by FMC_d / LWC target distribution.

Purpose
-------
Your dehydration stages can overlap strongly in FMC/LWC, producing an
unbalanced target distribution. This script balances the CSV by histogram bins
of the target variable, usually FMC_d, while preserving each original row
exactly.

Important safety rule
---------------------
The script NEVER recombines image filenames, spectra, species/stage labels, or
FMC values. It only selects or removes complete rows from the input CSV.

Therefore, for every selected sample:

    blue, green, red, nir, red_edge, spectral, Species/Stages, FMC_d

remain exactly as they were in the original CSV.

Expected CSV style
------------------
Compatible with the dataset.py convention used in pix2spectral:

    blue, green, red, nir, red_edge, spectral, Species, Stages, FMC_d

Also accepts lowercase aliases:

    species, stage

Main output
-----------
A balanced CSV with the same columns as the input CSV.

Debug outputs
-------------
With --debug, the script writes:

    histogram_fmc_before_after.svg
    fmc_bin_counts_before_after.svg
    stage_bin_counts_before.csv
    stage_bin_counts_after.csv
    selected_rows_debug.csv
    dropped_rows_debug.csv
    bin_counts.csv
    balance_manifest.json

With --debug-spectra, it additionally parses the spectral column and writes:

    spectra_mean_before_after.svg
    spectra_overlay_selected.svg
    spectral_integrity_report.csv

With --debug-images and --img-dir, it additionally checks image files and writes:

    image_integrity_report.csv
    image_preview_selected_rows.png

Example
-------
python balance_csv_by_fmc_histogram.py \
    --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --output-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train_balanced_FMC.csv \
    --target-column FMC_d \
    --n-bins 10 \
    --target-per-bin min \
    --random-state 42 \
    --debug \
    --debug-spectra \
    --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/"
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None


BAND_COLUMNS = ["blue", "green", "red", "nir", "red_edge"]


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced subset of a pix2spectral CSV by histogram bins "
            "of FMC_d or another target column."
        )
    )

    parser.add_argument("--input-csv", required=True, help="Input CSV file.")
    parser.add_argument("--output-csv", required=True, help="Balanced output CSV file.")

    parser.add_argument(
        "--target-column",
        default="FMC_d",
        help="Continuous target column used for balancing. Default: FMC_d.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of uniform FMC bins. Ignored if --bin-width or --bin-edges is used.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Optional fixed FMC bin width. Example: --bin-width 10.",
    )
    parser.add_argument(
        "--bin-edges",
        default=None,
        help="Optional comma-separated explicit bin edges. Example: 0,25,50,75,100,125.",
    )
    parser.add_argument(
        "--binning",
        choices=["uniform", "quantile"],
        default="uniform",
        help=(
            "uniform: equal-width target bins. "
            "quantile: equal-count bins before balancing. "
            "For true FMC balancing, uniform is usually preferred."
        ),
    )

    parser.add_argument(
        "--target-per-bin",
        default="min",
        help=(
            "Rows to keep per non-empty bin. Use 'min', 'median', 'mean', "
            "or an integer. Default: min."
        ),
    )
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=1,
        help=(
            "Ignore bins with fewer than this many rows before computing the "
            "balanced subset. Default: 1."
        ),
    )

    parser.add_argument(
        "--stratify-columns",
        nargs="*",
        default=None,
        help=(
            "Optional columns to balance separately inside each stratum, for "
            "example --stratify-columns Species or --stratify-columns Species Stages."
        ),
    )

    parser.add_argument(
        "--species-column",
        default="auto",
        help="Species column for reports/filtering. Use auto, Species, species, or none.",
    )
    parser.add_argument(
        "--stage-column",
        default="auto",
        help="Stage column for reports/filtering. Use auto, Stages, stage, or none.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species filter. Example: --species Avocado.",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Optional stage filter. Use all/any/* to disable. Example: --stage dry.",
    )

    parser.add_argument(
        "--image-columns",
        nargs="*",
        default=BAND_COLUMNS,
        help="Image filename columns. Default: blue green red nir red_edge.",
    )
    parser.add_argument(
        "--spectral-column",
        default="spectral",
        help="Spectral signature column. Default: spectral.",
    )
    parser.add_argument(
        "--img-dir",
        default=None,
        help="Optional root directory for image existence checks and previews.",
    )

    parser.add_argument(
        "--group-column",
        default="none",
        help=(
            "Optional grouping column for debug reports only. Use 'auto' to infer "
            "leaf id from blue/blue_basename. This script samples complete rows; "
            "it does not recombine groups."
        ),
    )

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shuffle-output", action="store_true")
    parser.add_argument(
        "--sort-output-by-original-index",
        action="store_true",
        default=True,
        help="Preserve original row order in the output CSV. Default: enabled.",
    )

    parser.add_argument("--debug", action="store_true", help="Write debug plots and CSV reports.")
    parser.add_argument("--debug-spectra", action="store_true", help="Parse and plot spectral signatures.")
    parser.add_argument("--debug-images", action="store_true", help="Check image paths and create image previews.")
    parser.add_argument(
        "--debug-preview-n",
        type=int,
        default=12,
        help="Number of selected rows to show in image/spectrum previews. Default: 12.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Debug output directory. Default: output_csv parent / fmc_balance_debug.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Column and path helpers
# -------------------------------------------------------------------------

def expand_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def choose_existing_column(
    df: pd.DataFrame,
    requested: str,
    candidates: Sequence[str],
    allow_none: bool = True,
) -> Optional[str]:
    if requested is None:
        return None

    requested_lower = str(requested).strip().lower()

    if requested_lower in ["none", "null", ""]:
        return None

    if requested_lower == "auto":
        for c in candidates:
            if c in df.columns:
                return c
        if allow_none:
            return None
        raise ValueError(f"None of the candidate columns exists: {candidates}")

    if requested in df.columns:
        return requested

    raise ValueError(f"Requested column '{requested}' not found. Available columns: {list(df.columns)}")


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def basename_any(path_like: Any) -> str:
    if pd.isna(path_like):
        return ""
    text = str(path_like).strip()
    if not text:
        return ""
    return Path(PureWindowsPath(text).name).name


def resolve_image_path(fname: Any, root_dir: Optional[str] = None) -> Path:
    text = str(fname).strip()

    if text == "" or text.lower() == "nan":
        return Path("")

    p = Path(text).expanduser()

    if p.is_absolute():
        return p

    candidates = []
    if root_dir is not None:
        root = Path(root_dir).expanduser()
        candidates.append(root / text)
        candidates.append(root / PureWindowsPath(text).name)
        candidates.append(root / Path(text).name)

    candidates.append(p)

    for c in candidates:
        if c.exists():
            return c.resolve()

    return candidates[0] if candidates else p


def infer_leaf_id_from_text(value: Any) -> Optional[str]:
    import re

    if pd.isna(value):
        return None

    text = basename_any(value).lower()
    patterns = [
        r"leaf[_-]?(\d+)",
        r"leaf(\d{3})d\d",
        r"(\d{3})d\d",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    return None


def add_debug_identity_columns(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[str]]:
    df = df.copy()

    if args.group_column.lower() == "none":
        return df, None

    if args.group_column != "auto":
        if args.group_column not in df.columns:
            raise ValueError(f"--group-column '{args.group_column}' was not found.")
        return df, args.group_column

    for candidate in ["leaf_id", "Leaf_ID", "leaf", "leafID"]:
        if candidate in df.columns:
            return df, candidate

    for candidate in ["blue_basename", "blue", "Blue", "filename", "image_name"]:
        if candidate in df.columns:
            inferred = df[candidate].map(infer_leaf_id_from_text)
            if inferred.notna().any():
                df["leaf_id_inferred"] = inferred
                return df, "leaf_id_inferred"

    return df, None


# -------------------------------------------------------------------------
# Spectral parsing and debug checks
# -------------------------------------------------------------------------

def parse_spectrum_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)

    if pd.isna(value):
        raise ValueError("Missing spectral value.")

    text = str(value).strip()
    if not text:
        raise ValueError("Empty spectral string.")

    try:
        parsed = json.loads(text)
        return np.asarray(parsed, dtype=float).reshape(-1)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        return np.asarray(parsed, dtype=float).reshape(-1)
    except Exception:
        pass

    cleaned = text.strip().strip("[]()").replace(";", ",")
    if "," in cleaned:
        arr = np.fromstring(cleaned, sep=",", dtype=float)
    else:
        arr = np.fromstring(cleaned, sep=" ", dtype=float)

    if arr.size == 0:
        raise ValueError(f"Could not parse spectral string: {text[:120]}...")

    return arr.reshape(-1)


def spectral_integrity_report(df: pd.DataFrame, spectral_col: str) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    rows = []
    spectra = []

    if spectral_col not in df.columns:
        return pd.DataFrame([{"error": f"Column '{spectral_col}' not found."}]), None

    for i, value in enumerate(df[spectral_col].values):
        try:
            spec = parse_spectrum_value(value)
            finite = np.isfinite(spec)
            rows.append({
                "row_index": int(df.iloc[i]["__original_index"]),
                "parse_ok": True,
                "length": int(spec.size),
                "finite_count": int(finite.sum()),
                "nonfinite_count": int((~finite).sum()),
                "min": float(np.nanmin(spec)) if spec.size else np.nan,
                "max": float(np.nanmax(spec)) if spec.size else np.nan,
                "mean": float(np.nanmean(spec)) if spec.size else np.nan,
            })
            spectra.append(spec)
        except Exception as exc:
            rows.append({
                "row_index": int(df.iloc[i]["__original_index"]),
                "parse_ok": False,
                "error": str(exc),
            })

    report = pd.DataFrame(rows)

    if not spectra:
        return report, None

    lengths = [s.size for s in spectra]
    if len(set(lengths)) != 1:
        return report, None

    return report, np.stack(spectra, axis=0)


# -------------------------------------------------------------------------
# Binning and balancing
# -------------------------------------------------------------------------

def make_bin_edges(y: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        raise ValueError("No finite target values available for binning.")

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if args.bin_edges is not None:
        edges = np.asarray([float(x) for x in str(args.bin_edges).split(",")], dtype=float)
        edges = np.unique(np.sort(edges))
        if edges.size < 2:
            raise ValueError("--bin-edges must contain at least two values.")
        return edges

    if args.binning == "quantile":
        q = np.linspace(0.0, 1.0, int(args.n_bins) + 1)
        edges = np.quantile(y, q)
        edges = np.unique(edges)
        if edges.size < 2:
            raise ValueError("Quantile binning produced fewer than two unique edges.")
        return edges

    if args.bin_width is not None:
        width = float(args.bin_width)
        if width <= 0:
            raise ValueError("--bin-width must be positive.")

        lo = math.floor(y_min / width) * width
        hi = math.ceil(y_max / width) * width
        if hi <= lo:
            hi = lo + width

        edges = np.arange(lo, hi + 0.5 * width, width, dtype=float)
        if edges[-1] < y_max:
            edges = np.append(edges, edges[-1] + width)
        return edges

    n_bins = int(args.n_bins)
    if n_bins < 1:
        raise ValueError("--n-bins must be >= 1.")

    if y_max <= y_min:
        return np.asarray([y_min - 0.5, y_max + 0.5], dtype=float)

    return np.linspace(y_min, y_max, n_bins + 1, dtype=float)


def assign_bins(df: pd.DataFrame, target_col: str, edges: np.ndarray) -> pd.DataFrame:
    df = df.copy()

    # Extend endpoints slightly to include exact min/max robustly.
    edges = np.asarray(edges, dtype=float).copy()
    eps = max(1e-9, 1e-9 * max(1.0, abs(edges[-1] - edges[0])))
    edges[0] -= eps
    edges[-1] += eps

    df["__fmc_bin"] = pd.cut(
        df[target_col].astype(float),
        bins=edges,
        labels=False,
        include_lowest=True,
        right=False,
    )

    if df["__fmc_bin"].isna().any():
        bad = int(df["__fmc_bin"].isna().sum())
        raise ValueError(
            f"{bad} rows could not be assigned to FMC bins. Check bin edges and target values."
        )

    df["__fmc_bin"] = df["__fmc_bin"].astype(int)
    df["__fmc_bin_left"] = df["__fmc_bin"].map(lambda i: float(edges[int(i)]))
    df["__fmc_bin_right"] = df["__fmc_bin"].map(lambda i: float(edges[int(i) + 1]))
    df["__fmc_bin_label"] = df.apply(
        lambda r: f"[{r['__fmc_bin_left']:.6g}, {r['__fmc_bin_right']:.6g})",
        axis=1,
    )

    return df


def target_count_from_counts(counts: pd.Series, target_per_bin: str, min_bin_count: int) -> int:
    counts = counts[counts >= int(min_bin_count)]

    if counts.empty:
        raise ValueError(
            f"No bins have at least min_bin_count={min_bin_count}. "
            "Lower --min-bin-count or use fewer/wider bins."
        )

    mode = str(target_per_bin).strip().lower()

    if mode == "min":
        return int(counts.min())

    if mode == "median":
        return int(np.floor(counts.median()))

    if mode == "mean":
        return int(np.floor(counts.mean()))

    try:
        requested = int(mode)
    except ValueError as exc:
        raise ValueError("--target-per-bin must be 'min', 'median', 'mean', or an integer.") from exc

    if requested < 1:
        raise ValueError("--target-per-bin integer must be >= 1.")

    return int(requested)


def balance_one_dataframe(
    df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    stratum_label: str = "global",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = df.groupby("__fmc_bin", observed=True).size().rename("count_before")
    target_n = target_count_from_counts(counts, args.target_per_bin, args.min_bin_count)

    selected_parts = []
    report_rows = []

    for bin_id in sorted(counts.index.tolist()):
        bin_df = df[df["__fmc_bin"] == bin_id].copy()
        count_before = len(bin_df)

        if count_before < int(args.min_bin_count):
            keep_n = 0
            status = "dropped_below_min_bin_count"
        else:
            keep_n = min(target_n, count_before)
            status = "sampled" if keep_n < count_before else "kept_all"

        if keep_n > 0:
            sampled = bin_df.sample(
                n=keep_n,
                replace=False,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            selected_parts.append(sampled)

        report_rows.append({
            "stratum": stratum_label,
            "bin_id": int(bin_id),
            "bin_left": float(bin_df["__fmc_bin_left"].iloc[0]),
            "bin_right": float(bin_df["__fmc_bin_right"].iloc[0]),
            "bin_label": str(bin_df["__fmc_bin_label"].iloc[0]),
            "count_before": int(count_before),
            "count_after": int(keep_n),
            "target_per_bin_effective": int(target_n),
            "status": status,
        })

    if selected_parts:
        selected_df = pd.concat(selected_parts, axis=0, ignore_index=False)
    else:
        selected_df = df.iloc[0:0].copy()

    return selected_df, pd.DataFrame(report_rows)


def balance_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(args.random_state))

    if args.stratify_columns:
        missing = [c for c in args.stratify_columns if c not in df.columns]
        if missing:
            raise ValueError(f"--stratify-columns missing from CSV: {missing}")

        selected_parts = []
        reports = []

        grouped = df.groupby(args.stratify_columns, dropna=False, observed=True)
        for key, sub in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            label = "|".join(f"{c}={v}" for c, v in zip(args.stratify_columns, key))
            selected, report = balance_one_dataframe(sub, args, rng, stratum_label=label)
            selected_parts.append(selected)
            reports.append(report)

        if selected_parts:
            selected_df = pd.concat(selected_parts, axis=0, ignore_index=False)
        else:
            selected_df = df.iloc[0:0].copy()

        report_df = pd.concat(reports, axis=0, ignore_index=True) if reports else pd.DataFrame()
    else:
        selected_df, report_df = balance_one_dataframe(df, args, rng, stratum_label="global")

    if args.shuffle_output:
        selected_df = selected_df.sample(
            frac=1.0,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
    elif args.sort_output_by_original_index:
        selected_df = selected_df.sort_values("__original_index")

    return selected_df.reset_index(drop=True), report_df


# -------------------------------------------------------------------------
# Debug plots and reports
# -------------------------------------------------------------------------

def plot_histogram_before_after(
    original_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    target_col: str,
    edges: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        original_df[target_col].to_numpy(dtype=float),
        bins=edges,
        alpha=0.45,
        label=f"Original n={len(original_df)}",
        edgecolor="black",
    )
    ax.hist(
        balanced_df[target_col].to_numpy(dtype=float),
        bins=edges,
        alpha=0.65,
        label=f"Balanced n={len(balanced_df)}",
        edgecolor="black",
    )
    ax.set_xlabel(target_col)
    ax.set_ylabel("Sample count")
    ax.set_title(f"{target_col} histogram before and after balancing")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_bin_counts(report_df: pd.DataFrame, out_path: Path) -> None:
    if report_df.empty:
        return

    if report_df["stratum"].nunique() == 1:
        x = np.arange(len(report_df))
        width = 0.42

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, report_df["count_before"], width=width, label="Before")
        ax.bar(x + width / 2, report_df["count_after"], width=width, label="After")
        ax.set_xticks(x)
        ax.set_xticklabels(report_df["bin_label"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Sample count")
        ax.set_xlabel("FMC bin")
        ax.set_title("FMC bin counts before and after balancing")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, format="svg")
        plt.close(fig)
        return

    # For many strata, plot aggregate counts by bin.
    agg = report_df.groupby(["bin_id", "bin_label"], observed=True)[["count_before", "count_after"]].sum().reset_index()
    x = np.arange(len(agg))
    width = 0.42

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, agg["count_before"], width=width, label="Before")
    ax.bar(x + width / 2, agg["count_after"], width=width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["bin_label"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Sample count")
    ax.set_xlabel("FMC bin")
    ax.set_title("Aggregate FMC bin counts before and after balancing")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def save_stage_bin_tables(
    original_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    stage_col: Optional[str],
    out_dir: Path,
) -> None:
    if stage_col is None or stage_col not in original_df.columns:
        return

    before = pd.crosstab(original_df[stage_col], original_df["__fmc_bin_label"])
    after = pd.crosstab(balanced_df[stage_col], balanced_df["__fmc_bin_label"])

    before.to_csv(out_dir / "stage_bin_counts_before.csv")
    after.to_csv(out_dir / "stage_bin_counts_after.csv")


def plot_spectra_debug(
    original_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    spectral_col: str,
    out_dir: Path,
    preview_n: int,
) -> None:
    before_report, before_spectra = spectral_integrity_report(original_df, spectral_col)
    after_report, after_spectra = spectral_integrity_report(balanced_df, spectral_col)

    before_report["split"] = "original"
    after_report["split"] = "balanced"
    pd.concat([before_report, after_report], axis=0, ignore_index=True).to_csv(
        out_dir / "spectral_integrity_report.csv",
        index=False,
    )

    if before_spectra is None or after_spectra is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_before = np.arange(before_spectra.shape[1])
    x_after = np.arange(after_spectra.shape[1])
    ax.plot(x_before, np.mean(before_spectra, axis=0), label="Original mean", linewidth=2)
    ax.plot(x_after, np.mean(after_spectra, axis=0), label="Balanced mean", linewidth=2)
    ax.fill_between(
        x_before,
        np.mean(before_spectra, axis=0) - np.std(before_spectra, axis=0),
        np.mean(before_spectra, axis=0) + np.std(before_spectra, axis=0),
        alpha=0.18,
        label="Original +/- 1 std",
    )
    ax.fill_between(
        x_after,
        np.mean(after_spectra, axis=0) - np.std(after_spectra, axis=0),
        np.mean(after_spectra, axis=0) + np.std(after_spectra, axis=0),
        alpha=0.18,
        label="Balanced +/- 1 std",
    )
    ax.set_xlabel("Spectral index")
    ax.set_ylabel("Reflectance")
    ax.set_title("Spectral signatures before and after FMC balancing")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "spectra_mean_before_after.svg", format="svg")
    plt.close(fig)

    n = min(int(preview_n), after_spectra.shape[0])
    if n > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        idx = np.linspace(0, after_spectra.shape[0] - 1, n).astype(int)
        for j in idx:
            ax.plot(np.arange(after_spectra.shape[1]), after_spectra[j], alpha=0.55, linewidth=0.9)
        ax.set_xlabel("Spectral index")
        ax.set_ylabel("Reflectance")
        ax.set_title(f"Preview of {n} selected balanced spectral signatures")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "spectra_overlay_selected.svg", format="svg")
        plt.close(fig)


def robust_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return (255 * out).astype(np.uint8)


def image_integrity_and_preview(
    balanced_df: pd.DataFrame,
    image_cols: Sequence[str],
    img_dir: Optional[str],
    out_dir: Path,
    preview_n: int,
) -> None:
    if not image_cols:
        return

    rows = []
    for i, row in balanced_df.iterrows():
        for col in image_cols:
            if col not in balanced_df.columns:
                rows.append({
                    "row_index": int(row["__original_index"]),
                    "column": col,
                    "exists": False,
                    "error": "column_missing",
                })
                continue

            p = resolve_image_path(row[col], img_dir)
            rows.append({
                "row_index": int(row["__original_index"]),
                "column": col,
                "csv_value": row[col],
                "resolved_path": str(p),
                "exists": bool(p.exists()),
            })

    pd.DataFrame(rows).to_csv(out_dir / "image_integrity_report.csv", index=False)

    if Image is None:
        return

    n = min(int(preview_n), len(balanced_df))
    if n <= 0:
        return

    thumbs = []
    thumb_w, thumb_h = 128, 128
    label_h = 28

    selected = balanced_df.head(n)

    for _, row in selected.iterrows():
        row_imgs = []
        for col in image_cols:
            try:
                p = resolve_image_path(row[col], img_dir)
                if not p.exists():
                    raise FileNotFoundError(str(p))
                img = Image.open(p)
                arr = np.asarray(img)
                if arr.ndim == 3:
                    arr = arr[..., :3].mean(axis=2)
                arr8 = robust_uint8(arr)
                im = Image.fromarray(arr8).convert("L").resize((thumb_w, thumb_h))
            except Exception:
                im = Image.new("L", (thumb_w, thumb_h), color=0)

            canvas = Image.new("RGB", (thumb_w, thumb_h + label_h), color="white")
            canvas.paste(im.convert("RGB"), (0, 0))
            if ImageDraw is not None:
                d = ImageDraw.Draw(canvas)
                d.text((4, thumb_h + 4), str(col), fill=(0, 0, 0))
            row_imgs.append(canvas)

        combined_row = Image.new("RGB", (thumb_w * len(row_imgs), thumb_h + label_h), color="white")
        for j, im in enumerate(row_imgs):
            combined_row.paste(im, (j * thumb_w, 0))
        thumbs.append(combined_row)

    total = Image.new(
        "RGB",
        (thumb_w * len(image_cols), (thumb_h + label_h) * len(thumbs)),
        color="white",
    )
    for i, im in enumerate(thumbs):
        total.paste(im, (0, i * (thumb_h + label_h)))

    total.save(out_dir / "image_preview_selected_rows.png")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    input_csv = expand_path(args.input_csv)
    output_csv = expand_path(args.output_csv)
    assert input_csv is not None and output_csv is not None

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.output_dir is None:
        out_dir = output_csv.parent / "fmc_balance_debug"
    else:
        out_dir = expand_path(args.output_dir)
        assert out_dir is not None

    if args.debug or args.debug_spectra or args.debug_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    df_original = pd.read_csv(input_csv)
    original_columns = list(df_original.columns)

    if args.target_column not in df_original.columns:
        raise ValueError(
            f"Target column '{args.target_column}' not found. "
            f"Available columns: {list(df_original.columns)}"
        )

    df = df_original.copy()
    df["__original_index"] = np.arange(len(df), dtype=int)

    species_col = choose_existing_column(
        df,
        args.species_column,
        candidates=["Species", "species", "SPECIES"],
        allow_none=True,
    )
    stage_col = choose_existing_column(
        df,
        args.stage_column,
        candidates=["Stages", "stage", "STAGE", "Stage"],
        allow_none=True,
    )

    if args.species is not None:
        if species_col is None:
            raise ValueError("--species was provided, but no species column was found.")
        wanted = normalize_text(args.species)
        before = len(df)
        df = df[df[species_col].map(normalize_text) == wanted].copy()
        print(f"Species filter kept {len(df)}/{before} rows.")

    if args.stage is not None and normalize_text(args.stage) not in ["all", "any", "*", ""]:
        if stage_col is None:
            raise ValueError("--stage was provided, but no stage column was found.")
        wanted = normalize_text(args.stage)
        before = len(df)
        df = df[df[stage_col].map(normalize_text) == wanted].copy()
        print(f"Stage filter kept {len(df)}/{before} rows.")

    if df.empty:
        raise ValueError("No rows left after filtering.")

    df[args.target_column] = pd.to_numeric(df[args.target_column], errors="coerce")
    missing_target = int(df[args.target_column].isna().sum())
    if missing_target > 0:
        raise ValueError(
            f"{missing_target} rows have missing/non-numeric {args.target_column}. "
            "Fix the CSV before balancing."
        )

    df, group_col = add_debug_identity_columns(df, args)

    edges = make_bin_edges(df[args.target_column].to_numpy(dtype=float), args)
    df_binned = assign_bins(df, args.target_column, edges)

    balanced_df, report_df = balance_dataframe(df_binned, args)

    if balanced_df.empty:
        raise ValueError("Balancing selected zero rows. Use fewer/wider bins or lower --min-bin-count.")

    dropped_df = df_binned[~df_binned["__original_index"].isin(balanced_df["__original_index"])].copy()

    # Save output with exactly the same original columns and no debug/helper columns.
    balanced_output = balanced_df[original_columns].copy()
    balanced_output.to_csv(output_csv, index=False)

    if args.debug or args.debug_spectra or args.debug_images:
        report_df.to_csv(out_dir / "bin_counts.csv", index=False)
        balanced_df.to_csv(out_dir / "selected_rows_debug.csv", index=False)
        dropped_df.to_csv(out_dir / "dropped_rows_debug.csv", index=False)

        plot_histogram_before_after(
            original_df=df_binned,
            balanced_df=balanced_df,
            target_col=args.target_column,
            edges=edges,
            out_path=out_dir / "histogram_fmc_before_after.svg",
        )
        plot_bin_counts(report_df, out_dir / "fmc_bin_counts_before_after.svg")
        save_stage_bin_tables(df_binned, balanced_df, stage_col, out_dir)

        if args.debug_spectra:
            plot_spectra_debug(
                original_df=df_binned,
                balanced_df=balanced_df,
                spectral_col=args.spectral_column,
                out_dir=out_dir,
                preview_n=args.debug_preview_n,
            )

        if args.debug_images:
            image_integrity_and_preview(
                balanced_df=balanced_df,
                image_cols=args.image_columns,
                img_dir=args.img_dir,
                out_dir=out_dir,
                preview_n=args.debug_preview_n,
            )

    before_counts = df_binned.groupby("__fmc_bin_label", observed=True).size()
    after_counts = balanced_df.groupby("__fmc_bin_label", observed=True).size()

    manifest = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "debug_output_dir": str(out_dir) if (args.debug or args.debug_spectra or args.debug_images) else None,
        "target_column": args.target_column,
        "n_rows_input": int(len(df_original)),
        "n_rows_after_filtering": int(len(df_binned)),
        "n_rows_balanced": int(len(balanced_df)),
        "n_rows_dropped": int(len(dropped_df)),
        "original_columns_preserved": original_columns,
        "species_column": species_col,
        "stage_column": stage_col,
        "group_column_debug_only": group_col,
        "bin_edges": [float(x) for x in edges.tolist()],
        "target_per_bin": args.target_per_bin,
        "min_bin_count": int(args.min_bin_count),
        "stratify_columns": args.stratify_columns,
        "before_counts_by_bin": {str(k): int(v) for k, v in before_counts.items()},
        "after_counts_by_bin": {str(k): int(v) for k, v in after_counts.items()},
        "safety_note": (
            "The balanced CSV is made by row selection only. Image filenames, "
            "spectral signatures, species/stage labels, and target values are "
            "not recombined or modified."
        ),
    }

    if args.debug or args.debug_spectra or args.debug_images:
        (out_dir / "balance_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("=" * 80)
    print("FMC-balanced CSV created")
    print("=" * 80)
    print(f"Input rows:              {len(df_original)}")
    print(f"Rows after filtering:    {len(df_binned)}")
    print(f"Balanced rows:           {len(balanced_df)}")
    print(f"Dropped rows:            {len(dropped_df)}")
    print(f"Output CSV:              {output_csv}")
    print("")
    print("Counts before balancing:")
    print(before_counts.to_string())
    print("")
    print("Counts after balancing:")
    print(after_counts.to_string())
    if args.debug or args.debug_spectra or args.debug_images:
        print("")
        print(f"Debug directory:         {out_dir}")
        print(f"Histogram SVG:           {out_dir / 'histogram_fmc_before_after.svg'}")
        print(f"Bin count SVG:           {out_dir / 'fmc_bin_counts_before_after.svg'}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
