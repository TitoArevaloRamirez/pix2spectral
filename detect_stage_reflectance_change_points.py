#!/usr/bin/env python3
"""
Detect reflectance change points from mean spectral signatures by dehydration stage.

Reads a pix2spectral-style CSV, computes the mean spectral signature for each
dehydration stage, and detects:
  - local maxima
  - local minima
  - slope/curvature change points

Outputs:
  - terminal list
  - stage_mean_spectra.csv
  - stage_reflectance_change_points.csv
  - stage_reflectance_change_points.json
  - stage_reflectance_change_points.svg
  - stage_mean_spectra_only.svg

python detect_stage_reflectance_change_points.py \
    --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/vineyard_train_val.csv \
    --output-dir ~/Code/pix2spectral/pix2spectral_stage_change_points/vineyard_train_common \
    --stage-column auto \
    --spectral-column spectral \
    --spectral-drop-first-n 50 \
    --wavelength-min 400 \
    --wavelength-max 2500 \
    --smoothing-window 31 \
    --min-distance 50 \
    --prominence-frac 0.03 \
    --curvature-prominence-frac 0.15 \
    --merge-window-nm 25 \
    --common-min-stage-count 1 \
    --debug


The script treats each CSV row as an intact sample. It does not recombine
spectra, image names, stages, or target values.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks as scipy_find_peaks
    from scipy.signal import savgol_filter as scipy_savgol_filter

    SCIPY_AVAILABLE = True
except Exception:
    scipy_find_peaks = None
    scipy_savgol_filter = None
    SCIPY_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect mean-spectrum local maxima, minima, and slope changes by dehydration stage."
    )
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage-column", default="auto")
    p.add_argument("--spectral-column", default="spectral")
    p.add_argument("--species-column", default="auto")
    p.add_argument("--species", default=None)
    p.add_argument("--stages", nargs="*", default=None)
    p.add_argument("--spectral-drop-first-n", type=int, default=50)
    p.add_argument(
        "--within-row-reduction", choices=["mean", "median", "first"], default="mean"
    )
    p.add_argument("--wavelength-min", type=float, default=400.0)
    p.add_argument("--wavelength-max", type=float, default=2500.0)
    p.add_argument("--wavelength-count", type=int, default=None)
    p.add_argument("--smoothing-window", type=int, default=31)
    p.add_argument("--smoothing-polyorder", type=int, default=3)
    p.add_argument("--min-distance", type=int, default=20)
    p.add_argument("--prominence-frac", type=float, default=0.02)
    p.add_argument("--curvature-prominence-frac", type=float, default=0.05)
    p.add_argument("--max-points-per-type", type=int, default=None)
    p.add_argument("--include-inflection-zero-crossings", action="store_true")
    p.add_argument("--stage-order", nargs="*", default=None)
    p.add_argument(
        "--merge-window-nm",
        type=float,
        default=25.0,
        help="Merge stage-specific change points closer than this distance into one common point. Default: 25 nm.",
    )
    p.add_argument(
        "--common-max-points",
        type=int,
        default=None,
        help="Optional maximum number of common change points to keep. Keeps strongest clusters.",
    )
    p.add_argument(
        "--common-min-stage-count",
        type=int,
        default=1,
        help="Minimum number of different stages represented in a common cluster. Use 2 for shared-by-at-least-two-stages.",
    )
    p.add_argument(
        "--common-types",
        nargs="*",
        default=None,
        help="Optional change types used for common list, e.g. --common-types slope_curvature_change.",
    )
    p.add_argument("--terminal-max-points", type=int, default=500)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def choose_column(
    df: pd.DataFrame, requested: str, candidates: Sequence[str], name: str
) -> Optional[str]:
    req = str(requested).strip()
    if req.lower() in ["none", "null", ""]:
        return None
    if req.lower() == "auto":
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"Could not find {name} column. Tried {list(candidates)}.")
    if req not in df.columns:
        raise ValueError(
            f"Requested {name} column '{req}' not found. Available: {list(df.columns)}"
        )
    return req


def choose_optional_column(
    df: pd.DataFrame, requested: str, candidates: Sequence[str]
) -> Optional[str]:
    req = str(requested).strip()
    if req.lower() in ["none", "null", ""]:
        return None
    if req.lower() == "auto":
        for c in candidates:
            if c in df.columns:
                return c
        return None
    if req not in df.columns:
        raise ValueError(
            f"Requested column '{req}' not found. Available: {list(df.columns)}"
        )
    return req


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def stage_label(x: Any) -> str:
    if pd.isna(x):
        return "missing"
    return str(x).strip()


def parse_spectrum_value(value: Any, reduction: str = "mean") -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        if pd.isna(value):
            raise ValueError("Missing spectral value.")
        text = str(value).strip()
        if not text:
            raise ValueError("Empty spectral string.")
        parsed = None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            pass
        if parsed is None:
            try:
                parsed = json.loads(text)
            except Exception:
                pass
        if parsed is not None:
            arr = np.asarray(parsed)
        else:
            cleaned = text.strip().strip("[]()").replace(";", ",")
            arr = np.fromstring(
                cleaned, sep="," if "," in cleaned else " ", dtype=np.float32
            )
            if arr.size == 0:
                raise ValueError(f"Could not parse spectral value: {text[:120]}...")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        if reduction == "mean":
            out = np.nanmean(arr, axis=0)
        elif reduction == "median":
            out = np.nanmedian(arr, axis=0)
        else:
            out = arr[0]
    else:
        flat = arr.reshape(-1, arr.shape[-1])
        if reduction == "mean":
            out = np.nanmean(flat, axis=0)
        elif reduction == "median":
            out = np.nanmedian(flat, axis=0)
        else:
            out = flat[0]

    out = np.asarray(out, dtype=np.float32).reshape(-1)
    if out.size == 0:
        raise ValueError("Empty spectrum.")
    if not np.isfinite(out).all():
        idx = np.arange(out.size)
        finite = np.isfinite(out)
        if finite.sum() < 2:
            raise ValueError("Spectrum has fewer than 2 finite values.")
        out = np.interp(idx, idx[finite], out[finite]).astype(np.float32)
    return out


def parse_all_spectra(
    df: pd.DataFrame, spectral_col: str, drop_first_n: int, reduction: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    rows = []
    spectra = []
    for i, v in enumerate(df[spectral_col].values):
        try:
            full = parse_spectrum_value(v, reduction=reduction)
            if full.size <= drop_first_n:
                raise ValueError(
                    f"Spectrum length {full.size} <= drop_first_n={drop_first_n}."
                )
            retained = full[int(drop_first_n) :].astype(np.float32)
            spectra.append(retained)
            rows.append(
                {
                    "row": i,
                    "parse_ok": True,
                    "original_length": int(full.size),
                    "retained_length": int(retained.size),
                    "min": float(np.min(retained)),
                    "max": float(np.max(retained)),
                    "mean": float(np.mean(retained)),
                }
            )
        except Exception as exc:
            rows.append({"row": i, "parse_ok": False, "error": str(exc)})

    report = pd.DataFrame(rows)
    bad = report[report["parse_ok"] == False]
    if len(bad) > 0:
        raise ValueError(
            f"Failed to parse {len(bad)} spectra. First error: {bad.iloc[0].to_dict()}"
        )

    lengths = [s.size for s in spectra]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Inconsistent retained spectral lengths: {pd.Series(lengths).value_counts().to_dict()}"
        )

    return np.stack(spectra, axis=0).astype(np.float32), report


def odd_window(requested: int, length: int, polyorder: int) -> int:
    if requested <= 1:
        return 1
    w = min(int(requested), int(length) if int(length) % 2 == 1 else int(length) - 1)
    if w % 2 == 0:
        w -= 1
    min_w = int(polyorder) + 2
    if min_w % 2 == 0:
        min_w += 1
    if w < min_w:
        return 1
    return max(1, w)


def smooth_signal(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    w = odd_window(window, len(y), polyorder)
    if w <= 1:
        return y.copy()
    if SCIPY_AVAILABLE and scipy_savgol_filter is not None:
        return scipy_savgol_filter(
            y, window_length=w, polyorder=int(polyorder), mode="interp"
        )
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="reflect")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(ypad, kernel, mode="valid")


def fallback_find_peaks(
    y: np.ndarray, distance: int, prominence: float
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    distance = max(1, int(distance))
    candidates, scores = [], []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            left = np.min(y[max(0, i - distance) : i + 1])
            right = np.min(y[i : min(len(y), i + distance + 1)])
            prom = y[i] - max(left, right)
            if prom >= prominence:
                candidates.append(i)
                scores.append(prom)
    if not candidates:
        return np.array([], dtype=int), np.array([], dtype=float)
    order = np.argsort(scores)[::-1]
    kept = []
    for j in order:
        p = candidates[j]
        if all(abs(p - q) >= distance for q in kept):
            kept.append(p)
    kept = sorted(kept)
    score_map = {p: scores[candidates.index(p)] for p in kept}
    return np.asarray(kept, dtype=int), np.asarray(
        [score_map[p] for p in kept], dtype=float
    )


def detect_peaks(
    y: np.ndarray, distance: int, prominence: float
) -> Tuple[np.ndarray, np.ndarray]:
    if SCIPY_AVAILABLE and scipy_find_peaks is not None:
        idx, props = scipy_find_peaks(
            np.asarray(y, dtype=np.float64),
            distance=max(1, int(distance)),
            prominence=max(0.0, float(prominence)),
        )
        return idx.astype(int), np.asarray(
            props.get("prominences", np.zeros_like(idx)), dtype=float
        )
    return fallback_find_peaks(y, distance=distance, prominence=prominence)


def cap_points(
    idx: np.ndarray, scores: np.ndarray, max_points: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points is None or len(idx) <= int(max_points):
        order = np.argsort(idx)
        return idx[order], scores[order]
    if int(max_points) < 1:
        return np.array([], dtype=int), np.array([], dtype=float)
    strongest = np.argsort(scores)[::-1][: int(max_points)]
    idx2, scores2 = idx[strongest], scores[strongest]
    order = np.argsort(idx2)
    return idx2[order], scores2[order]


def detect_stage_points(
    stage: str,
    y_mean: np.ndarray,
    wavelengths: np.ndarray,
    smoothing_window: int,
    smoothing_polyorder: int,
    min_distance: int,
    prominence_frac: float,
    curvature_prominence_frac: float,
    max_points_per_type: Optional[int],
    include_inflections: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(y_mean, dtype=np.float64)
    wl = np.asarray(wavelengths, dtype=np.float64)
    ys = smooth_signal(y, smoothing_window, smoothing_polyorder)
    d1 = np.gradient(ys, wl)
    d2 = np.gradient(d1, wl)

    y_range = float(np.max(ys) - np.min(ys))
    prominence = max(float(prominence_frac) * y_range, 1e-12)

    max_idx, max_scores = detect_peaks(ys, min_distance, prominence)
    min_idx, min_scores = detect_peaks(-ys, min_distance, prominence)
    max_idx, max_scores = cap_points(max_idx, max_scores, max_points_per_type)
    min_idx, min_scores = cap_points(min_idx, min_scores, max_points_per_type)

    curvature = np.abs(d2)
    c_range = float(np.max(curvature) - np.min(curvature))
    c_prom = max(float(curvature_prominence_frac) * c_range, 1e-18)
    slope_idx, slope_scores = detect_peaks(curvature, min_distance, c_prom)
    slope_idx, slope_scores = cap_points(slope_idx, slope_scores, max_points_per_type)

    rows = []

    def add(indices: np.ndarray, scores: np.ndarray, typ: str) -> None:
        for i, score in zip(indices.tolist(), scores.tolist()):
            rows.append(
                {
                    "stage": stage,
                    "change_type": typ,
                    "spectral_index_retained": int(i),
                    "wavelength_nm": float(wl[i]),
                    "mean_reflectance": float(y[i]),
                    "smoothed_reflectance": float(ys[i]),
                    "first_derivative": float(d1[i]),
                    "second_derivative": float(d2[i]),
                    "score": float(score),
                }
            )

    add(max_idx, max_scores, "local_maximum")
    add(min_idx, min_scores, "local_minimum")
    add(slope_idx, slope_scores, "slope_curvature_change")

    if include_inflections:
        signs = np.sign(d2)
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        z = np.where(signs[:-1] * signs[1:] < 0)[0] + 1
        z_scores = curvature[z]
        z, z_scores = cap_points(
            z.astype(int), z_scores.astype(float), max_points_per_type
        )
        add(z, z_scores, "inflection_zero_crossing")

    cp = pd.DataFrame(rows)
    if not cp.empty:
        cp = cp.sort_values(["stage", "wavelength_nm", "change_type"]).reset_index(
            drop=True
        )

    debug = pd.DataFrame(
        {
            "stage": stage,
            "spectral_index_retained": np.arange(len(wl), dtype=int),
            "wavelength_nm": wl,
            "mean_reflectance": y,
            "smoothed_reflectance": ys,
            "first_derivative": d1,
            "second_derivative": d2,
            "abs_second_derivative": curvature,
        }
    )
    return cp, debug


def order_stages(
    stages: Sequence[str], requested_order: Optional[Sequence[str]]
) -> List[str]:
    stages = list(stages)
    if requested_order:
        ordered = [s for s in requested_order if s in stages]
        return ordered + sorted([s for s in stages if s not in ordered])
    default = ["fresh", "stage1", "stage2", "stage3", "dry"]
    lower = {str(s).lower(): s for s in stages}
    ordered = [lower[s] for s in default if s in lower]
    return ordered + sorted([s for s in stages if s not in ordered])


def add_normalized_change_scores(change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize scores within each change_type so local maxima, local minima,
    and curvature points can be compared more safely during common-point merging.
    """
    if change_df.empty:
        return change_df.copy()

    out = change_df.copy()
    out["score_abs"] = out["score"].abs().astype(float)
    out["score_norm"] = 0.0

    for change_type, idx in out.groupby("change_type").groups.items():
        vals = out.loc[idx, "score_abs"].to_numpy(dtype=float)
        vmax = float(np.nanmax(vals)) if vals.size else 0.0
        if np.isfinite(vmax) and vmax > 0:
            out.loc[idx, "score_norm"] = vals / vmax
        else:
            out.loc[idx, "score_norm"] = 0.0

    return out


def build_common_change_points(
    change_df: pd.DataFrame,
    merge_window_nm: float = 25.0,
    common_max_points: Optional[int] = None,
    common_min_stage_count: int = 1,
    common_types: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge nearby stage-specific change points into a common wavelength list.

    If multiple stages have close points, the representative wavelength is the
    wavelength of the most prominent normalized point in that cluster.
    """
    if change_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = add_normalized_change_scores(change_df)

    if common_types:
        wanted = {str(x) for x in common_types}
        df = df[df["change_type"].isin(wanted)].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df.sort_values("wavelength_nm").reset_index(drop=True)
    merge_window_nm = float(merge_window_nm)

    clusters: List[List[int]] = []
    current: List[int] = []

    for row_idx, row in df.iterrows():
        wl = float(row["wavelength_nm"])
        if not current:
            current = [int(row_idx)]
            continue

        prev_wl = float(df.loc[current[-1], "wavelength_nm"])
        if abs(wl - prev_wl) <= merge_window_nm:
            current.append(int(row_idx))
        else:
            clusters.append(current)
            current = [int(row_idx)]

    if current:
        clusters.append(current)

    common_rows: List[Dict[str, Any]] = []
    member_rows: List[Dict[str, Any]] = []

    for cluster_id, idxs in enumerate(clusters):
        sub = df.loc[idxs].copy()
        stage_count = int(sub["stage"].nunique())

        if stage_count < int(common_min_stage_count):
            continue

        sub_sorted = sub.sort_values(
            ["score_norm", "score_abs"],
            ascending=[False, False],
        )
        rep = sub_sorted.iloc[0]

        wl_values = sub["wavelength_nm"].to_numpy(dtype=float)
        weights = sub["score_norm"].to_numpy(dtype=float)

        if np.sum(weights) > 0:
            weighted_wl = float(np.average(wl_values, weights=weights))
        else:
            weighted_wl = float(np.mean(wl_values))

        common_rows.append(
            {
                "common_id": int(cluster_id),
                "representative_wavelength_nm": float(rep["wavelength_nm"]),
                "weighted_mean_wavelength_nm": weighted_wl,
                "wavelength_min_nm": float(np.min(wl_values)),
                "wavelength_max_nm": float(np.max(wl_values)),
                "wavelength_span_nm": float(np.max(wl_values) - np.min(wl_values)),
                "representative_stage": str(rep["stage"]),
                "representative_change_type": str(rep["change_type"]),
                "representative_score": float(rep["score"]),
                "representative_score_norm": float(rep["score_norm"]),
                "cluster_score_sum_norm": float(np.sum(weights)),
                "cluster_score_max_norm": float(np.max(weights))
                if len(weights)
                else 0.0,
                "n_points_in_cluster": int(len(sub)),
                "n_stages_in_cluster": int(stage_count),
                "stages_present": ",".join(sorted(map(str, sub["stage"].unique()))),
                "change_types_present": ",".join(
                    sorted(map(str, sub["change_type"].unique()))
                ),
            }
        )

        for _, m in sub.iterrows():
            item = m.to_dict()
            item["common_id"] = int(cluster_id)
            item["common_representative_wavelength_nm"] = float(rep["wavelength_nm"])
            member_rows.append(item)

    common_df = pd.DataFrame(common_rows)
    members_df = pd.DataFrame(member_rows)

    if common_df.empty:
        return common_df, members_df

    if common_max_points is not None:
        common_max_points = int(common_max_points)
        if common_max_points > 0 and len(common_df) > common_max_points:
            keep_ids = (
                common_df.sort_values(
                    [
                        "cluster_score_sum_norm",
                        "cluster_score_max_norm",
                        "n_stages_in_cluster",
                    ],
                    ascending=[False, False, False],
                )
                .head(common_max_points)["common_id"]
                .tolist()
            )
            common_df = common_df[common_df["common_id"].isin(keep_ids)].copy()
            members_df = members_df[members_df["common_id"].isin(keep_ids)].copy()

    common_df = common_df.sort_values("representative_wavelength_nm").reset_index(
        drop=True
    )
    members_df = members_df.sort_values(["common_id", "wavelength_nm"]).reset_index(
        drop=True
    )

    return common_df, members_df


def plot_common_change_points_all_stages(
    mean_df: pd.DataFrame,
    common_df: pd.DataFrame,
    stage_order: Sequence[str],
    out_path: Path,
) -> None:
    """
    Plot all stage mean spectra together and mark common representative wavelengths.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    for stage in stage_order:
        sdf = mean_df[mean_df["stage"] == stage]
        if sdf.empty:
            continue
        ax.plot(
            sdf["wavelength_nm"],
            sdf["mean_reflectance"],
            linewidth=1.5,
            label=str(stage),
        )

    if not common_df.empty:
        for _, row in common_df.iterrows():
            wl = float(row["representative_wavelength_nm"])
            ax.axvline(wl, linestyle="--", linewidth=0.9, alpha=0.55)

        ymax = ax.get_ylim()[1]
        for _, row in common_df.iterrows():
            wl = float(row["representative_wavelength_nm"])
            ax.text(
                wl,
                ymax,
                f"{wl:.0f}",
                rotation=90,
                va="top",
                ha="center",
                fontsize=7,
                alpha=0.75,
            )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean reflectance")
    ax.set_title("Common prominent reflectance change points across dehydration stages")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_common_change_points_stage_panels(
    mean_df: pd.DataFrame,
    common_df: pd.DataFrame,
    stage_order: Sequence[str],
    out_path: Path,
) -> None:
    """
    Plot one panel per stage and mark the same common wavelengths in all panels.
    """
    n = len(stage_order)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(13, max(4, 3.0 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    common_wls = []
    if not common_df.empty:
        common_wls = (
            common_df["representative_wavelength_nm"].to_numpy(dtype=float).tolist()
        )

    for ax, stage in zip(axes, stage_order):
        sdf = mean_df[mean_df["stage"] == stage]
        if sdf.empty:
            continue

        ax.plot(
            sdf["wavelength_nm"],
            sdf["mean_reflectance"],
            linewidth=1.5,
            label=f"{stage} mean",
        )

        if "smoothed_reflectance" in sdf.columns:
            ax.plot(
                sdf["wavelength_nm"],
                sdf["smoothed_reflectance"],
                linewidth=1.1,
                linestyle="--",
                label="smoothed",
            )

        for wl in common_wls:
            ax.axvline(wl, linestyle="--", linewidth=0.8, alpha=0.45)

        ax.set_ylabel("Reflectance")
        ax.set_title(f"Stage: {stage}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Common prominent change points plotted for all dehydration stages", y=0.995
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def print_common_points(common_df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("Common prominent change-point wavelength list")
    print("=" * 88)

    if common_df.empty:
        print("No common change points were selected.")
        print("=" * 88)
        return

    for _, row in common_df.iterrows():
        print(
            f"wl={row['representative_wavelength_nm']:8.2f} nm | "
            f"weighted={row['weighted_mean_wavelength_nm']:8.2f} nm | "
            f"type={row['representative_change_type']} | "
            f"stage={row['representative_stage']} | "
            f"n_points={int(row['n_points_in_cluster'])} | "
            f"n_stages={int(row['n_stages_in_cluster'])} | "
            f"score_sum={row['cluster_score_sum_norm']:.3f} | "
            f"stages={row['stages_present']}"
        )

    print("=" * 88)


def plot_change_points(
    mean_df: pd.DataFrame,
    change_df: pd.DataFrame,
    stage_order: Sequence[str],
    out_path: Path,
) -> None:
    n = len(stage_order)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(4, 3.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    markers = {
        "local_maximum": "^",
        "local_minimum": "v",
        "slope_curvature_change": "o",
        "inflection_zero_crossing": "x",
    }

    for ax, stage in zip(axes, stage_order):
        sdf = mean_df[mean_df["stage"] == stage]
        ax.plot(
            sdf["wavelength_nm"], sdf["mean_reflectance"], linewidth=1.5, label="mean"
        )
        ax.plot(
            sdf["wavelength_nm"],
            sdf["smoothed_reflectance"],
            linewidth=1.2,
            linestyle="--",
            label="smoothed",
        )

        cdf = (
            change_df[change_df["stage"] == stage]
            if not change_df.empty
            else pd.DataFrame()
        )
        for typ, marker in markers.items():
            sub = cdf[cdf["change_type"] == typ] if not cdf.empty else pd.DataFrame()
            if len(sub) > 0:
                ax.scatter(
                    sub["wavelength_nm"],
                    sub["smoothed_reflectance"],
                    marker=marker,
                    s=40,
                    label=typ,
                    zorder=5,
                )

        ax.set_ylabel("Reflectance")
        ax.set_title(f"Stage: {stage}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Stage mean spectral signatures and detected reflectance change points", y=0.995
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_mean_only(
    mean_df: pd.DataFrame, stage_order: Sequence[str], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for stage in stage_order:
        sdf = mean_df[mean_df["stage"] == stage]
        ax.plot(
            sdf["wavelength_nm"],
            sdf["mean_reflectance"],
            linewidth=1.5,
            label=str(stage),
        )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean reflectance")
    ax.set_title("Mean spectral signature by dehydration stage")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def print_points(change_df: pd.DataFrame, terminal_max_points: int) -> None:
    if change_df.empty:
        print("No change points detected.")
        return

    print("=" * 88)
    print("Detected reflectance change points by dehydration stage")
    print("=" * 88)
    printed = 0

    for stage, sdf in change_df.groupby("stage", sort=False):
        print(f"\nStage: {stage}")
        print("-" * 88)
        for typ, tdf in sdf.groupby("change_type", sort=False):
            print(f"  {typ}: {len(tdf)} point(s)")
            for _, row in tdf.sort_values("wavelength_nm").iterrows():
                if printed >= int(terminal_max_points):
                    print(f"\n[terminal output truncated. Full list saved to CSV.]")
                    print("=" * 88)
                    return
                print(
                    "    "
                    f"wl={row['wavelength_nm']:8.2f} nm | "
                    f"idx={int(row['spectral_index_retained']):5d} | "
                    f"mean={row['mean_reflectance']:.6f} | "
                    f"d1={row['first_derivative']:.6e} | "
                    f"d2={row['second_derivative']:.6e} | "
                    f"score={row['score']:.6e}"
                )
                printed += 1
    print("=" * 88)


def main() -> int:
    args = parse_args()

    input_csv = expand_path(args.input_csv)
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    stage_col = choose_column(
        df, args.stage_column, ["Stages", "stage", "Stage", "STAGE"], "stage"
    )
    species_col = choose_optional_column(
        df, args.species_column, ["Species", "species", "SPECIES"]
    )

    if args.spectral_column not in df.columns:
        raise ValueError(
            f"Spectral column '{args.spectral_column}' not found. Available: {list(df.columns)}"
        )

    if args.species is not None:
        if species_col is None:
            raise ValueError("--species was provided, but no species column was found.")
        before = len(df)
        df = df[df[species_col].map(norm_text) == norm_text(args.species)].copy()
        print(f"Species filter kept {len(df)}/{before} rows.")

    if args.stages:
        wanted = {norm_text(s) for s in args.stages}
        before = len(df)
        df = df[df[stage_col].map(norm_text).isin(wanted)].copy()
        print(f"Stage filter kept {len(df)}/{before} rows.")

    if df.empty:
        raise ValueError("No rows left after filtering.")

    spectra, parse_report = parse_all_spectra(
        df,
        spectral_col=args.spectral_column,
        drop_first_n=args.spectral_drop_first_n,
        reduction=args.within_row_reduction,
    )
    parse_report.to_csv(output_dir / "spectral_parse_report.csv", index=False)

    L = spectra.shape[1]
    if args.wavelength_count is not None and int(args.wavelength_count) != L:
        raise ValueError(
            f"--wavelength-count={args.wavelength_count} does not match retained length {L}."
        )

    wavelengths = np.linspace(
        float(args.wavelength_min), float(args.wavelength_max), L, dtype=np.float64
    )
    labels = df[stage_col].map(stage_label).to_numpy()
    unique = list(pd.unique(labels))
    stage_order = order_stages(unique, args.stage_order)

    mean_rows = []
    cp_list = []
    debug_list = []

    for stage in stage_order:
        mask = labels == stage
        stage_spectra = spectra[mask]
        if stage_spectra.shape[0] == 0:
            continue

        y_mean = np.mean(stage_spectra, axis=0)
        y_std = np.std(stage_spectra, axis=0)

        cp, dbg = detect_stage_points(
            stage=stage,
            y_mean=y_mean,
            wavelengths=wavelengths,
            smoothing_window=args.smoothing_window,
            smoothing_polyorder=args.smoothing_polyorder,
            min_distance=args.min_distance,
            prominence_frac=args.prominence_frac,
            curvature_prominence_frac=args.curvature_prominence_frac,
            max_points_per_type=args.max_points_per_type,
            include_inflections=args.include_inflection_zero_crossings,
        )
        cp_list.append(cp)
        debug_list.append(dbg)

        for i in range(L):
            mean_rows.append(
                {
                    "stage": stage,
                    "n_samples": int(stage_spectra.shape[0]),
                    "spectral_index_retained": int(i),
                    "wavelength_nm": float(wavelengths[i]),
                    "mean_reflectance": float(y_mean[i]),
                    "std_reflectance": float(y_std[i]),
                    "smoothed_reflectance": float(dbg["smoothed_reflectance"].iloc[i]),
                    "first_derivative": float(dbg["first_derivative"].iloc[i]),
                    "second_derivative": float(dbg["second_derivative"].iloc[i]),
                }
            )

    mean_df = pd.DataFrame(mean_rows)
    change_df = pd.concat(cp_list, ignore_index=True) if cp_list else pd.DataFrame()
    debug_df = (
        pd.concat(debug_list, ignore_index=True) if debug_list else pd.DataFrame()
    )

    mean_csv = output_dir / "stage_mean_spectra.csv"
    change_csv = output_dir / "stage_reflectance_change_points.csv"
    change_json = output_dir / "stage_reflectance_change_points.json"
    plot_svg = output_dir / "stage_reflectance_change_points.svg"
    mean_svg = output_dir / "stage_mean_spectra_only.svg"

    mean_df.to_csv(mean_csv, index=False)
    change_df.to_csv(change_csv, index=False)
    if args.debug:
        debug_df.to_csv(
            output_dir / "stage_mean_spectra_derivatives_debug.csv", index=False
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    if not change_df.empty:
        for stage, sdf in change_df.groupby("stage", sort=False):
            grouped[str(stage)] = []
            for _, r in sdf.iterrows():
                grouped[str(stage)].append(
                    {
                        "change_type": str(r["change_type"]),
                        "spectral_index_retained": int(r["spectral_index_retained"]),
                        "wavelength_nm": float(r["wavelength_nm"]),
                        "mean_reflectance": float(r["mean_reflectance"]),
                        "smoothed_reflectance": float(r["smoothed_reflectance"]),
                        "first_derivative": float(r["first_derivative"]),
                        "second_derivative": float(r["second_derivative"]),
                        "score": float(r["score"]),
                    }
                )
    change_json.write_text(json.dumps(grouped, indent=2))

    plot_change_points(mean_df, change_df, stage_order, plot_svg)
    plot_mean_only(mean_df, stage_order, mean_svg)

    common_df, common_members_df = build_common_change_points(
        change_df=change_df,
        merge_window_nm=args.merge_window_nm,
        common_max_points=args.common_max_points,
        common_min_stage_count=args.common_min_stage_count,
        common_types=args.common_types,
    )

    common_csv = output_dir / "common_reflectance_change_points.csv"
    common_members_csv = output_dir / "common_reflectance_change_point_members.csv"
    common_json = output_dir / "common_reflectance_change_points.json"
    common_plot_svg = output_dir / "common_reflectance_change_points_all_stages.svg"
    common_panels_svg = output_dir / "common_reflectance_change_points_stage_panels.svg"

    common_df.to_csv(common_csv, index=False)
    common_members_df.to_csv(common_members_csv, index=False)

    common_json_list = []
    if not common_df.empty:
        for _, r in common_df.iterrows():
            common_json_list.append(
                {
                    "common_id": int(r["common_id"]),
                    "representative_wavelength_nm": float(
                        r["representative_wavelength_nm"]
                    ),
                    "weighted_mean_wavelength_nm": float(
                        r["weighted_mean_wavelength_nm"]
                    ),
                    "representative_change_type": str(r["representative_change_type"]),
                    "representative_stage": str(r["representative_stage"]),
                    "n_points_in_cluster": int(r["n_points_in_cluster"]),
                    "n_stages_in_cluster": int(r["n_stages_in_cluster"]),
                    "stages_present": str(r["stages_present"]),
                    "change_types_present": str(r["change_types_present"]),
                    "cluster_score_sum_norm": float(r["cluster_score_sum_norm"]),
                }
            )
    common_json.write_text(json.dumps(common_json_list, indent=2))

    plot_common_change_points_all_stages(
        mean_df=mean_df,
        common_df=common_df,
        stage_order=stage_order,
        out_path=common_plot_svg,
    )
    plot_common_change_points_stage_panels(
        mean_df=mean_df,
        common_df=common_df,
        stage_order=stage_order,
        out_path=common_panels_svg,
    )

    manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "stage_column": stage_col,
        "species_column": species_col,
        "spectral_column": args.spectral_column,
        "n_rows_used": int(len(df)),
        "spectral_drop_first_n": int(args.spectral_drop_first_n),
        "retained_spectral_length": int(L),
        "wavelength_min": float(args.wavelength_min),
        "wavelength_max": float(args.wavelength_max),
        "smoothing_window": int(args.smoothing_window),
        "smoothing_polyorder": int(args.smoothing_polyorder),
        "min_distance": int(args.min_distance),
        "prominence_frac": float(args.prominence_frac),
        "curvature_prominence_frac": float(args.curvature_prominence_frac),
        "merge_window_nm": float(args.merge_window_nm),
        "common_max_points": None
        if args.common_max_points is None
        else int(args.common_max_points),
        "common_min_stage_count": int(args.common_min_stage_count),
        "common_types": args.common_types,
        "scipy_available": bool(SCIPY_AVAILABLE),
        "outputs": {
            "stage_mean_spectra_csv": str(mean_csv),
            "change_points_csv": str(change_csv),
            "change_points_json": str(change_json),
            "change_points_plot": str(plot_svg),
            "mean_spectra_plot": str(mean_svg),
            "common_change_points_csv": str(common_csv),
            "common_change_point_members_csv": str(common_members_csv),
            "common_change_points_json": str(common_json),
            "common_change_points_all_stages_plot": str(common_plot_svg),
            "common_change_points_stage_panels_plot": str(common_panels_svg),
        },
    }
    (output_dir / "change_point_detection_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print_points(change_df, args.terminal_max_points)
    print_common_points(common_df)

    print("\nOutput files:")
    print(f"  Mean spectra CSV:                 {mean_csv}")
    print(f"  Stage change points CSV:          {change_csv}")
    print(f"  Stage change points JSON:         {change_json}")
    print(f"  Stage change points plot:         {plot_svg}")
    print(f"  Mean spectra plot:                {mean_svg}")
    print(f"  Common change points CSV:         {common_csv}")
    print(f"  Common change point members CSV:  {common_members_csv}")
    print(f"  Common change points JSON:        {common_json}")
    print(f"  Common all-stages plot:           {common_plot_svg}")
    print(f"  Common stage-panel plot:          {common_panels_svg}")
    if args.debug:
        print(
            f"  Derivative debug CSV:             {output_dir / 'stage_mean_spectra_derivatives_debug.csv'}"
        )
    print(
        f"  Manifest:                         {output_dir / 'change_point_detection_manifest.json'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
