#!/usr/bin/env python3
"""
Train LWC/FMC regressors using fused features:

    generated spectra + PROSPECT parameters + multispectral channel statistics

For each multispectral channel, the script computes:
    mean, median, standard deviation, 10th percentile, 90th percentile

The default protocol follows the previous scripts:
    - train + validation are merged as development data
    - 5-fold CV on development data
    - final fit on development data
    - independent test evaluation
    - optional grid search
    - feature and target normalization fitted on all leaf samples by default
    - metrics and plots reported in original target units
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR


MODEL_ORDER = [
    "Elastic Net",
    "Gradient Boosting",
    "Random Forest",
    "Ridge Regressor",
    "SVR RBF",
]

PROSPECT_PARAM_NAMES = ["N", "Cab", "Car", "Cbrown", "Cw", "Cm", "Ant"]

HYPERPARAMETERS = {
    "avocado": {
        "Elastic Net": {"alpha": 0.01, "l1_ratio": 0.1},
        "Gradient Boosting": {
            "learning_rate": 0.03,
            "max_depth": 3,
            "n_estimators": 400,
            "subsample": 0.8,
        },
        "Random Forest": {
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 300,
        },
        "Ridge Regressor": {"alpha": 10.0},
        "SVR RBF": {"C": 1000.0, "epsilon": 0.3, "gamma": "scale"},
    },
    "olive": {
        "Elastic Net": {"alpha": 0.01, "l1_ratio": 0.1},
        "Gradient Boosting": {
            "learning_rate": 0.03,
            "max_depth": 3,
            "n_estimators": 400,
            "subsample": 0.8,
        },
        "Random Forest": {
            "max_depth": 40,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 600,
        },
        "Ridge Regressor": {"alpha": 1.0},
        "SVR RBF": {"C": 100.0, "epsilon": 0.01, "gamma": "scale"},
    },
    "grape": {
        "Elastic Net": {"alpha": 0.01, "l1_ratio": 0.5},
        "Gradient Boosting": {
            "learning_rate": 0.03,
            "max_depth": 3,
            "n_estimators": 400,
            "subsample": 0.8,
        },
        "Random Forest": {
            "max_depth": 20,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 600,
        },
        "Ridge Regressor": {"alpha": 1.0},
        "SVR RBF": {"C": 1000.0, "epsilon": 0.3, "gamma": "scale"},
    },
}

BAND_ALIASES = {
    "blue": ["blue", "Blue", "BLUE", "blue_path", "B"],
    "green": ["green", "Green", "GREEN", "green_path", "G"],
    "red": ["red", "Red", "RED", "red_path", "R"],
    "red_edge": [
        "red_edge",
        "rededge",
        "redEdge",
        "RedEdge",
        "RED_EDGE",
        "red_edge_path",
        "RE",
        "re",
    ],
    "nir": ["nir", "NIR", "Nir", "nir_path", "near_ir", "nearinfrared"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train regressors using spectra + PROSPECT params + image statistics."
    )

    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--target-column", default="LWC_d")
    p.add_argument("--species", default="Avocado")
    p.add_argument("--species-column", default=None)
    p.add_argument("--filter-species", action="store_true")

    p.add_argument("--spectrum-column", default="generated_spectrum_json")
    p.add_argument("--fallback-spectrum-column", default="spectral")
    p.add_argument(
        "--wl-min", "--wavelength-min", dest="wavelength_min", type=float, default=None
    )
    p.add_argument(
        "--wl-max", "--wavelength-max", dest="wavelength_max", type=float, default=None
    )
    p.add_argument("--spectrum-wavelength-min", type=float, default=400.0)
    p.add_argument("--spectrum-wavelength-max", type=float, default=2500.0)

    p.add_argument("--params-column", default="params_json")
    p.add_argument(
        "--param-feature-mode",
        choices=["flatten", "mean", "mean_std", "flatten_mean_std"],
        default="flatten",
    )
    p.add_argument("--train-params-csv", default=None)
    p.add_argument("--val-params-csv", default=None)
    p.add_argument("--test-params-csv", default=None)
    p.add_argument(
        "--merge-keys", nargs="+", default=["species", "stage", "blue_basename"]
    )

    p.add_argument("--img-dir", default=None)
    p.add_argument(
        "--channel-columns",
        nargs="*",
        default=None,
        help="Explicit mapping band:column, e.g. blue:Blue green:Green red:Red red_edge:RedEdge nir:NIR",
    )
    p.add_argument(
        "--image-mask-mode",
        choices=["nonzero", "positive", "finite", "none"],
        default="nonzero",
    )

    p.add_argument("--include-spectra", action="store_true", default=True)
    p.add_argument("--no-spectra", action="store_false", dest="include_spectra")
    p.add_argument("--include-prospect", action="store_true", default=True)
    p.add_argument("--no-prospect", action="store_false", dest="include_prospect")
    p.add_argument("--include-image-stats", action="store_true", default=True)
    p.add_argument("--no-image-stats", action="store_false", dest="include_image_stats")

    p.add_argument("--group-column", default="auto")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument("--grid_search", action="store_true")
    p.add_argument("--grid-size", choices=["small", "medium", "wide"], default="medium")
    p.add_argument("--grid-refit-metric", choices=["rmse", "mae", "r2"], default="rmse")

    p.add_argument(
        "--x-normalization", choices=["standard", "minmax", "none"], default="standard"
    )
    p.add_argument(
        "--y-normalization", choices=["standard", "minmax", "none"], default="standard"
    )
    p.add_argument(
        "--normalization-scope",
        choices=["all_leaf_samples", "development_only"],
        default="all_leaf_samples",
    )

    p.add_argument("--grape-gb-n-estimators", type=int, default=400)
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--write-latex", action="store_true")
    p.add_argument("--drop-na-target", action="store_true")
    p.add_argument("--max-samples-debug", type=int, default=None)
    return p.parse_args()


def expand_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def normalize_species_key(species: str) -> str:
    s = str(species).strip().lower()
    aliases = {
        "avo": "avocado",
        "avocado": "avocado",
        "palta": "avocado",
        "olive": "olive",
        "olivo": "olive",
        "grape": "grape",
        "grapes": "grape",
        "vineyard": "grape",
        "vine": "grape",
        "vid": "grape",
    }
    if s not in aliases:
        raise ValueError(
            f"Unknown species '{species}'. Use Avocado/Avo, Olive, or Grape/Vineyard."
        )
    return aliases[s]


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def basename_any(path_like: Any) -> str:
    if pd.isna(path_like):
        return ""
    text = str(path_like).strip()
    if not text:
        return ""
    return PurePosixPath(PureWindowsPath(text).name).name


def infer_leaf_id_from_text(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = basename_any(value).lower()
    for pat in [r"leaf[_-]?(\d+)", r"leaf(\d{3})d\d", r"(\d{3})d\d"]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def find_species_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["species", "Species", "SPECIES"]:
        if c in df.columns:
            return c
    return None


def maybe_add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "blue_basename" not in df.columns:
        for c in ["blue", "Blue", "BLUE", "blue_path"]:
            if c in df.columns:
                df["blue_basename"] = df[c].map(basename_any)
                break
    if "stage" not in df.columns and "Stages" in df.columns:
        df["stage"] = df["Stages"]
    if "species" not in df.columns and "Species" in df.columns:
        df["species"] = df["Species"]
    if "leaf_id" not in df.columns:
        for c in ["blue_basename", "blue", "Blue", "filename", "image_name"]:
            if c in df.columns:
                inferred = df[c].map(infer_leaf_id_from_text)
                if inferred.notna().any():
                    df["leaf_id_inferred"] = inferred
                    break
    return df


def prepare_dataframe(
    csv_path: str, args: argparse.Namespace, split_name: str
) -> pd.DataFrame:
    path = expand_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{split_name} CSV not found: {path}")
    df = pd.read_csv(path).copy()
    df["__source_split"] = split_name
    df["__source_csv"] = str(path)
    if args.max_samples_debug is not None:
        df = df.head(args.max_samples_debug).copy()

    if args.filter_species:
        species_col = args.species_column or find_species_column(df)
        if species_col is None:
            raise ValueError(
                f"--filter-species requested but no species column found in {path}"
            )
        target_species = normalize_species_key(args.species)

        def key(x):
            try:
                return normalize_species_key(str(x))
            except Exception:
                return str(x).strip().lower()

        before = len(df)
        df = df[df[species_col].map(key) == target_species].copy()
        if len(df) == 0:
            raise ValueError(f"No rows remain in {split_name} after species filtering.")
        print(f"{split_name}: species filter kept {len(df)}/{before} rows.")

    if args.target_column not in df.columns:
        raise ValueError(
            f"{split_name} CSV lacks target column '{args.target_column}'. Columns: {list(df.columns)}"
        )
    df[args.target_column] = pd.to_numeric(df[args.target_column], errors="coerce")
    if df[args.target_column].isna().any():
        n_bad = int(df[args.target_column].isna().sum())
        if args.drop_na_target:
            print(f"{split_name}: dropping {n_bad} rows with missing target.")
            df = df.dropna(subset=[args.target_column]).copy()
        else:
            raise ValueError(
                f"{split_name}: {n_bad} missing/non-numeric target values. Use --drop-na-target."
            )
    df = maybe_add_keys(df).reset_index(drop=True)
    df["__row_id_within_split"] = np.arange(len(df), dtype=int)
    return df


def merge_params_if_needed(
    df: pd.DataFrame,
    params_csv: Optional[str],
    args: argparse.Namespace,
    split_name: str,
) -> pd.DataFrame:
    if not args.include_prospect or args.params_column in df.columns:
        return df
    if params_csv is None:
        raise ValueError(
            f"{split_name}: '{args.params_column}' missing and no separate params CSV provided."
        )
    p = expand_path(params_csv)
    if not p.exists():
        raise FileNotFoundError(f"{split_name} params CSV not found: {p}")
    params_df = maybe_add_keys(pd.read_csv(p).copy())
    df = maybe_add_keys(df)
    keys = [k for k in args.merge_keys if k in df.columns and k in params_df.columns]
    if not keys:
        raise ValueError(
            f"{split_name}: no usable merge keys. Requested {args.merge_keys}"
        )
    merged = df.merge(
        params_df[keys + [args.params_column]], on=keys, how="left", validate="m:1"
    )
    missing = int(merged[args.params_column].isna().sum())
    if missing:
        raise ValueError(
            f"{split_name}: {missing}/{len(merged)} rows missing params after merge on {keys}."
        )
    print(f"{split_name}: merged params using keys {keys} from {p}")
    return merged


# ----------------------------- spectra -----------------------------


def parse_vector_value(value: Any, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    if pd.isna(value):
        raise ValueError(f"Missing {name} value")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty {name} string")
    for parser in [json.loads, ast.literal_eval]:
        try:
            return np.asarray(parser(text), dtype=float).reshape(-1)
        except Exception:
            pass
    cleaned = text.strip().strip("[]()").replace(";", ",")
    arr = np.fromstring(cleaned, sep="," if "," in cleaned else " ", dtype=float)
    if arr.size == 0:
        raise ValueError(f"Could not parse {name}: {text[:120]}")
    return arr.reshape(-1)


def find_wavelength_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).startswith("wl_")]

    def key(c):
        try:
            return float(str(c)[3:])
        except Exception:
            return math.inf

    return sorted(cols, key=key)


def select_wavelength_range(X, wavelengths, names, wl_min, wl_max):
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wl_min is None and wl_max is None:
        return X, wavelengths, names
    if wl_min is not None and wl_max is not None and wl_min > wl_max:
        raise ValueError("Invalid wavelength range: wl-min > wl-max")
    mask = np.ones_like(wavelengths, dtype=bool)
    if wl_min is not None:
        mask &= wavelengths >= wl_min
    if wl_max is not None:
        mask &= wavelengths <= wl_max
    if not mask.any():
        raise ValueError("Selected wavelength range contains zero features.")
    return X[:, mask], wavelengths[mask], [n for n, keep in zip(names, mask) if keep]


def build_spectrum_features(df: pd.DataFrame, args: argparse.Namespace):
    if args.spectrum_column in df.columns:
        rows = [
            parse_vector_value(v, args.spectrum_column)
            for v in df[args.spectrum_column].values
        ]
        source = args.spectrum_column
        lengths = [len(r) for r in rows]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Inconsistent spectrum lengths: {pd.Series(lengths).value_counts().to_dict()}"
            )
        X = np.stack(rows).astype(np.float64)
        wavelengths = np.linspace(
            args.spectrum_wavelength_min, args.spectrum_wavelength_max, X.shape[1]
        )
        names = [f"spectrum_wl_{w:g}" for w in wavelengths]
    elif args.fallback_spectrum_column in df.columns:
        rows = [
            parse_vector_value(v, args.fallback_spectrum_column)
            for v in df[args.fallback_spectrum_column].values
        ]
        source = args.fallback_spectrum_column
        lengths = [len(r) for r in rows]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Inconsistent spectrum lengths: {pd.Series(lengths).value_counts().to_dict()}"
            )
        X = np.stack(rows).astype(np.float64)
        wavelengths = np.linspace(
            args.spectrum_wavelength_min, args.spectrum_wavelength_max, X.shape[1]
        )
        names = [f"spectrum_wl_{w:g}" for w in wavelengths]
    else:
        wl_cols = find_wavelength_columns(df)
        if not wl_cols:
            raise ValueError("No generated spectrum column or wl_* columns found.")
        source = "wl_columns"
        X = df[wl_cols].to_numpy(dtype=float)
        wavelengths = np.asarray([float(str(c)[3:]) for c in wl_cols])
        names = [f"spectrum_{c}" for c in wl_cols]
    X, wavelengths, names = select_wavelength_range(
        X, wavelengths, names, args.wavelength_min, args.wavelength_max
    )
    return (
        X,
        names,
        {
            "source": source,
            "n_features": X.shape[1],
            "wavelength_min": float(wavelengths.min()),
            "wavelength_max": float(wavelengths.max()),
        },
    )


# ----------------------------- PROSPECT params -----------------------------


def parse_params_value(value: Any) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        for parser in [json.loads, ast.literal_eval]:
            try:
                return np.asarray(parser(text), dtype=float)
            except Exception:
                pass
    return parse_vector_value(value, "params_json")


def params_to_features(params: np.ndarray, mode: str):
    params = np.asarray(params, dtype=float)
    shape = tuple(params.shape)
    if not np.isfinite(params).all():
        raise ValueError("PROSPECT parameters contain non-finite values.")
    if params.ndim == 1:
        if params.size == 7:
            mat = params.reshape(1, 7)
        elif params.size % 7 == 0:
            mat = params.reshape(params.size // 7, 7)
        else:
            if mode != "flatten":
                raise ValueError(
                    f"1D params length {params.size} is not compatible with mode {mode}"
                )
            vec = params.reshape(-1)
            return vec, [f"prospect_param_{i}" for i in range(vec.size)], shape
    elif params.ndim == 2 and params.shape[1] == 7:
        mat = params
    elif params.ndim > 2 and params.shape[-1] == 7:
        mat = params.reshape(-1, 7)
    else:
        if mode != "flatten":
            raise ValueError(f"Unsupported params shape {params.shape}")
        vec = params.reshape(-1)
        return vec, [f"prospect_param_{i}" for i in range(vec.size)], shape

    nseg = mat.shape[0]
    flat = mat.reshape(-1)
    flat_names = [
        f"prospect_seg{s + 1}_{p}" for s in range(nseg) for p in PROSPECT_PARAM_NAMES
    ]
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=0)
    mean_names = [f"prospect_mean_{p}" for p in PROSPECT_PARAM_NAMES]
    std_names = [f"prospect_std_{p}" for p in PROSPECT_PARAM_NAMES]
    if mode == "flatten":
        return flat, flat_names, shape
    if mode == "mean":
        return mean, mean_names, shape
    if mode == "mean_std":
        return np.concatenate([mean, std]), mean_names + std_names, shape
    if mode == "flatten_mean_std":
        return (
            np.concatenate([flat, mean, std]),
            flat_names + mean_names + std_names,
            shape,
        )
    raise ValueError(f"Unknown param feature mode {mode}")


def build_prospect_features(df: pd.DataFrame, args: argparse.Namespace):
    if args.params_column not in df.columns:
        raise ValueError(f"Column {args.params_column} not found.")
    rows, shapes, ref_names = [], [], None
    for i, val in enumerate(df[args.params_column].values):
        vec, names, shape = params_to_features(
            parse_params_value(val), args.param_feature_mode
        )
        if ref_names is None:
            ref_names = names
        elif len(names) != len(ref_names):
            raise ValueError(f"Inconsistent PROSPECT feature length at row {i}")
        rows.append(vec)
        shapes.append(str(shape))
    X = np.stack(rows).astype(np.float64)
    return (
        X,
        ref_names or [],
        {
            "n_features": X.shape[1],
            "mode": args.param_feature_mode,
            "shape_counts": pd.Series(shapes).value_counts().to_dict(),
        },
    )


# ----------------------------- image stats -----------------------------


def parse_channel_columns(
    df: pd.DataFrame, explicit: Optional[Sequence[str]]
) -> Dict[str, str]:
    if explicit:
        mapping = {}
        for item in explicit:
            if ":" not in item:
                raise ValueError(
                    f"Invalid channel mapping '{item}', expected band:column"
                )
            band, col = item.split(":", 1)
            if col not in df.columns:
                raise ValueError(f"Channel column {col} for band {band} not found.")
            mapping[band] = col
        return mapping
    norm_to_col = {normalize_colname(c): c for c in df.columns}
    mapping = {}
    for band, aliases in BAND_ALIASES.items():
        for alias in aliases:
            key = normalize_colname(alias)
            if key in norm_to_col:
                mapping[band] = norm_to_col[key]
                break
    return mapping


def resolve_image_path(value: Any, img_dir: Optional[str]) -> Path:
    if pd.isna(value):
        raise ValueError("Missing image path")
    text = str(value).strip()
    p = Path(text).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    if img_dir:
        base = expand_path(img_dir)
        candidate = base / text
        if candidate.exists():
            return candidate.resolve()
        candidate2 = base / basename_any(text)
        if candidate2.exists():
            return candidate2.resolve()
        return candidate
    return p


def read_image_array(path: Path) -> np.ndarray:
    try:
        import tifffile

        arr = tifffile.imread(str(path))
    except Exception:
        try:
            import imageio.v3 as iio

            arr = iio.imread(str(path))
        except Exception:
            from PIL import Image

            arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.astype(np.float64).mean(axis=-1)
    return arr.astype(np.float64)


def valid_pixels(arr: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(arr)
    if mode in ["none", "finite"]:
        mask = finite
    elif mode == "nonzero":
        mask = finite & (arr != 0)
    elif mode == "positive":
        mask = finite & (arr > 0)
    else:
        raise ValueError(f"Unknown image mask mode {mode}")
    vals = arr[mask]
    if vals.size == 0:
        vals = arr[finite]
    if vals.size == 0:
        raise ValueError("No finite pixels found")
    return vals.reshape(-1)


def compute_channel_stats(vals: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.mean(vals),
            np.median(vals),
            np.std(vals, ddof=0),
            np.percentile(vals, 10),
            np.percentile(vals, 90),
        ],
        dtype=np.float64,
    )


def build_image_stats_features(
    df: pd.DataFrame,
    channel_mapping: Dict[str, str],
    args: argparse.Namespace,
    split_label: str,
):
    if not channel_mapping:
        raise ValueError(
            "No image channel columns detected. Use --channel-columns to provide explicit mapping."
        )
    stat_names = ["mean", "median", "std", "p10", "p90"]
    names = [f"image_{band}_{stat}" for band in channel_mapping for stat in stat_names]
    rows, audit, cache = [], [], {}
    for row_idx, row in df.iterrows():
        feats = []
        for band, col in channel_mapping.items():
            path = resolve_image_path(row[col], args.img_dir)
            key = str(path)
            if key in cache:
                stats = cache[key]
            else:
                if not path.exists():
                    raise FileNotFoundError(
                        f"{split_label}: image for {band} not found: {path}"
                    )
                arr = read_image_array(path)
                vals = valid_pixels(arr, args.image_mask_mode)
                stats = compute_channel_stats(vals)
                cache[key] = stats
            feats.extend(stats.tolist())
            audit.append(
                {
                    "split": split_label,
                    "row_index": row_idx,
                    "band": band,
                    "column": col,
                    "path": str(path),
                    "mean": stats[0],
                    "median": stats[1],
                    "std": stats[2],
                    "p10": stats[3],
                    "p90": stats[4],
                }
            )
        rows.append(feats)
    return np.asarray(rows, dtype=np.float64), names, pd.DataFrame(audit)


# ----------------------------- fusion -----------------------------


def build_fused_features(
    df: pd.DataFrame,
    args: argparse.Namespace,
    channel_mapping: Optional[Dict[str, str]],
    split_label: str,
):
    matrices, names, info = [], [], {}
    audit_df = None
    if args.include_spectra:
        X, n, i = build_spectrum_features(df, args)
        matrices.append(X)
        names.extend(n)
        info["spectra"] = i
    else:
        info["spectra"] = {"enabled": False}
    if args.include_prospect:
        X, n, i = build_prospect_features(df, args)
        matrices.append(X)
        names.extend(n)
        info["prospect"] = i
    else:
        info["prospect"] = {"enabled": False}
    if args.include_image_stats:
        X, n, audit_df = build_image_stats_features(
            df, channel_mapping or {}, args, split_label
        )
        matrices.append(X)
        names.extend(n)
        info["image_stats"] = {
            "n_features": X.shape[1],
            "channel_mapping": channel_mapping,
            "mask_mode": args.image_mask_mode,
        }
    else:
        info["image_stats"] = {"enabled": False}
    if not matrices:
        raise ValueError("No feature groups enabled.")
    X = np.concatenate(matrices, axis=1)
    info["total_features"] = int(X.shape[1])
    return X, names, info, audit_df


# ----------------------------- normalization and models -----------------------------


def make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return None
    raise ValueError(kind)


def fit_global_normalizers(X_dev, y_dev, X_test, y_test, x_kind, y_kind, scope):
    if scope == "all_leaf_samples":
        X_fit = np.vstack([X_dev, X_test])
        y_fit = np.concatenate([y_dev, y_test]).reshape(-1, 1)
    elif scope == "development_only":
        X_fit = X_dev
        y_fit = y_dev.reshape(-1, 1)
    else:
        raise ValueError(scope)
    xs, ys = make_scaler(x_kind), make_scaler(y_kind)
    Xd = xs.fit_transform(X_dev) if xs is not None else X_dev.copy()
    Xt = xs.transform(X_test) if xs is not None else X_test.copy()
    if xs is not None:
        xs.fit(X_fit)
        Xd = xs.transform(X_dev)
        Xt = xs.transform(X_test)
    yd = (
        ys.fit_transform(y_dev.reshape(-1, 1)).reshape(-1)
        if ys is not None
        else y_dev.copy()
    )
    yt = (
        ys.transform(y_test.reshape(-1, 1)).reshape(-1)
        if ys is not None
        else y_test.copy()
    )
    if ys is not None:
        ys.fit(y_fit)
        yd = ys.transform(y_dev.reshape(-1, 1)).reshape(-1)
        yt = ys.transform(y_test.reshape(-1, 1)).reshape(-1)
    return Xd, yd, Xt, yt, xs, ys


def inverse_y(y, scaler):
    y = np.asarray(y, dtype=float).reshape(-1)
    return (
        scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        if scaler is not None
        else y
    )


def scaler_summary(scaler, name):
    if scaler is None:
        return {"name": name, "type": "none"}
    out = {"name": name, "type": scaler.__class__.__name__}
    for attr in ["mean_", "scale_", "data_min_", "data_max_", "min_", "data_range_"]:
        if hasattr(scaler, attr):
            arr = np.asarray(getattr(scaler, attr))
            out[attr] = {
                "shape": list(arr.shape),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
            }
    return out


def get_hyperparameters(species, grape_gb_n_estimators):
    key = normalize_species_key(species)
    hp = {m: dict(v) for m, v in HYPERPARAMETERS[key].items()}
    if key == "grape":
        hp["Gradient Boosting"]["n_estimators"] = grape_gb_n_estimators
    return hp


def make_models(args):
    hp = get_hyperparameters(args.species, args.grape_gb_n_estimators)
    return {
        "Elastic Net": ElasticNet(
            **hp["Elastic Net"], max_iter=10000, random_state=args.random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            **hp["Gradient Boosting"], random_state=args.random_state
        ),
        "Random Forest": RandomForestRegressor(
            **hp["Random Forest"], random_state=args.random_state, n_jobs=args.n_jobs
        ),
        "Ridge Regressor": Ridge(**hp["Ridge Regressor"]),
        "SVR RBF": SVR(kernel="rbf", **hp["SVR RBF"]),
    }


def make_param_grids(args):
    if args.grid_size == "small":
        return {
            "Elastic Net": {"alpha": [0.003, 0.01, 0.03], "l1_ratio": [0.1, 0.5]},
            "Ridge Regressor": {"alpha": [0.1, 1.0, 10.0]},
            "SVR RBF": {
                "C": [100.0, 1000.0],
                "epsilon": [0.01, 0.1, 0.3],
                "gamma": ["scale"],
            },
            "Random Forest": {
                "n_estimators": [300, 600],
                "max_depth": [None, 20, 40],
                "max_features": ["sqrt"],
                "min_samples_leaf": [1],
                "min_samples_split": [2],
            },
            "Gradient Boosting": {
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "n_estimators": [200, 400],
                "subsample": [0.8, 1.0],
            },
        }
    if args.grid_size == "wide":
        return {
            "Elastic Net": {
                "alpha": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                "l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "Ridge Regressor": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
            "SVR RBF": {
                "C": [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
                "epsilon": [0.001, 0.01, 0.03, 0.1, 0.3],
                "gamma": ["scale", "auto"],
            },
            "Random Forest": {
                "n_estimators": [200, 300, 600, 900],
                "max_depth": [None, 10, 20, 40, 80],
                "max_features": ["sqrt", 0.5, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5],
                "n_estimators": [100, 200, 400, 600, 800],
                "subsample": [0.6, 0.8, 1.0],
            },
        }
    return {
        "Elastic Net": {
            "alpha": [0.001, 0.003, 0.01, 0.03, 0.1],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7],
        },
        "Ridge Regressor": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "SVR RBF": {
            "C": [30.0, 100.0, 300.0, 1000.0],
            "epsilon": [0.01, 0.03, 0.1, 0.3],
            "gamma": ["scale", "auto"],
        },
        "Random Forest": {
            "n_estimators": [300, 600],
            "max_depth": [None, 20, 40],
            "max_features": ["sqrt", 0.5],
            "min_samples_leaf": [1, 2],
            "min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [2, 3, 4],
            "n_estimators": [200, 400, 600],
            "subsample": [0.8, 1.0],
        },
    }


def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cv_splitter(groups, n_splits, random_state):
    if groups is not None:
        if len(np.unique(groups)) < n_splits:
            raise ValueError("Not enough groups for GroupKFold")
        return GroupKFold(n_splits), groups, "GroupKFold"
    return KFold(n_splits, shuffle=True, random_state=random_state), None, "KFold"


def groups_from_df(df, group_column):
    if group_column.lower() == "none":
        return None, "none"
    if group_column != "auto":
        return df[group_column].astype(str).values, group_column
    for c in ["leaf_id", "Leaf_ID", "leaf", "leafID", "leaf_id_inferred"]:
        if c in df.columns and df[c].notna().all():
            return df[c].astype(str).values, c
    return None, "none"


def metadata(df, target_col):
    cols = [
        c
        for c in [
            "__source_split",
            "__source_csv",
            "__row_id_within_split",
            "stage",
            "Stages",
            "species",
            "Species",
            "blue_basename",
            "blue",
            "leaf_id",
            "leaf_id_inferred",
            target_col,
        ]
        if c in df.columns
    ]
    return df[cols].copy()


def plot_scatter(y_true, y_pred, title, target, path, m):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.7, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Real {target}")
    ax.set_ylabel(f"Estimated {target}")
    ax.set_title(title)
    ax.text(
        0.05,
        0.95,
        f"RMSE={m['RMSE']:.4f}\nMAE={m['MAE']:.4f}\nR²={m['R2']:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    args = parse_args()
    out = expand_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Fused-feature LWC/FMC regression")
    print("=" * 80)

    train = prepare_dataframe(args.train_csv, args, "train")
    val = prepare_dataframe(args.val_csv, args, "val")
    test = prepare_dataframe(args.test_csv, args, "test")
    train = merge_params_if_needed(train, args.train_params_csv, args, "train")
    val = merge_params_if_needed(val, args.val_params_csv, args, "val")
    test = merge_params_if_needed(test, args.test_params_csv, args, "test")

    dev = pd.concat([train, val], ignore_index=True)
    dev = maybe_add_keys(dev)
    test = maybe_add_keys(test)

    channel_mapping = (
        parse_channel_columns(dev, args.channel_columns)
        if args.include_image_stats
        else None
    )
    if channel_mapping:
        print(f"Channel mapping: {channel_mapping}")

    X_dev_raw, feature_names, dev_info, audit_dev = build_fused_features(
        dev, args, channel_mapping, "dev"
    )
    X_test_raw, feature_names_test, test_info, audit_test = build_fused_features(
        test, args, channel_mapping, "test"
    )
    if feature_names != feature_names_test or X_dev_raw.shape[1] != X_test_raw.shape[1]:
        raise ValueError("Train+val and test feature spaces do not match")

    y_dev = dev[args.target_column].to_numpy(float)
    y_test = test[args.target_column].to_numpy(float)
    X_dev, y_dev_n, X_test, y_test_n, x_scaler, y_scaler = fit_global_normalizers(
        X_dev_raw,
        y_dev,
        X_test_raw,
        y_test,
        args.x_normalization,
        args.y_normalization,
        args.normalization_scope,
    )

    groups, group_source = groups_from_df(dev, args.group_column)
    cv, cv_groups, cv_name = cv_splitter(groups, args.n_splits, args.random_state)

    pd.DataFrame(
        {"feature_index": np.arange(len(feature_names)), "feature_name": feature_names}
    ).to_csv(out / "fused_feature_names.csv", index=False)
    if audit_dev is not None:
        audit_dev.to_csv(out / "image_stats_audit_dev.csv", index=False)
    if audit_test is not None:
        audit_test.to_csv(out / "image_stats_audit_test.csv", index=False)

    norm_summary = {
        "scope": args.normalization_scope,
        "x_normalization": args.x_normalization,
        "y_normalization": args.y_normalization,
        "x_scaler": scaler_summary(x_scaler, "x_scaler"),
        "y_scaler": scaler_summary(y_scaler, "y_scaler"),
    }
    (out / "normalization_summary.json").write_text(
        json.dumps(norm_summary, indent=2, default=str)
    )

    print(f"Train rows: {len(train)} | Val rows: {len(val)} | Test rows: {len(test)}")
    print(
        f"Feature count: {X_dev.shape[1]} | Group source: {group_source} | CV: {cv_name}"
    )
    print(
        f"Feature groups: spectra={args.include_spectra}, prospect={args.include_prospect}, image_stats={args.include_image_stats}"
    )

    models = make_models(args)
    grids = make_param_grids(args) if args.grid_search else None
    if grids:
        (out / "grid_search_param_grids.json").write_text(
            json.dumps(grids, indent=2, default=str)
        )
    (out / "fixed_table_hyperparameters.json").write_text(
        json.dumps(
            get_hyperparameters(args.species, args.grape_gb_n_estimators),
            indent=2,
            default=str,
        )
    )

    scatter_cv = out / "scatter_cv_svg"
    scatter_test = out / "scatter_test_svg"
    scatter_cv.mkdir(exist_ok=True)
    scatter_test.mkdir(exist_ok=True)
    grid_dir = out / "grid_search_results"
    if args.grid_search:
        grid_dir.mkdir(exist_ok=True)

    cv_rows = []
    test_rows = []
    cv_preds = []
    test_preds = []
    best_params = {}
    fitted = {}
    for name in MODEL_ORDER:
        print("=" * 80)
        print(name)
        print("=" * 80)
        est = models[name]
        if args.grid_search:
            gs = GridSearchCV(
                est,
                grids[name],
                scoring={
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                    "r2": "r2",
                },
                refit=args.grid_refit_metric,
                cv=cv,
                n_jobs=args.n_jobs,
                return_train_score=True,
                verbose=2,
                error_score="raise",
            )
            if cv_groups is not None:
                gs.fit(X_dev, y_dev_n, groups=cv_groups)
            else:
                gs.fit(X_dev, y_dev_n)
            pd.DataFrame(gs.cv_results_).to_csv(
                grid_dir / f"grid_search_results_{safe_model_filename(name)}.csv",
                index=False,
            )
            est = gs.best_estimator_
            best_params[name] = dict(gs.best_params_)
        else:
            best_params[name] = "fixed_table_hyperparameters"

        if cv_groups is not None:
            y_cv_n = cross_val_predict(
                clone(est), X_dev, y_dev_n, cv=cv, groups=cv_groups, n_jobs=args.n_jobs
            )
        else:
            y_cv_n = cross_val_predict(
                clone(est), X_dev, y_dev_n, cv=cv, n_jobs=args.n_jobs
            )
        y_cv = inverse_y(y_cv_n, y_scaler)
        mcv = metrics(y_dev, y_cv)
        cv_rows.append({"Model": name, **mcv})
        pred_df = metadata(dev, args.target_column)
        pred_df["Model"] = name
        pred_df["evaluation"] = "5fold_cv"
        pred_df["y_true"] = y_dev
        pred_df["y_pred"] = y_cv
        pred_df["residual"] = y_cv - y_dev
        pred_df["y_true_normalized"] = y_dev_n
        pred_df["y_pred_normalized"] = np.asarray(y_cv_n).reshape(-1)
        cv_preds.append(pred_df)
        plot_scatter(
            y_dev,
            y_cv,
            f"{name} | 5-fold CV",
            args.target_column,
            scatter_cv / f"cv_scatter_{safe_model_filename(name)}.svg",
            mcv,
        )

        final = clone(est)
        final.fit(X_dev, y_dev_n)
        fitted[name] = final
        y_te_n = final.predict(X_test)
        y_te = inverse_y(y_te_n, y_scaler)
        mte = metrics(y_test, y_te)
        test_rows.append({"Model": name, **mte})
        pred_df = metadata(test, args.target_column)
        pred_df["Model"] = name
        pred_df["evaluation"] = "test"
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_te
        pred_df["residual"] = y_te - y_test
        pred_df["y_true_normalized"] = y_test_n
        pred_df["y_pred_normalized"] = np.asarray(y_te_n).reshape(-1)
        test_preds.append(pred_df)
        plot_scatter(
            y_test,
            y_te,
            f"{name} | Test",
            args.target_column,
            scatter_test / f"test_scatter_{safe_model_filename(name)}.svg",
            mte,
        )
        print("CV:", mcv)
        print("Test:", mte)

    cv_metrics = pd.DataFrame(cv_rows).set_index("Model").loc[MODEL_ORDER].reset_index()
    test_metrics = (
        pd.DataFrame(test_rows).set_index("Model").loc[MODEL_ORDER].reset_index()
    )
    cv_pred_all = pd.concat(cv_preds, ignore_index=True)
    test_pred_all = pd.concat(test_preds, ignore_index=True)
    cv_metrics.to_csv(out / "cv_metrics.csv", index=False)
    test_metrics.to_csv(out / "test_metrics.csv", index=False)
    (out / "cv_metrics.md").write_text(cv_metrics.to_markdown(index=False))
    (out / "test_metrics.md").write_text(test_metrics.to_markdown(index=False))
    if args.write_latex:
        (out / "cv_metrics.tex").write_text(
            cv_metrics.to_latex(index=False, float_format="%.4f")
        )
        (out / "test_metrics.tex").write_text(
            test_metrics.to_latex(index=False, float_format="%.4f")
        )
    cv_pred_all.to_csv(out / "cv_predictions.csv", index=False)
    test_pred_all.to_csv(out / "test_predictions.csv", index=False)
    best_df = pd.DataFrame(
        [
            {"Model": k, **v}
            if isinstance(v, dict)
            else {"Model": k, "params_source": v}
            for k, v in best_params.items()
        ]
    )
    best_df.to_csv(out / "selected_best_hyperparameters.csv", index=False)

    if args.save_models:
        model_dir = out / "fitted_models_joblib"
        model_dir.mkdir(exist_ok=True)
        artifact = {
            "models": fitted,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_names": feature_names,
            "channel_mapping": channel_mapping,
            "target_column": args.target_column,
            "normalization_scope": args.normalization_scope,
        }
        joblib.dump(
            artifact,
            model_dir / "fused_feature_regressors_with_global_normalizers.joblib",
        )
        joblib.dump(fitted, model_dir / "sklearn_regressors_normalized_input.joblib")
        for name, est in fitted.items():
            joblib.dump(
                est, model_dir / f"{safe_model_filename(name)}_normalized_input.joblib"
            )

    manifest = {
        "target_column": args.target_column,
        "n_features": int(X_dev.shape[1]),
        "feature_groups": {
            "spectra": args.include_spectra,
            "prospect": args.include_prospect,
            "image_stats": args.include_image_stats,
        },
        "dev_feature_info": dev_info,
        "test_feature_info": test_info,
        "channel_mapping": channel_mapping,
        "normalization_scope": args.normalization_scope,
        "x_normalization": args.x_normalization,
        "y_normalization": args.y_normalization,
        "grid_search": args.grid_search,
        "grid_size": args.grid_size if args.grid_search else None,
        "cv": {"cv_name": cv_name, "n_splits": args.n_splits},
        "best_params": best_params,
        "outputs": {
            "cv_metrics": str(out / "cv_metrics.csv"),
            "test_metrics": str(out / "test_metrics.csv"),
            "feature_names": str(out / "fused_feature_names.csv"),
        },
    }
    (out / "regression_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print("=" * 80)
    print("Finished")
    print("CV metrics:")
    print(cv_metrics.to_string(index=False))
    print("\nTest metrics:")
    print(test_metrics.to_string(index=False))
    print(f"Output directory: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
