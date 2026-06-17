#!/usr/bin/env python3
"""
Fused-feature regression for LWC_d/FMC_d estimation.

Input vector:

    generated_spectrum
  + flattened PROSPECT parameters
  + first-order multispectral image-channel statistics

This version explicitly supports your current naming convention:

    blue_basename example: leaf028d0_1

where the suffix after "_" identifies the multispectral channel:

    blue     -> _1
    green    -> _2
    red      -> _3
    nir      -> _4
    red_edge -> _5

If the CSV only contains blue_basename, the other channel basenames are
constructed automatically by replacing the final channel suffix.

PROSPECT parameter convention:
    The generator-derived PROSPECT parameter array is expected to be 4 x 7:
        4 segments x 7 parameters
    It is flattened by default into 28 features:
        seg1_N, seg1_Cab, ..., seg4_Ant

The script keeps the established evaluation protocol:
    - train + validation are merged as development set
    - 5-fold CV on train + validation
    - final fit on train + validation
    - independent test evaluation
    - optional grid search
    - all-leaf normalization by default: train + val + test
    - metrics and plots are reported in original target units

Example:

python train_lwc_regressors_from_generated_spectra_fusion_features_v2_fixed.py \
    --train-csv ~/Results/pix2spectral_inference/train/generated_spectra_with_FMC_d.csv \
    --val-csv ~/Results/pix2spectral_inference/val/generated_spectra_with_FMC_d.csv \
    --test-csv ~/Results/pix2spectral_inference/test/generated_spectra_with_FMC_d.csv \
    --target-column FMC_d \
    --species Avocado \
    --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --output-dir ~/Results/lwc_regression/avocado_fusion_features_v2 \
    --grid_search \
    --grid-size medium \
    --save-models
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


# -------------------------------------------------------------------------
# Fixed model order and hyperparameters
# -------------------------------------------------------------------------

MODEL_ORDER = [
    "Elastic Net",
    "Gradient Boosting",
    "Random Forest",
    "Ridge Regressor",
    "SVR RBF",
]

PROSPECT_PARAM_NAMES = ["N", "Cab", "Car", "Cbrown", "Cw", "Cm", "Ant"]

CHANNEL_SUFFIX = {
    "blue": "1",
    "green": "2",
    "red": "3",
    "nir": "4",
    "red_edge": "5",
}

CHANNEL_ORDER = ["blue", "green", "red", "nir", "red_edge"]

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


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LWC/FMC regressors using generated spectra + flattened "
            "PROSPECT params + multispectral channel statistics."
        )
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--target-column", default="LWC_d")
    parser.add_argument("--species", default="Avocado")
    parser.add_argument("--species-column", default=None)
    parser.add_argument("--filter-species", action="store_true")

    parser.add_argument("--spectrum-column", default="generated_spectrum_json")
    parser.add_argument("--fallback-spectrum-column", default="spectral")
    parser.add_argument("--spectrum-wavelength-min", type=float, default=400.0)
    parser.add_argument("--spectrum-wavelength-max", type=float, default=2500.0)
    parser.add_argument(
        "--wl-min", "--wavelength-min", dest="wavelength_min", type=float, default=None
    )
    parser.add_argument(
        "--wl-max", "--wavelength-max", dest="wavelength_max", type=float, default=None
    )

    parser.add_argument(
        "--params-column",
        default="params_json",
        help="Column containing the 4x7 PROSPECT parameter array.",
    )
    parser.add_argument(
        "--param-feature-mode",
        choices=["flatten", "mean", "mean_std", "flatten_mean_std"],
        default="flatten",
        help="Default flatten converts 4x7 to 28 features.",
    )
    parser.add_argument("--train-params-csv", default=None)
    parser.add_argument("--val-params-csv", default=None)
    parser.add_argument("--test-params-csv", default=None)
    parser.add_argument(
        "--merge-keys",
        nargs="+",
        default=["species", "stage", "blue_basename"],
        help="Keys used if PROSPECT params are in separate CSV files.",
    )

    parser.add_argument(
        "--blue-basename-column",
        default="blue_basename",
        help="Column containing blue basename/path, e.g. leaf028d0_1.",
    )
    parser.add_argument(
        "--img-dir",
        default=None,
        help="Base directory where multispectral channel images are located.",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=[".tif", ".tiff", ".TIF", ".TIFF", ".png", ".jpg", ".jpeg"],
        help="Extensions to try when blue_basename has no extension.",
    )
    parser.add_argument(
        "--recursive-image-search",
        action="store_true",
        help="Search recursively under --img-dir when direct path candidates fail.",
    )
    parser.add_argument(
        "--image-mask-mode",
        choices=["nonzero", "positive", "finite", "none"],
        default="nonzero",
        help="Pixel mask for image statistics.",
    )

    parser.add_argument("--include-spectra", action="store_true", default=True)
    parser.add_argument("--no-spectra", action="store_false", dest="include_spectra")
    parser.add_argument("--include-prospect", action="store_true", default=True)
    parser.add_argument("--no-prospect", action="store_false", dest="include_prospect")
    parser.add_argument("--include-image-stats", action="store_true", default=True)
    parser.add_argument(
        "--no-image-stats", action="store_false", dest="include_image_stats"
    )

    parser.add_argument("--group-column", default="auto")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument(
        "--grid-size", choices=["small", "medium", "wide"], default="medium"
    )
    parser.add_argument(
        "--grid-refit-metric", choices=["rmse", "mae", "r2"], default="rmse"
    )

    parser.add_argument(
        "--x-normalization", choices=["standard", "minmax", "none"], default="standard"
    )
    parser.add_argument(
        "--y-normalization", choices=["standard", "minmax", "none"], default="standard"
    )
    parser.add_argument(
        "--normalization-scope",
        choices=["all_leaf_samples", "development_only"],
        default="all_leaf_samples",
    )

    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--gpu-backend", choices=["auto", "none", "cuml", "xgboost"], default="auto"
    )
    parser.add_argument("--grape-gb-n-estimators", type=int, default=400)

    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--write-latex", action="store_true")
    parser.add_argument("--drop-na-target", action="store_true")
    parser.add_argument("--max-samples-debug", type=int, default=None)

    return parser.parse_args()


# -------------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------------


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
            f"Unknown species '{species}'. Expected Avocado/Avo, Olive, or Grape/Vineyard."
        )
    return aliases[s]


def basename_any(path_like: Any) -> str:
    if pd.isna(path_like):
        return ""
    text = str(path_like).strip()
    if not text:
        return ""
    return PurePosixPath(PureWindowsPath(text).name).name


def stem_and_suffix(path_like: Any) -> Tuple[str, str]:
    name = basename_any(path_like)
    p = Path(name)
    return p.stem, p.suffix


def replace_channel_suffix(blue_value: Any, channel: str) -> str:
    """
    Convert a blue basename/path such as leaf028d0_1 or leaf028d0_1.tif into
    the requested channel basename/path using:
        blue=1, green=2, red=3, nir=4, red_edge=5
    """
    suffix = CHANNEL_SUFFIX[channel]
    text = str(blue_value).strip()
    if not text:
        raise ValueError("Empty blue_basename value.")

    # Preserve directory if the CSV value contains one.
    original = Path(text)
    parent = "" if str(original.parent) == "." else str(original.parent)

    stem, ext = stem_and_suffix(text)
    new_stem = re.sub(r"_(1|2|3|4|5)$", f"_{suffix}", stem)
    if new_stem == stem and not re.search(r"_(1|2|3|4|5)$", stem):
        # No channel suffix found; append the requested channel suffix.
        new_stem = f"{stem}_{suffix}"

    new_name = new_stem + ext
    if parent:
        return str(Path(parent) / new_name)
    return new_name


def infer_leaf_id_from_text(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = basename_any(value).lower()
    for pat in [r"leaf[_-]?(\d+)", r"leaf(\d{3})d\d", r"(\d{3})d\d"]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def safe_model_filename(model_name: str) -> str:
    """Convert model names into safe filenames."""
    return (
        str(model_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_species_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["species", "Species", "SPECIES"]:
        if col in df.columns:
            return col
    return None


# -------------------------------------------------------------------------
# CSV loading and merge
# -------------------------------------------------------------------------


def maybe_add_canonical_columns(
    df: pd.DataFrame, blue_basename_column: str
) -> pd.DataFrame:
    df = df.copy()

    if "stage" not in df.columns and "Stages" in df.columns:
        df["stage"] = df["Stages"]
    if "species" not in df.columns and "Species" in df.columns:
        df["species"] = df["Species"]

    if "blue_basename" not in df.columns:
        for candidate in [
            blue_basename_column,
            "blue",
            "Blue",
            "BLUE",
            "filename",
            "image_name",
        ]:
            if candidate in df.columns:
                df["blue_basename"] = df[candidate].map(basename_any)
                break
    else:
        df["blue_basename"] = df["blue_basename"].map(basename_any)

    if "leaf_id" not in df.columns:
        for candidate in [
            "blue_basename",
            blue_basename_column,
            "blue",
            "Blue",
            "filename",
            "image_name",
        ]:
            if candidate in df.columns:
                inferred = df[candidate].map(infer_leaf_id_from_text)
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
        df = df.head(int(args.max_samples_debug)).copy()

    df = maybe_add_canonical_columns(df, args.blue_basename_column)

    species_col = args.species_column or find_species_column(df)
    if args.filter_species:
        if species_col is None:
            raise ValueError(
                f"{split_name}: --filter-species requested, but no species column found."
            )
        target_species = normalize_species_key(args.species)

        def row_species_key(x):
            if pd.isna(x):
                return ""
            s = str(x).strip().lower()
            try:
                return normalize_species_key(s)
            except ValueError:
                return s

        before = len(df)
        df = df[df[species_col].map(row_species_key) == target_species].copy()
        after = len(df)
        if after == 0:
            raise ValueError(f"{split_name}: no rows left after species filtering.")
        print(f"{split_name}: species filter kept {after}/{before} rows.")

    if args.target_column not in df.columns:
        raise ValueError(
            f"{split_name}: target column '{args.target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[args.target_column] = pd.to_numeric(df[args.target_column], errors="coerce")
    if df[args.target_column].isna().any():
        n_bad = int(df[args.target_column].isna().sum())
        if args.drop_na_target:
            df = df.dropna(subset=[args.target_column]).copy()
            print(f"{split_name}: dropped {n_bad} rows with missing target.")
        else:
            raise ValueError(
                f"{split_name}: {n_bad} missing/non-numeric target values."
            )

    df = df.reset_index(drop=True)
    df["__row_id_within_split"] = np.arange(len(df), dtype=int)
    return df


def merge_params_if_needed(
    df: pd.DataFrame,
    params_csv: Optional[str],
    params_column: str,
    merge_keys: Sequence[str],
    split_name: str,
    blue_basename_column: str,
) -> pd.DataFrame:
    if params_column in df.columns:
        return df

    if params_csv is None:
        raise ValueError(
            f"{split_name}: params column '{params_column}' not found. "
            f"Provide --{split_name}-params-csv or include the column in the main CSV."
        )

    params_path = expand_path(params_csv)
    if not params_path.exists():
        raise FileNotFoundError(f"{split_name}: params CSV not found: {params_path}")

    params_df = pd.read_csv(params_path).copy()
    params_df = maybe_add_canonical_columns(params_df, blue_basename_column)
    df = maybe_add_canonical_columns(df, blue_basename_column)

    usable_keys = [k for k in merge_keys if k in df.columns and k in params_df.columns]
    if not usable_keys:
        raise ValueError(
            f"{split_name}: no usable merge keys. Requested {merge_keys}; "
            f"main columns={list(df.columns)}; params columns={list(params_df.columns)}"
        )

    keep_cols = list(dict.fromkeys(usable_keys + [params_column]))
    merged = df.merge(params_df[keep_cols], on=usable_keys, how="left", validate="m:1")
    missing = int(merged[params_column].isna().sum())
    if missing > 0:
        raise ValueError(
            f"{split_name}: missing '{params_column}' for {missing}/{len(merged)} rows "
            f"after merge using keys {usable_keys}."
        )

    print(f"{split_name}: merged params from {params_path} using keys {usable_keys}.")
    return merged


# -------------------------------------------------------------------------
# Spectrum features
# -------------------------------------------------------------------------


def parse_vector_value(value: Any, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    if pd.isna(value):
        raise ValueError(f"Missing {name} value.")

    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty {name} string.")

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
        raise ValueError(f"Could not parse {name}: {text[:120]}...")
    return arr.reshape(-1)


def find_wavelength_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).startswith("wl_")]

    def key(c):
        try:
            return float(str(c)[3:])
        except ValueError:
            return math.inf

    return sorted(cols, key=key)


def select_wavelength_range(
    X: np.ndarray,
    wavelengths: np.ndarray,
    names: List[str],
    wl_min: Optional[float],
    wl_max: Optional[float],
):
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wl_min is None and wl_max is None:
        return X, wavelengths, names

    if wl_min is not None and wl_max is not None and wl_min > wl_max:
        raise ValueError(f"Invalid wavelength range: {wl_min} > {wl_max}")

    mask = np.ones_like(wavelengths, dtype=bool)
    if wl_min is not None:
        mask &= wavelengths >= float(wl_min)
    if wl_max is not None:
        mask &= wavelengths <= float(wl_max)

    if int(mask.sum()) == 0:
        raise ValueError(
            f"No wavelengths selected. Requested [{wl_min}, {wl_max}], "
            f"available [{float(np.min(wavelengths))}, {float(np.max(wavelengths))}]."
        )

    return X[:, mask], wavelengths[mask], [n for n, keep in zip(names, mask) if keep]


def build_spectrum_features(df: pd.DataFrame, args: argparse.Namespace):
    if args.spectrum_column in df.columns:
        spectra = [
            parse_vector_value(v, args.spectrum_column)
            for v in df[args.spectrum_column].values
        ]
        source = args.spectrum_column
        lengths = [len(s) for s in spectra]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Inconsistent spectrum lengths: {pd.Series(lengths).value_counts().to_dict()}"
            )
        X = np.stack(spectra, axis=0).astype(np.float64)
        wl = np.linspace(
            args.spectrum_wavelength_min, args.spectrum_wavelength_max, X.shape[1]
        )
        names = [f"spectrum_wl_{w:g}" for w in wl]
    elif args.fallback_spectrum_column in df.columns:
        spectra = [
            parse_vector_value(v, args.fallback_spectrum_column)
            for v in df[args.fallback_spectrum_column].values
        ]
        source = args.fallback_spectrum_column
        lengths = [len(s) for s in spectra]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Inconsistent fallback spectrum lengths: {pd.Series(lengths).value_counts().to_dict()}"
            )
        X = np.stack(spectra, axis=0).astype(np.float64)
        wl = np.linspace(
            args.spectrum_wavelength_min, args.spectrum_wavelength_max, X.shape[1]
        )
        names = [f"spectrum_wl_{w:g}" for w in wl]
    else:
        wl_cols = find_wavelength_columns(df)
        if not wl_cols:
            raise ValueError(
                f"No generated spectrum found. Tried '{args.spectrum_column}', "
                f"'{args.fallback_spectrum_column}', and wl_* columns."
            )
        source = "wl_columns"
        X = df[wl_cols].to_numpy(dtype=float)
        wl = np.asarray([float(str(c)[3:]) for c in wl_cols], dtype=float)
        names = [f"spectrum_{c}" for c in wl_cols]

    X, wl, names = select_wavelength_range(
        X, wl, names, args.wavelength_min, args.wavelength_max
    )
    return (
        X,
        names,
        {
            "source": source,
            "wavelength_min": float(np.min(wl)),
            "wavelength_max": float(np.max(wl)),
            "n_features": int(X.shape[1]),
        },
    )


# -------------------------------------------------------------------------
# PROSPECT params: explicit 4x7 flattening support
# -------------------------------------------------------------------------


def parse_params_preserve_shape(value: Any) -> np.ndarray:
    """
    Parse params_json while preserving the 4x7 shape.

    Supported examples:
        [[...7...], [...7...], [...7...], [...7...]]
        "[[...], [...], [...], [...]]"
        flattened list length 28
    """
    if isinstance(value, np.ndarray):
        arr = value.astype(float)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
    else:
        if pd.isna(value):
            raise ValueError("Missing params_json value.")
        text = str(value).strip()
        if not text:
            raise ValueError("Empty params_json value.")
        try:
            arr = np.asarray(json.loads(text), dtype=float)
        except Exception:
            try:
                arr = np.asarray(ast.literal_eval(text), dtype=float)
            except Exception as exc:
                # Last-resort numeric parser for flattened values.
                cleaned = text.strip().strip("[]()").replace(";", ",")
                arr_flat = np.fromstring(cleaned, sep=",", dtype=float)
                if arr_flat.size == 0:
                    arr_flat = np.fromstring(cleaned, sep=" ", dtype=float)
                if arr_flat.size == 0:
                    raise ValueError(
                        f"Could not parse PROSPECT params: {text[:120]}..."
                    ) from exc
                arr = arr_flat

    if not np.isfinite(arr).all():
        raise ValueError("PROSPECT params contain non-finite values.")

    return arr


def params_to_feature_vector(params: np.ndarray, mode: str):
    params = np.asarray(params, dtype=float)
    original_shape = tuple(params.shape)

    # Your expected case: 4 segments x 7 parameters.
    if params.ndim == 2:
        if params.shape != (4, 7):
            if params.shape[1] != 7:
                raise ValueError(
                    f"Expected PROSPECT params shape [S,7], got {params.shape}."
                )
        mat = params
    elif params.ndim == 1:
        if params.size == 28:
            mat = params.reshape(4, 7)
            original_shape = (4, 7)
        elif params.size == 7:
            mat = params.reshape(1, 7)
        elif params.size % 7 == 0:
            mat = params.reshape(params.size // 7, 7)
        else:
            raise ValueError(
                f"Expected flattened PROSPECT params length 28 or a multiple of 7, got {params.size}."
            )
    else:
        if params.shape[-1] != 7:
            raise ValueError(f"Unsupported PROSPECT params shape {params.shape}.")
        mat = params.reshape(-1, 7)

    n_segments = mat.shape[0]

    flat = mat.reshape(-1)
    flat_names = []
    for seg in range(n_segments):
        for p in PROSPECT_PARAM_NAMES:
            flat_names.append(f"prospect_seg{seg + 1}_{p}")

    mean = mat.mean(axis=0)
    mean_names = [f"prospect_mean_{p}" for p in PROSPECT_PARAM_NAMES]
    std = mat.std(axis=0, ddof=0)
    std_names = [f"prospect_std_{p}" for p in PROSPECT_PARAM_NAMES]

    if mode == "flatten":
        return flat, flat_names, original_shape
    if mode == "mean":
        return mean, mean_names, original_shape
    if mode == "mean_std":
        return np.concatenate([mean, std]), mean_names + std_names, original_shape
    if mode == "flatten_mean_std":
        return (
            np.concatenate([flat, mean, std]),
            flat_names + mean_names + std_names,
            original_shape,
        )

    raise ValueError(f"Unsupported param feature mode: {mode}")


def build_prospect_features(df: pd.DataFrame, args: argparse.Namespace):
    if args.params_column not in df.columns:
        raise ValueError(f"Column '{args.params_column}' not found.")

    vectors = []
    shapes = []
    names_ref = None

    for idx, value in enumerate(df[args.params_column].values):
        arr = parse_params_preserve_shape(value)
        vec, names, shape = params_to_feature_vector(arr, args.param_feature_mode)
        if names_ref is None:
            names_ref = names
        elif len(names) != len(names_ref):
            raise ValueError(f"Inconsistent PROSPECT feature length at row {idx}.")
        vectors.append(vec)
        shapes.append(shape)

    X = np.stack(vectors, axis=0).astype(np.float64)
    return (
        X,
        names_ref or [],
        {
            "n_features": int(X.shape[1]),
            "shape_counts": pd.Series([str(s) for s in shapes])
            .value_counts()
            .to_dict(),
            "mode": args.param_feature_mode,
        },
    )


# -------------------------------------------------------------------------
# Image stats from blue_basename convention
# -------------------------------------------------------------------------


def image_path_candidates(
    basename_or_path: str, img_dir: Optional[str], extensions: Sequence[str]
) -> List[Path]:
    text = str(basename_or_path).strip()
    p = Path(text).expanduser()
    candidates: List[Path] = []

    def add_with_ext(base_path: Path):
        candidates.append(base_path)
        if base_path.suffix == "":
            for ext in extensions:
                candidates.append(Path(str(base_path) + ext))

    if p.is_absolute():
        add_with_ext(p)
    else:
        add_with_ext(p)
        if img_dir is not None:
            base = expand_path(img_dir)
            add_with_ext(base / text)
            add_with_ext(base / basename_any(text))

    # De-duplicate while preserving order.
    out = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def resolve_image_path_from_blue(
    blue_value: Any,
    channel: str,
    img_dir: Optional[str],
    extensions: Sequence[str],
    recursive: bool,
) -> Path:
    channel_name = replace_channel_suffix(blue_value, channel)

    for cand in image_path_candidates(channel_name, img_dir, extensions):
        if cand.exists():
            return cand.resolve()

    if recursive and img_dir is not None:
        base = expand_path(img_dir)
        stem, ext = stem_and_suffix(channel_name)
        patterns = []
        if ext:
            patterns.append(stem + ext)
        else:
            patterns.extend([stem + e for e in extensions])
        for pattern in patterns:
            matches = list(base.rglob(pattern))
            if matches:
                return matches[0].resolve()

    tried = image_path_candidates(channel_name, img_dir, extensions)
    raise FileNotFoundError(
        f"Could not find image for channel '{channel}' constructed from blue_basename='{blue_value}'. "
        f"Constructed channel name='{channel_name}'. Tried first candidates: {[str(x) for x in tried[:8]]}"
    )


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


def channel_pixels_for_stats(arr: np.ndarray, mask_mode: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)

    if mask_mode == "none":
        mask = finite
    elif mask_mode == "finite":
        mask = finite
    elif mask_mode == "nonzero":
        mask = finite & (arr != 0)
    elif mask_mode == "positive":
        mask = finite & (arr > 0)
    else:
        raise ValueError(f"Unsupported mask mode: {mask_mode}")

    vals = arr[mask]
    if vals.size == 0:
        vals = arr[finite]
    if vals.size == 0:
        raise ValueError("No valid pixels available for image statistics.")
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


def build_image_stats_features(df: pd.DataFrame, args: argparse.Namespace):
    if "blue_basename" not in df.columns:
        raise ValueError(
            f"blue_basename column not found. Provide --blue-basename-column or include blue_basename in CSV."
        )

    stat_names = ["mean", "median", "std", "p10", "p90"]
    names = [f"image_{band}_{stat}" for band in CHANNEL_ORDER for stat in stat_names]

    cache: Dict[Tuple[str, str], np.ndarray] = {}
    rows = []
    audit_rows = []

    for row_idx, blue_value in enumerate(df["blue_basename"].values):
        feats = []
        for band in CHANNEL_ORDER:
            path = resolve_image_path_from_blue(
                blue_value=blue_value,
                channel=band,
                img_dir=args.img_dir,
                extensions=args.image_extensions,
                recursive=args.recursive_image_search,
            )
            key = (band, str(path))
            if key in cache:
                stats = cache[key]
            else:
                arr = read_image_array(path)
                vals = channel_pixels_for_stats(arr, args.image_mask_mode)
                stats = compute_channel_stats(vals)
                cache[key] = stats

            feats.extend(stats.tolist())
            audit_rows.append(
                {
                    "row_index": row_idx,
                    "blue_basename": blue_value,
                    "band": band,
                    "channel_suffix": CHANNEL_SUFFIX[band],
                    "path": str(path),
                    "mean": stats[0],
                    "median": stats[1],
                    "std": stats[2],
                    "p10": stats[3],
                    "p90": stats[4],
                }
            )

        rows.append(feats)

    X = np.asarray(rows, dtype=np.float64)
    return (
        X,
        names,
        pd.DataFrame(audit_rows),
        {
            "n_features": int(X.shape[1]),
            "channel_suffix": CHANNEL_SUFFIX,
            "channels": CHANNEL_ORDER,
            "mask_mode": args.image_mask_mode,
        },
    )


# -------------------------------------------------------------------------
# Fused features
# -------------------------------------------------------------------------


def build_fused_features(df: pd.DataFrame, args: argparse.Namespace):
    matrices = []
    names = []
    info = {}
    image_audit = None

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
        X, n, audit, i = build_image_stats_features(df, args)
        matrices.append(X)
        names.extend(n)
        info["image_stats"] = i
        image_audit = audit
    else:
        info["image_stats"] = {"enabled": False}

    if not matrices:
        raise ValueError("No feature groups enabled.")

    X = np.concatenate(matrices, axis=1)
    info["total_features"] = int(X.shape[1])
    return X, names, info, image_audit


# -------------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------------


def make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return None
    raise ValueError(f"Unsupported normalization: {kind}")


def fit_global_normalizers(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    x_normalization: str,
    y_normalization: str,
    normalization_scope: str,
):
    if normalization_scope == "all_leaf_samples":
        X_fit = np.vstack([X_dev, X_test])
        y_fit = np.concatenate([y_dev, y_test]).reshape(-1, 1)
    elif normalization_scope == "development_only":
        X_fit = X_dev
        y_fit = y_dev.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown normalization scope: {normalization_scope}")

    x_scaler = make_scaler(x_normalization)
    y_scaler = make_scaler(y_normalization)

    if x_scaler is not None:
        x_scaler.fit(X_fit)
        X_dev_norm = x_scaler.transform(X_dev)
        X_test_norm = x_scaler.transform(X_test)
    else:
        X_dev_norm = X_dev.copy()
        X_test_norm = X_test.copy()

    if y_scaler is not None:
        y_scaler.fit(y_fit)
        y_dev_norm = y_scaler.transform(y_dev.reshape(-1, 1)).reshape(-1)
        y_test_norm = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    else:
        y_dev_norm = y_dev.copy()
        y_test_norm = y_test.copy()

    return X_dev_norm, y_dev_norm, X_test_norm, y_test_norm, x_scaler, y_scaler


def inverse_y(y_values: np.ndarray, y_scaler) -> np.ndarray:
    y_values = np.asarray(y_values, dtype=float).reshape(-1)
    if y_scaler is None:
        return y_values
    return y_scaler.inverse_transform(y_values.reshape(-1, 1)).reshape(-1)


def scaler_to_summary(scaler, name: str) -> Dict[str, Any]:
    if scaler is None:
        return {"name": name, "type": "none"}
    summary = {"name": name, "type": scaler.__class__.__name__}
    for attr in ["mean_", "scale_", "data_min_", "data_max_", "min_", "data_range_"]:
        if hasattr(scaler, attr):
            arr = np.asarray(getattr(scaler, attr))
            summary[attr] = {
                "shape": list(arr.shape),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
    return summary


# -------------------------------------------------------------------------
# Models, grids, and evaluation
# -------------------------------------------------------------------------


def get_hyperparameters(species: str, grape_gb_n_estimators: int):
    key = normalize_species_key(species)
    hp = {m: dict(v) for m, v in HYPERPARAMETERS[key].items()}
    if key == "grape":
        hp["Gradient Boosting"]["n_estimators"] = int(grape_gb_n_estimators)
    return hp


def import_optional_gpu_backends(use_gpu: bool, gpu_backend: str):
    backends = {"cuml": None, "xgboost": None, "messages": []}
    if not use_gpu or gpu_backend == "none":
        backends["messages"].append("GPU disabled; using CPU scikit-learn estimators.")
        return backends

    if gpu_backend in ["auto", "cuml"]:
        try:
            from cuml.linear_model import ElasticNet as cuElasticNet
            from cuml.linear_model import Ridge as cuRidge
            from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
            from cuml.svm import SVR as cuSVR

            backends["cuml"] = {
                "ElasticNet": cuElasticNet,
                "Ridge": cuRidge,
                "RandomForestRegressor": cuRandomForestRegressor,
                "SVR": cuSVR,
            }
            backends["messages"].append("RAPIDS/cuML detected for ENet/Ridge/RF/SVR.")
        except Exception as exc:
            backends["messages"].append(
                f"RAPIDS/cuML unavailable; CPU fallback. Reason: {exc}"
            )

    if gpu_backend in ["auto", "xgboost"]:
        try:
            from xgboost import XGBRegressor

            backends["xgboost"] = XGBRegressor
            backends["messages"].append("XGBoost detected for GPU gradient boosting.")
        except Exception as exc:
            backends["messages"].append(
                f"XGBoost unavailable; CPU GB fallback. Reason: {exc}"
            )

    return backends


def make_models(args: argparse.Namespace):
    hp = get_hyperparameters(args.species, args.grape_gb_n_estimators)
    backends = import_optional_gpu_backends(args.use_gpu, args.gpu_backend)
    for msg in backends["messages"]:
        print(msg)

    use_cuml = backends["cuml"] is not None
    use_xgb = backends["xgboost"] is not None

    if use_cuml:
        try:
            cuElasticNet = backends["cuml"]["ElasticNet"]
            cuRidge = backends["cuml"]["Ridge"]
            cuRF = backends["cuml"]["RandomForestRegressor"]
            cuSVR = backends["cuml"]["SVR"]

            enet = cuElasticNet(
                alpha=hp["Elastic Net"]["alpha"],
                l1_ratio=hp["Elastic Net"]["l1_ratio"],
                max_iter=10000,
            )
            ridge = cuRidge(alpha=hp["Ridge Regressor"]["alpha"])
            rf = cuRF(
                max_depth=hp["Random Forest"]["max_depth"],
                max_features=hp["Random Forest"]["max_features"],
                min_samples_leaf=hp["Random Forest"]["min_samples_leaf"],
                min_samples_split=hp["Random Forest"]["min_samples_split"],
                n_estimators=hp["Random Forest"]["n_estimators"],
                random_state=args.random_state,
            )
            svr = cuSVR(
                kernel="rbf",
                C=hp["SVR RBF"]["C"],
                epsilon=hp["SVR RBF"]["epsilon"],
                gamma=hp["SVR RBF"]["gamma"],
            )
        except Exception as exc:
            print(f"cuML construction failed; CPU fallback. Reason: {exc}")
            use_cuml = False

    if not use_cuml:
        enet = ElasticNet(
            alpha=hp["Elastic Net"]["alpha"],
            l1_ratio=hp["Elastic Net"]["l1_ratio"],
            max_iter=10000,
            random_state=args.random_state,
        )
        ridge = Ridge(alpha=hp["Ridge Regressor"]["alpha"])
        rf = RandomForestRegressor(
            max_depth=hp["Random Forest"]["max_depth"],
            max_features=hp["Random Forest"]["max_features"],
            min_samples_leaf=hp["Random Forest"]["min_samples_leaf"],
            min_samples_split=hp["Random Forest"]["min_samples_split"],
            n_estimators=hp["Random Forest"]["n_estimators"],
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        svr = SVR(
            kernel="rbf",
            C=hp["SVR RBF"]["C"],
            epsilon=hp["SVR RBF"]["epsilon"],
            gamma=hp["SVR RBF"]["gamma"],
        )

    if use_xgb:
        try:
            XGBRegressor = backends["xgboost"]
            gb = XGBRegressor(
                objective="reg:squarederror",
                learning_rate=hp["Gradient Boosting"]["learning_rate"],
                max_depth=hp["Gradient Boosting"]["max_depth"],
                n_estimators=hp["Gradient Boosting"]["n_estimators"],
                subsample=hp["Gradient Boosting"]["subsample"],
                random_state=args.random_state,
                n_jobs=1,
                tree_method="hist",
                device="cuda",
            )
        except Exception as exc:
            print(f"XGBoost construction failed; CPU fallback. Reason: {exc}")
            use_xgb = False

    if not use_xgb:
        gb = GradientBoostingRegressor(
            learning_rate=hp["Gradient Boosting"]["learning_rate"],
            max_depth=hp["Gradient Boosting"]["max_depth"],
            n_estimators=hp["Gradient Boosting"]["n_estimators"],
            subsample=hp["Gradient Boosting"]["subsample"],
            random_state=args.random_state,
        )

    models = {
        "Elastic Net": enet,
        "Gradient Boosting": gb,
        "Random Forest": rf,
        "Ridge Regressor": ridge,
        "SVR RBF": svr,
    }
    backend_info = {
        "use_gpu_requested": bool(args.use_gpu),
        "gpu_backend": args.gpu_backend,
        "cuml_enabled": bool(use_cuml),
        "xgboost_gpu_enabled": bool(use_xgb),
        "messages": backends["messages"],
    }
    return {m: models[m] for m in MODEL_ORDER}, backend_info


def make_param_grids(args: argparse.Namespace):
    hp = get_hyperparameters(args.species, args.grape_gb_n_estimators)
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
                "epsilon": [0.001, 0.01, 0.03, 0.1, 0.3, 0.5],
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_cv_splitter_and_groups(
    df: pd.DataFrame, group_column: str, n_splits: int, random_state: int
):
    groups = None
    source = "none"
    if group_column.lower() != "none":
        if group_column != "auto":
            if group_column not in df.columns:
                raise ValueError(f"Group column '{group_column}' not found.")
            groups = df[group_column].astype(str).values
            source = group_column
        else:
            for c in ["leaf_id", "Leaf_ID", "leaf", "leafID", "leaf_id_inferred"]:
                if c in df.columns and df[c].notna().all():
                    groups = df[c].astype(str).values
                    source = c
                    break

    if groups is not None:
        if len(np.unique(groups)) < n_splits:
            raise ValueError(
                f"Only {len(np.unique(groups))} unique groups for n_splits={n_splits}."
            )
        return GroupKFold(n_splits=n_splits), groups, "GroupKFold", source

    return (
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        None,
        "KFold",
        source,
    )


def scoring_dict():
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }


def build_metadata(df: pd.DataFrame, target_column: str):
    preferred = [
        "__source_split",
        "__source_csv",
        "__row_id_within_split",
        "species",
        "stage",
        "blue_basename",
        "leaf_id",
        "leaf_id_inferred",
        target_column,
    ]
    return df[[c for c in preferred if c in df.columns]].copy()


def plot_scatter(y_true, y_pred, model_name, eval_name, target_name, out_path, metrics):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    min_v = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
    max_v = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
    pad = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
    lo, hi = min_v - pad, max_v + pad
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.7, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Real {target_name}")
    ax.set_ylabel(f"Estimated {target_name}")
    ax.set_title(f"{model_name} | {eval_name}")
    ax.text(
        0.05,
        0.95,
        f"RMSE = {metrics['RMSE']:.4f}\nMAE = {metrics['MAE']:.4f}\nR² = {metrics['R2']:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_combined_scatter_grid(pred_df, model_names, eval_name, target_name, out_path):
    n = len(model_names)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), squeeze=False
    )
    for ax, model_name in zip(axes.ravel(), model_names):
        sub = pred_df[pred_df["Model"] == model_name]
        y_true = sub["y_true"].to_numpy(float)
        y_pred = sub["y_pred"].to_numpy(float)
        m = regression_metrics(y_true, y_pred)
        min_v = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
        max_v = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
        pad = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
        lo, hi = min_v - pad, max_v + pad
        ax.scatter(y_true, y_pred, s=14, alpha=0.7, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{model_name}\nRMSE={m['RMSE']:.3f}, R²={m['R2']:.3f}")
        ax.set_xlabel(f"Real {target_name}")
        ax.set_ylabel(f"Estimated {target_name}")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(model_names) :]:
        ax.axis("off")
    fig.suptitle(f"{eval_name}: real vs estimated {target_name}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def run_model_evaluation(
    models,
    grids,
    X_dev,
    y_dev,
    X_test,
    y_test,
    y_dev_orig,
    y_test_orig,
    y_scaler,
    dev_meta,
    test_meta,
    args,
    output_dir,
):
    cv, cv_groups, cv_name, group_source = make_cv_splitter_and_groups(
        dev_meta, args.group_column, args.n_splits, args.random_state
    )

    scatter_cv_dir = output_dir / "scatter_cv_svg"
    scatter_cv_dir.mkdir(exist_ok=True, parents=True)
    scatter_test_dir = output_dir / "scatter_test_svg"
    scatter_test_dir.mkdir(exist_ok=True, parents=True)
    grid_dir = output_dir / "grid_search_results"
    if args.grid_search:
        grid_dir.mkdir(exist_ok=True, parents=True)

    cv_rows, test_rows, cv_pred_rows, test_pred_rows = [], [], [], []
    fitted_models, best_params = {}, {}

    for model_name in MODEL_ORDER:
        print("=" * 80)
        print(f"Model: {model_name}")
        estimator = models[model_name]

        if args.grid_search:
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=grids[model_name],
                scoring=scoring_dict(),
                refit=args.grid_refit_metric,
                cv=cv,
                n_jobs=args.n_jobs,
                return_train_score=True,
                verbose=2,
                error_score="raise",
            )
            fit_kwargs = {"groups": cv_groups} if cv_groups is not None else {}
            grid.fit(X_dev, y_dev, **fit_kwargs)
            pd.DataFrame(grid.cv_results_).to_csv(
                grid_dir / f"grid_search_results_{safe_model_filename(model_name)}.csv",
                index=False,
            )
            estimator = grid.best_estimator_
            best_params[model_name] = dict(grid.best_params_)
            print(f"Best params: {grid.best_params_}")
        else:
            best_params[model_name] = "fixed_table_hyperparameters"

        est_cv = clone(estimator)
        if cv_groups is not None:
            y_dev_pred_norm = cross_val_predict(
                est_cv, X_dev, y_dev, cv=cv, groups=cv_groups, n_jobs=args.n_jobs
            )
        else:
            y_dev_pred_norm = cross_val_predict(
                est_cv, X_dev, y_dev, cv=cv, n_jobs=args.n_jobs
            )

        y_dev_pred = inverse_y(y_dev_pred_norm, y_scaler)
        cv_m = regression_metrics(y_dev_orig, y_dev_pred)
        cv_rows.append({"Model": model_name, **cv_m})

        cv_df = dev_meta.copy()
        cv_df["Model"] = model_name
        cv_df["evaluation"] = "5fold_cv"
        cv_df["y_true"] = y_dev_orig
        cv_df["y_pred"] = y_dev_pred
        cv_df["residual"] = y_dev_pred - y_dev_orig
        cv_df["y_true_normalized"] = y_dev
        cv_df["y_pred_normalized"] = np.asarray(y_dev_pred_norm).reshape(-1)
        cv_pred_rows.append(cv_df)

        plot_scatter(
            y_dev_orig,
            y_dev_pred,
            model_name,
            "5-fold CV",
            args.target_column,
            scatter_cv_dir / f"cv_scatter_{safe_model_filename(model_name)}.svg",
            cv_m,
        )

        final = clone(estimator)
        final.fit(X_dev, y_dev)
        fitted_models[model_name] = final

        y_test_pred_norm = final.predict(X_test)
        y_test_pred = inverse_y(y_test_pred_norm, y_scaler)
        test_m = regression_metrics(y_test_orig, y_test_pred)
        test_rows.append({"Model": model_name, **test_m})

        test_df = test_meta.copy()
        test_df["Model"] = model_name
        test_df["evaluation"] = "test"
        test_df["y_true"] = y_test_orig
        test_df["y_pred"] = y_test_pred
        test_df["residual"] = y_test_pred - y_test_orig
        test_df["y_true_normalized"] = y_test
        test_df["y_pred_normalized"] = np.asarray(y_test_pred_norm).reshape(-1)
        test_pred_rows.append(test_df)

        plot_scatter(
            y_test_orig,
            y_test_pred,
            model_name,
            "Test set",
            args.target_column,
            scatter_test_dir / f"test_scatter_{safe_model_filename(model_name)}.svg",
            test_m,
        )

        print("CV metrics:", cv_m)
        print("Test metrics:", test_m)

    cv_metrics = pd.DataFrame(cv_rows).set_index("Model").loc[MODEL_ORDER].reset_index()
    test_metrics = (
        pd.DataFrame(test_rows).set_index("Model").loc[MODEL_ORDER].reset_index()
    )
    cv_pred = pd.concat(cv_pred_rows, ignore_index=True)
    test_pred = pd.concat(test_pred_rows, ignore_index=True)

    plot_combined_scatter_grid(
        cv_pred,
        MODEL_ORDER,
        "5-fold CV",
        args.target_column,
        output_dir / "cv_scatter_all_models.svg",
    )
    plot_combined_scatter_grid(
        test_pred,
        MODEL_ORDER,
        "Test set",
        args.target_column,
        output_dir / "test_scatter_all_models.svg",
    )

    return (
        cv_metrics,
        test_metrics,
        cv_pred,
        test_pred,
        best_params,
        fitted_models,
        {"cv_name": cv_name, "group_source": group_source, "n_splits": args.n_splits},
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    out = expand_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Fused-feature LWC/FMC regression v2")
    print("Features = generated spectra + flattened PROSPECT params + channel stats")
    print("=" * 80)

    train_df = prepare_dataframe(args.train_csv, args, "train")
    val_df = prepare_dataframe(args.val_csv, args, "val")
    test_df = prepare_dataframe(args.test_csv, args, "test")

    if args.include_prospect:
        train_df = merge_params_if_needed(
            train_df,
            args.train_params_csv,
            args.params_column,
            args.merge_keys,
            "train",
            args.blue_basename_column,
        )
        val_df = merge_params_if_needed(
            val_df,
            args.val_params_csv,
            args.params_column,
            args.merge_keys,
            "val",
            args.blue_basename_column,
        )
        test_df = merge_params_if_needed(
            test_df,
            args.test_params_csv,
            args.params_column,
            args.merge_keys,
            "test",
            args.blue_basename_column,
        )

    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    dev_df = maybe_add_canonical_columns(dev_df, args.blue_basename_column)
    test_df = maybe_add_canonical_columns(test_df, args.blue_basename_column)

    X_dev_raw, feature_names, info_dev, audit_dev = build_fused_features(dev_df, args)
    X_test_raw, feature_names_test, info_test, audit_test = build_fused_features(
        test_df, args
    )

    if X_dev_raw.shape[1] != X_test_raw.shape[1]:
        raise ValueError(
            f"Feature mismatch: dev={X_dev_raw.shape[1]}, test={X_test_raw.shape[1]}"
        )
    if feature_names != feature_names_test:
        raise ValueError("Feature names differ between dev and test.")

    y_dev_orig = dev_df[args.target_column].to_numpy(dtype=float)
    y_test_orig = test_df[args.target_column].to_numpy(dtype=float)

    X_dev, y_dev, X_test, y_test, x_scaler, y_scaler = fit_global_normalizers(
        X_dev_raw,
        y_dev_orig,
        X_test_raw,
        y_test_orig,
        args.x_normalization,
        args.y_normalization,
        args.normalization_scope,
    )

    print("Data summary:")
    print(f"  Train rows:      {len(train_df)}")
    print(f"  Val rows:        {len(val_df)}")
    print(f"  Dev rows:        {len(dev_df)}")
    print(f"  Test rows:       {len(test_df)}")
    print(f"  Total features:  {X_dev.shape[1]}")
    print(f"  Feature info:    {info_dev}")
    print(f"  Target dev:      {np.min(y_dev_orig):.4f} to {np.max(y_dev_orig):.4f}")
    print(f"  Target test:     {np.min(y_test_orig):.4f} to {np.max(y_test_orig):.4f}")

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
        "x_scaler": scaler_to_summary(x_scaler, "x_scaler"),
        "y_scaler": scaler_to_summary(y_scaler, "y_scaler"),
    }
    (out / "normalization_summary.json").write_text(
        json.dumps(norm_summary, indent=2, default=str)
    )

    models, backend_info = make_models(args)
    grids = make_param_grids(args) if args.grid_search else None
    if grids is not None:
        (out / "grid_search_param_grids.json").write_text(
            json.dumps(grids, indent=2, default=str)
        )

    dev_meta = build_metadata(dev_df, args.target_column)
    test_meta = build_metadata(test_df, args.target_column)

    (
        cv_metrics,
        test_metrics,
        cv_pred,
        test_pred,
        best_params,
        fitted_models,
        cv_info,
    ) = run_model_evaluation(
        models,
        grids,
        X_dev,
        y_dev,
        X_test,
        y_test,
        y_dev_orig,
        y_test_orig,
        y_scaler,
        dev_meta,
        test_meta,
        args,
        out,
    )

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

    pd.DataFrame(
        [
            {"Model": m, **p}
            if isinstance(p, dict)
            else {"Model": m, "params_source": str(p)}
            for m, p in best_params.items()
        ]
    ).to_csv(out / "selected_best_hyperparameters.csv", index=False)

    cv_pred.to_csv(out / "cv_predictions.csv", index=False)
    test_pred.to_csv(out / "test_predictions.csv", index=False)

    if args.save_models:
        model_dir = out / "fitted_models_joblib"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "models": fitted_models,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_names": feature_names,
            "feature_info": info_dev,
            "channel_suffix": CHANNEL_SUFFIX,
            "channel_order": CHANNEL_ORDER,
            "params_shape_expected": "4x7 flattened to 28 features by default",
            "target_column": args.target_column,
            "normalization_scope": args.normalization_scope,
        }
        joblib.dump(
            artifact,
            model_dir / "fused_feature_regressors_with_global_normalizers.joblib",
        )
        joblib.dump(
            fitted_models, model_dir / "sklearn_regressors_normalized_input.joblib"
        )
        for name, model in fitted_models.items():
            joblib.dump(
                model,
                model_dir / f"{safe_model_filename(name)}_normalized_input.joblib",
            )

    manifest = {
        "train_csv": str(expand_path(args.train_csv)),
        "val_csv": str(expand_path(args.val_csv)),
        "test_csv": str(expand_path(args.test_csv)),
        "output_dir": str(out),
        "target_column": args.target_column,
        "n_features": int(X_dev.shape[1]),
        "feature_info_dev": info_dev,
        "feature_info_test": info_test,
        "blue_basename_rule": "construct channel basenames by replacing final _1 with _2,_3,_4,_5",
        "channel_suffix": CHANNEL_SUFFIX,
        "prospect_rule": "4x7 PROSPECT parameter array flattened to 28 features when --param-feature-mode flatten",
        "normalization_scope": args.normalization_scope,
        "x_normalization": args.x_normalization,
        "y_normalization": args.y_normalization,
        "cv": cv_info,
        "grid_search": bool(args.grid_search),
        "backend_info": backend_info,
        "selected_best_hyperparameters": best_params,
        "outputs": {
            "cv_metrics": str(out / "cv_metrics.csv"),
            "test_metrics": str(out / "test_metrics.csv"),
            "cv_predictions": str(out / "cv_predictions.csv"),
            "test_predictions": str(out / "test_predictions.csv"),
            "feature_names": str(out / "fused_feature_names.csv"),
            "image_stats_audit_dev": str(out / "image_stats_audit_dev.csv"),
            "image_stats_audit_test": str(out / "image_stats_audit_test.csv"),
            "normalization_summary": str(out / "normalization_summary.json"),
            "model_artifact": str(
                out
                / "fitted_models_joblib"
                / "fused_feature_regressors_with_global_normalizers.joblib"
            )
            if args.save_models
            else None,
        },
    }
    (out / "regression_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print("CV metrics:")
    print(cv_metrics.to_string(index=False))
    print("")
    print("Test metrics:")
    print(test_metrics.to_string(index=False))
    print("")
    print(f"Feature names:         {out / 'fused_feature_names.csv'}")
    print(f"Image stats audit dev: {out / 'image_stats_audit_dev.csv'}")
    print(f"Image stats audit test:{out / 'image_stats_audit_test.csv'}")
    print(f"Output directory:      {out}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
