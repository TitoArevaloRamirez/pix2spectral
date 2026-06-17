#!/usr/bin/env python3
"""
Train and evaluate classical regression models for LWC_d/FMC_d inference
using PROSPECT parameters derived from the pix2spectral generator.

Input features
--------------
Instead of spectral reflectance vectors, this script uses generator-derived
PROSPECT parameters, typically from:

    prospect_parameters.csv

created by infer_generate_spectra.py.

Expected parameter column:

    params_json

Typical shapes:
    [7]          -> one full-spectrum PROSPECT parameter vector
    [S, 7]       -> segmented PROSPECT parameters, e.g. [4, 7]
    [S*7]        -> already flattened segmented parameters

Feature modes:
    flatten      -> use all parameters as one vector, e.g. [4,7] -> 28 features
    mean         -> average segmented parameters across segments -> 7 features
    mean_std     -> mean and std across segments -> 14 features
    flatten_mean_std -> flattened parameters plus mean/std summary

Workflow
--------
1. Read train CSV, validation CSV, and test CSV.
2. Merge train + validation into the development set.
3. Parse params_json into feature matrix X.
4. Fit normalization using all leaf samples by default:
       train + validation + test
5. Train on normalized X and normalized y.
6. Run 5-fold CV on train + validation.
7. Fit final models on full train + validation.
8. Evaluate on independent test set.
9. Report metrics and plots in original target units.

Models
------
    - Elastic Net
    - Gradient Boosting Regressor
    - Random Forest Regressor
    - Ridge Regressor
    - Support Vector Regressor with RBF kernel

Example
-------
python train_lwc_regressors_from_prospect_params.py \
    --train-csv ~/Results/pix2spectral_inference/train/prospect_parameters_with_FMC_d.csv \
    --val-csv ~/Results/pix2spectral_inference/val/prospect_parameters_with_FMC_d.csv \
    --test-csv ~/Results/pix2spectral_inference/test/prospect_parameters_with_FMC_d.csv \
    --target-column FMC_d \
    --species Avocado \
    --param-feature-mode flatten \
    --output-dir ~/Results/lwc_regression/avocado_prospect_params \
    --grid_search \
    --grid-size medium \
    --save-models

If your target column is LWC_d, use:

    --target-column LWC_d

Plot update
-----------
The script saves one SVG subplot figure:

    cv_test_scatter_subplots_all_models.svg

The figure has five rows, one row per machine-learning model, and two columns:
5-fold cross-validation and independent test. All subplots share the same x/y
limits and the same tick positions.
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

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.base import clone


# -------------------------------------------------------------------------
# Fixed hyperparameters from the provided table
# -------------------------------------------------------------------------

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

MODEL_ORDER = [
    "Elastic Net",
    "Gradient Boosting",
    "Random Forest",
    "Ridge Regressor",
    "SVR RBF",
]

PROSPECT_PARAM_NAMES = ["N", "Cab", "Car", "Cbrown", "Cw", "Cm", "Ant"]


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train classical regressors for LWC_d/FMC_d estimation using "
            "generator-derived PROSPECT parameters."
        )
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--target-column", default="LWC_d")
    parser.add_argument(
        "--params-column",
        default="params_json",
        help="Column containing PROSPECT parameters as JSON/list string.",
    )
    parser.add_argument(
        "--param-feature-mode",
        choices=["flatten", "mean", "mean_std", "flatten_mean_std"],
        default="flatten",
        help=(
            "How to convert segmented PROSPECT parameters to ML features. "
            "For [S,7] params, flatten gives S*7 features."
        ),
    )

    parser.add_argument("--species", default="Avocado")
    parser.add_argument("--species-column", default=None)
    parser.add_argument("--filter-species", action="store_true")

    parser.add_argument(
        "--group-column",
        default="auto",
        help=(
            "Grouping column for GroupKFold. Use 'auto' to detect leaf_id or "
            "infer it from blue_basename/blue. Use 'none' to force ordinary KFold."
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument(
        "--grid-size", choices=["small", "medium", "wide"], default="medium"
    )
    parser.add_argument(
        "--grid-refit-metric",
        choices=["rmse", "mae", "r2"],
        default="rmse",
        help="Metric used to select best grid-search parameters.",
    )

    parser.add_argument(
        "--x-normalization",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="PROSPECT-parameter feature normalization.",
    )
    parser.add_argument(
        "--y-normalization",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Target normalization.",
    )
    parser.add_argument(
        "--normalization-scope",
        choices=["all_leaf_samples", "development_only"],
        default="all_leaf_samples",
        help=(
            "all_leaf_samples: fit X/y normalizers on train+val+test. "
            "development_only: fit normalizers only on train+val."
        ),
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help=(
            "Try GPU backends. Requires RAPIDS/cuML for ENet/Ridge/RF/SVR and "
            "xgboost for GPU gradient boosting. Falls back to CPU if unavailable."
        ),
    )
    parser.add_argument(
        "--gpu-backend",
        choices=["auto", "none", "cuml", "xgboost"],
        default="auto",
    )

    parser.add_argument(
        "--grape-gb-n-estimators",
        type=int,
        default=400,
        help=(
            "n_estimators for Grape/Vineyard Gradient Boosting. The provided "
            "LaTeX table has an invalid value 0.8 for GB n, so default is 400."
        ),
    )

    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--write-latex", action="store_true")
    parser.add_argument("--drop-na-target", action="store_true")
    parser.add_argument("--max-samples-debug", type=int, default=None)

    return parser.parse_args()


# -------------------------------------------------------------------------
# Generic helpers
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


def find_species_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["species", "Species", "SPECIES"]:
        if col in df.columns:
            return col
    return None


def normalize_species_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def basename_any(path_like: Any) -> str:
    if pd.isna(path_like):
        return ""
    text = str(path_like).strip()
    if not text:
        return ""
    win_name = PureWindowsPath(text).name
    posix_name = PurePosixPath(win_name).name
    return posix_name


def infer_leaf_id_from_text(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    text = basename_any(value).lower()
    patterns = [
        r"leaf[_-]?(\d+)",
        r"leaf(\d{3})d\d",
        r"(\d{3})d\d",
    ]

    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)

    return None


# -------------------------------------------------------------------------
# PROSPECT parameter parsing and feature construction
# -------------------------------------------------------------------------


def parse_params_value(value: Any) -> np.ndarray:
    """
    Parse params_json.

    Supported:
        [7]
        [[...7...], [...7...], ...]
        flattened list [S*7]
        stringified JSON/list
    """
    if isinstance(value, np.ndarray):
        arr = value.astype(float)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
    else:
        if pd.isna(value):
            raise ValueError("Encountered missing params_json value.")

        text = str(value).strip()
        if not text:
            raise ValueError("Encountered empty params_json string.")

        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception as exc:
                raise ValueError(
                    f"Could not parse params_json value: {text[:120]}..."
                ) from exc

        arr = np.asarray(parsed, dtype=float)

    if arr.ndim == 0:
        raise ValueError("Parsed params_json is scalar; expected [7] or [S,7].")

    if not np.isfinite(arr).all():
        raise ValueError("params_json contains non-finite values.")

    return arr


def params_to_feature_vector(
    params: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, List[str], Tuple[int, ...]]:
    """
    Convert one params array into a feature vector and feature names.
    """
    params = np.asarray(params, dtype=float)
    original_shape = tuple(params.shape)

    if params.ndim == 1:
        if params.size == 7:
            mat = params.reshape(1, 7)
        elif params.size % 7 == 0:
            mat = params.reshape(params.size // 7, 7)
        else:
            # Unknown 1D feature vector; allow flatten only.
            if mode != "flatten":
                raise ValueError(
                    f"1D params length {params.size} is not divisible by 7; "
                    f"mode={mode} is not supported."
                )
            feature = params.reshape(-1)
            names = [f"param_{i}" for i in range(feature.size)]
            return feature, names, original_shape

    elif params.ndim == 2:
        if params.shape[1] != 7:
            raise ValueError(
                f"Expected segmented params shape [S,7], got {params.shape}."
            )
        mat = params
    else:
        # Flatten higher dimensions if last dimension is 7.
        if params.shape[-1] == 7:
            mat = params.reshape(-1, 7)
        else:
            if mode != "flatten":
                raise ValueError(
                    f"Unsupported params shape {params.shape}; last dimension is not 7."
                )
            feature = params.reshape(-1)
            names = [f"param_{i}" for i in range(feature.size)]
            return feature, names, original_shape

    n_segments = mat.shape[0]

    flat = mat.reshape(-1)
    flat_names = []
    for seg in range(n_segments):
        for p_name in PROSPECT_PARAM_NAMES:
            flat_names.append(f"seg{seg + 1}_{p_name}")

    mean = mat.mean(axis=0)
    mean_names = [f"mean_{p}" for p in PROSPECT_PARAM_NAMES]

    std = mat.std(axis=0, ddof=0)
    std_names = [f"std_{p}" for p in PROSPECT_PARAM_NAMES]

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


def build_param_feature_matrix(
    df: pd.DataFrame,
    params_column: str,
    feature_mode: str,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, ...]]]:
    if params_column not in df.columns:
        raise ValueError(
            f"CSV does not contain params column '{params_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    vectors = []
    shapes = []
    feature_names_ref = None

    for idx, value in enumerate(df[params_column].values):
        params = parse_params_value(value)
        vec, names, shape = params_to_feature_vector(params, mode=feature_mode)

        if feature_names_ref is None:
            feature_names_ref = names
        else:
            if len(names) != len(feature_names_ref):
                raise ValueError(
                    f"Inconsistent PROSPECT parameter feature lengths. "
                    f"First row has {len(feature_names_ref)} features, row {idx} has {len(names)}. "
                    f"Check that all checkpoints use the same segmented/full generator configuration."
                )

        vectors.append(vec)
        shapes.append(shape)

    X = np.stack(vectors, axis=0).astype(np.float64)
    return X, feature_names_ref or [f"param_{i}" for i in range(X.shape[1])], shapes


# -------------------------------------------------------------------------
# Data loading and metadata
# -------------------------------------------------------------------------


def prepare_dataframe(
    csv_path: str, args: argparse.Namespace, split_name: str
) -> pd.DataFrame:
    path = expand_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{split_name} CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.copy()
    df["__source_split"] = split_name
    df["__source_csv"] = str(path)

    if args.max_samples_debug is not None:
        df = df.head(int(args.max_samples_debug)).copy()

    species_col = args.species_column or find_species_column(df)
    if args.filter_species:
        if species_col is None:
            raise ValueError(
                f"--filter-species requested, but no species column was found in {path}."
            )

        target_species = normalize_species_key(args.species)

        def row_species_key(x):
            s = normalize_species_value(x)
            try:
                return normalize_species_key(s)
            except ValueError:
                return s

        before = len(df)
        df = df[df[species_col].map(row_species_key) == target_species].copy()
        after = len(df)
        if after == 0:
            raise ValueError(
                f"No rows left in {split_name} after species filtering. "
                f"Species column={species_col}, requested={args.species}."
            )
        print(f"{split_name}: species filter kept {after}/{before} rows.")

    if args.target_column not in df.columns:
        raise ValueError(
            f"{split_name} CSV does not contain target column '{args.target_column}'. "
            f"If you are using raw prospect_parameters.csv, first merge the target "
            f"FMC_d/LWC_d into it. Available columns: {list(df.columns)}"
        )

    df[args.target_column] = pd.to_numeric(df[args.target_column], errors="coerce")
    if df[args.target_column].isna().any():
        n_bad = int(df[args.target_column].isna().sum())
        if args.drop_na_target:
            print(
                f"Warning: dropping {n_bad} rows with missing {args.target_column} in {split_name}."
            )
            df = df.dropna(subset=[args.target_column]).copy()
        else:
            bad = df[df[args.target_column].isna()].head(10)
            raise ValueError(
                f"{split_name} CSV contains {n_bad} rows with missing/non-numeric "
                f"target '{args.target_column}'. First bad rows:\n"
                f"{bad.to_string(index=False)}\n"
                "Use --drop-na-target to drop them."
            )

    df = df.reset_index(drop=True)
    df["__row_id_within_split"] = np.arange(len(df), dtype=int)
    return df


def maybe_add_inferred_leaf_id(df: pd.DataFrame) -> pd.DataFrame:
    if "leaf_id" in df.columns:
        return df

    for candidate in ["blue_basename", "blue", "Blue", "filename", "image_name"]:
        if candidate in df.columns:
            inferred = df[candidate].map(infer_leaf_id_from_text)
            if inferred.notna().any():
                df = df.copy()
                df["leaf_id_inferred"] = inferred
                return df

    return df


def add_or_infer_groups(
    df: pd.DataFrame, group_column: str
) -> Tuple[Optional[np.ndarray], str]:
    if group_column.lower() == "none":
        return None, "none"

    if group_column != "auto":
        if group_column not in df.columns:
            raise ValueError(f"Requested group column '{group_column}' not found.")
        return df[group_column].astype(str).values, group_column

    for candidate in ["leaf_id", "Leaf_ID", "leaf", "leafID", "leaf_id_inferred"]:
        if candidate in df.columns:
            values = df[candidate]
            if values.notna().all():
                return values.astype(str).values, candidate

    for candidate in ["blue_basename", "blue", "Blue", "filename", "image_name"]:
        if candidate in df.columns:
            inferred = df[candidate].map(infer_leaf_id_from_text)
            if inferred.notna().all():
                return inferred.astype(str).values, f"inferred_from_{candidate}"

    return None, "none"


def build_metadata(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    preferred = [
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
        "params_shape",
        target_column,
    ]
    cols = [c for c in preferred if c in df.columns]
    return df[cols].copy()


# -------------------------------------------------------------------------
# Global normalization
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
# Model construction
# -------------------------------------------------------------------------


def get_hyperparameters(
    species: str, grape_gb_n_estimators: int
) -> Dict[str, Dict[str, Any]]:
    key = normalize_species_key(species)
    hp = {m: dict(v) for m, v in HYPERPARAMETERS[key].items()}

    if key == "grape":
        hp["Gradient Boosting"]["n_estimators"] = int(grape_gb_n_estimators)

    return hp


def import_optional_gpu_backends(use_gpu: bool, gpu_backend: str) -> Dict[str, Any]:
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
                f"RAPIDS/cuML not available or incomplete; using CPU for ENet/Ridge/RF/SVR. Reason: {exc}"
            )

    if gpu_backend in ["auto", "xgboost"]:
        try:
            from xgboost import XGBRegressor

            backends["xgboost"] = XGBRegressor
            backends["messages"].append("XGBoost detected for GPU gradient boosting.")
        except Exception as exc:
            backends["messages"].append(
                f"XGBoost not available; using CPU sklearn GradientBoosting. Reason: {exc}"
            )

    return backends


def make_models(
    species: str,
    random_state: int,
    n_jobs: int,
    grape_gb_n_estimators: int,
    use_gpu: bool,
    gpu_backend: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hp = get_hyperparameters(species, grape_gb_n_estimators)
    backends = import_optional_gpu_backends(use_gpu, gpu_backend)

    for msg in backends["messages"]:
        print(msg)

    use_cuml = backends["cuml"] is not None
    use_xgb = backends["xgboost"] is not None

    if use_gpu and n_jobs != 1:
        print(
            "Note: GPU backends usually perform best with --n-jobs 1 to avoid "
            "multiple processes competing for the same GPU."
        )

    if use_cuml:
        try:
            cuElasticNet = backends["cuml"]["ElasticNet"]
            cuRidge = backends["cuml"]["Ridge"]
            cuRandomForestRegressor = backends["cuml"]["RandomForestRegressor"]
            cuSVR = backends["cuml"]["SVR"]

            enet = cuElasticNet(
                alpha=hp["Elastic Net"]["alpha"],
                l1_ratio=hp["Elastic Net"]["l1_ratio"],
                max_iter=10000,
            )
            ridge = cuRidge(alpha=hp["Ridge Regressor"]["alpha"])
            rf = cuRandomForestRegressor(
                max_depth=hp["Random Forest"]["max_depth"],
                max_features=hp["Random Forest"]["max_features"],
                min_samples_leaf=hp["Random Forest"]["min_samples_leaf"],
                min_samples_split=hp["Random Forest"]["min_samples_split"],
                n_estimators=hp["Random Forest"]["n_estimators"],
                random_state=random_state,
            )
            svr = cuSVR(
                kernel="rbf",
                C=hp["SVR RBF"]["C"],
                epsilon=hp["SVR RBF"]["epsilon"],
                gamma=hp["SVR RBF"]["gamma"],
            )
        except Exception as exc:
            print(
                f"cuML estimator construction failed; falling back to CPU. Reason: {exc}"
            )
            use_cuml = False

    if not use_cuml:
        enet = ElasticNet(
            alpha=hp["Elastic Net"]["alpha"],
            l1_ratio=hp["Elastic Net"]["l1_ratio"],
            max_iter=10000,
            random_state=random_state,
        )
        ridge = Ridge(alpha=hp["Ridge Regressor"]["alpha"])
        rf = RandomForestRegressor(
            max_depth=hp["Random Forest"]["max_depth"],
            max_features=hp["Random Forest"]["max_features"],
            min_samples_leaf=hp["Random Forest"]["min_samples_leaf"],
            min_samples_split=hp["Random Forest"]["min_samples_split"],
            n_estimators=hp["Random Forest"]["n_estimators"],
            random_state=random_state,
            n_jobs=n_jobs,
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
                random_state=random_state,
                n_jobs=1,
                tree_method="hist",
                device="cuda",
            )
        except Exception as exc:
            print(
                f"XGBoost GPU estimator construction failed; falling back to sklearn GB. Reason: {exc}"
            )
            use_xgb = False

    if not use_xgb:
        gb = GradientBoostingRegressor(
            learning_rate=hp["Gradient Boosting"]["learning_rate"],
            max_depth=hp["Gradient Boosting"]["max_depth"],
            n_estimators=hp["Gradient Boosting"]["n_estimators"],
            subsample=hp["Gradient Boosting"]["subsample"],
            random_state=random_state,
        )

    models = {
        "Elastic Net": enet,
        "Gradient Boosting": gb,
        "Random Forest": rf,
        "Ridge Regressor": ridge,
        "SVR RBF": svr,
    }

    backend_info = {
        "use_gpu_requested": bool(use_gpu),
        "gpu_backend": gpu_backend,
        "cuml_enabled": bool(use_cuml),
        "xgboost_gpu_enabled": bool(use_xgb),
        "messages": backends["messages"],
    }

    return {name: models[name] for name in MODEL_ORDER}, backend_info


# -------------------------------------------------------------------------
# Hyperparameter grids
# -------------------------------------------------------------------------


def make_param_grids(
    species: str,
    grid_size: str,
    grape_gb_n_estimators: int,
) -> Dict[str, Dict[str, List[Any]]]:
    hp = get_hyperparameters(species, grape_gb_n_estimators)

    if grid_size == "small":
        return {
            "Elastic Net": {
                "alpha": sorted(set([hp["Elastic Net"]["alpha"], 0.003, 0.01, 0.03])),
                "l1_ratio": sorted(set([hp["Elastic Net"]["l1_ratio"], 0.1, 0.5])),
            },
            "Ridge Regressor": {
                "alpha": sorted(set([hp["Ridge Regressor"]["alpha"], 0.1, 1.0, 10.0])),
            },
            "SVR RBF": {
                "C": sorted(set([hp["SVR RBF"]["C"], 100.0, 1000.0])),
                "epsilon": sorted(set([hp["SVR RBF"]["epsilon"], 0.01, 0.1, 0.3])),
                "gamma": ["scale"],
            },
            "Random Forest": {
                "n_estimators": sorted(
                    set([hp["Random Forest"]["n_estimators"], 300, 600])
                ),
                "max_depth": [None, 20, 40],
                "max_features": ["sqrt"],
                "min_samples_leaf": [1],
                "min_samples_split": [2],
            },
            "Gradient Boosting": {
                "learning_rate": sorted(
                    set([hp["Gradient Boosting"]["learning_rate"], 0.03, 0.05])
                ),
                "max_depth": sorted(set([hp["Gradient Boosting"]["max_depth"], 2, 3])),
                "n_estimators": sorted(
                    set([hp["Gradient Boosting"]["n_estimators"], 200, 400])
                ),
                "subsample": sorted(
                    set([hp["Gradient Boosting"]["subsample"], 0.8, 1.0])
                ),
            },
        }

    if grid_size == "wide":
        return {
            "Elastic Net": {
                "alpha": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                "l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "Ridge Regressor": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            },
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
        "Ridge Regressor": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
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


# -------------------------------------------------------------------------
# Metrics and plotting
# -------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def safe_model_filename(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    eval_name: str,
    target_name: str,
    out_path: Path,
    metrics: Dict[str, float],
) -> None:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    min_v = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
    max_v = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
    # pad = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
    # lo, hi = min_v - pad, max_v + pad

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.70, edgecolors="none")
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5)
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel(f"Real {target_name}")
    ax.set_ylabel(f"Estimated {target_name}")
    ax.set_title(f"{model_name} | {eval_name}")

    text = f"RMSE = {metrics['RMSE']:.4f}\nMAE = {metrics['MAE']:.4f}\nR² = {metrics['R2']:.4f}"
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_combined_scatter_grid(
    predictions_df: pd.DataFrame,
    model_names: Sequence[str],
    eval_name: str,
    target_name: str,
    out_path: Path,
) -> None:
    n = len(model_names)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), squeeze=False
    )

    for ax, model_name in zip(axes.ravel(), model_names):
        sub = predictions_df[predictions_df["Model"] == model_name]
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        metrics = regression_metrics(y_true, y_pred)

        min_v = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
        max_v = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
        # pad = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
        # lo, hi = min_v - pad, max_v + pad

        ax.scatter(y_true, y_pred, s=14, alpha=0.70, edgecolors="none")
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.2)
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_title(
            f"{model_name}\nRMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f}"
        )
        ax.set_xlabel(f"Real {target_name}")
        ax.set_ylabel(f"Estimated {target_name}")
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(model_names) :]:
        ax.axis("off")

    fig.suptitle(f"{eval_name}: real vs estimated {target_name}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_cv_test_scatter_subplots(
    cv_predictions_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    model_names: Sequence[str],
    target_name: str,
    out_path: Path,
) -> None:
    """
    Save one SVG figure with five rows and two columns.

    Rows:
        one row per machine-learning model.

    Columns:
        column 1 = 5-fold cross-validation predictions
        column 2 = independent test-set predictions

    All subplots use the same x/y limits and the same tick positions.
    """
    all_values = np.concatenate(
        [
            cv_predictions_df["y_true"].to_numpy(dtype=float),
            cv_predictions_df["y_pred"].to_numpy(dtype=float),
            test_predictions_df["y_true"].to_numpy(dtype=float),
            test_predictions_df["y_pred"].to_numpy(dtype=float),
        ]
    )
    all_values = all_values[np.isfinite(all_values)]

    if all_values.size == 0:
        raise ValueError("No finite values available for shared subplot axis scaling.")

    min_v = float(np.min(all_values))
    max_v = float(np.max(all_values))
    pad = 0.05 * (max_v - min_v) if max_v > min_v else 1.0
    lo, hi = min_v - pad, max_v + pad

    shared_ticks = np.linspace(lo, hi, 6)

    nrows = len(model_names)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(10.0, 3.0 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    columns = [
        ("5-fold CV", cv_predictions_df),
        ("Independent test", test_predictions_df),
    ]

    for row_idx, model_name in enumerate(model_names):
        for col_idx, (eval_name, pred_df) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            sub = pred_df[pred_df["Model"] == model_name]

            y_true = sub["y_true"].to_numpy(dtype=float)
            y_pred = sub["y_pred"].to_numpy(dtype=float)
            metrics = regression_metrics(y_true, y_pred)

            ax.scatter(y_true, y_pred, s=16, alpha=0.70, edgecolors="none")
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xticks(shared_ticks)
            ax.set_yticks(shared_ticks)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)

            if row_idx == 0:
                ax.set_title(eval_name)

            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nEstimated {target_name}")

            if row_idx == nrows - 1:
                ax.set_xlabel(f"Real {target_name}")

            ax.text(
                0.05,
                0.95,
                f"RMSE={metrics['RMSE']:.3f}\nMAE={metrics['MAE']:.3f}\nR²={metrics['R2']:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.12),
            )

    fig.suptitle(
        f"Real vs estimated {target_name}: cross-validation and test set",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# CV/grid evaluation
# -------------------------------------------------------------------------


def make_cv_splitter_and_groups(
    groups: Optional[np.ndarray],
    n_splits: int,
    random_state: int,
):
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"GroupKFold requested but only {len(unique_groups)} unique groups "
                f"are available for n_splits={n_splits}."
            )
        return GroupKFold(n_splits=n_splits), groups, "GroupKFold"

    return (
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        None,
        "KFold",
    )


def scoring_dict() -> Dict[str, str]:
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }


def run_grid_search_for_model(
    model_name: str,
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    X_dev_norm: np.ndarray,
    y_dev_norm: np.ndarray,
    cv,
    groups: Optional[np.ndarray],
    refit_metric: str,
    n_jobs: int,
    output_dir: Path,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    print("-" * 80)
    print(f"Grid search: {model_name}")
    print(f"Parameter grid: {param_grid}")
    print("-" * 80)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring_dict(),
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=2,
        error_score="raise",
    )

    fit_kwargs = {}
    if groups is not None:
        fit_kwargs["groups"] = groups

    grid.fit(X_dev_norm, y_dev_norm, **fit_kwargs)

    results_df = pd.DataFrame(grid.cv_results_)
    results_path = (
        output_dir / f"grid_search_results_{safe_model_filename(model_name)}.csv"
    )
    results_df.to_csv(results_path, index=False)

    print(f"Best params for {model_name}: {grid.best_params_}")
    print(f"Best CV {refit_metric} on normalized target: {grid.best_score_}")

    return grid.best_estimator_, dict(grid.best_params_), results_df


def run_model_evaluation(
    models: Dict[str, Any],
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]],
    X_dev_norm: np.ndarray,
    y_dev_norm: np.ndarray,
    X_test_norm: np.ndarray,
    y_test_norm: np.ndarray,
    y_dev_original: np.ndarray,
    y_test_original: np.ndarray,
    y_scaler,
    dev_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    groups_dev: Optional[np.ndarray],
    n_splits: int,
    random_state: int,
    n_jobs: int,
    output_dir: Path,
    target_column: str,
    save_models: bool,
    grid_search: bool,
    grid_refit_metric: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    cv, cv_groups, cv_name = make_cv_splitter_and_groups(
        groups_dev, n_splits, random_state
    )

    cv_metric_rows = []
    test_metric_rows = []
    cv_pred_rows = []
    test_pred_rows = []
    fitted_models = {}
    best_params_by_model = {}

    scatter_cv_dir = output_dir / "scatter_cv_svg"
    scatter_test_dir = output_dir / "scatter_test_svg"
    scatter_cv_dir.mkdir(parents=True, exist_ok=True)
    scatter_test_dir.mkdir(parents=True, exist_ok=True)

    grid_dir = None
    if grid_search:
        grid_dir = output_dir / "grid_search_results"
        grid_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_ORDER:
        print("=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)

        estimator = models[model_name]

        if grid_search:
            estimator, best_params, _ = run_grid_search_for_model(
                model_name=model_name,
                estimator=estimator,
                param_grid=param_grids[model_name],
                X_dev_norm=X_dev_norm,
                y_dev_norm=y_dev_norm,
                cv=cv,
                groups=cv_groups,
                refit_metric=grid_refit_metric,
                n_jobs=n_jobs,
                output_dir=grid_dir,
            )
            best_params_by_model[model_name] = best_params
        else:
            best_params_by_model[model_name] = "fixed_table_hyperparameters"

        estimator_for_cv = clone(estimator)
        if cv_groups is not None:
            y_dev_pred_norm = cross_val_predict(
                estimator_for_cv,
                X_dev_norm,
                y_dev_norm,
                cv=cv,
                groups=cv_groups,
                n_jobs=n_jobs,
            )
        else:
            y_dev_pred_norm = cross_val_predict(
                estimator_for_cv,
                X_dev_norm,
                y_dev_norm,
                cv=cv,
                n_jobs=n_jobs,
            )

        y_dev_pred_original = inverse_y(y_dev_pred_norm, y_scaler)
        cv_metrics = regression_metrics(y_dev_original, y_dev_pred_original)
        cv_metric_rows.append({"Model": model_name, **cv_metrics})

        cv_pred_df = dev_meta.copy()
        cv_pred_df["Model"] = model_name
        cv_pred_df["evaluation"] = "5fold_cv"
        cv_pred_df["y_true"] = y_dev_original
        cv_pred_df["y_pred"] = y_dev_pred_original
        cv_pred_df["residual"] = y_dev_pred_original - y_dev_original
        cv_pred_df["y_true_normalized"] = y_dev_norm
        cv_pred_df["y_pred_normalized"] = np.asarray(
            y_dev_pred_norm, dtype=float
        ).reshape(-1)
        cv_pred_rows.append(cv_pred_df)

        plot_scatter(
            y_true=y_dev_original,
            y_pred=y_dev_pred_original,
            model_name=model_name,
            eval_name="5-fold CV",
            target_name=target_column,
            out_path=scatter_cv_dir
            / f"cv_scatter_{safe_model_filename(model_name)}.svg",
            metrics=cv_metrics,
        )

        final_estimator = clone(estimator)
        final_estimator.fit(X_dev_norm, y_dev_norm)
        fitted_models[model_name] = final_estimator

        y_test_pred_norm = final_estimator.predict(X_test_norm)
        y_test_pred_original = inverse_y(y_test_pred_norm, y_scaler)
        test_metrics = regression_metrics(y_test_original, y_test_pred_original)
        test_metric_rows.append({"Model": model_name, **test_metrics})

        test_pred_df = test_meta.copy()
        test_pred_df["Model"] = model_name
        test_pred_df["evaluation"] = "test"
        test_pred_df["y_true"] = y_test_original
        test_pred_df["y_pred"] = y_test_pred_original
        test_pred_df["residual"] = y_test_pred_original - y_test_original
        test_pred_df["y_true_normalized"] = y_test_norm
        test_pred_df["y_pred_normalized"] = np.asarray(
            y_test_pred_norm, dtype=float
        ).reshape(-1)
        test_pred_rows.append(test_pred_df)

        plot_scatter(
            y_true=y_test_original,
            y_pred=y_test_pred_original,
            model_name=model_name,
            eval_name="Test set",
            target_name=target_column,
            out_path=scatter_test_dir
            / f"test_scatter_{safe_model_filename(model_name)}.svg",
            metrics=test_metrics,
        )

        print("Selected params:", best_params_by_model[model_name])
        print("5-fold CV metrics in original units:", cv_metrics)
        print("Test metrics in original units:     ", test_metrics)

    cv_metrics_df = (
        pd.DataFrame(cv_metric_rows).set_index("Model").loc[MODEL_ORDER].reset_index()
    )
    test_metrics_df = (
        pd.DataFrame(test_metric_rows).set_index("Model").loc[MODEL_ORDER].reset_index()
    )
    cv_predictions_df = pd.concat(cv_pred_rows, axis=0, ignore_index=True)
    test_predictions_df = pd.concat(test_pred_rows, axis=0, ignore_index=True)

    plot_combined_scatter_grid(
        predictions_df=cv_predictions_df,
        model_names=MODEL_ORDER,
        eval_name="5-fold CV",
        target_name=target_column,
        out_path=output_dir / "cv_scatter_all_models.svg",
    )
    plot_combined_scatter_grid(
        predictions_df=test_predictions_df,
        model_names=MODEL_ORDER,
        eval_name="Test set",
        target_name=target_column,
        out_path=output_dir / "test_scatter_all_models.svg",
    )

    plot_cv_test_scatter_subplots(
        cv_predictions_df=cv_predictions_df,
        test_predictions_df=test_predictions_df,
        model_names=MODEL_ORDER,
        target_name=target_column,
        out_path=output_dir / "cv_test_scatter_subplots_all_models.svg",
    )

    if save_models:
        model_dir = output_dir / "fitted_models_joblib"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            fitted_models, model_dir / "sklearn_regressors_normalized_input.joblib"
        )
        for model_name, estimator in fitted_models.items():
            joblib.dump(
                estimator,
                model_dir
                / f"{safe_model_filename(model_name)}_normalized_input.joblib",
            )

    cv_info = {
        "cv_name": cv_name,
        "n_splits": n_splits,
        "grid_search": bool(grid_search),
        "grid_refit_metric": grid_refit_metric if grid_search else None,
        "training_target_units": "normalized",
        "predictions_are_original_units": True,
        "metrics_are_original_units": True,
    }

    return (
        cv_metrics_df,
        test_metrics_df,
        cv_predictions_df,
        test_predictions_df,
        cv_info,
        best_params_by_model,
        fitted_models,
    )


def save_metrics_tables(
    cv_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    output_dir: Path,
    write_latex: bool,
) -> None:
    cv_path = output_dir / "cv_metrics.csv"
    test_path = output_dir / "test_metrics.csv"

    cv_metrics_df.to_csv(cv_path, index=False)
    test_metrics_df.to_csv(test_path, index=False)

    (output_dir / "cv_metrics.md").write_text(cv_metrics_df.to_markdown(index=False))
    (output_dir / "test_metrics.md").write_text(
        test_metrics_df.to_markdown(index=False)
    )

    if write_latex:
        (output_dir / "cv_metrics.tex").write_text(
            cv_metrics_df.to_latex(index=False, float_format="%.4f")
        )
        (output_dir / "test_metrics.tex").write_text(
            test_metrics_df.to_latex(index=False, float_format="%.4f")
        )

    print(f"Saved CV metrics:   {cv_path}")
    print(f"Saved test metrics: {test_path}")


def best_params_to_dataframe(best_params_by_model: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        params = best_params_by_model.get(model_name, {})
        if isinstance(params, dict):
            row = {"Model": model_name, **params}
        else:
            row = {"Model": model_name, "params_source": str(params)}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LWC/FMC regression from generator-derived PROSPECT parameters")
    print("=" * 80)
    print(f"Train CSV:             {expand_path(args.train_csv)}")
    print(f"Val CSV:               {expand_path(args.val_csv)}")
    print(f"Test CSV:              {expand_path(args.test_csv)}")
    print(f"Output dir:            {output_dir}")
    print(f"Target column:         {args.target_column}")
    print(f"Params column:         {args.params_column}")
    print(f"Param feature mode:    {args.param_feature_mode}")
    print(f"Species:               {args.species}")
    print(f"Grid search:           {args.grid_search}")
    print(f"X normalization:       {args.x_normalization}")
    print(f"Y normalization:       {args.y_normalization}")
    print(f"Normalization scope:   {args.normalization_scope}")
    print(f"Use GPU:               {args.use_gpu}")
    print("=" * 80)

    if args.normalization_scope == "all_leaf_samples":
        print(
            "Normalization will be fitted using all leaf samples: train + validation + test."
        )

    if args.use_gpu and args.n_jobs != 1:
        print(
            "Recommendation: with --use-gpu, set --n-jobs 1 to avoid launching "
            "multiple GPU jobs simultaneously. Continuing with current n_jobs."
        )

    train_df = prepare_dataframe(args.train_csv, args, "train")
    val_df = prepare_dataframe(args.val_csv, args, "val")
    test_df = prepare_dataframe(args.test_csv, args, "test")

    dev_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    dev_df = maybe_add_inferred_leaf_id(dev_df)
    test_df = maybe_add_inferred_leaf_id(test_df)

    X_dev_raw, feature_names, param_shapes_dev = build_param_feature_matrix(
        dev_df,
        params_column=args.params_column,
        feature_mode=args.param_feature_mode,
    )
    X_test_raw, feature_names_test, param_shapes_test = build_param_feature_matrix(
        test_df,
        params_column=args.params_column,
        feature_mode=args.param_feature_mode,
    )

    if X_dev_raw.shape[1] != X_test_raw.shape[1]:
        raise ValueError(
            f"Feature length mismatch: train+val has {X_dev_raw.shape[1]} features, "
            f"test has {X_test_raw.shape[1]} features."
        )

    if feature_names != feature_names_test:
        raise ValueError("Feature names differ between train+val and test.")

    y_dev_original = dev_df[args.target_column].to_numpy(dtype=float)
    y_test_original = test_df[args.target_column].to_numpy(dtype=float)

    (
        X_dev_norm,
        y_dev_norm,
        X_test_norm,
        y_test_norm,
        x_scaler,
        y_scaler,
    ) = fit_global_normalizers(
        X_dev=X_dev_raw,
        y_dev=y_dev_original,
        X_test=X_test_raw,
        y_test=y_test_original,
        x_normalization=args.x_normalization,
        y_normalization=args.y_normalization,
        normalization_scope=args.normalization_scope,
    )

    groups_dev, group_source = add_or_infer_groups(dev_df, args.group_column)

    param_shape_counts = pd.Series(
        [str(s) for s in param_shapes_dev + param_shapes_test]
    ).value_counts()

    print("Data summary:")
    print(f"  Train rows:          {len(train_df)}")
    print(f"  Validation rows:     {len(val_df)}")
    print(f"  Train+val rows:      {len(dev_df)}")
    print(f"  Test rows:           {len(test_df)}")
    print(f"  Number of features:  {X_dev_norm.shape[1]}")
    print(f"  Feature mode:        {args.param_feature_mode}")
    print(f"  Parameter shapes:    {param_shape_counts.to_dict()}")
    print(
        f"  Target min/max dev:  {np.min(y_dev_original):.6f} / {np.max(y_dev_original):.6f}"
    )
    print(
        f"  Target min/max test: {np.min(y_test_original):.6f} / {np.max(y_test_original):.6f}"
    )
    print(f"  Normalized y dev:    {np.min(y_dev_norm):.6f} / {np.max(y_dev_norm):.6f}")
    print(
        f"  Normalized y test:   {np.min(y_test_norm):.6f} / {np.max(y_test_norm):.6f}"
    )
    print(f"  Group source:        {group_source}")
    if groups_dev is not None:
        print(f"  Unique groups:       {len(np.unique(groups_dev))}")

    feature_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names), dtype=int),
            "feature_name": feature_names,
        }
    )
    feature_df.to_csv(output_dir / "prospect_parameter_features.csv", index=False)

    normalization_summary = {
        "scope": args.normalization_scope,
        "x_normalization": args.x_normalization,
        "y_normalization": args.y_normalization,
        "x_scaler": scaler_to_summary(x_scaler, "x_scaler"),
        "y_scaler": scaler_to_summary(y_scaler, "y_scaler"),
    }
    (output_dir / "normalization_summary.json").write_text(
        json.dumps(normalization_summary, indent=2, default=str)
    )

    models, backend_info = make_models(
        species=args.species,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        grape_gb_n_estimators=args.grape_gb_n_estimators,
        use_gpu=args.use_gpu,
        gpu_backend=args.gpu_backend,
    )

    param_grids = None
    if args.grid_search:
        param_grids = make_param_grids(
            species=args.species,
            grid_size=args.grid_size,
            grape_gb_n_estimators=args.grape_gb_n_estimators,
        )
        (output_dir / "grid_search_param_grids.json").write_text(
            json.dumps(param_grids, indent=2, default=str)
        )

    fixed_hyperparams = get_hyperparameters(args.species, args.grape_gb_n_estimators)
    (output_dir / "fixed_table_hyperparameters.json").write_text(
        json.dumps(fixed_hyperparams, indent=2, default=str)
    )

    (
        cv_metrics_df,
        test_metrics_df,
        cv_pred_df,
        test_pred_df,
        cv_info,
        best_params_by_model,
        fitted_models,
    ) = run_model_evaluation(
        models=models,
        param_grids=param_grids,
        X_dev_norm=X_dev_norm,
        y_dev_norm=y_dev_norm,
        X_test_norm=X_test_norm,
        y_test_norm=y_test_norm,
        y_dev_original=y_dev_original,
        y_test_original=y_test_original,
        y_scaler=y_scaler,
        dev_meta=build_metadata(dev_df, args.target_column),
        test_meta=build_metadata(test_df, args.target_column),
        groups_dev=groups_dev,
        n_splits=args.n_splits,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        output_dir=output_dir,
        target_column=args.target_column,
        save_models=args.save_models,
        grid_search=args.grid_search,
        grid_refit_metric=args.grid_refit_metric,
    )

    if args.save_models:
        model_dir = output_dir / "fitted_models_joblib"
        full_artifact = {
            "models": fitted_models,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_names": feature_names,
            "target_column": args.target_column,
            "params_column": args.params_column,
            "param_feature_mode": args.param_feature_mode,
            "normalization_scope": args.normalization_scope,
            "x_normalization": args.x_normalization,
            "y_normalization": args.y_normalization,
            "note": (
                "For raw PROSPECT-parameter input: X_norm = x_scaler.transform(X_raw) "
                "if x_scaler is not None. Then y_norm = model.predict(X_norm). "
                "Finally y_original = y_scaler.inverse_transform(y_norm.reshape(-1,1)) "
                "if y_scaler is not None."
            ),
        }
        joblib.dump(
            full_artifact,
            model_dir / "prospect_param_regressors_with_global_normalizers.joblib",
        )

    save_metrics_tables(
        cv_metrics_df=cv_metrics_df,
        test_metrics_df=test_metrics_df,
        output_dir=output_dir,
        write_latex=args.write_latex,
    )

    best_params_df = best_params_to_dataframe(best_params_by_model)
    best_params_df.to_csv(output_dir / "selected_best_hyperparameters.csv", index=False)

    cv_pred_path = output_dir / "cv_predictions.csv"
    test_pred_path = output_dir / "test_predictions.csv"
    cv_pred_df.to_csv(cv_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)

    manifest = {
        "train_csv": str(expand_path(args.train_csv)),
        "val_csv": str(expand_path(args.val_csv)),
        "test_csv": str(expand_path(args.test_csv)),
        "output_dir": str(output_dir),
        "target_column": args.target_column,
        "params_column": args.params_column,
        "param_feature_mode": args.param_feature_mode,
        "n_features": int(X_dev_norm.shape[1]),
        "feature_names_csv": str(output_dir / "prospect_parameter_features.csv"),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "species": args.species,
        "cv": cv_info,
        "group_source": group_source,
        "grid_search": bool(args.grid_search),
        "grid_size": args.grid_size if args.grid_search else None,
        "grid_refit_metric": args.grid_refit_metric if args.grid_search else None,
        "normalization_scope": args.normalization_scope,
        "x_normalization": args.x_normalization,
        "y_normalization": args.y_normalization,
        "metrics_and_predictions_units": "original_target_units",
        "plots_units": "original_target_units",
        "backend_info": backend_info,
        "fixed_table_hyperparameters": fixed_hyperparams,
        "selected_best_hyperparameters": best_params_by_model,
        "outputs": {
            "cv_metrics": str(output_dir / "cv_metrics.csv"),
            "test_metrics": str(output_dir / "test_metrics.csv"),
            "cv_predictions": str(cv_pred_path),
            "test_predictions": str(test_pred_path),
            "selected_best_hyperparameters": str(
                output_dir / "selected_best_hyperparameters.csv"
            ),
            "prospect_parameter_features": str(
                output_dir / "prospect_parameter_features.csv"
            ),
            "normalization_summary": str(output_dir / "normalization_summary.json"),
            "cv_scatter_all_models": str(output_dir / "cv_scatter_all_models.svg"),
            "test_scatter_all_models": str(output_dir / "test_scatter_all_models.svg"),
            "cv_test_scatter_subplots_all_models": str(
                output_dir / "cv_test_scatter_subplots_all_models.svg"
            ),
            "scatter_cv_dir": str(output_dir / "scatter_cv_svg"),
            "scatter_test_dir": str(output_dir / "scatter_test_svg"),
            "models": str(output_dir / "fitted_models_joblib")
            if args.save_models
            else None,
        },
    }
    (output_dir / "regression_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print("CV metrics, original target units:")
    print(cv_metrics_df.to_string(index=False))
    print("")
    print("Test metrics, original target units:")
    print(test_metrics_df.to_string(index=False))
    print("")
    print("Selected hyperparameters:")
    print(best_params_df.to_string(index=False))
    print("")
    print(
        f"Feature list:                {output_dir / 'prospect_parameter_features.csv'}"
    )
    print(f"Normalization summary:       {output_dir / 'normalization_summary.json'}")
    print(f"CV predictions:              {cv_pred_path}")
    print(f"Test predictions:            {test_pred_path}")
    print(
        f"5x2 subplot SVG:             {output_dir / 'cv_test_scatter_subplots_all_models.svg'}"
    )
    print(f"Output directory:            {output_dir}")
    if args.save_models:
        print(f"Model checkpoints:           {output_dir / 'fitted_models_joblib'}")
        print(
            f"Global normalizer artifact:  {output_dir / 'fitted_models_joblib' / 'prospect_param_regressors_with_global_normalizers.joblib'}"
        )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
