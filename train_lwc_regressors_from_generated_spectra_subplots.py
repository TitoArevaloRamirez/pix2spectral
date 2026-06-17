#!/usr/bin/env python3
"""
Train and evaluate classical regression models for LWC_d/FMC_d inference
from generated spectra.

Models:
    - Elastic Net
    - Gradient Boosting Regressor
    - Random Forest Regressor
    - Ridge Regressor
    - Support Vector Regressor with RBF kernel

Workflow:
    1. Read train CSV, validation CSV, and test CSV.
    2. Merge train + validation into a development set.
    3. Parse spectrum vectors from generated_spectrum_json, spectral, or wl_* columns.
    4. Run 5-fold CV on the development set.
       - Uses GroupKFold if a group/leaf_id column can be found or inferred.
       - Otherwise uses ordinary KFold.
    5. Fit each model on the full development set.
    6. Evaluate final models on the independent test set.
    7. Save:
       - cv_metrics.csv
       - test_metrics.csv
       - cv_predictions.csv
       - test_predictions.csv
       - per-model SVG scatter plots
       - combined SVG scatter plots
       - fitted sklearn models as joblib, optionally

Example
-------
python train_lwc_regressors_from_generated_spectra.py \
    --train-csv ~/Results/pix2spectral_inference/train/generated_spectra_with_FMC_d.csv \
    --val-csv ~/Results/pix2spectral_inference/val/generated_spectra_with_FMC_d.csv \
    --test-csv ~/Results/pix2spectral_inference/test/generated_spectra_with_FMC_d.csv \
    --target-column FMC_d \
    --species Avocado \
    --output-dir ~/Results/lwc_regression/avocado_generated_spectra

If your target column is LWC_d:

python train_lwc_regressors_from_generated_spectra.py \
    --train-csv train_generated_with_LWC_d.csv \
    --val-csv val_generated_with_LWC_d.csv \
    --test-csv test_generated_with_LWC_d.csv \
    --target-column LWC_d \
    --species Avocado \
    --output-dir ~/Results/lwc_regression/avocado_LWC_d

Plot update
-----------
The script saves one SVG subplot figure:

    cv_test_scatter_subplots_all_models.svg

The figure has five rows, one row per machine-learning model, and two columns:
5-fold cross-validation and independent test. All subplots share the same x/y
limits and the same tick positions.
"""

from __future__ import annotations

import os
import argparse
import ast
import json
import math
import re
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# -------------------------------------------------------------------------
# Hyperparameters from the provided table
# -------------------------------------------------------------------------

# Notes:
# - Species aliases are handled by normalize_species_key().
# - The provided LaTeX table reports GB "n" for Grape as 0.8, which cannot be
#   n_estimators. This script uses 400 by default for Grape/Vineyard GB and
#   exposes --grape-gb-n-estimators to override it.
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


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train classical regressors for LWC_d/FMC_d estimation from generated spectra."
        )
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--target-column",
        default="LWC_d",
        help="Target variable column. Use LWC_d or FMC_d depending on your CSV.",
    )
    parser.add_argument(
        "--spectrum-column",
        default="generated_spectrum_json",
        help=(
            "Column containing spectrum vectors. If not found, the script can "
            "use wl_* columns or --fallback-spectrum-column."
        ),
    )
    parser.add_argument(
        "--fallback-spectrum-column",
        default="spectral",
        help="Fallback spectrum column if --spectrum-column is not present.",
    )

    parser.add_argument(
        "--species",
        default="Avocado",
        help="Species used to select hyperparameters: Avocado/Avo, Olive, Grape/Vineyard.",
    )
    parser.add_argument(
        "--species-column",
        default=None,
        help="Optional species column for filtering. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--filter-species",
        action="store_true",
        help="Filter rows by --species if a species column is present.",
    )

    parser.add_argument(
        "--group-column",
        default="auto",
        help=(
            "Grouping column for GroupKFold. Use 'auto' to detect leaf_id or "
            "infer it from blue_basename/blue. Use 'none' to force ordinary KFold."
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds. Default: 5.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="n_jobs for RandomForest and cross_val_predict where supported.",
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

    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save final fitted sklearn models as joblib.",
    )
    parser.add_argument(
        "--write-latex",
        action="store_true",
        help="Also write cv_metrics.tex and test_metrics.tex.",
    )
    parser.add_argument(
        "--drop-na-target",
        action="store_true",
        help="Drop rows with missing target values instead of failing.",
    )
    parser.add_argument(
        "--max-samples-debug",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Data parsing helpers
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
    """
    Infer leaf id from names such as:
        leaf003d2_1.tif
        leaf_003_...
        something_leaf003...
    Returns a string group id.
    """
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


def parse_spectrum_value(value: Any) -> np.ndarray:
    """
    Parse one spectrum vector from a CSV cell.

    Supported formats:
        - JSON list: "[0.1, 0.2, ...]"
        - Python list-like string
        - whitespace/comma/semicolon separated string
        - list/tuple/np.ndarray
    """
    if isinstance(value, np.ndarray):
        arr = value.astype(float).reshape(-1)
        return arr

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)

    if pd.isna(value):
        raise ValueError("Encountered missing spectrum value.")

    text = str(value).strip()
    if not text:
        raise ValueError("Encountered empty spectrum string.")

    # JSON first.
    try:
        parsed = json.loads(text)
        return np.asarray(parsed, dtype=float).reshape(-1)
    except Exception:
        pass

    # Python literal list fallback.
    try:
        parsed = ast.literal_eval(text)
        return np.asarray(parsed, dtype=float).reshape(-1)
    except Exception:
        pass

    # Generic numeric string fallback.
    cleaned = text.strip()
    cleaned = cleaned.strip("[]()")
    cleaned = cleaned.replace(";", ",")
    if "," in cleaned:
        arr = np.fromstring(cleaned, sep=",", dtype=float)
    else:
        arr = np.fromstring(cleaned, sep=" ", dtype=float)

    if arr.size == 0:
        raise ValueError(f"Could not parse spectrum string: {text[:120]}...")

    return arr.reshape(-1)


def find_wavelength_columns(df: pd.DataFrame) -> List[str]:
    wl_cols = [c for c in df.columns if str(c).startswith("wl_")]

    def wl_key(col: str) -> float:
        raw = str(col)[3:]
        try:
            return float(raw)
        except ValueError:
            return math.inf

    return sorted(wl_cols, key=wl_key)


def build_feature_matrix(
    df: pd.DataFrame,
    spectrum_column: str,
    fallback_spectrum_column: str,
) -> Tuple[np.ndarray, List[str], str]:
    """
    Return X, feature_names, feature_source.
    """
    if spectrum_column in df.columns:
        spectra = [parse_spectrum_value(v) for v in df[spectrum_column].values]
        source = spectrum_column
    elif fallback_spectrum_column in df.columns:
        spectra = [parse_spectrum_value(v) for v in df[fallback_spectrum_column].values]
        source = fallback_spectrum_column
    else:
        wl_cols = find_wavelength_columns(df)
        if not wl_cols:
            raise ValueError(
                f"No spectrum source found. Tried columns '{spectrum_column}', "
                f"'{fallback_spectrum_column}', and wl_* columns."
            )
        X = df[wl_cols].to_numpy(dtype=float)
        return X, wl_cols, "wl_columns"

    lengths = [len(s) for s in spectra]
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        counts = pd.Series(lengths).value_counts().sort_index()
        raise ValueError(f"Spectra have inconsistent lengths:\n{counts.to_string()}")

    X = np.stack(spectra, axis=0).astype(np.float64)
    feature_names = [f"lambda_{i}" for i in range(X.shape[1])]
    return X, feature_names, source


def prepare_dataframe(
    csv_path: str,
    args: argparse.Namespace,
    split_name: str,
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

    # Optional species filtering.
    species_col = args.species_column or find_species_column(df)
    if args.filter_species:
        if species_col is None:
            raise ValueError(
                f"--filter-species was requested, but no species column was found in {path}."
            )
        target_species = normalize_species_key(args.species)

        # Map only known aliases where possible, otherwise compare lowercase.
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
            f"Available columns: {list(df.columns)}"
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


def add_or_infer_groups(
    df: pd.DataFrame, group_column: str
) -> Tuple[Optional[np.ndarray], str]:
    """
    Return groups and group_source. If no groups are available, return None.
    """
    if group_column.lower() == "none":
        return None, "none"

    if group_column != "auto":
        if group_column not in df.columns:
            raise ValueError(f"Requested group column '{group_column}' not found.")
        return df[group_column].astype(str).values, group_column

    for candidate in ["leaf_id", "Leaf_ID", "leaf", "leafID", "leaf_id_inferred"]:
        if candidate in df.columns:
            return df[candidate].astype(str).values, candidate

    # Try image/filename columns.
    for candidate in ["blue_basename", "blue", "Blue", "filename", "image_name"]:
        if candidate in df.columns:
            inferred = df[candidate].map(infer_leaf_id_from_text)
            if inferred.notna().all():
                return inferred.astype(str).values, f"inferred_from_{candidate}"

    return None, "none"


# -------------------------------------------------------------------------
# Model construction
# -------------------------------------------------------------------------


def get_hyperparameters(
    species: str, grape_gb_n_estimators: int
) -> Dict[str, Dict[str, Any]]:
    key = normalize_species_key(species)
    hp = json.loads(
        json.dumps(HYPERPARAMETERS[key])
    )  # simple deep copy for JSON-compatible values

    # Restore Python None after JSON copy.
    for model_hp in hp.values():
        for k, v in list(model_hp.items()):
            if v is None:
                model_hp[k] = None

    if key == "grape":
        hp["Gradient Boosting"]["n_estimators"] = int(grape_gb_n_estimators)

    return hp


def make_models(
    species: str,
    random_state: int,
    n_jobs: int,
    grape_gb_n_estimators: int,
) -> Dict[str, Any]:
    hp = get_hyperparameters(species, grape_gb_n_estimators)

    models = {
        "Elastic Net": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "enet",
                    ElasticNet(
                        alpha=hp["Elastic Net"]["alpha"],
                        l1_ratio=hp["Elastic Net"]["l1_ratio"],
                        max_iter=10000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            learning_rate=hp["Gradient Boosting"]["learning_rate"],
            max_depth=hp["Gradient Boosting"]["max_depth"],
            n_estimators=hp["Gradient Boosting"]["n_estimators"],
            subsample=hp["Gradient Boosting"]["subsample"],
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            max_depth=hp["Random Forest"]["max_depth"],
            max_features=hp["Random Forest"]["max_features"],
            min_samples_leaf=hp["Random Forest"]["min_samples_leaf"],
            min_samples_split=hp["Random Forest"]["min_samples_split"],
            n_estimators=hp["Random Forest"]["n_estimators"],
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Ridge Regressor": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=hp["Ridge Regressor"]["alpha"])),
            ]
        ),
        "SVR RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svr",
                    SVR(
                        kernel="rbf",
                        C=hp["SVR RBF"]["C"],
                        epsilon=hp["SVR RBF"]["epsilon"],
                        gamma=hp["SVR RBF"]["gamma"],
                    ),
                ),
            ]
        ),
    }

    return {name: models[name] for name in MODEL_ORDER}


# -------------------------------------------------------------------------
# Metrics and plots
# -------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


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
    if min_v < 0.0:
        min_v = 0
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
    ax.set_xticks([0, 25, 50, 100, 150, 200])
    ax.set_yticks([0, 25, 50, 100, 150, 200])

    text = (
        f"RMSE = {metrics['RMSE']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}\n"
        f"R² = {metrics['R2']:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    # ax.grid(alpha=0.25)
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
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    for ax, model_name in zip(axes.ravel(), model_names):
        sub = predictions_df[predictions_df["Model"] == model_name]
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)

        metrics = regression_metrics(y_true, y_pred)

        min_v = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
        if min_v < 0.0:
            min_v = 0

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
        ax.set_xticks([0, 25, 50, 100, 150, 200])
        ax.set_yticks([0, 25, 50, 100, 150, 200])
        # ax.grid(alpha=0.25)

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

    # Keep the lower bound at zero when the target is non-negative, but do not
    # force zero if all values are above zero and the data are not intended to be clipped.
    # if min_v >= 0.0:
    #    lo_base = 0.0
    # else:
    #    lo_base = min_v

    if min_v < 0.0:
        min_v = 0

    # pad = 0.05 * (max_v - lo_base) if max_v > lo_base else 1.0
    # lo = max(0.0, lo_base - pad) if lo_base >= 0.0 else lo_base - pad
    # hi = max_v + pad

    # Same ticks for x and y in every subplot.
    shared_ticks = np.round(np.linspace(min_v, max_v, 6), 0)

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
            ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.2)

            ax.set_xlim(min_v, max_v)
            ax.set_ylim(min_v, max_v)
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
# Core evaluation
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


def run_model_evaluation(
    models: Dict[str, Any],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dev_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    groups_dev: Optional[np.ndarray],
    n_splits: int,
    random_state: int,
    n_jobs: int,
    output_dir: Path,
    target_column: str,
    save_models: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cv, cv_groups, cv_name = make_cv_splitter_and_groups(
        groups_dev, n_splits, random_state
    )

    cv_metric_rows = []
    test_metric_rows = []
    cv_pred_rows = []
    test_pred_rows = []
    fitted_models = {}

    scatter_cv_dir = output_dir / "scatter_cv_svg"
    scatter_test_dir = output_dir / "scatter_test_svg"
    scatter_cv_dir.mkdir(parents=True, exist_ok=True)
    scatter_test_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_ORDER:
        estimator = models[model_name]
        print("=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)

        # Out-of-fold CV predictions on train+validation.
        estimator_for_cv = clone(estimator)
        if cv_groups is not None:
            y_dev_pred = cross_val_predict(
                estimator_for_cv,
                X_dev,
                y_dev,
                cv=cv,
                groups=cv_groups,
                n_jobs=n_jobs,
            )
        else:
            y_dev_pred = cross_val_predict(
                estimator_for_cv,
                X_dev,
                y_dev,
                cv=cv,
                n_jobs=n_jobs,
            )

        cv_metrics = regression_metrics(y_dev, y_dev_pred)
        cv_metric_rows.append({"Model": model_name, **cv_metrics})

        cv_pred_df = dev_meta.copy()
        cv_pred_df["Model"] = model_name
        cv_pred_df["evaluation"] = "5fold_cv"
        cv_pred_df["y_true"] = y_dev
        cv_pred_df["y_pred"] = y_dev_pred
        cv_pred_df["residual"] = y_dev_pred - y_dev
        cv_pred_rows.append(cv_pred_df)

        plot_scatter(
            y_true=y_dev,
            y_pred=y_dev_pred,
            model_name=model_name,
            eval_name="5-fold CV",
            target_name=target_column,
            out_path=scatter_cv_dir
            / f"cv_scatter_{safe_model_filename(model_name)}.svg",
            metrics=cv_metrics,
        )

        # Final model trained on all train+validation, then tested.
        final_estimator = clone(estimator)
        final_estimator.fit(X_dev, y_dev)
        fitted_models[model_name] = final_estimator

        y_test_pred = final_estimator.predict(X_test)
        test_metrics = regression_metrics(y_test, y_test_pred)
        test_metric_rows.append({"Model": model_name, **test_metrics})

        test_pred_df = test_meta.copy()
        test_pred_df["Model"] = model_name
        test_pred_df["evaluation"] = "test"
        test_pred_df["y_true"] = y_test
        test_pred_df["y_pred"] = y_test_pred
        test_pred_df["residual"] = y_test_pred - y_test
        test_pred_rows.append(test_pred_df)

        plot_scatter(
            y_true=y_test,
            y_pred=y_test_pred,
            model_name=model_name,
            eval_name="Test set",
            target_name=target_column,
            out_path=scatter_test_dir
            / f"test_scatter_{safe_model_filename(model_name)}.svg",
            metrics=test_metrics,
        )

        print("5-fold CV:", cv_metrics)
        print("Test:     ", test_metrics)

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
        joblib.dump(fitted_models, model_dir / "sklearn_regressors.joblib")

    return (
        cv_metrics_df,
        test_metrics_df,
        cv_predictions_df,
        test_predictions_df,
        {
            "cv_name": cv_name,
            "n_splits": n_splits,
        },
    )


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
        target_column,
    ]
    cols = [c for c in preferred if c in df.columns]
    meta = df[cols].copy()
    return meta


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

    # Human-readable markdown tables.
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


def main() -> int:
    args = parse_args()
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LWC/FMC regression from generated spectra")
    print("=" * 80)
    print(f"Train CSV:     {expand_path(args.train_csv)}")
    print(f"Val CSV:       {expand_path(args.val_csv)}")
    print(f"Test CSV:      {expand_path(args.test_csv)}")
    print(f"Output dir:    {output_dir}")
    print(f"Target column: {args.target_column}")
    print(f"Species:       {args.species}")
    print("=" * 80)

    train_df = prepare_dataframe(args.train_csv, args, "train")
    val_df = prepare_dataframe(args.val_csv, args, "val")
    test_df = prepare_dataframe(args.test_csv, args, "test")

    dev_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    # dev_df = maybe_add_inferred_leaf_id(dev_df)
    # test_df = maybe_add_inferred_leaf_id(test_df)

    X_dev, feature_names, feature_source = build_feature_matrix(
        dev_df,
        spectrum_column=args.spectrum_column,
        fallback_spectrum_column=args.fallback_spectrum_column,
    )
    X_test, feature_names_test, feature_source_test = build_feature_matrix(
        test_df,
        spectrum_column=args.spectrum_column,
        fallback_spectrum_column=args.fallback_spectrum_column,
    )

    if X_dev.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature length mismatch: train+val has {X_dev.shape[1]} features, "
            f"test has {X_test.shape[1]} features."
        )

    y_dev = dev_df[args.target_column].to_numpy(dtype=float)
    y_test = test_df[args.target_column].to_numpy(dtype=float)

    groups_dev, group_source = add_or_infer_groups(dev_df, args.group_column)

    print("Data summary:")
    print(f"  Train rows:        {len(train_df)}")
    print(f"  Validation rows:   {len(val_df)}")
    print(f"  Train+val rows:    {len(dev_df)}")
    print(f"  Test rows:         {len(test_df)}")
    print(f"  Spectrum source:   {feature_source}")
    print(f"  Number of features:{X_dev.shape[1]}")
    print(f"  Group source:      {group_source}")
    if groups_dev is not None:
        print(f"  Unique groups:     {len(np.unique(groups_dev))}")

    std_scaler = StandardScaler()
    X_dev_std = std_scaler.fit_transform(X_dev)
    X_test_std = std_scaler.transform(X_test)

    models = make_models(
        species=args.species,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        grape_gb_n_estimators=args.grape_gb_n_estimators,
    )

    hyperparams = get_hyperparameters(args.species, args.grape_gb_n_estimators)
    hyperparams_path = output_dir / "fixed_hyperparameters.json"
    hyperparams_path.write_text(json.dumps(hyperparams, indent=2))

    cv_metrics_df, test_metrics_df, cv_pred_df, test_pred_df, cv_info = (
        run_model_evaluation(
            models=models,
            X_dev=X_dev_std,
            y_dev=y_dev,
            X_test=X_test_std,
            y_test=y_test,
            dev_meta=build_metadata(dev_df, args.target_column),
            test_meta=build_metadata(test_df, args.target_column),
            groups_dev=groups_dev,
            n_splits=args.n_splits,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            output_dir=output_dir,
            target_column=args.target_column,
            save_models=args.save_models,
        )
    )

    save_metrics_tables(
        cv_metrics_df=cv_metrics_df,
        test_metrics_df=test_metrics_df,
        output_dir=output_dir,
        write_latex=args.write_latex,
    )

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
        "spectrum_column": args.spectrum_column,
        "fallback_spectrum_column": args.fallback_spectrum_column,
        "feature_source_dev": feature_source,
        "feature_source_test": feature_source_test,
        "n_features": int(X_dev.shape[1]),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "species": args.species,
        "cv": cv_info,
        "group_source": group_source,
        "hyperparameters": hyperparams,
        "outputs": {
            "cv_metrics": str(output_dir / "cv_metrics.csv"),
            "test_metrics": str(output_dir / "test_metrics.csv"),
            "cv_predictions": str(cv_pred_path),
            "test_predictions": str(test_pred_path),
            "cv_scatter_all_models": str(output_dir / "cv_scatter_all_models.svg"),
            "test_scatter_all_models": str(output_dir / "test_scatter_all_models.svg"),
            "cv_test_scatter_subplots_all_models": str(
                output_dir / "cv_test_scatter_subplots_all_models.svg"
            ),
            "scatter_cv_dir": str(output_dir / "scatter_cv_svg"),
            "scatter_test_dir": str(output_dir / "scatter_test_svg"),
        },
    }
    (output_dir / "regression_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print("CV metrics:")
    print(cv_metrics_df.to_string(index=False))
    print("")
    print("Test metrics:")
    print(test_metrics_df.to_string(index=False))
    print("")
    print(f"CV predictions:   {cv_pred_path}")
    print(f"Test predictions: {test_pred_path}")
    print(f"5x2 subplot SVG:  {output_dir / 'cv_test_scatter_subplots_all_models.svg'}")
    print(f"Scatter SVGs:     {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
