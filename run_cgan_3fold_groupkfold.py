#!/usr/bin/env python3
"""
3-fold GroupKFold cross-validation runner for pix2spectral conditional cGAN.

This script creates grouped train/validation splits from one pix2spectral-style
CSV and runs the conditional cGAN training script once per fold.

Main idea
---------
The CSV row is the atomic sample. Each row keeps its own:

    Species / stage
    multispectral image filenames
    spectral signature
    FMC_d / LWC target if present

No image/spectrum/FMC information is recombined. Rows are only assigned to
train or validation folds.

Recommended use
---------------
Use this on your balanced CSV:

python run_cgan_3fold_groupkfold.py \
    --input-csv ~/Code/pix2spectral/Data/avocado_train_balanced_FMC.csv \
    --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --train-script train_with_physics_losses_conditional_cgan_metrics.py \
    --output-root ~/Results/pix2spectral_cgan_3fold_cv \
    --experiment-prefix avocado_cgan_balanced \
    --group-column auto \
    --n-splits 3 \
    --batch-size 2 \
    --num-epochs 300 \
    --num-workers 0 \
    --max-patches-per-band 100 \
    --stop-on-failure

Outputs
-------
output-root/
    cv_splits/
        fold_00_train.csv
        fold_00_val.csv
        fold_01_train.csv
        fold_01_val.csv
        fold_02_train.csv
        fold_02_val.csv
        fold_assignments.csv
        group_counts.csv
    fold_00/
        logs/
        plots/
        checkpoints...
    fold_01/
        ...
    fold_02/
        ...
    cv_run_manifest.json
    cv_results_summary.csv
    cv_results_summary.md

The fold training script is controlled through environment variables, so it
works with the environment-driven config.py used in the pix2spectral project.

Important
---------
This runner does not average generator weights across folds. Each fold is a
separate trained model. You should average metrics across folds, not model
weights.

Metric-enabled version
----------------------
When used with train_with_physics_losses_conditional_cgan_metrics.py, each fold
logs train and validation:

    RMSE, MAE, Bias, MRE, SAM_deg, R2

The runner summarizes these metrics in cv_results_summary.csv and
cv_results_aggregate.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3-fold GroupKFold cross-validation for pix2spectral conditional cGAN."
    )

    parser.add_argument(
        "--input-csv", required=True, help="Balanced or full CSV used for CV."
    )
    parser.add_argument(
        "--img-dir", required=True, help="Root directory for multispectral images."
    )
    parser.add_argument(
        "--train-script",
        default="train_with_physics_losses_conditional_cgan_metrics.py",
        help="Conditional cGAN training script.",
    )
    parser.add_argument(
        "--output-root", required=True, help="Root directory for CV outputs."
    )
    parser.add_argument("--experiment-prefix", default="pix2spectral_cgan_cv")

    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--group-column",
        default="auto",
        help=(
            "Column used for GroupKFold. Use 'auto' to detect leaf_id or infer it "
            "from blue/blue_basename/filename. Use a concrete column name otherwise."
        ),
    )

    parser.add_argument(
        "--species-filter",
        default="all",
        help=(
            "Value passed to PIX2SPECTRAL_SPECIES_FILTER. Use all to disable. "
            "If the input CSV is already species-specific, use all."
        ),
    )
    parser.add_argument(
        "--stage-filter",
        default="all",
        help="Value passed to PIX2SPECTRAL_STAGE_FILTER. Usually all for CV.",
    )

    parser.add_argument(
        "--test-csv",
        default=None,
        help="Optional independent test CSV passed to config.",
    )
    parser.add_argument(
        "--test-img-dir", default=None, help="Optional image root for independent test."
    )

    parser.add_argument(
        "--device", default=None, help="Optional PIX2SPECTRAL_DEVICE override."
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches-per-band", type=int, default=None)
    parser.add_argument("--min-patches", type=int, default=None)

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--l1-lambda", type=float, default=None)

    parser.add_argument(
        "--lambda-segment-continuity",
        type=float,
        default=None,
        help="Optional PIX2SPECTRAL_LAMBDA_SEGMENT_CONTINUITY override.",
    )
    parser.add_argument(
        "--lambda-mismatch",
        type=float,
        default=0.5,
        help="Weight for mismatched condition discriminator loss.",
    )
    parser.add_argument(
        "--disable-mismatch-loss",
        action="store_true",
        help="Disable D(real spectrum, wrong image condition) fake loss.",
    )

    parser.add_argument(
        "--discriminator-mode",
        default="global",
        choices=["global", "segmented", "global_plus_segmented"],
        help="Recommended for conditional cGAN: global.",
    )
    parser.add_argument(
        "--band-encoder-mode",
        default="separate",
        choices=["shared", "separate"],
        help="Recommended final architecture: separate.",
    )
    parser.add_argument("--normalization-scope", default="global_band")
    parser.add_argument("--normalization-method", default="robust_zscore")

    parser.add_argument("--use-segmented-prospect", type=int, default=1)
    parser.add_argument("--use-segment-residual", type=int, default=1)

    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-epochs", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--plot-interval", type=int, default=None)

    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the training script.",
    )

    parser.add_argument(
        "--copy-source-files",
        action="store_true",
        help=(
            "Copy train script/config/generator/discriminator into each fold directory "
            "for reproducibility when files are available."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create splits and print commands without launching training.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failed fold.",
    )
    parser.add_argument(
        "--overwrite-splits",
        action="store_true",
        help="Overwrite existing fold CSV split files.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def expand_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def basename_any(path_like: Any) -> str:
    if pd.isna(path_like):
        return ""
    text = str(path_like).strip()
    if not text:
        return ""
    return Path(PureWindowsPath(text).name).name


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
        m = re.search(pat, text)
        if m:
            return m.group(1)

    return None


def resolve_group_column(
    df: pd.DataFrame, group_column: str
) -> Tuple[pd.Series, str, pd.DataFrame]:
    df = df.copy()

    if group_column.lower() != "auto":
        if group_column not in df.columns:
            raise ValueError(f"Requested group column '{group_column}' not found.")
        groups = df[group_column].astype(str)
        if groups.isna().any() or (groups.str.len() == 0).any():
            raise ValueError(f"Group column '{group_column}' contains empty values.")
        return groups, group_column, df

    candidates = ["leaf_id", "Leaf_ID", "leaf", "leafID", "plant_id", "sample_id"]
    for col in candidates:
        if col in df.columns:
            groups = df[col].astype(str)
            if groups.notna().all() and (groups.str.len() > 0).all():
                return groups, col, df

    image_candidates = [
        "blue_basename",
        "blue",
        "Blue",
        "filename",
        "image_name",
        "image",
    ]
    for col in image_candidates:
        if col in df.columns:
            inferred = df[col].map(infer_leaf_id_from_text)
            if inferred.notna().all():
                df["leaf_id_inferred"] = inferred.astype(str)
                return df["leaf_id_inferred"], f"inferred_from_{col}", df

    raise ValueError(
        "Could not infer groups automatically. Provide --group-column with a valid "
        "column such as leaf_id. Available columns: " + ", ".join(map(str, df.columns))
    )


def safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("_")
    return text or "experiment"


def env_set(env: Dict[str, str], key: str, value: Any) -> None:
    if value is None:
        return
    env[key] = str(value)


def normalize_filter_value(value: str) -> str:
    if value is None:
        return "all"
    text = str(value).strip()
    if text.lower() in ["", "none", "null", "any", "*"]:
        return "all"
    return text


def write_csv_if_allowed(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def copy_if_exists(src_name: str, dst_dir: Path) -> Optional[str]:
    src = Path(src_name)
    if src.exists():
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return str(dst)
    return None


def parse_jsonl_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()

    rows = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return pd.DataFrame(rows)


def summarize_fold_log(
    log_path: Path, fold_id: int, monitor_metric: str = "val_l1"
) -> Dict[str, Any]:
    df = parse_jsonl_log(log_path)

    base = {
        "fold": int(fold_id),
        "log_path": str(log_path),
        "log_found": bool(log_path.exists()),
    }

    if df.empty:
        return {**base, "status": "no_log_or_empty"}

    if "is_best" in df.columns and df["is_best"].fillna(False).any():
        best_row = df[df["is_best"].fillna(False)].iloc[-1]
    elif monitor_metric in df.columns:
        best_idx = df[monitor_metric].astype(float).idxmin()
        best_row = df.loc[best_idx]
    else:
        best_row = df.iloc[-1]

    keys = [
        "epoch",
        "best_epoch",
        "monitor_metric",
        "monitor_value",
        "best_monitor_value",
        # Train spectral metrics
        "train_rmse",
        "train_mae",
        "train_bias",
        "train_mre",
        "train_sam_rad",
        "train_sam_deg",
        "train_r2",
        # Validation spectral metrics
        "val_l1",
        "val_rmse",
        "val_mae",
        "val_bias",
        "val_mre",
        "val_sam_rad",
        "val_sam_deg",
        "val_r2",
        # Physics and adversarial metrics
        "val_physics_total",
        "val_spectral_l1",
        "val_weighted_l1",
        "val_param_penalty",
        "val_smoothness",
        "val_derivative",
        "d_loss",
        "d_mismatch",
        "g_loss",
        "g_adv",
        "g_physics",
    ]

    out = {**base, "status": "ok", "n_logged_epochs": int(len(df))}
    for k in keys:
        if k in best_row.index:
            v = best_row[k]
            if pd.isna(v):
                out[k] = None
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                out[k] = float(v)
            else:
                out[k] = v

    return out


# -------------------------------------------------------------------------
# CV split creation
# -------------------------------------------------------------------------


def create_groupkfold_splits(
    df: pd.DataFrame,
    groups: pd.Series,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_groups = pd.Series(groups).astype(str).unique()
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} unique groups, but found only "
            f"{len(unique_groups)} unique groups."
        )

    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros(len(df), dtype=np.float32)
    y_dummy = np.zeros(len(df), dtype=np.float32)
    return list(gkf.split(X_dummy, y_dummy, groups=groups.astype(str).to_numpy()))


def write_split_reports(
    df: pd.DataFrame,
    groups: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    split_dir: Path,
    target_col: Optional[str],
    stage_col: Optional[str],
) -> None:
    assign_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        for idx in train_idx:
            assign_rows.append(
                {
                    "fold": fold_id,
                    "row_index": int(idx),
                    "role": "train",
                    "group": str(groups.iloc[idx]),
                }
            )
        for idx in val_idx:
            assign_rows.append(
                {
                    "fold": fold_id,
                    "row_index": int(idx),
                    "role": "val",
                    "group": str(groups.iloc[idx]),
                }
            )

    assignments = pd.DataFrame(assign_rows)
    assignments.to_csv(split_dir / "fold_assignments.csv", index=False)

    group_counts = (
        pd.DataFrame({"group": groups.astype(str)})
        .groupby("group", observed=True)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["n_rows", "group"], ascending=[False, True])
    )
    group_counts.to_csv(split_dir / "group_counts.csv", index=False)

    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        row = {
            "fold": fold_id,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_train_groups": int(groups.iloc[train_idx].nunique()),
            "n_val_groups": int(groups.iloc[val_idx].nunique()),
        }

        if target_col is not None and target_col in df.columns:
            y_train = pd.to_numeric(df.iloc[train_idx][target_col], errors="coerce")
            y_val = pd.to_numeric(df.iloc[val_idx][target_col], errors="coerce")
            row.update(
                {
                    f"{target_col}_train_min": float(y_train.min()),
                    f"{target_col}_train_max": float(y_train.max()),
                    f"{target_col}_train_mean": float(y_train.mean()),
                    f"{target_col}_val_min": float(y_val.min()),
                    f"{target_col}_val_max": float(y_val.max()),
                    f"{target_col}_val_mean": float(y_val.mean()),
                }
            )

        fold_rows.append(row)

    pd.DataFrame(fold_rows).to_csv(split_dir / "fold_summary.csv", index=False)

    if stage_col is not None and stage_col in df.columns:
        stage_rows = []
        for fold_id, (train_idx, val_idx) in enumerate(splits):
            for role, idxs in [("train", train_idx), ("val", val_idx)]:
                counts = df.iloc[idxs][stage_col].astype(str).value_counts()
                for stage, count in counts.items():
                    stage_rows.append(
                        {
                            "fold": fold_id,
                            "role": role,
                            "stage": stage,
                            "count": int(count),
                        }
                    )
        pd.DataFrame(stage_rows).to_csv(
            split_dir / "fold_stage_counts.csv", index=False
        )


# -------------------------------------------------------------------------
# Training launch
# -------------------------------------------------------------------------


def build_fold_env(
    args: argparse.Namespace,
    fold_id: int,
    fold_dir: Path,
    train_csv: Path,
    val_csv: Path,
) -> Dict[str, str]:
    env = os.environ.copy()

    experiment_name = safe_name(f"{args.experiment_prefix}_fold{fold_id:02d}")

    env_set(env, "PIX2SPECTRAL_RESULTS_DIR", fold_dir)
    env_set(env, "PIX2SPECTRAL_EXPERIMENT_NAME", experiment_name)

    env_set(env, "PIX2SPECTRAL_TRAIN_CSV", train_csv)
    env_set(env, "PIX2SPECTRAL_VAL_CSV", val_csv)
    env_set(
        env,
        "PIX2SPECTRAL_TEST_CSV",
        expand_path(args.test_csv) if args.test_csv else val_csv,
    )

    img_dir = expand_path(args.img_dir)
    env_set(env, "PIX2SPECTRAL_IMG_DIR", img_dir)
    env_set(env, "PIX2SPECTRAL_TRAIN_IMG_DIR", img_dir)
    env_set(env, "PIX2SPECTRAL_VAL_IMG_DIR", img_dir)
    env_set(
        env,
        "PIX2SPECTRAL_TEST_IMG_DIR",
        expand_path(args.test_img_dir) if args.test_img_dir else img_dir,
    )

    env_set(
        env, "PIX2SPECTRAL_SPECIES_FILTER", normalize_filter_value(args.species_filter)
    )
    env_set(env, "PIX2SPECTRAL_STAGE_FILTER", normalize_filter_value(args.stage_filter))

    env_set(env, "PIX2SPECTRAL_RANDOM_SEED", int(args.random_seed) + int(fold_id))
    env_set(env, "PIX2SPECTRAL_DEVICE", args.device)

    env_set(env, "PIX2SPECTRAL_BATCH_SIZE", args.batch_size)
    env_set(env, "PIX2SPECTRAL_NUM_EPOCHS", args.num_epochs)
    env_set(env, "PIX2SPECTRAL_NUM_WORKERS", args.num_workers)
    env_set(env, "PIX2SPECTRAL_PERSISTENT_WORKERS", 0)
    env_set(env, "PIX2SPECTRAL_PREFETCH_FACTOR", 1)
    env_set(env, "PIX2SPECTRAL_MAX_PATCHES_PER_BAND", args.max_patches_per_band)
    env_set(env, "PIX2SPECTRAL_MIN_PATCHES", args.min_patches)

    env_set(env, "PIX2SPECTRAL_LEARNING_RATE", args.learning_rate)
    env_set(env, "PIX2SPECTRAL_L1_LAMBDA", args.l1_lambda)
    env_set(
        env, "PIX2SPECTRAL_LAMBDA_SEGMENT_CONTINUITY", args.lambda_segment_continuity
    )

    env_set(env, "PIX2SPECTRAL_IMAGE_NORMALIZATION_SCOPE", args.normalization_scope)
    env_set(env, "PIX2SPECTRAL_IMAGE_NORMALIZATION_METHOD", args.normalization_method)

    env_set(env, "PIX2SPECTRAL_DISCRIMINATOR_MODE", args.discriminator_mode)
    env_set(env, "PIX2SPECTRAL_BAND_ENCODER_MODE", args.band_encoder_mode)
    env_set(
        env, "PIX2SPECTRAL_USE_SEGMENTED_PROSPECT", int(args.use_segmented_prospect)
    )
    env_set(env, "PIX2SPECTRAL_USE_SEGMENT_RESIDUAL", int(args.use_segment_residual))

    # Conditional-cGAN controls.
    env_set(env, "PIX2SPECTRAL_USE_CONDITIONAL_DISCRIMINATOR", 1)
    env_set(
        env,
        "PIX2SPECTRAL_USE_MISMATCHED_CONDITION_LOSS",
        0 if args.disable_mismatch_loss else 1,
    )
    env_set(env, "PIX2SPECTRAL_LAMBDA_MISMATCH", args.lambda_mismatch)

    env_set(env, "PIX2SPECTRAL_EARLY_STOP_PATIENCE", args.early_stop_patience)
    env_set(env, "PIX2SPECTRAL_EARLY_STOP_MIN_EPOCHS", args.early_stop_min_epochs)
    env_set(env, "PIX2SPECTRAL_SAVE_INTERVAL", args.save_interval)
    env_set(env, "PIX2SPECTRAL_PLOT_INTERVAL", args.plot_interval)

    # Always start each fold from scratch unless the caller overrides by shell env after launch.
    env_set(env, "PIX2SPECTRAL_LOAD_MODEL", 0)
    env_set(env, "PIX2SPECTRAL_SAVE_MODEL", 1)

    return env


def run_one_fold(
    args: argparse.Namespace,
    fold_id: int,
    train_csv: Path,
    val_csv: Path,
    fold_dir: Path,
) -> Dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)

    env = build_fold_env(args, fold_id, fold_dir, train_csv, val_csv)

    cmd = [str(args.python), str(expand_path(args.train_script) or args.train_script)]

    env_preview_keys = [
        "PIX2SPECTRAL_RESULTS_DIR",
        "PIX2SPECTRAL_EXPERIMENT_NAME",
        "PIX2SPECTRAL_TRAIN_CSV",
        "PIX2SPECTRAL_VAL_CSV",
        "PIX2SPECTRAL_TEST_CSV",
        "PIX2SPECTRAL_IMG_DIR",
        "PIX2SPECTRAL_SPECIES_FILTER",
        "PIX2SPECTRAL_STAGE_FILTER",
        "PIX2SPECTRAL_USE_CONDITIONAL_DISCRIMINATOR",
        "PIX2SPECTRAL_USE_MISMATCHED_CONDITION_LOSS",
        "PIX2SPECTRAL_LAMBDA_MISMATCH",
        "PIX2SPECTRAL_DISCRIMINATOR_MODE",
        "PIX2SPECTRAL_BAND_ENCODER_MODE",
        "PIX2SPECTRAL_USE_SEGMENTED_PROSPECT",
        "PIX2SPECTRAL_USE_SEGMENT_RESIDUAL",
        "PIX2SPECTRAL_NUM_EPOCHS",
        "PIX2SPECTRAL_BATCH_SIZE",
        "PIX2SPECTRAL_NUM_WORKERS",
    ]

    print("=" * 80)
    print(f"Fold {fold_id:02d}")
    print("=" * 80)
    print("Command:")
    print(" ".join(cmd))
    print("Environment preview:")
    for k in env_preview_keys:
        if k in env:
            print(f"  {k}={env[k]}")

    run_info = {
        "fold": int(fold_id),
        "fold_dir": str(fold_dir),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "command": cmd,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        run_info["returncode"] = None
        run_info["status"] = "dry_run"
        return run_info

    stdout_path = fold_dir / "training_stdout_stderr.log"
    t0_info = {"fold": fold_id, "stdout_stderr_log": str(stdout_path)}
    print(f"Writing training log to: {stdout_path}")

    with open(stdout_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
        )

    run_info.update(t0_info)
    run_info["returncode"] = int(proc.returncode)
    run_info["status"] = "ok" if proc.returncode == 0 else "failed"

    return run_info


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    input_csv = expand_path(args.input_csv)
    output_root = expand_path(args.output_root)
    train_script = expand_path(args.train_script)

    if input_csv is None or not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if train_script is None or not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    if int(args.n_splits) != 3:
        print(
            f"Warning: requested n_splits={args.n_splits}. This runner defaults to 3 but will use your value."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    split_dir = output_root / "cv_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.copy()
    df["__cv_original_index"] = np.arange(len(df), dtype=int)

    groups, group_source, df_with_groups = resolve_group_column(df, args.group_column)

    splits = create_groupkfold_splits(df_with_groups, groups, int(args.n_splits))

    target_col = None
    for c in ["FMC_d", "LWC_d", "fmc", "lwc"]:
        if c in df_with_groups.columns:
            target_col = c
            break

    stage_col = None
    for c in ["Stages", "stage", "Stage", "STAGE"]:
        if c in df_with_groups.columns:
            stage_col = c
            break

    write_split_reports(
        df=df_with_groups,
        groups=groups,
        splits=splits,
        split_dir=split_dir,
        target_col=target_col,
        stage_col=stage_col,
    )

    manifest: Dict[str, Any] = {
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "train_script": str(train_script),
        "n_rows": int(len(df_with_groups)),
        "n_splits": int(args.n_splits),
        "group_source": group_source,
        "n_unique_groups": int(groups.nunique()),
        "target_column_detected": target_col,
        "stage_column_detected": stage_col,
        "conditional_cgan": True,
        "mismatched_condition_loss": not bool(args.disable_mismatch_loss),
        "lambda_mismatch": float(args.lambda_mismatch),
        "folds": [],
    }

    if args.copy_source_files:
        source_copy_dir = output_root / "source_files"
        source_copy_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in [
            args.train_script,
            "config_cgan.py",
            "generator_model_cgan.py",
            "discriminator_model_cgan.py",
            "physics_losses.py",
            "dataset.py",
        ]:
            copied_path = copy_if_exists(str(name), source_copy_dir)
            if copied_path is not None:
                copied.append(copied_path)
        manifest["copied_source_files"] = copied

    run_records = []

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        fold_dir = output_root / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_csv = split_dir / f"fold_{fold_id:02d}_train.csv"
        val_csv = split_dir / f"fold_{fold_id:02d}_val.csv"

        # Remove CV helper columns from training CSVs unless they were real input columns.
        input_columns = [c for c in df.columns if c != "__cv_original_index"]
        train_df = df_with_groups.iloc[train_idx][input_columns].copy()
        val_df = df_with_groups.iloc[val_idx][input_columns].copy()

        write_csv_if_allowed(train_df, train_csv, overwrite=args.overwrite_splits)
        write_csv_if_allowed(val_df, val_csv, overwrite=args.overwrite_splits)

        manifest["folds"].append(
            {
                "fold": int(fold_id),
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "fold_dir": str(fold_dir),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_train_groups": int(groups.iloc[train_idx].nunique()),
                "n_val_groups": int(groups.iloc[val_idx].nunique()),
            }
        )

        run_info = run_one_fold(args, fold_id, train_csv, val_csv, fold_dir)
        run_records.append(run_info)

        (output_root / "cv_run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        pd.DataFrame(run_records).to_csv(output_root / "cv_run_status.csv", index=False)

        if run_info["status"] == "failed" and args.stop_on_failure:
            print(
                f"Fold {fold_id:02d} failed. Stopping because --stop-on-failure was set."
            )
            break

    # Summarize logs from completed folds.
    summary_rows = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        exp_name = safe_name(f"{args.experiment_prefix}_fold{fold_id:02d}")
        log_path = Path(fold["fold_dir"]) / "logs" / f"{exp_name}_training_log.json"
        summary_rows.append(
            summarize_fold_log(log_path, fold_id, monitor_metric="val_l1")
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "cv_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    try:
        (output_root / "cv_results_summary.md").write_text(
            summary_df.to_markdown(index=False)
        )
    except Exception:
        pass

    # Add aggregate numeric statistics for key metrics.
    aggregate_rows = []
    for metric in [
        "train_rmse",
        "train_mae",
        "train_bias",
        "train_mre",
        "train_sam_deg",
        "train_r2",
        "val_rmse",
        "val_mae",
        "val_bias",
        "val_mre",
        "val_sam_deg",
        "val_r2",
        "val_physics_total",
        "g_loss",
        "d_loss",
    ]:
        if metric in summary_df.columns:
            vals = pd.to_numeric(summary_df[metric], errors="coerce").dropna()
            if len(vals) > 0:
                aggregate_rows.append(
                    {
                        "metric": metric,
                        "n_folds": int(len(vals)),
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                    }
                )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = output_root / "cv_results_aggregate.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    manifest["run_records"] = run_records
    manifest["cv_results_summary"] = str(summary_path)
    manifest["cv_results_aggregate"] = str(aggregate_path)
    (output_root / "cv_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    print("=" * 80)
    print("3-fold GroupKFold conditional cGAN CV finished")
    print("=" * 80)
    print(f"Output root:        {output_root}")
    print(f"Split directory:    {split_dir}")
    print(f"Run status CSV:     {output_root / 'cv_run_status.csv'}")
    print(f"Summary CSV:        {summary_path}")
    print(f"Aggregate CSV:      {aggregate_path}")
    print("")
    print("Fold summary:")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print("")
    print("Aggregate metrics:")
    if not aggregate_df.empty:
        print(aggregate_df.to_string(index=False))
    print("=" * 80)

    failed = [r for r in run_records if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
