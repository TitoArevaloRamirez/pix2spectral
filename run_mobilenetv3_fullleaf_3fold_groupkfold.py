#!/usr/bin/env python3
"""
3-fold GroupKFold runner for the clean MobileNetV3 full-leaf cGAN.

This runner is specialized for:
    train_with_physics_losses_mobilenetv3_fullleaf_clean_stageaux.py

It creates group-disjoint fold CSVs and launches the MobileNetV3 full-leaf
training script for each fold. It passes explicit environment variables for the
latest clean MobileNetV3 implementation, including stage auxiliary loss.

It does not use patch terminology.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


STAGE_CANONICAL = {
    "fresh": "fresh",
    "d0": "fresh",
    "stage0": "fresh",
    "stage1": "stage1",
    "d1": "stage1",
    "stage2": "stage2",
    "d2": "stage2",
    "stage3": "stage3",
    "d3": "stage3",
    "dry": "dry",
    "d4": "dry",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run 3-fold GroupKFold CV for MobileNetV3 full-leaf cGAN."
    )
    p.add_argument("--input-csv", required=True, help="Input CSV to split into folds.")
    p.add_argument(
        "--img-dir",
        required=True,
        help="Root directory containing preprocessed resized/padded/normalized full-leaf images.",
    )
    p.add_argument(
        "--train-script",
        default="train_with_physics_losses_mobilenetv3_fullleaf_clean_stageaux.py",
        help="Training script to launch.",
    )
    p.add_argument(
        "--output-root", required=True, help="Output directory for fold CSVs and runs."
    )
    p.add_argument("--experiment-prefix", default="mobilenetv3_fullleaf")
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--group-column", default="auto")
    p.add_argument("--species", default=None)
    p.add_argument("--stage-column", default="auto")
    p.add_argument("--spectral-drop-first-n", type=int, default=50)
    p.add_argument("--full-image-size", type=int, default=220)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=2e-5)

    p.add_argument("--mobilenet-pretrained", type=int, default=1)
    p.add_argument("--mobilenet-freeze-all-except-last", type=int, default=0)
    p.add_argument("--mobilenet-token-dim", type=int, default=64)
    p.add_argument("--mobilenet-attention-layers", type=int, default=1)
    p.add_argument("--mobilenet-attention-heads", type=int, default=2)
    p.add_argument("--mobilenet-dropout", type=float, default=0.40)
    p.add_argument("--mobilenet-adapter-hidden-channels", type=int, default=8)

    p.add_argument("--stage-aux-weight", type=float, default=0.02)
    p.add_argument("--disable-stage-aux", action="store_true")

    p.add_argument("--lambda-mismatch", type=float, default=0.1)
    p.add_argument("--disable-mismatch-loss", action="store_true")

    p.add_argument("--best-model-metric", default="val_rmse")
    p.add_argument("--best-model-mode", default="min")
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--early-stop-min-epochs", type=int, default=20)

    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stream-output", action="store_true", default=True)
    p.add_argument("--no-stream-output", action="store_false", dest="stream_output")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def canonical_stage(x) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[\s\-]+", "", s)
    return STAGE_CANONICAL.get(s, s)


def infer_stage_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested stage column not found: {requested}")
        return requested

    for c in ["Stages", "Stage", "stage", "dehydration_stage", "DehydrationStage"]:
        if c in df.columns:
            return c
    return None


def infer_leaf_id_from_path(path_value) -> str:
    name = Path(str(path_value).replace("\\", "/")).name
    stem = Path(name).stem.lower()

    # Common examples: leaf028d0_1, leaf028_d0_1, leaf028-stage1_1
    stem = re.sub(r"[_\-](1|2|3|4|5)$", "", stem)

    # Remove terminal dehydration marker if it is appended to leaf id.
    stem = re.sub(r"[_\-]?d[0-4]$", "", stem)
    stem = re.sub(r"[_\-]?stage[0-3]$", "", stem)
    stem = re.sub(r"[_\-]?fresh$", "", stem)
    stem = re.sub(r"[_\-]?dry$", "", stem)
    return stem


def choose_group_column(df: pd.DataFrame, group_column: str) -> Tuple[str, pd.Series]:
    if group_column and group_column != "auto":
        if group_column not in df.columns:
            raise ValueError(f"Requested group column not found: {group_column}")
        return group_column, df[group_column].astype(str)

    candidates = [
        "leaf_id",
        "LeafID",
        "leaf",
        "Leaf",
        "plant_id",
        "PlantID",
        "sample_id",
        "SampleID",
        "id",
        "ID",
    ]
    for c in candidates:
        if c in df.columns:
            return c, df[c].astype(str)

    if "blue" in df.columns:
        groups = df["blue"].apply(infer_leaf_id_from_path).astype(str)
        return "auto_from_blue_basename", groups

    raise ValueError(
        "Could not infer groups. Provide --group-column or include a blue image column."
    )


def make_groupkfold_indices(groups: pd.Series, n_splits: int, seed: int):
    unique_groups = np.array(sorted(groups.astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    # Greedy distribute groups by row count to balance fold sizes.
    counts = groups.astype(str).value_counts().to_dict()
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]

    for g in sorted(unique_groups, key=lambda x: counts.get(x, 0), reverse=True):
        k = int(np.argmin(fold_sizes))
        fold_groups[k].append(g)
        fold_sizes[k] += int(counts.get(g, 0))

    folds = []
    group_arr = groups.astype(str).to_numpy()
    for k in range(n_splits):
        val_groups = set(fold_groups[k])
        val_mask = np.array([g in val_groups for g in group_arr])
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        folds.append((train_idx, val_idx, sorted(val_groups)))

    return folds


def write_stage_counts(df, stage_col, path):
    if stage_col is None:
        return
    tmp = df.copy()
    tmp[stage_col] = tmp[stage_col].apply(canonical_stage)
    counts = (
        tmp[stage_col].value_counts().rename_axis("stage").reset_index(name="count")
    )
    counts.to_csv(path, index=False)


def run_command(
    cmd: List[str], env: Dict[str, str], log_path: Path, stream: bool
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if stream:
        with log_path.open("w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
                f.flush()
            return int(proc.wait())
    else:
        with log_path.open("w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        return int(proc.returncode)


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    img_dir = Path(args.img_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.species is not None and "Species" in df.columns:
        df = df[
            df["Species"].astype(str).str.lower() == str(args.species).lower()
        ].reset_index(drop=True)

    group_name, groups = choose_group_column(df, args.group_column)
    stage_col = infer_stage_column(df, args.stage_column)

    folds = make_groupkfold_indices(groups, args.n_splits, args.seed)

    manifest_rows = []
    print("=" * 80)
    print("MobileNetV3 full-leaf GroupKFold runner")
    print("=" * 80)
    print(f"Input CSV:      {input_csv}")
    print(f"Image dir:      {img_dir}")
    print(f"Output root:    {output_root}")
    print(f"Train script:   {args.train_script}")
    print(f"Rows:           {len(df)}")
    print(f"Group source:   {group_name}")
    print(f"Groups:         {groups.nunique()}")
    print(f"Stage column:   {stage_col}")
    print(f"Splits:         {args.n_splits}")
    print("=" * 80)

    for fold_id, (train_idx, val_idx, val_groups) in enumerate(folds):
        fold_dir = output_root / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_csv = fold_dir / f"{args.experiment_prefix}_fold{fold_id:02d}_train.csv"
        val_csv = fold_dir / f"{args.experiment_prefix}_fold{fold_id:02d}_val.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        write_stage_counts(train_df, stage_col, fold_dir / "train_stage_counts.csv")
        write_stage_counts(val_df, stage_col, fold_dir / "val_stage_counts.csv")

        env = os.environ.copy()
        env.update(
            {
                "PIX2SPECTRAL_TRAIN_CSV": str(train_csv),
                "PIX2SPECTRAL_VAL_CSV": str(val_csv),
                "PIX2SPECTRAL_TEST_CSV": str(val_csv),
                "PIX2SPECTRAL_IMG_DIR": str(img_dir),
                "PIX2SPECTRAL_TRAIN_IMG_DIR": str(img_dir),
                "PIX2SPECTRAL_VAL_IMG_DIR": str(img_dir),
                "PIX2SPECTRAL_TEST_IMG_DIR": str(img_dir),
                "PIX2SPECTRAL_RESULTS_DIR": str(fold_dir),
                "PIX2SPECTRAL_EXPERIMENT_NAME": f"{args.experiment_prefix}_fold{fold_id:02d}",
                "PIX2SPECTRAL_FULL_IMAGE_SIZE": str(args.full_image_size),
                "PIX2SPECTRAL_BATCH_SIZE": str(args.batch_size),
                "PIX2SPECTRAL_NUM_EPOCHS": str(args.num_epochs),
                "PIX2SPECTRAL_NUM_WORKERS": str(args.num_workers),
                "PIX2SPECTRAL_LEARNING_RATE": str(args.learning_rate),
                "PIX2SPECTRAL_SPECTRAL_DROP_FIRST_N": str(args.spectral_drop_first_n),
                "PIX2SPECTRAL_MOBILENET_PRETRAINED": str(
                    int(args.mobilenet_pretrained)
                ),
                "PIX2SPECTRAL_MOBILENET_FREEZE_ALL_EXCEPT_LAST": str(
                    int(args.mobilenet_freeze_all_except_last)
                ),
                "PIX2SPECTRAL_MOBILENET_TOKEN_DIM": str(args.mobilenet_token_dim),
                "PIX2SPECTRAL_MOBILENET_ATTENTION_LAYERS": str(
                    args.mobilenet_attention_layers
                ),
                "PIX2SPECTRAL_MOBILENET_ATTENTION_HEADS": str(
                    args.mobilenet_attention_heads
                ),
                "PIX2SPECTRAL_MOBILENET_DROPOUT": str(args.mobilenet_dropout),
                "PIX2SPECTRAL_MOBILENET_ADAPTER_HIDDEN_CHANNELS": str(
                    args.mobilenet_adapter_hidden_channels
                ),
                "PIX2SPECTRAL_USE_STAGE_AUXILIARY_LOSS": "0"
                if args.disable_stage_aux
                else "1",
                "PIX2SPECTRAL_STAGE_AUX_WEIGHT": str(args.stage_aux_weight),
                "PIX2SPECTRAL_USE_STAGE_AS_CONDITION": "0",
                "PIX2SPECTRAL_USE_CONDITIONAL_DISCRIMINATOR": "1",
                "PIX2SPECTRAL_USE_MISMATCHED_CONDITION_LOSS": "0"
                if args.disable_mismatch_loss
                else "1",
                "PIX2SPECTRAL_LAMBDA_MISMATCH": "0.0"
                if args.disable_mismatch_loss
                else str(args.lambda_mismatch),
                "PIX2SPECTRAL_BEST_MODEL_METRIC": str(args.best_model_metric),
                "PIX2SPECTRAL_BEST_MODEL_MODE": str(args.best_model_mode),
                "PIX2SPECTRAL_EARLY_STOP_PATIENCE": str(args.early_stop_patience),
                "PIX2SPECTRAL_EARLY_STOP_MIN_EPOCHS": str(args.early_stop_min_epochs),
                "SPECIES_FILTER ": str(args.species),
            }
        )

        cmd = [args.python, args.train_script]

        print("\n" + "=" * 80)
        print(f"Fold {fold_id:02d}")
        print("=" * 80)
        print(f"Train rows: {len(train_df)}")
        print(f"Val rows:   {len(val_df)}")
        print(f"Val groups: {len(val_groups)}")
        print("Command:")
        print(" ".join(shlex.quote(x) for x in cmd))
        print("=" * 80)

        fold_env_path = fold_dir / "fold_environment.txt"
        with fold_env_path.open("w") as f:
            for k in sorted(env):
                if k.startswith("PIX2SPECTRAL_"):
                    f.write(f"{k}={env[k]}\n")

        if args.dry_run:
            rc = 0
        else:
            rc = run_command(
                cmd=cmd,
                env=env,
                log_path=fold_dir / "training_stdout_stderr.log",
                stream=bool(args.stream_output),
            )

        manifest_rows.append(
            {
                "fold": fold_id,
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "n_val_groups": len(val_groups),
                "return_code": rc,
            }
        )

        if rc != 0 and args.stop_on_failure:
            print(f"Stopping because fold {fold_id:02d} failed with return code {rc}.")
            break

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(
        output_root / "mobilenetv3_fullleaf_3fold_manifest.csv", index=False
    )

    print("\n" + "=" * 80)
    print("Runner finished")
    print(f"Manifest: {output_root / 'mobilenetv3_fullleaf_3fold_manifest.csv'}")
    print("=" * 80)

    failed = manifest[manifest["return_code"] != 0] if len(manifest) else pd.DataFrame()
    return 1 if len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
