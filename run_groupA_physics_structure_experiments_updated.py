#!/usr/bin/env python3
"""
Run Group A physics-structure ablation experiments for pix2spectral.

Fixed setting:
  discriminator = global
  loss profile  = L5 full physics-informed loss

Generator variants:
  G1 full PROSPECT, no residual
  G2 full PROSPECT, residual
  G3 segmented PROSPECT, no residual
  G4 segmented PROSPECT, residual

This script launches one training process per variant/stage, then optionally
launches the test-set evaluation script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_STAGES = ["fresh", "stage1", "stage2", "stage3", "dry"]

GROUP_A_VARIANTS = [
    {
        "id": "G1",
        "name": "full_prospect_no_residual",
        "dir": "G1_full_prospect_no_residual",
        "label": "G1_full_no_residual",
        "use_segmented_prospect": False,
        "use_segment_residual": False,
        "description": "Full-spectrum PROSPECT, no learned residual",
    },
    {
        "id": "G2",
        "name": "full_prospect_residual",
        "dir": "G2_full_prospect_residual",
        "label": "G2_full_residual",
        "use_segmented_prospect": False,
        "use_segment_residual": True,
        "description": "Full-spectrum PROSPECT with learned residual correction",
    },
    {
        "id": "G3",
        "name": "segmented_prospect_no_residual",
        "dir": "G3_segmented_prospect_no_residual",
        "label": "G3_segmented_no_residual",
        "use_segmented_prospect": True,
        "use_segment_residual": False,
        "description": "Segmented PROSPECT, no learned residual",
    },
    {
        "id": "G4",
        "name": "segmented_prospect_residual",
        "dir": "G4_segmented_prospect_residual",
        "label": "G4_segmented_residual",
        "use_segmented_prospect": True,
        "use_segment_residual": True,
        "description": "Segmented PROSPECT with learned residual correction",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Group A pix2spectral physics-structure ablation."
    )
    parser.add_argument("--train-script", default="train_with_physics_losses.py")
    parser.add_argument(
        "--eval-script", default="evaluate_test_set_export_spectra_smallN.py"
    )
    parser.add_argument(
        "--results-root", default="~/Results/pix2spectral_groupA_globalD"
    )
    parser.add_argument("--experiment-prefix", default="avocado")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--img-dir", default=None)
    parser.add_argument("--species-filter", default=None)
    parser.add_argument("--variants", nargs="+", default=["G1", "G2", "G3", "G4"])
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--normalization-scope",
        default=None,
        choices=["none", "stage_band", "global_band"],
    )
    parser.add_argument(
        "--normalization-method",
        default=None,
        choices=["zscore", "robust_zscore", "minmax"],
    )
    parser.add_argument(
        "--band-encoder-mode", default=None, choices=["shared", "separate"]
    )
    parser.add_argument(
        "--pooling-type",
        default=None,
        choices=["mean", "mean_std", "attention", "attention_stats"],
    )
    parser.add_argument("--segment-residual-scale", type=float, default=None)
    parser.add_argument("--run-test-after-training", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-testing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-best", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--allow-train-test-overlap",
        action="store_true",
        help="Forward this to the evaluation script if you intentionally use overlapping CSVs for debugging.",
    )
    return parser.parse_args()


def bool_env(value: bool) -> str:
    return "1" if bool(value) else "0"


def selected_variants(requested_ids):
    requested = {v.upper() for v in requested_ids}
    variants = [v for v in GROUP_A_VARIANTS if v["id"].upper() in requested]
    found = {v["id"].upper() for v in variants}
    missing = requested.difference(found)
    if missing:
        raise ValueError(f"Unknown variant id(s): {sorted(missing)}")
    return variants


def write_manifest(results_root: Path, variants):
    manifest_json = results_root / "groupA_physics_structure_manifest.json"
    manifest_csv = results_root / "groupA_physics_structure_manifest.csv"
    with open(manifest_json, "w") as f:
        json.dump(variants, f, indent=2)
    cols = [
        "id",
        "name",
        "dir",
        "label",
        "use_segmented_prospect",
        "use_segment_residual",
        "description",
    ]
    lines = [",".join(cols)]
    for v in variants:
        lines.append(",".join(str(v[c]).replace(",", ";") for c in cols))
    manifest_csv.write_text("\n".join(lines) + "\n")
    return manifest_json, manifest_csv


def build_train_env(args, variant, stage, variant_dir: Path):
    env = os.environ.copy()
    env["PIX2SPECTRAL_RESULTS_DIR"] = str(variant_dir)
    env["PIX2SPECTRAL_EXPERIMENT_NAME"] = f"{args.experiment_prefix}_{stage}"
    env["PIX2SPECTRAL_STAGE_FILTER"] = stage

    # Dataset paths and filters for training/validation.
    # These make config.py fully controlled by this runner.
    if args.train_csv is not None:
        env["PIX2SPECTRAL_TRAIN_CSV"] = str(Path(args.train_csv).expanduser())

    if args.val_csv is not None:
        env["PIX2SPECTRAL_VAL_CSV"] = str(Path(args.val_csv).expanduser())

    if args.test_csv is not None:
        env["PIX2SPECTRAL_TEST_CSV"] = str(Path(args.test_csv).expanduser())

    if args.img_dir is not None:
        env["PIX2SPECTRAL_IMG_DIR"] = args.img_dir
        env["PIX2SPECTRAL_TRAIN_IMG_DIR"] = args.img_dir
        env["PIX2SPECTRAL_VAL_IMG_DIR"] = args.img_dir
        env["PIX2SPECTRAL_TEST_IMG_DIR"] = args.img_dir

    if args.species_filter is not None:
        env["PIX2SPECTRAL_SPECIES_FILTER"] = args.species_filter

    # Fixed for Group A.
    env["PIX2SPECTRAL_DISCRIMINATOR_MODE"] = "global"
    env["PIX2SPECTRAL_LOSS_PROFILE"] = "L5_FULL_PHYSICS"

    # Generator physics structure.
    env["PIX2SPECTRAL_GROUPA_VARIANT_ID"] = variant["id"]
    env["PIX2SPECTRAL_GROUPA_VARIANT_NAME"] = variant["name"]
    env["PIX2SPECTRAL_USE_SEGMENTED_PROSPECT"] = bool_env(
        variant["use_segmented_prospect"]
    )
    env["PIX2SPECTRAL_USE_SEGMENT_RESIDUAL"] = bool_env(variant["use_segment_residual"])

    # L5 full physics-informed loss.
    env["PIX2SPECTRAL_LAMBDA_SPECTRAL"] = "1.0"
    env["PIX2SPECTRAL_LAMBDA_WEIGHTED"] = "0.5"
    env["PIX2SPECTRAL_LAMBDA_PARAM_PENALTY"] = "0.1"
    env["PIX2SPECTRAL_LAMBDA_SMOOTHNESS"] = "0.01"
    env["PIX2SPECTRAL_LAMBDA_DERIVATIVE"] = "0.01"
    env["PIX2SPECTRAL_LAMBDA_SEGMENT_CONTINUITY"] = "0.1"

    env["PIX2SPECTRAL_LOAD_MODEL"] = bool_env(args.resume)
    env["PIX2SPECTRAL_RESUME_FROM_BEST"] = bool_env(args.resume_from_best)

    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.num_epochs is not None:
        env["PIX2SPECTRAL_NUM_EPOCHS"] = str(args.num_epochs)
    if args.batch_size is not None:
        env["PIX2SPECTRAL_BATCH_SIZE"] = str(args.batch_size)
    if args.num_workers is not None:
        env["PIX2SPECTRAL_NUM_WORKERS"] = str(args.num_workers)
    if args.normalization_scope is not None:
        env["PIX2SPECTRAL_IMAGE_NORMALIZATION_SCOPE"] = args.normalization_scope
    if args.normalization_method is not None:
        env["PIX2SPECTRAL_IMAGE_NORMALIZATION_METHOD"] = args.normalization_method
    if args.band_encoder_mode is not None:
        env["PIX2SPECTRAL_BAND_ENCODER_MODE"] = args.band_encoder_mode
    if args.pooling_type is not None:
        env["PIX2SPECTRAL_POOLING_TYPE"] = args.pooling_type
    if args.segment_residual_scale is not None:
        env["PIX2SPECTRAL_SEGMENT_RESIDUAL_SCALE"] = str(args.segment_residual_scale)
    return env


def run_command(cmd, env, cwd, stdout_path, stderr_path, dry_run=False):
    print(f"Command: {' '.join(cmd)}")
    print(f"stdout:  {stdout_path}")
    print(f"stderr:  {stderr_path}")
    if dry_run:
        keys = [
            "PIX2SPECTRAL_RESULTS_DIR",
            "PIX2SPECTRAL_EXPERIMENT_NAME",
            "PIX2SPECTRAL_STAGE_FILTER",
            "PIX2SPECTRAL_DISCRIMINATOR_MODE",
            "PIX2SPECTRAL_USE_SEGMENTED_PROSPECT",
            "PIX2SPECTRAL_USE_SEGMENT_RESIDUAL",
            "PIX2SPECTRAL_LOSS_PROFILE",
            "PIX2SPECTRAL_LAMBDA_SPECTRAL",
            "PIX2SPECTRAL_LAMBDA_WEIGHTED",
            "PIX2SPECTRAL_LAMBDA_PARAM_PENALTY",
            "PIX2SPECTRAL_LAMBDA_SMOOTHNESS",
            "PIX2SPECTRAL_LAMBDA_DERIVATIVE",
            "PIX2SPECTRAL_LAMBDA_SEGMENT_CONTINUITY",
            "PIX2SPECTRAL_NUM_EPOCHS",
            "PIX2SPECTRAL_BATCH_SIZE",
            "PIX2SPECTRAL_NUM_WORKERS",
            "CUDA_VISIBLE_DEVICES",
        ]
        print("Environment overrides:")
        for k in keys:
            if k in env:
                print(f"  {k}={env[k]}")
        return 0
    with (
        open(stdout_path, "w", buffering=1) as stdout_f,
        open(stderr_path, "w", buffering=1) as stderr_f,
    ):
        proc = subprocess.run(
            cmd, env=env, cwd=str(cwd), stdout=stdout_f, stderr=stderr_f, text=True
        )
    return int(proc.returncode)


def run_training(args, variants, results_root: Path):
    train_script = Path(args.train_script).expanduser().resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    runner_log_dir = results_root / "runner_logs"
    runner_log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for variant in variants:
        variant_dir = results_root / variant["dir"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        for stage in args.stages:
            stage = stage.strip().lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stdout_path = (
                runner_log_dir / f"{variant['id']}_{stage}_{timestamp}.stdout.log"
            )
            stderr_path = (
                runner_log_dir / f"{variant['id']}_{stage}_{timestamp}.stderr.log"
            )
            env = build_train_env(args, variant, stage, variant_dir)
            cmd = [sys.executable, str(train_script)]
            print("\n" + "=" * 80)
            print(f"TRAIN | {variant['id']} {variant['name']} | stage={stage}")
            print("=" * 80)
            code = run_command(
                cmd, env, train_script.parent, stdout_path, stderr_path, args.dry_run
            )
            if code != 0:
                failures.append(
                    {
                        "phase": "training",
                        "variant": variant["id"],
                        "stage": stage,
                        "return_code": code,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                    }
                )
                print(
                    f"FAILED training: variant={variant['id']} stage={stage} code={code}"
                )
                if args.stop_on_failure:
                    return failures
            else:
                print(f"OK training: variant={variant['id']} stage={stage}")
    return failures


def run_testing(args, variants, results_root: Path):
    eval_script = Path(args.eval_script).expanduser().resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")
    if args.test_csv is None or args.train_csv is None or args.img_dir is None:
        raise ValueError(
            "--test-csv, --train-csv, and --img-dir are required for testing."
        )
    output_dir = results_root / "groupA_test_evaluation"
    experiment_dirs = [v["dir"] for v in variants]
    mode_labels = [v["label"] for v in variants]
    cmd = [
        sys.executable,
        str(eval_script),
        "--test-csv",
        str(Path(args.test_csv).expanduser()),
        "--train-csv",
        str(Path(args.train_csv).expanduser()),
        "--img-dir",
        args.img_dir,
        "--results-root",
        str(results_root),
        "--experiment-prefix",
        args.experiment_prefix,
        "--experiment-dirs",
        *experiment_dirs,
        "--mode-labels",
        *mode_labels,
        "--stages",
        *args.stages,
        "--output-dir",
        str(output_dir),
        "--num-workers",
        str(args.num_workers if args.num_workers is not None else 0),
    ]
    if args.allow_train_test_overlap:
        cmd.append("--allow-train-test-overlap")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_log_dir = results_root / "runner_logs"
    runner_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = runner_log_dir / f"groupA_testing_{timestamp}.stdout.log"
    stderr_path = runner_log_dir / f"groupA_testing_{timestamp}.stderr.log"
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print("\n" + "=" * 80)
    print("TEST | Group A physics-structure ablation")
    print("=" * 80)
    code = run_command(
        cmd, env, eval_script.parent, stdout_path, stderr_path, args.dry_run
    )
    if code != 0:
        return [
            {
                "phase": "testing",
                "return_code": code,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        ]
    return []


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    variants = selected_variants(args.variants)
    manifest_json, manifest_csv = write_manifest(results_root, variants)
    print("=" * 80)
    print("pix2spectral Group A physics-structure ablation")
    print("=" * 80)
    print(f"Results root: {results_root}")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest CSV:  {manifest_csv}")
    for v in variants:
        print(
            f"  {v['id']}: {v['name']} | segmented={v['use_segmented_prospect']} | residual={v['use_segment_residual']}"
        )
    print(f"Stages: {args.stages}")
    print("=" * 80)
    failures = []
    if not args.skip_training:
        failures.extend(run_training(args, variants, results_root))
        if failures and args.stop_on_failure:
            (results_root / "groupA_failures.json").write_text(
                json.dumps(failures, indent=2)
            )
            return 1
    if args.run_test_after_training and not args.skip_testing:
        failures.extend(run_testing(args, variants, results_root))
        if failures and args.stop_on_failure:
            (results_root / "groupA_failures.json").write_text(
                json.dumps(failures, indent=2)
            )
            return 1
    print("\n" + "=" * 80)
    print("Group A run finished")
    print("=" * 80)
    if failures:
        failure_path = results_root / "groupA_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2))
        print(f"Failures found. Details saved to: {failure_path}")
        return 1
    print("All requested jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
