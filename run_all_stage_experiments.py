#!/usr/bin/env python3
"""
Run pix2spectral experiments for multiple dehydration stages with one command.

One-time requirement:
    config.py must read the PIX2SPECTRAL_* environment variables shown in the
    companion config snippet.

Example:
    python run_all_stage_experiments.py \
        --train-script train_with_physics_losses.py \
        --results-dir ~/Results/pix2spectral \
        --experiment-prefix avocado \
        --stages fresh stage1 stage2 stage3 dry all

Useful variants:
    # Run only two stages
    python run_all_stage_experiments.py --stages fresh dry

    # Resume from each stage's last checkpoint
    python run_all_stage_experiments.py --resume

    # Use GPU 0
    python run_all_stage_experiments.py --cuda-visible-devices 0

    # Stop all remaining experiments if one fails
    python run_all_stage_experiments.py --stop-on-failure
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_STAGES = ["fresh", "stage1", "stage2", "stage3", "dry", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pix2spectral training for multiple dehydration stages."
    )

    parser.add_argument(
        "--train-script",
        default="train_with_physics_losses.py",
        help="Training script to execute.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        help="Stages to run sequentially.",
    )
    parser.add_argument(
        "--results-dir",
        default="~/Results/pix2spectral",
        help="Base directory for checkpoints, logs, and plots.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="pix2spectral",
        help="Prefix used to build each experiment name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each experiment from its stage-specific last checkpoint.",
    )
    parser.add_argument(
        "--resume-from-best",
        action="store_true",
        help="Resume each experiment from its best checkpoint instead of last.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0' or '1'.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional override for NUM_EPOCHS.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for BATCH_SIZE.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for NUM_WORKERS.",
    )
    parser.add_argument(
        "--normalization-scope",
        default=None,
        choices=["none", "stage_band", "global_band"],
        help="Optional override for IMAGE_NORMALIZATION_SCOPE.",
    )
    parser.add_argument(
        "--encoder-mode",
        default=None,
        choices=["shared", "separate"],
        help="Optional override for BAND_ENCODER_MODE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands/environment without running training.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the sweep immediately if one experiment fails.",
    )

    return parser.parse_args()


def stage_to_experiment_name(prefix: str, stage: str) -> str:
    clean_stage = stage.strip().lower().replace(" ", "_")
    return f"{prefix}_{clean_stage}"


def main() -> int:
    args = parse_args()

    train_script = Path(args.train_script).expanduser().resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    results_dir = Path(args.results_dir).expanduser().resolve()
    runner_log_dir = results_dir / "runner_logs"
    runner_log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("pix2spectral stage experiment runner")
    print("=" * 80)
    print(f"Training script: {train_script}")
    print(f"Results dir:     {results_dir}")
    print(f"Stages:          {args.stages}")
    print(f"Resume:          {args.resume}")
    print(f"Resume best:     {args.resume_from_best}")
    print("=" * 80)

    failures = []

    for stage in args.stages:
        stage = stage.strip().lower()
        exp_name = stage_to_experiment_name(args.experiment_prefix, stage)

        env = os.environ.copy()
        env["PIX2SPECTRAL_STAGE_FILTER"] = stage
        env["PIX2SPECTRAL_EXPERIMENT_NAME"] = exp_name
        env["PIX2SPECTRAL_RESULTS_DIR"] = str(results_dir)

        # Each stage should usually train from scratch unless --resume is set.
        env["PIX2SPECTRAL_LOAD_MODEL"] = "1" if args.resume else "0"
        env["PIX2SPECTRAL_RESUME_FROM_BEST"] = "1" if args.resume_from_best else "0"

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

        if args.encoder_mode is not None:
            env["PIX2SPECTRAL_BAND_ENCODER_MODE"] = args.encoder_mode

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_path = runner_log_dir / f"{exp_name}_{timestamp}.stdout.log"
        stderr_path = runner_log_dir / f"{exp_name}_{timestamp}.stderr.log"

        cmd = [sys.executable, str(train_script)]

        print("\n" + "-" * 80)
        print(f"Starting experiment: {exp_name}")
        print(f"Stage filter:        {stage}")
        print(f"stdout log:          {stdout_path}")
        print(f"stderr log:          {stderr_path}")
        print(f"Command:             {' '.join(cmd)}")
        print("-" * 80)

        if args.dry_run:
            print("Dry run only. Environment overrides:")
            keys = [
                "PIX2SPECTRAL_STAGE_FILTER",
                "PIX2SPECTRAL_EXPERIMENT_NAME",
                "PIX2SPECTRAL_RESULTS_DIR",
                "PIX2SPECTRAL_LOAD_MODEL",
                "PIX2SPECTRAL_RESUME_FROM_BEST",
                "PIX2SPECTRAL_NUM_EPOCHS",
                "PIX2SPECTRAL_BATCH_SIZE",
                "PIX2SPECTRAL_NUM_WORKERS",
                "PIX2SPECTRAL_IMAGE_NORMALIZATION_SCOPE",
                "PIX2SPECTRAL_BAND_ENCODER_MODE",
                "CUDA_VISIBLE_DEVICES",
            ]
            for key in keys:
                if key in env:
                    print(f"  {key}={env[key]}")
            continue

        with open(stdout_path, "w", buffering=1) as stdout_f, open(
            stderr_path, "w", buffering=1
        ) as stderr_f:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=str(train_script.parent),
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
            )

        if proc.returncode != 0:
            failures.append((stage, exp_name, proc.returncode, stdout_path, stderr_path))
            print(f"FAILED: {exp_name} with return code {proc.returncode}")

            if args.stop_on_failure:
                break
        else:
            print(f"FINISHED OK: {exp_name}")

    print("\n" + "=" * 80)
    print("Experiment sweep finished")
    print("=" * 80)

    if failures:
        print("Failures:")
        for stage, exp_name, code, stdout_path, stderr_path in failures:
            print(f"  {exp_name} stage={stage} code={code}")
            print(f"    stdout: {stdout_path}")
            print(f"    stderr: {stderr_path}")
        return 1

    print("All experiments completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
