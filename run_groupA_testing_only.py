#!/usr/bin/env python3
"""
Test-only runner for Group A physics-structure ablation.

Expected trained checkpoint layout:

    ~/Results/pix2spectral_groupA_globalD/
        G1_full_prospect_no_residual/avocado_fresh_gen_best.pth.tar
        G1_full_prospect_no_residual/avocado_stage1_gen_best.pth.tar
        ...
        G4_segmented_prospect_residual/avocado_dry_gen_best.pth.tar

This script calls evaluate_test_set_export_spectra_smallN.py using the correct
Group A experiment folders and labels.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_STAGES = ["fresh", "stage1", "stage2", "stage3", "dry"]

GROUP_A_VARIANTS = {
    "G1": {
        "dir": "G1_full_prospect_no_residual",
        "label": "G1_full_no_residual",
    },
    "G2": {
        "dir": "G2_full_prospect_residual",
        "label": "G2_full_residual",
    },
    "G3": {
        "dir": "G3_segmented_prospect_no_residual",
        "label": "G3_segmented_no_residual",
    },
    "G4": {
        "dir": "G4_segmented_prospect_residual",
        "label": "G4_segmented_residual",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-set evaluation for trained Group A pix2spectral models."
    )

    parser.add_argument(
        "--eval-script",
        default="evaluate_test_set_export_spectra_smallN.py",
        help="Path to the evaluation script.",
    )
    parser.add_argument("--config-module", default="config")
    parser.add_argument("--dataset-module", default="dataset")
    parser.add_argument("--generator-module", default="generator_model")

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--img-dir", required=True)

    parser.add_argument(
        "--results-root",
        default="~/Results/pix2spectral_groupA_globalD",
        help="Root folder containing G1/G2/G3/G4 experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <results-root>/groupA_test_evaluation",
    )
    parser.add_argument("--experiment-prefix", default="avocado")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES)
    parser.add_argument("--variants", nargs="+", default=["G1", "G2", "G3", "G4"])

    parser.add_argument(
        "--checkpoint-kind",
        choices=["best", "last", "final_best"],
        default="best",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--non-strict-load", action="store_true")
    parser.add_argument("--allow-train-test-overlap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-checkpoint-check", action="store_true")

    return parser.parse_args()


def expand(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def selected_variant_info(variant_ids):
    ids = [v.upper() for v in variant_ids]
    missing = [v for v in ids if v not in GROUP_A_VARIANTS]
    if missing:
        raise ValueError(f"Unknown Group A variant(s): {missing}")

    experiment_dirs = [GROUP_A_VARIANTS[v]["dir"] for v in ids]
    mode_labels = [GROUP_A_VARIANTS[v]["label"] for v in ids]
    return ids, experiment_dirs, mode_labels


def checkpoint_suffix(kind: str) -> str:
    if kind == "best":
        return "gen_best.pth.tar"
    if kind == "last":
        return "gen_last.pth.tar"
    if kind == "final_best":
        return "gen_final_best.pth.tar"
    raise ValueError(f"Unknown checkpoint kind: {kind}")


def build_checkpoint_template(kind: str) -> str:
    suffix = checkpoint_suffix(kind)
    return "{results_root}/{experiment_dir}/{experiment_prefix}_{stage}_" + suffix


def check_checkpoints(results_root, experiment_prefix, variant_ids, stages, kind):
    suffix = checkpoint_suffix(kind)
    missing = []

    for variant_id in variant_ids:
        exp_dir = GROUP_A_VARIANTS[variant_id]["dir"]
        for stage in stages:
            ckpt = Path(results_root) / exp_dir / f"{experiment_prefix}_{stage}_{suffix}"
            if not ckpt.exists():
                missing.append(str(ckpt))

    if missing:
        msg = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(
            "Missing expected checkpoint files:\n"
            f"{msg}\n\n"
            "Check --results-root, --experiment-prefix, --stages, and --checkpoint-kind. "
            "Use --skip-checkpoint-check only if you want the evaluation script to handle it."
        )


def main() -> int:
    args = parse_args()

    eval_script = Path(args.eval_script).expanduser().resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else results_root / "groupA_test_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_ids, experiment_dirs, mode_labels = selected_variant_info(args.variants)

    if not args.skip_checkpoint_check:
        check_checkpoints(
            results_root=results_root,
            experiment_prefix=args.experiment_prefix,
            variant_ids=variant_ids,
            stages=args.stages,
            kind=args.checkpoint_kind,
        )

    cmd = [
        sys.executable,
        str(eval_script),

        "--config-module",
        args.config_module,
        "--dataset-module",
        args.dataset_module,
        "--generator-module",
        args.generator_module,

        "--train-csv",
        expand(args.train_csv),
        "--test-csv",
        expand(args.test_csv),
        "--img-dir",
        args.img_dir,

        "--results-root",
        str(results_root),
        "--output-dir",
        str(output_dir),
        "--experiment-prefix",
        args.experiment_prefix,

        "--experiment-dirs",
        *experiment_dirs,
        "--mode-labels",
        *mode_labels,
        "--stages",
        *args.stages,

        "--checkpoint-template",
        build_checkpoint_template(args.checkpoint_kind),

        "--num-workers",
        str(args.num_workers),
    ]

    if args.device is not None:
        cmd += ["--device", args.device]
    if args.non_strict_load:
        cmd += ["--non-strict-load"]
    if args.allow_train_test_overlap:
        cmd += ["--allow-train-test-overlap"]

    print("=" * 80)
    print("Group A test-set evaluation")
    print("=" * 80)
    print(f"Results root:      {results_root}")
    print(f"Output dir:        {output_dir}")
    print(f"Variants:          {variant_ids}")
    print(f"Experiment dirs:   {experiment_dirs}")
    print(f"Mode labels:       {mode_labels}")
    print(f"Stages:            {args.stages}")
    print(f"Checkpoint kind:   {args.checkpoint_kind}")
    print(f"Evaluation script: {eval_script}")
    print("=" * 80)
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(eval_script.parent), text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
