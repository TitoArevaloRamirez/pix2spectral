#!/usr/bin/env python3
"""
Evaluate trained pix2spectral generator models on a held-out testing set.

Outputs:
  1. CSV with per-sample quantitative metrics:
       stage, sample index, RMSE, MAE, SAM radians/degrees, relative error, etc.

  2. Qualitative figure with 5 subplots, one per dehydration stage:
       - average real spectral signature: black solid line
       - real standard deviation: dark gray shaded region
       - generated/estimated spectra: thin black patterned lines

Typical usage, per-stage trained models from run_all_stage_experiments.py:

    python evaluate_test_set.py \
        --test-csv ./Data/dataset_splits_70_20_10/avocado_test_minimal.csv \
        --train-csv ./Data/dataset_splits_70_20_10/avocado_train_minimal.csv \
        --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
        --results-dir ~/Results/pix2spectral \
        --experiment-prefix avocado \
        --model-mode per-stage \
        --stages fresh stage1 stage2 stage3 dry

Evaluate one "all stages" model on every stage:

    python evaluate_test_set.py \
        --model-mode single \
        --checkpoint ~/Results/pix2spectral/avocado_all_gen_best.pth.tar \
        --stages fresh stage1 stage2 stage3 dry

Notes:
  - This script evaluates the generator only.
  - It assumes your dataset.py exposes MultiSpectralCSVPatchDataset and patch_collate_fn.
  - It supports both old and improved generator constructors using safe fallback logic.
  - It computes normalization statistics from TRAIN_CSV only, then applies them to TEST_CSV.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STAGES = ["fresh", "stage1", "stage2", "stage3", "dry"]


# -------------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------------


def expand_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def import_from_module(module_name: str, attr_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_cfg_value(cfg, names: Iterable[str], default=None):
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def maybe_set_config_value(cfg, name: str, value: Any):
    if value is not None:
        setattr(cfg, name, value)


def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by fn/class constructor."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove common wrappers such as 'module.'."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


def load_generator_checkpoint(
    checkpoint_path: str,
    gen: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint_path = expand_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = normalize_state_dict_keys(state_dict)
    missing, unexpected = gen.load_state_dict(state_dict, strict=strict)

    if not strict:
        if missing:
            print(f"Warning: missing keys while loading {checkpoint_path}: {missing}")
        if unexpected:
            print(
                f"Warning: unexpected keys while loading {checkpoint_path}: {unexpected}"
            )

    return ckpt if isinstance(ckpt, dict) else {"state_dict": state_dict}


def move_batch_bands_to_device(batch_bands, device, non_blocking=False):
    out = {}
    for band, lst in batch_bands.items():
        out[band] = [t.to(device, non_blocking=non_blocking) for t in lst]
    return out


# -------------------------------------------------------------------------
# Wavelength helpers
# -------------------------------------------------------------------------


def make_wavelengths_from_config(
    cfg, fallback_count: Optional[int] = None
) -> np.ndarray:
    wl_min = float(get_cfg_value(cfg, ["WAVELENGTH_MIN", "wavelength_min"], 400.0))
    wl_max = float(get_cfg_value(cfg, ["WAVELENGTH_MAX", "wavelength_max"], 2500.0))
    wl_count = int(
        get_cfg_value(
            cfg,
            ["WAVELENGTH_COUNT", "wavelength_count"],
            2101 if fallback_count is None else fallback_count,
        )
    )

    if fallback_count is not None and wl_count != fallback_count:
        print(
            f"Warning: config wavelength count={wl_count}, "
            f"but spectra length={fallback_count}. Using spectra length."
        )
        wl_count = fallback_count

    return np.linspace(wl_min, wl_max, wl_count, dtype=np.float64)


# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------


def compute_sample_metrics(y_fake: np.ndarray, y_real: np.ndarray, eps: float = 1e-8):
    """Metrics for one sample."""
    y_fake = np.asarray(y_fake, dtype=np.float64).reshape(-1)
    y_real = np.asarray(y_real, dtype=np.float64).reshape(-1)

    diff = y_fake - y_real
    abs_diff = np.abs(diff)

    mse = np.mean(diff**2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_diff))

    denom = np.maximum(np.abs(y_real), eps)
    rel = abs_diff / denom

    mean_relative_error = float(np.mean(rel))
    median_relative_error = float(np.median(rel))
    max_relative_error = float(np.max(rel))

    real_mean_abs = float(np.mean(np.abs(y_real)))
    relative_rmse = float(rmse / max(real_mean_abs, eps))

    dot = float(np.dot(y_fake, y_real))
    norm_fake = float(np.linalg.norm(y_fake))
    norm_real = float(np.linalg.norm(y_real))
    cos_sim = dot / max(norm_fake * norm_real, eps)
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    sam_rad = float(np.arccos(cos_sim))
    sam_deg = float(sam_rad * 180.0 / np.pi)

    return {
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": relative_rmse,
        "mean_relative_error": mean_relative_error,
        "median_relative_error": median_relative_error,
        "max_relative_error": max_relative_error,
        "sam_rad": sam_rad,
        "sam_deg": sam_deg,
    }


# -------------------------------------------------------------------------
# Model and dataset construction
# -------------------------------------------------------------------------


def build_generator(cfg, generator_module: str, device: torch.device):
    GeneratorClass = import_from_module(
        generator_module,
        "MultiSpectralPatchToProspectGenerator",
    )

    bands = get_cfg_value(cfg, ["BANDS"], ["blue", "green", "red", "nir", "red_edge"])

    kwargs = {
        "bands": bands,
        "base_features": get_cfg_value(cfg, ["BASE_FEATURES"], 8),
        "embed_dim": get_cfg_value(cfg, ["EMBED_DIM"], 64),
        "mins": get_cfg_value(cfg, ["PROSPECT_PARAM_MINS"], None),
        "maxs": get_cfg_value(cfg, ["PROSPECT_PARAM_MAXS"], None),
        "wavelength_min": get_cfg_value(
            cfg, ["WAVELENGTH_MIN", "wavelength_min"], 400.0
        ),
        "wavelength_max": get_cfg_value(
            cfg, ["WAVELENGTH_MAX", "wavelength_max"], 2500.0
        ),
        "wavelength_count": get_cfg_value(
            cfg, ["WAVELENGTH_COUNT", "wavelength_count"], 2101
        ),
        "spectral_segments": get_cfg_value(
            cfg,
            ["SPECTRAL_SEGMENTS"],
            [(400.0, 900.0), (900.0, 1000.0), (1000.0, 2000.0), (2000.0, 2500.0)],
        ),
        "use_segmented_prospect": get_cfg_value(cfg, ["USE_SEGMENTED_PROSPECT"], True),
        "use_segment_residual": get_cfg_value(cfg, ["USE_SEGMENT_RESIDUAL"], True),
        "segment_residual_scale": get_cfg_value(cfg, ["SEGMENT_RESIDUAL_SCALE"], 0.05),
        "segment_blend_width": get_cfg_value(cfg, ["SEGMENT_BLEND_WIDTH"], 0),
        "patch_encoder_type": get_cfg_value(cfg, ["PATCH_ENCODER_TYPE"], "cnn"),
        "pooling_type": get_cfg_value(cfg, ["POOLING_TYPE"], "attention_stats"),
        "band_encoder_mode": get_cfg_value(cfg, ["BAND_ENCODER_MODE"], "separate"),
        "norm_type": get_cfg_value(cfg, ["NORM_TYPE"], "group"),
    }

    kwargs = filter_kwargs_for_callable(GeneratorClass, kwargs)
    gen = GeneratorClass(**kwargs).to(device)
    gen.eval()
    return gen


def build_dataset(
    cfg,
    dataset_module: str,
    csv_path: str,
    img_dir: str,
    stage: str,
    normalization_stats=None,
    compute_normalization_stats: bool = False,
    cache_patches: bool = False,
):
    DatasetClass = import_from_module(dataset_module, "MultiSpectralCSVPatchDataset")

    # Support both the older normalization_mode API and the newer scope/method API.
    norm_scope = get_cfg_value(cfg, ["IMAGE_NORMALIZATION_SCOPE"], "stage_band")
    norm_method = get_cfg_value(cfg, ["IMAGE_NORMALIZATION_METHOD"], "robust_zscore")
    norm_mode = get_cfg_value(
        cfg, ["IMAGE_NORMALIZATION_MODE"], "stage_band_robust_zscore"
    )
    norm_clip = get_cfg_value(cfg, ["IMAGE_NORMALIZATION_OUTPUT_CLIP"], (-5.0, 5.0))

    kwargs = {
        "csv_path": csv_path,
        "root_dir": img_dir,
        "species": get_cfg_value(cfg, ["SPECIES_FILTER"], None),
        "stage": stage,
        "patch_h": get_cfg_value(cfg, ["PATCH_H"], 32),
        "patch_w": get_cfg_value(cfg, ["PATCH_W"], 32),
        "stride_h": get_cfg_value(cfg, ["VAL_STRIDE_H", "STRIDE_H"], 32),
        "stride_w": get_cfg_value(cfg, ["VAL_STRIDE_W", "STRIDE_W"], 32),
        "black_thr": get_cfg_value(cfg, ["BLACK_THR"], 0.0),
        "min_leaf_coverage": get_cfg_value(cfg, ["LEAF_COVERAGE"], 0.9),
        "min_patches_per_band": get_cfg_value(cfg, ["MIN_PATCHES"], 10),
        "max_patches_per_band": get_cfg_value(cfg, ["MAX_PATCHES_PER_BAND"], 10),
        "border_erode_px": get_cfg_value(cfg, ["BORDER_ERODE_PX"], 2),
        "mask_method": get_cfg_value(cfg, ["MASK_METHOD"], "contour"),
        "random_seed": get_cfg_value(cfg, ["RANDOM_SEED"], 42),
        "return_debug": False,
        "spectral_drop_first_n": get_cfg_value(cfg, ["SPECTRAL_DROP_FIRST_N"], 50),
        "normalization_stats": normalization_stats,
        "compute_normalization_stats": compute_normalization_stats,
        "normalization_scope": norm_scope,
        "normalization_method": norm_method,
        "normalization_mode": norm_mode,
        "normalization_output_clip": norm_clip,
        "cache_patches": cache_patches,
        "clone_cached_items": False,
    }

    kwargs = filter_kwargs_for_callable(DatasetClass, kwargs)
    return DatasetClass(**kwargs)


def choose_stats_stage(args, cfg, eval_stage: str) -> str:
    """
    Which training rows should be used to compute normalization statistics?

    auto:
      - if per-stage model and stage_band normalization: use that stage
      - otherwise: use all stages
    """
    if args.stats_source == "stage":
        return eval_stage
    if args.stats_source == "all":
        return "all"

    norm_scope = get_cfg_value(cfg, ["IMAGE_NORMALIZATION_SCOPE"], "stage_band")

    if args.model_mode == "per-stage" and norm_scope == "stage_band":
        return eval_stage

    return "all"


def checkpoint_for_stage(args, stage: str, cfg) -> str:
    if args.model_mode == "single":
        if args.checkpoint is not None:
            return expand_path(args.checkpoint)

        ckpt = get_cfg_value(cfg, ["BEST_CHECKPOINT_GEN", "CHECKPOINT_GEN"], None)
        if ckpt is None:
            raise ValueError(
                "No checkpoint provided and config has no BEST_CHECKPOINT_GEN/CHECKPOINT_GEN."
            )
        return expand_path(ckpt)

    template = args.checkpoint_template
    if template is None:
        template = "{results_dir}/{experiment_prefix}_{stage}_gen_best.pth.tar"

    return expand_path(
        template.format(
            results_dir=str(Path(args.results_dir).expanduser().resolve()),
            experiment_prefix=args.experiment_prefix,
            stage=stage,
        )
    )


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------


def evaluate_one_stage(
    args,
    cfg,
    stage: str,
    device: torch.device,
    normalization_stats_cache: Dict[str, Any],
):
    dataset_module = args.dataset_module
    generator_module = args.generator_module

    train_csv = expand_path(args.train_csv)
    test_csv = expand_path(args.test_csv)
    img_dir = expand_path(args.img_dir)

    stats_stage = choose_stats_stage(args, cfg, stage)

    if stats_stage not in normalization_stats_cache:
        print(
            f"Computing normalization stats from TRAIN set with stage='{stats_stage}'..."
        )
        stats_dataset = build_dataset(
            cfg=cfg,
            dataset_module=dataset_module,
            csv_path=train_csv,
            img_dir=img_dir,
            stage=stats_stage,
            normalization_stats=None,
            compute_normalization_stats=True,
            cache_patches=False,
        )
        normalization_stats_cache[stats_stage] = getattr(
            stats_dataset,
            "normalization_stats",
            None,
        )

    normalization_stats = normalization_stats_cache[stats_stage]

    print(f"Building TEST dataset for stage='{stage}'...")
    test_dataset = build_dataset(
        cfg=cfg,
        dataset_module=dataset_module,
        csv_path=test_csv,
        img_dir=img_dir,
        stage=stage,
        normalization_stats=normalization_stats,
        compute_normalization_stats=False,
        cache_patches=args.cache_patches,
    )

    if len(test_dataset) == 0:
        print(f"Warning: no test samples found for stage='{stage}'.")
        return [], None, None

    patch_collate_fn = import_from_module(dataset_module, "patch_collate_fn")

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=patch_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = checkpoint_for_stage(args, stage, cfg)
    print(f"Loading generator for stage='{stage}': {checkpoint_path}")

    gen = build_generator(cfg, generator_module, device)
    load_generator_checkpoint(
        checkpoint_path=checkpoint_path,
        gen=gen,
        device=device,
        strict=not args.non_strict_load,
    )
    gen.eval()

    rows = []
    all_real = []
    all_pred = []

    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for local_idx, (batch_bands, y_real) in enumerate(loader):
            batch_bands = move_batch_bands_to_device(
                batch_bands,
                device,
                non_blocking=non_blocking,
            )
            y_real = y_real.to(device, non_blocking=non_blocking).float()

            # Evaluation in float32 is safer for physics models.
            with torch.amp.autocast("cuda", enabled=False):
                y_fake, p_params = gen.forward_batch_list(batch_bands)

            y_fake = y_fake.float()

            y_real_np = y_real[0].detach().cpu().numpy().reshape(-1)
            y_fake_np = y_fake[0].detach().cpu().numpy().reshape(-1)

            if not np.isfinite(y_real_np).all():
                raise FloatingPointError(
                    f"Non-finite y_real at stage={stage}, index={local_idx}"
                )
            if not np.isfinite(y_fake_np).all():
                raise FloatingPointError(
                    f"Non-finite y_fake at stage={stage}, index={local_idx}"
                )

            metrics = compute_sample_metrics(y_fake_np, y_real_np)

            row_meta = {}
            if hasattr(test_dataset, "df"):
                df_row = test_dataset.df.iloc[local_idx]
                for col in [
                    "Species",
                    "Stages",
                    "blue",
                    "green",
                    "red",
                    "nir",
                    "red_edge",
                ]:
                    if col in df_row:
                        row_meta[col] = df_row[col]

            row = {
                "stage_eval": stage,
                "sample_index_within_stage": local_idx,
                "checkpoint": checkpoint_path,
                **row_meta,
                **metrics,
            }

            # Store predicted PROSPECT params if compact enough.
            p_np = p_params[0].detach().cpu().numpy()
            row["params_shape"] = str(tuple(p_np.shape))
            row["params_json"] = json.dumps(np.asarray(p_np, dtype=float).tolist())

            rows.append(row)
            all_real.append(y_real_np)
            all_pred.append(y_fake_np)

    real_arr = np.stack(all_real, axis=0)
    pred_arr = np.stack(all_pred, axis=0)

    return rows, real_arr, pred_arr


def plot_qualitative_by_stage(
    stage_to_real: Dict[str, np.ndarray],
    stage_to_pred: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
    stages: List[str],
    out_path: str,
    max_pred_lines: int = 80,
):
    n_stages = len(stages)

    fig_height = max(3.0 * n_stages, 10.0)
    fig, axes = plt.subplots(
        n_stages,
        1,
        figsize=(14, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    line_styles = [":", "-.", "--", (0, (1, 1)), (0, (3, 1, 1, 1))]

    for ax, stage in zip(axes, stages):
        real = stage_to_real.get(stage)
        pred = stage_to_pred.get(stage)

        if real is None or pred is None or real.size == 0:
            ax.set_title(f"{stage}: no samples")
            ax.grid(alpha=0.25)
            continue

        mean_real = np.mean(real, axis=0)
        std_real = np.std(real, axis=0)

        ax.fill_between(
            wavelengths,
            mean_real - std_real,
            mean_real + std_real,
            color="0.35",
            alpha=0.35,
            linewidth=0,
            label="Real +/- 1 std",
        )

        ax.plot(
            wavelengths,
            mean_real,
            color="black",
            linewidth=2.0,
            linestyle="-",
            label="Mean real",
        )

        n_pred = pred.shape[0]
        if n_pred > max_pred_lines:
            idx = np.linspace(0, n_pred - 1, max_pred_lines).astype(int)
            pred_to_plot = pred[idx]
        else:
            pred_to_plot = pred

        for i, y_hat in enumerate(pred_to_plot):
            ax.plot(
                wavelengths,
                y_hat,
                color="black",
                linewidth=0.65,
                alpha=0.45,
                linestyle=line_styles[i % len(line_styles)],
            )

        ax.set_title(f"{stage} | n={real.shape[0]}")
        ax.set_ylabel("Reflectance")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Wavelength (nm)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(
        "Real spectral signatures and generated estimates by dehydration stage",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained pix2spectral generators on a testing set."
    )

    parser.add_argument("--config-module", default="config")
    parser.add_argument("--dataset-module", default="dataset")
    parser.add_argument("--generator-module", default="generator_model")

    parser.add_argument(
        "--test-csv",
        default="~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv",
    )
    parser.add_argument(
        "--train-csv",
        default="~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv",
    )
    parser.add_argument(
        "--img-dir",
        default="/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/",
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        help="Stages to evaluate and plot.",
    )

    parser.add_argument(
        "--model-mode",
        choices=["per-stage", "single"],
        default="per-stage",
        help="per-stage loads one checkpoint per stage; single loads one checkpoint for all stages.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint for --model-mode single.",
    )

    parser.add_argument(
        "--checkpoint-template",
        default=None,
        help=(
            "Template for per-stage checkpoints. Available fields: "
            "{results_dir}, {experiment_prefix}, {stage}. "
            "Default: {results_dir}/{experiment_prefix}_{stage}_gen_best.pth.tar"
        ),
    )

    parser.add_argument(
        "--results-dir",
        default="~/Results/pix2spectral_segmented/avocado_segmentedDiscriminator",
    )
    parser.add_argument("--experiment-prefix", default="avocado")
    parser.add_argument("--output-dir", default=None)

    parser.add_argument(
        "--stats-source",
        choices=["auto", "stage", "all"],
        default="auto",
        help="How to compute normalization stats from training set.",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-patches", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--non-strict-load", action="store_true")

    parser.add_argument("--max-pred-lines", type=int, default=80)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = importlib.import_module(args.config_module)

    # Resolve config/default paths.
    if args.test_csv is None:
        args.test_csv = get_cfg_value(cfg, ["TEST_CSV", "VAL_CSV"], None)
    if args.train_csv is None:
        args.train_csv = get_cfg_value(cfg, ["TRAIN_CSV"], None)
    if args.img_dir is None:
        args.img_dir = get_cfg_value(
            cfg, ["TEST_IMG_DIR", "VAL_IMG_DIR", "TRAIN_IMG_DIR"], None
        )

    if args.test_csv is None:
        raise ValueError("No --test-csv provided and config has no TEST_CSV/VAL_CSV.")
    if args.train_csv is None:
        raise ValueError("No --train-csv provided and config has no TRAIN_CSV.")
    if args.img_dir is None:
        raise ValueError(
            "No --img-dir provided and config has no TEST_IMG_DIR/VAL_IMG_DIR/TRAIN_IMG_DIR."
        )

    results_dir = ensure_dir(args.results_dir)

    if args.output_dir is None:
        args.output_dir = str(results_dir / "test_evaluation")
    output_dir = ensure_dir(args.output_dir)

    # Force evaluation-related config overrides for deterministic test evaluation.
    maybe_set_config_value(cfg, "STAGE_FILTER", "all")
    if hasattr(cfg, "LOAD_MODEL"):
        setattr(cfg, "LOAD_MODEL", False)

    if args.device is None:
        requested = get_cfg_value(
            cfg, ["DEVICE"], "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        requested = args.device

    if "cuda" in str(requested) and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested = "cpu"

    device = torch.device(requested)

    print("=" * 80)
    print("pix2spectral test-set evaluation")
    print("=" * 80)
    print(f"Device:            {device}")
    print(f"Train CSV:         {expand_path(args.train_csv)}")
    print(f"Test CSV:          {expand_path(args.test_csv)}")
    print(f"Image dir:         {expand_path(args.img_dir)}")
    print(f"Model mode:        {args.model_mode}")
    print(f"Stages:            {args.stages}")
    print(f"Output dir:        {output_dir}")
    print(f"Dataset module:    {args.dataset_module}")
    print(f"Generator module:  {args.generator_module}")
    print("=" * 80)

    normalization_stats_cache = {}
    all_rows = []
    stage_to_real = {}
    stage_to_pred = {}

    for stage in args.stages:
        stage = stage.strip().lower()
        print("\n" + "-" * 80)
        print(f"Evaluating stage: {stage}")
        print("-" * 80)

        rows, real_arr, pred_arr = evaluate_one_stage(
            args=args,
            cfg=cfg,
            stage=stage,
            device=device,
            normalization_stats_cache=normalization_stats_cache,
        )

        if rows:
            all_rows.extend(rows)
            stage_to_real[stage] = real_arr
            stage_to_pred[stage] = pred_arr

            stage_df = pd.DataFrame(rows)
            stage_csv = output_dir / f"test_metrics_{stage}.csv"
            stage_df.to_csv(stage_csv, index=False)
            print(f"Saved stage metrics: {stage_csv}")

    if not all_rows:
        raise RuntimeError(
            "No evaluation rows were produced. Check test CSV/stage filters."
        )

    metrics_df = pd.DataFrame(all_rows)
    metrics_csv = output_dir / "test_metrics_all_stages.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Summary by stage.
    metric_cols = [
        "rmse",
        "mae",
        "relative_rmse",
        "mean_relative_error",
        "median_relative_error",
        "sam_rad",
        "sam_deg",
    ]

    summary_df = (
        metrics_df.groupby("stage_eval")[metric_cols]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary_csv = output_dir / "test_metrics_summary_by_stage.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Determine wavelengths from first real array length.
    first_stage = next(iter(stage_to_real))
    spectrum_len = stage_to_real[first_stage].shape[1]
    wavelengths = make_wavelengths_from_config(cfg, fallback_count=spectrum_len)

    fig_path = output_dir / "qualitative_spectra_by_stage.png"
    plot_qualitative_by_stage(
        stage_to_real=stage_to_real,
        stage_to_pred=stage_to_pred,
        wavelengths=wavelengths,
        stages=[s.strip().lower() for s in args.stages],
        out_path=str(fig_path),
        max_pred_lines=args.max_pred_lines,
    )

    print("\n" + "=" * 80)
    print("Evaluation finished")
    print("=" * 80)
    print(f"Per-sample metrics: {metrics_csv}")
    print(f"Summary metrics:    {summary_csv}")
    print(f"Qualitative plot:   {fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
