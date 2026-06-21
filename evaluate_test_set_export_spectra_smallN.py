#!/usr/bin/env python3
"""
Evaluate pix2spectral generators on a held-out testing set and compare
discriminator modes with small-test-set-friendly outputs.

This version assumes experiments are stored as:

    ~/Results/pix2spectral/
        avocado_global/
        avocado_segmented/
        avocado_global_plus_segmented/

Expected checkpoints:

    avocado_<stage>_gen_best.pth.tar

Main outputs:
  - Per-sample metrics CSV.
  - Generated spectra CSV.
  - Real spectra CSV.
  - Mean/std model-comparison tables for RMSE, MAE, bias, MRE.
  - Qualitative spectra plots.
  - Wavelength-wise reflectance error diagnostics:
      bias(lambda), MAE(lambda), RMSE(lambda), MRE(lambda).
  - Wavelength-wise SAM contribution proxy:
      NOT true SAM per wavelength, because SAM is a vector-angle metric.
      This proxy shows which wavelengths contribute most to angular mismatch.

Example:
    python evaluate_test_set_export_spectra.py \
        --test-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
        --train-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
        --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
        --results-root ~/Results/pix2spectral \
        --experiment-prefix avocado \
        --experiment-dirs avocado_global avocado_segmented avocado_global_plus_segmented \
        --mode-labels global segmented global_plus_segmented \
        --stages fresh stage1 stage2 stage3 dry
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STAGES = ["fresh", "stage1", "stage2", "stage3", "dry"]
DEFAULT_EXPERIMENT_DIRS = [
    "avocado_global",
    "avocado_segmented",
    "avocado_global_plus_segmented",
]
DEFAULT_MODE_LABELS = ["global", "segmented", "global_plus_segmented"]


# -------------------------------------------------------------------------
# Generic helpers
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
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
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


def canonical_stage_name(value):
    s = str(value).strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")

    aliases = {
        "fresh": "fresh",
        "f": "fresh",
        "stage1": "stage1",
        "stage01": "stage1",
        "s1": "stage1",
        "1": "stage1",
        "stage2": "stage2",
        "stage02": "stage2",
        "s2": "stage2",
        "2": "stage2",
        "stage3": "stage3",
        "stage03": "stage3",
        "s3": "stage3",
        "3": "stage3",
        "dry": "dry",
        "d": "dry",
        "all": "all",
        "any": "all",
        "*": "all",
        "": "all",
        "none": "all",
    }
    return aliases.get(s, s)


def safe_filename(text: str) -> str:
    return (
        str(text)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


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


def wavelength_columns(wavelengths: np.ndarray) -> List[str]:
    cols = []
    for wl in wavelengths:
        if abs(wl - round(wl)) < 1e-8:
            cols.append(f"wl_{int(round(wl))}")
        else:
            cols.append(f"wl_{wl:.2f}")
    return cols


# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------


def compute_sample_metrics(
    y_fake: np.ndarray,
    y_real: np.ndarray,
    relative_error_eps: float = 1e-3,
    numerical_eps: float = 1e-12,
):
    y_fake = np.asarray(y_fake, dtype=np.float64).reshape(-1)
    y_real = np.asarray(y_real, dtype=np.float64).reshape(-1)

    if y_fake.shape != y_real.shape:
        raise ValueError(
            f"Shape mismatch: y_fake={y_fake.shape}, y_real={y_real.shape}"
        )

    diff = y_fake - y_real
    abs_diff = np.abs(diff)

    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_diff))
    bias = float(np.mean(diff))

    denom = np.maximum(np.abs(y_real), float(relative_error_eps))
    rel = abs_diff / denom

    mean_relative_error = float(np.mean(rel))
    median_relative_error = float(np.median(rel))
    max_relative_error = float(np.max(rel))
    mape_percent = float(100.0 * mean_relative_error)

    real_mean_abs = float(np.mean(np.abs(y_real)))
    real_range = float(np.max(y_real) - np.min(y_real))

    relative_rmse = float(rmse / max(real_mean_abs, numerical_eps))
    nrmse_mean = relative_rmse
    nrmse_range = float(rmse / max(real_range, numerical_eps))

    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, numerical_eps))

    dot = float(np.dot(y_fake, y_real))
    norm_fake = float(np.linalg.norm(y_fake))
    norm_real = float(np.linalg.norm(y_real))
    cos_sim = dot / max(norm_fake * norm_real, numerical_eps)
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    sam_rad = float(np.arccos(cos_sim))
    sam_deg = float(sam_rad * 180.0 / np.pi)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "relative_rmse": relative_rmse,
        "nrmse_mean": nrmse_mean,
        "nrmse_range": nrmse_range,
        "mean_relative_error": mean_relative_error,
        "median_relative_error": median_relative_error,
        "max_relative_error": max_relative_error,
        "mape_percent": mape_percent,
        "r2": r2,
        "sam_rad": sam_rad,
        "sam_deg": sam_deg,
        "relative_error_eps": float(relative_error_eps),
    }


def compute_wavelength_error_summary(
    y_pred: np.ndarray,
    y_real: np.ndarray,
    wavelengths: np.ndarray,
    relative_error_eps: float = 1e-3,
    numerical_eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute wavelength-wise error diagnostics.

    Important:
        True SAM is a vector angle and is not defined per wavelength.
        The column `sam_contribution_proxy` is a diagnostic proxy:
            abs( y_pred/||y_pred|| - y_real/||y_real|| )
        It indicates which wavelengths contribute strongly to angular mismatch.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_real = np.asarray(y_real, dtype=np.float64)

    if y_pred.shape != y_real.shape:
        raise ValueError(f"Shape mismatch: pred={y_pred.shape}, real={y_real.shape}")

    err = y_pred - y_real
    abs_err = np.abs(err)
    denom = np.maximum(np.abs(y_real), float(relative_error_eps))
    rel_err = abs_err / denom

    pred_norm = y_pred / np.maximum(
        np.linalg.norm(y_pred, axis=1, keepdims=True),
        numerical_eps,
    )
    real_norm = y_real / np.maximum(
        np.linalg.norm(y_real, axis=1, keepdims=True),
        numerical_eps,
    )
    sam_proxy = np.abs(pred_norm - real_norm)

    return pd.DataFrame(
        {
            "wavelength": wavelengths,
            "bias_mean": np.mean(err, axis=0),
            "bias_std": np.std(err, axis=0, ddof=1)
            if err.shape[0] > 1
            else np.zeros(err.shape[1]),
            "mae_mean": np.mean(abs_err, axis=0),
            "mae_std": np.std(abs_err, axis=0, ddof=1)
            if err.shape[0] > 1
            else np.zeros(err.shape[1]),
            "rmse": np.sqrt(np.mean(err**2, axis=0)),
            "mre_mean": np.mean(rel_err, axis=0),
            "mre_std": np.std(rel_err, axis=0, ddof=1)
            if rel_err.shape[0] > 1
            else np.zeros(rel_err.shape[1]),
            "sam_contribution_proxy_mean": np.mean(sam_proxy, axis=0),
            "sam_contribution_proxy_std": np.std(sam_proxy, axis=0, ddof=1)
            if sam_proxy.shape[0] > 1
            else np.zeros(sam_proxy.shape[1]),
        }
    )


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
    if args.stats_source == "stage":
        return eval_stage
    if args.stats_source == "all":
        return "all"

    norm_scope = get_cfg_value(cfg, ["IMAGE_NORMALIZATION_SCOPE"], "stage_band")
    if norm_scope == "stage_band":
        return eval_stage
    return "all"


def _normalize_key_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().replace("\\", "/").lower()
    return value


def check_train_test_overlap(
    train_csv: str,
    test_csv: str,
    key_cols: Optional[List[str]] = None,
):
    train_csv = expand_path(train_csv)
    test_csv = expand_path(test_csv)

    if Path(train_csv).resolve() == Path(test_csv).resolve():
        raise ValueError(
            "Data leakage risk: train_csv and test_csv are the same file. "
            "Pass a true held-out TEST_CSV, or use --allow-train-test-overlap only for debugging."
        )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if key_cols is None:
        candidate_cols = ["blue", "green", "red", "nir", "red_edge"]
        key_cols = [
            c for c in candidate_cols if c in train_df.columns and c in test_df.columns
        ]

    if not key_cols:
        print("Warning: no common image filename columns found for overlap check.")
        return

    def make_keys(df):
        return set(
            tuple(_normalize_key_value(row[c]) for c in key_cols)
            for _, row in df[key_cols].iterrows()
        )

    overlap = make_keys(train_df).intersection(make_keys(test_df))
    if overlap:
        examples = list(overlap)[:5]
        raise ValueError(
            f"Data leakage risk: found {len(overlap)} exact image-file tuple(s) "
            f"appearing in both train and test CSVs using columns {key_cols}. "
            f"Examples: {examples}. "
            "Fix the split, or use --allow-train-test-overlap only for debugging."
        )

    print(
        f"Split-overlap check passed: no shared image-file tuples between train and test "
        f"using columns {key_cols}."
    )


# -------------------------------------------------------------------------
# Checkpoint/model-loop helpers
# -------------------------------------------------------------------------


def build_model_specs(args) -> List[Dict[str, str]]:
    if len(args.experiment_dirs) != len(args.mode_labels):
        raise ValueError(
            "--experiment-dirs and --mode-labels must have the same length. "
            f"Got {len(args.experiment_dirs)} dirs and {len(args.mode_labels)} labels."
        )

    specs = []
    results_root = Path(args.results_root).expanduser().resolve()

    for exp_dir, mode_label in zip(args.experiment_dirs, args.mode_labels):
        exp_path = results_root / exp_dir
        specs.append(
            {
                "mode_label": mode_label,
                "experiment_dir": exp_dir,
                "experiment_path": str(exp_path),
            }
        )

    return specs


def checkpoint_for_model_stage(args, model_spec: Dict[str, str], stage: str) -> str:
    if args.checkpoint_template is None:
        template = (
            "{results_root}/{experiment_dir}/"
            "{experiment_prefix}_{stage}_gen_best.pth.tar"
        )
    else:
        template = args.checkpoint_template

    path = template.format(
        results_root=str(Path(args.results_root).expanduser().resolve()),
        experiment_dir=model_spec["experiment_dir"],
        experiment_path=model_spec["experiment_path"],
        mode_label=model_spec["mode_label"],
        experiment_prefix=args.experiment_prefix,
        stage=stage,
    )
    return expand_path(path)


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------


def evaluate_model_on_stage(
    args,
    cfg,
    model_spec: Dict[str, str],
    stage: str,
    device: torch.device,
    normalization_stats_cache: Dict[str, Any],
):
    dataset_module = args.dataset_module
    generator_module = args.generator_module

    train_csv = expand_path(args.train_csv)
    test_csv = expand_path(args.test_csv)
    img_dir = expand_path(args.img_dir)

    stage = canonical_stage_name(stage)
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

    print(
        f"Building TEST dataset for mode='{model_spec['mode_label']}', "
        f"stage='{stage}'..."
    )
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
        return [], None, None, [], []

    patch_collate_fn = import_from_module(dataset_module, "patch_collate_fn")

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=patch_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = checkpoint_for_model_stage(args, model_spec, stage)
    print(
        f"Loading generator: mode='{model_spec['mode_label']}', "
        f"stage='{stage}', checkpoint='{checkpoint_path}'"
    )

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
    generated_rows = []
    real_rows = []

    non_blocking = device.type == "cuda"
    wavelengths = None
    wl_cols = None

    with torch.no_grad():
        for local_idx, (batch_bands, y_real) in enumerate(loader):
            batch_bands = move_batch_bands_to_device(
                batch_bands,
                device,
                non_blocking=non_blocking,
            )
            y_real = y_real.to(device, non_blocking=non_blocking).float()

            with torch.amp.autocast("cuda", enabled=False):
                y_fake, p_params = gen.forward_batch_list(batch_bands)

            y_fake = y_fake.float()

            y_real_np = y_real[0].detach().cpu().numpy().reshape(-1)
            y_fake_np = y_fake[0].detach().cpu().numpy().reshape(-1)

            if wavelengths is None:
                wavelengths = make_wavelengths_from_config(
                    cfg, fallback_count=len(y_fake_np)
                )
                wl_cols = wavelength_columns(wavelengths)

            if not np.isfinite(y_real_np).all():
                raise FloatingPointError(
                    f"Non-finite y_real at mode={model_spec['mode_label']}, "
                    f"stage={stage}, index={local_idx}"
                )
            if not np.isfinite(y_fake_np).all():
                raise FloatingPointError(
                    f"Non-finite y_fake at mode={model_spec['mode_label']}, "
                    f"stage={stage}, index={local_idx}"
                )

            metrics = compute_sample_metrics(
                y_fake_np,
                y_real_np,
                relative_error_eps=args.relative_error_eps,
            )

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

            base_meta = {
                "discriminator_mode": model_spec["mode_label"],
                "experiment_dir": model_spec["experiment_dir"],
                "stage_eval": stage,
                "sample_index_within_stage": local_idx,
                "checkpoint": checkpoint_path,
                **row_meta,
            }

            p_np = p_params[0].detach().cpu().numpy()

            rows.append(
                {
                    **base_meta,
                    **metrics,
                    "params_shape": str(tuple(p_np.shape)),
                    "params_json": json.dumps(np.asarray(p_np, dtype=float).tolist()),
                }
            )

            generated_rows.append(
                {
                    **base_meta,
                    "spectrum_type": "generated",
                    **{c: float(v) for c, v in zip(wl_cols, y_fake_np)},
                }
            )

            real_rows.append(
                {
                    **base_meta,
                    "spectrum_type": "real",
                    **{c: float(v) for c, v in zip(wl_cols, y_real_np)},
                }
            )

            all_real.append(y_real_np)
            all_pred.append(y_fake_np)

    real_arr = np.stack(all_real, axis=0)
    pred_arr = np.stack(all_pred, axis=0)

    return rows, real_arr, pred_arr, generated_rows, real_rows


# -------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------


def make_mean_std_tables(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Create compact model-comparison tables using mean and std.

    Main table requested by user:
        RMSE, MAE, bias, MRE
    """
    primary_metrics = ["rmse", "mae", "bias", "mean_relative_error"]
    optional_metrics = ["sam_deg", "r2", "relative_rmse", "mape_percent"]

    group_cols = ["discriminator_mode", "stage_eval"]

    rows = []
    for (mode, stage), g in metrics_df.groupby(group_cols):
        row = {
            "discriminator_mode": mode,
            "stage_eval": stage,
            "n": int(len(g)),
        }
        for metric in primary_metrics + optional_metrics:
            if metric not in g.columns:
                continue
            row[f"{metric}_mean"] = round(float(g[metric].mean()), 3)
            row[f"{metric}_std"] = (
                round(float(g[metric].std(ddof=1)), 3) if len(g) > 1 else 0.00
            )
        rows.append(row)

    by_stage_mode = pd.DataFrame(rows)
    by_stage_mode.to_csv(
        output_dir / "model_comparison_mean_std_by_stage.csv", index=False
    )

    rows = []
    for mode, g in metrics_df.groupby("discriminator_mode"):
        row = {
            "discriminator_mode": mode,
            "n": int(len(g)),
        }
        for metric in primary_metrics + optional_metrics:
            if metric not in g.columns:
                continue
            row[f"{metric}_mean"] = round(float(g[metric].mean()), 3)
            row[f"{metric}_std"] = (
                round(float(g[metric].std(ddof=1)), 3) if len(g) > 1 else 0.00
            )
        rows.append(row)

    by_mode = pd.DataFrame(rows)
    by_mode.to_csv(output_dir / "model_comparison_mean_std_by_mode.csv", index=False)

    # Compact markdown table for reports.
    md_cols = [
        "discriminator_mode",
        "stage_eval",
        "n",
        "rmse_mean_std",
        "mae_mean_std",
        "bias_mean_std",
        "mean_relative_error_mean_std",
    ]
    md_cols = [c for c in md_cols if c in by_stage_mode.columns]
    md = by_stage_mode[md_cols].to_markdown(index=False)
    (output_dir / "model_comparison_mean_std_by_stage.md").write_text(md)

    return by_stage_mode, by_mode


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------


def plot_qualitative_by_stage(
    stage_to_real: Dict[Tuple[str, str], np.ndarray],
    stage_to_pred: Dict[Tuple[str, str], np.ndarray],
    wavelengths: np.ndarray,
    stages: List[str],
    mode_label: str,
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
        key = (mode_label, stage)
        real = stage_to_real.get(key)
        pred = stage_to_pred.get(key)

        if real is None or pred is None or real.size == 0:
            ax.set_title(f"{mode_label} | {stage}: no samples")
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

        ax.set_title(f"{mode_label} | {stage} | n={real.shape[0]}")
        ax.set_ylabel("Reflectance")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Wavelength (nm)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(
        f"Real spectral signatures and generated estimates | {mode_label}",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_wavelength_error_curves(
    wavelength_error_df: pd.DataFrame,
    mode_labels: List[str],
    stages: List[str],
    output_dir: Path,
):
    """
    Plot wavelength-wise error curves per stage, comparing modes.

    The SAM-related plot is a contribution proxy, not true per-wavelength SAM.
    """
    plots_dir = ensure_dir(output_dir / "plots")

    plot_specs = [
        ("rmse", "Wavelength-wise RMSE", "RMSE"),
        ("mae_mean", "Wavelength-wise MAE", "MAE"),
        ("bias_mean", "Wavelength-wise bias", "Bias"),
        ("mre_mean", "Wavelength-wise mean relative error", "MRE"),
        (
            "sam_contribution_proxy_mean",
            "Wavelength-wise SAM contribution proxy",
            "SAM contribution proxy",
        ),
    ]

    for value_col, title_base, ylabel in plot_specs:
        for stage in stages:
            fig, ax = plt.subplots(figsize=(14, 5))

            for mode in mode_labels:
                sub = wavelength_error_df[
                    (wavelength_error_df["stage_eval"] == stage)
                    & (wavelength_error_df["discriminator_mode"] == mode)
                ].sort_values("wavelength")

                if sub.empty:
                    continue

                ax.plot(
                    sub["wavelength"].to_numpy(),
                    sub[value_col].to_numpy(),
                    linewidth=1.4,
                    label=mode,
                )

            ax.set_title(f"{title_base} | {stage}")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(
                plots_dir
                / f"wavelength_{safe_filename(value_col)}_{safe_filename(stage)}.svg",
                dpi=220,
            )
            plt.close(fig)

        # Aggregate over stages for each mode.
        fig, ax = plt.subplots(figsize=(14, 5))
        for mode in mode_labels:
            sub = wavelength_error_df[wavelength_error_df["discriminator_mode"] == mode]
            if sub.empty:
                continue

            agg = (
                sub.groupby("wavelength", as_index=False)[value_col]
                .mean(numeric_only=True)
                .sort_values("wavelength")
            )
            ax.plot(
                agg["wavelength"].to_numpy(),
                agg[value_col].to_numpy(),
                linewidth=1.6,
                label=mode,
            )

        ax.set_title(f"{title_base} | average across stages")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"wavelength_{safe_filename(value_col)}_all_stages_mean.svg",
            dpi=220,
        )
        plt.close(fig)


def make_all_plots(
    stage_to_real: Dict[Tuple[str, str], np.ndarray],
    stage_to_pred: Dict[Tuple[str, str], np.ndarray],
    wavelengths: np.ndarray,
    stages: List[str],
    mode_labels: List[str],
    output_dir: Path,
    max_pred_lines: int,
):
    plots_dir = ensure_dir(output_dir / "plots")

    for mode in mode_labels:
        out_path = plots_dir / f"qualitative_spectra_{safe_filename(mode)}.svg"
        plot_qualitative_by_stage(
            stage_to_real=stage_to_real,
            stage_to_pred=stage_to_pred,
            wavelengths=wavelengths,
            stages=stages,
            mode_label=mode,
            out_path=str(out_path),
            max_pred_lines=max_pred_lines,
        )

    return plots_dir


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained pix2spectral generators on a testing set, export "
            "generated spectra, and compare discriminator modes."
        )
    )

    parser.add_argument("--config-module", default="config_cgan")
    parser.add_argument("--dataset-module", default="dataset")
    parser.add_argument("--generator-module", default="generator_model_cgan")

    parser.add_argument(
        "--test-csv",
        default="~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv",
    )
    parser.add_argument(
        "--train-csv",
        default="~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv",
    )
    parser.add_argument(
        "--img-dir",
        default="/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/",
    )

    parser.add_argument(
        "--results-root",
        default="~/Results/pix2spectral",
        help="Root folder containing avocado_global/, avocado_segmented/, etc.",
    )
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        default=DEFAULT_EXPERIMENT_DIRS,
        help="Experiment folders inside --results-root.",
    )
    parser.add_argument(
        "--mode-labels",
        nargs="+",
        default=DEFAULT_MODE_LABELS,
        help="Labels corresponding to --experiment-dirs.",
    )
    parser.add_argument("--experiment-prefix", default="avocado")

    parser.add_argument(
        "--checkpoint-template",
        default=None,
        help=(
            "Optional checkpoint template. Available fields: "
            "{results_root}, {experiment_dir}, {experiment_path}, "
            "{mode_label}, {experiment_prefix}, {stage}. "
            "Default: {results_root}/{experiment_dir}/"
            "{experiment_prefix}_{stage}_gen_best.pth.tar"
        ),
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        help="Stages to evaluate and plot.",
    )

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
    parser.add_argument(
        "--relative-error-eps",
        type=float,
        default=1e-3,
        help=(
            "Denominator floor for relative-error metrics. "
            "Use a nonzero value because reflectance can be near zero."
        ),
    )
    parser.add_argument(
        "--allow-train-test-overlap",
        action="store_true",
        help=(
            "Disable train/test exact image-file overlap checks. "
            "Use only for debugging, not final reporting."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = importlib.import_module(args.config_module)

    if args.test_csv is None:
        args.test_csv = get_cfg_value(cfg, ["TEST_CSV"], None)
    if args.train_csv is None:
        args.train_csv = get_cfg_value(cfg, ["TRAIN_CSV"], None)
    if args.img_dir is None:
        args.img_dir = get_cfg_value(
            cfg, ["TEST_IMG_DIR", "VAL_IMG_DIR", "TRAIN_IMG_DIR"], None
        )

    if args.test_csv is None:
        raise ValueError("No --test-csv provided and config has no TEST_CSV.")
    if args.train_csv is None:
        raise ValueError("No --train-csv provided and config has no TRAIN_CSV.")
    if args.img_dir is None:
        raise ValueError(
            "No --img-dir provided and config has no TEST_IMG_DIR/VAL_IMG_DIR/TRAIN_IMG_DIR."
        )

    stages = [canonical_stage_name(s) for s in args.stages]
    mode_labels = [str(m).strip() for m in args.mode_labels]

    results_root = ensure_dir(args.results_root)

    if args.output_dir is None:
        args.output_dir = str(results_root / "test_evaluation")
    output_dir = ensure_dir(args.output_dir)

    if not args.allow_train_test_overlap:
        check_train_test_overlap(args.train_csv, args.test_csv)

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
    model_specs = build_model_specs(args)

    print("=" * 80)
    print("pix2spectral test-set evaluation and discriminator comparison")
    print("=" * 80)
    print(f"Device:            {device}")
    print(f"Train CSV:         {expand_path(args.train_csv)}")
    print(f"Test CSV:          {expand_path(args.test_csv)}")
    print(f"Image dir:         {expand_path(args.img_dir)}")
    print(f"Results root:      {results_root}")
    print(f"Output dir:        {output_dir}")
    print(f"Stages:            {stages}")
    print(f"Modes:             {mode_labels}")
    print(f"Dataset module:    {args.dataset_module}")
    print(f"Generator module:  {args.generator_module}")
    print("=" * 80)

    normalization_stats_cache = {}

    all_metric_rows = []
    all_generated_rows = []
    all_real_rows = []
    all_wavelength_error_rows = []

    stage_to_real = {}
    stage_to_pred = {}
    first_spectrum_len = None
    wavelengths = None

    for model_spec in model_specs:
        mode = model_spec["mode_label"]

        print("\n" + "#" * 80)
        print(f"Evaluating discriminator mode: {mode}")
        print(f"Experiment folder: {model_spec['experiment_path']}")
        print("#" * 80)

        for stage in stages:
            print("\n" + "-" * 80)
            print(f"Evaluating mode='{mode}' on matching test stage='{stage}'")
            print("-" * 80)

            rows, real_arr, pred_arr, gen_rows, real_rows = evaluate_model_on_stage(
                args=args,
                cfg=cfg,
                model_spec=model_spec,
                stage=stage,
                device=device,
                normalization_stats_cache=normalization_stats_cache,
            )

            if not rows:
                continue

            all_metric_rows.extend(rows)
            all_generated_rows.extend(gen_rows)
            all_real_rows.extend(real_rows)

            stage_to_real[(mode, stage)] = real_arr
            stage_to_pred[(mode, stage)] = pred_arr

            if first_spectrum_len is None:
                first_spectrum_len = pred_arr.shape[1]
                wavelengths = make_wavelengths_from_config(
                    cfg, fallback_count=first_spectrum_len
                )

            wl_summary = compute_wavelength_error_summary(
                y_pred=pred_arr,
                y_real=real_arr,
                wavelengths=wavelengths,
                relative_error_eps=args.relative_error_eps,
            )
            wl_summary.insert(0, "stage_eval", stage)
            wl_summary.insert(0, "discriminator_mode", mode)
            wl_summary.insert(0, "experiment_dir", model_spec["experiment_dir"])
            all_wavelength_error_rows.extend(wl_summary.to_dict("records"))

            mode_stage = f"{safe_filename(mode)}_{safe_filename(stage)}"
            pd.DataFrame(rows).to_csv(
                output_dir / f"test_metrics_{mode_stage}.csv",
                index=False,
            )
            pd.DataFrame(gen_rows).to_csv(
                output_dir / f"generated_spectra_{mode_stage}.csv",
                index=False,
            )
            pd.DataFrame(real_rows).to_csv(
                output_dir / f"real_spectra_{mode_stage}.csv",
                index=False,
            )
            wl_summary.to_csv(
                output_dir / f"wavelength_error_summary_{mode_stage}.csv",
                index=False,
            )

    if not all_metric_rows:
        raise RuntimeError(
            "No evaluation rows were produced. Check test CSV/stage filters and checkpoint paths."
        )

    metrics_df = pd.DataFrame(all_metric_rows)
    generated_df = pd.DataFrame(all_generated_rows)
    real_df = pd.DataFrame(all_real_rows)
    wavelength_error_df = pd.DataFrame(all_wavelength_error_rows)

    metrics_csv = output_dir / "test_metrics_all_modes_all_stages.csv"
    generated_csv = output_dir / "generated_spectra_all_modes_all_stages.csv"
    real_csv = output_dir / "real_spectra_all_modes_all_stages.csv"
    wl_error_csv = output_dir / "wavelength_error_summary_all_modes_all_stages.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    generated_df.to_csv(generated_csv, index=False)
    real_df.to_csv(real_csv, index=False)
    wavelength_error_df.to_csv(wl_error_csv, index=False)

    by_stage_mode, by_mode = make_mean_std_tables(metrics_df, output_dir)

    plots_dir = make_all_plots(
        stage_to_real=stage_to_real,
        stage_to_pred=stage_to_pred,
        wavelengths=wavelengths,
        stages=stages,
        mode_labels=mode_labels,
        output_dir=output_dir,
        max_pred_lines=args.max_pred_lines,
    )

    plot_wavelength_error_curves(
        wavelength_error_df=wavelength_error_df,
        mode_labels=mode_labels,
        stages=stages,
        output_dir=output_dir,
    )

    # Add a small text note explaining the SAM proxy.
    note = (
        "True SAM is a vector-angle metric and is not mathematically defined per wavelength.\\n"
        "The plotted sam_contribution_proxy is abs(y_pred/||y_pred|| - y_real/||y_real||) per wavelength,\\n"
        "averaged over test samples. It should be interpreted as a wavelength-wise angular-mismatch diagnostic,\\n"
        "not as a true SAM angle in degrees.\\n"
    )
    (output_dir / "sam_contribution_proxy_note.txt").write_text(note)

    print("\n" + "=" * 80)
    print("Evaluation finished")
    print("=" * 80)
    print(f"Per-sample metrics:          {metrics_csv}")
    print(f"Generated spectra:           {generated_csv}")
    print(f"Real spectra:                {real_csv}")
    print(f"Wavelength error summary:    {wl_error_csv}")
    print(
        f"Mean/std by stage:           {output_dir / 'model_comparison_mean_std_by_stage.csv'}"
    )
    print(
        f"Mean/std by mode:            {output_dir / 'model_comparison_mean_std_by_mode.csv'}"
    )
    print(f"Plots directory:             {plots_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
