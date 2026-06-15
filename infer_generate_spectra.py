#!/usr/bin/env python3
"""
Inference-only spectrum generation for pix2spectral.

Purpose
-------
Given a CSV file with multispectral image filenames, generate leaf reflectance
spectra using a trained pix2spectral generator checkpoint.

Intended model configuration
----------------------------
This script forces the inference generator to use the selected/best architecture:

    Generator: segmented PROSPECT + residual branch + separate band encoders
    Discriminator during training: global spectral discriminator

The discriminator is not needed during inference, but the discriminator mode is
recorded in the output metadata for traceability.

CSV input
---------
Preferred CSV columns:

    blue, green, red, nir, red_edge, Species, Stages

The dataset class also expects a `spectral` column. For pure inference, if the
input CSV does not contain `spectral`, this script automatically creates a
temporary dummy spectral column only to satisfy the dataloader interface. The
dummy spectrum is not used for prediction.

If `Species` or `Stages` are missing, they are also created using:
    Species -> --species-filter or "unknown"
    Stages  -> --stage or "all"

Outputs
-------
Inside --output-dir:

    generated_spectra_wide.csv
        one row per input sample, with metadata/image filenames and wl_* columns

    generated_spectra_long.csv
        optional; one row per sample-wavelength pair if --write-long is used

    generated_spectra.npy
        optional; array [N, wavelength_count] if --save-npy is used

    prospect_parameters.csv
        generated PROSPECT/segment parameters per sample

    inference_manifest.json
        configuration and checkpoint metadata

Example: stage-specific checkpoints
-----------------------------------

python infer_generate_spectra.py \
    --input-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
    --stats-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --results-root ~/Results/pix2spectral_groupB_loss_ablation_separate \
    --experiment-dir L5_full_physics_informed_loss \
    --experiment-prefix avocado \
    --stages auto \
    --output-dir ~/Results/pix2spectral_inference/avocado_L5

Example: one checkpoint for all samples
---------------------------------------

python infer_generate_spectra.py \
    --input-csv new_samples.csv \
    --stats-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_train.csv \
    --img-dir "/home/usr3/Data/EstradaDataset/Avocado/Multispectral Images/" \
    --checkpoint ~/Results/my_model/avocado_all_gen_best.pth.tar \
    --stage all \
    --output-dir ~/Results/pix2spectral_inference/new_samples
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


DEFAULT_BANDS = ["blue", "green", "red", "nir", "red_edge"]
DEFAULT_STAGE_ORDER = ["fresh", "stage1", "stage2", "stage3", "dry"]


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


def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def canonical_stage_name(value: Any) -> str:
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


def ordered_unique_stages(values: Iterable[Any]) -> List[str]:
    found = []
    for value in values:
        stage = canonical_stage_name(value)
        if stage not in found:
            found.append(stage)

    ordered = [s for s in DEFAULT_STAGE_ORDER if s in found]
    ordered += [s for s in found if s not in ordered and s != "all"]

    if not ordered:
        ordered = ["all"]

    return ordered


def wavelength_columns(wavelengths: np.ndarray) -> List[str]:
    cols = []
    for wl in wavelengths:
        if abs(float(wl) - round(float(wl))) < 1e-8:
            cols.append(f"wl_{int(round(float(wl)))}")
        else:
            cols.append(f"wl_{float(wl):.2f}")
    return cols


def make_wavelengths_from_config(cfg, fallback_count: Optional[int] = None) -> np.ndarray:
    wl_min = float(get_cfg_value(cfg, ["WAVELENGTH_MIN", "wavelength_min"], 400.0))
    wl_max = float(get_cfg_value(cfg, ["WAVELENGTH_MAX", "wavelength_max"], 2500.0))
    wl_count = int(
        get_cfg_value(
            cfg,
            ["WAVELENGTH_COUNT", "wavelength_count"],
            2101 if fallback_count is None else fallback_count,
        )
    )
    if fallback_count is not None:
        wl_count = int(fallback_count)
    return np.linspace(wl_min, wl_max, wl_count, dtype=np.float64)


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        out[key] = value
    return out


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in [
            "state_dict",
            "generator_state_dict",
            "gen_state_dict",
            "model_state_dict",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        # Some checkpoints are directly state_dict-like dictionaries.
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            tensor_like = [torch.is_tensor(v) for v in ckpt.values()]
            if any(tensor_like):
                return ckpt

    raise ValueError("Unsupported checkpoint format: could not locate state_dict.")


def load_generator_checkpoint(
    checkpoint_path: str,
    gen: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint_path = expand_path(checkpoint_path)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(ckpt))

    missing, unexpected = gen.load_state_dict(state_dict, strict=strict)
    if not strict:
        if missing:
            print(f"Warning: missing keys while loading {checkpoint_path}: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys while loading {checkpoint_path}: {unexpected}")

    return ckpt if isinstance(ckpt, dict) else {"state_dict": state_dict}


def inspect_checkpoint_architecture(checkpoint_path: str) -> Dict[str, Any]:
    ckpt = torch.load(expand_path(checkpoint_path), map_location="cpu")
    sd = normalize_state_dict_keys(extract_state_dict(ckpt))
    keys = list(sd.keys())

    has_shared = any(k.startswith("patch_encoder.") for k in keys)
    has_separate = any(k.startswith("patch_encoders.") for k in keys)
    has_residual = any(k.startswith("residual_mlp.") for k in keys)

    if has_separate and not has_shared:
        encoder_mode = "separate"
    elif has_shared and not has_separate:
        encoder_mode = "shared"
    elif has_shared and has_separate:
        encoder_mode = "mixed"
    else:
        encoder_mode = "unknown"

    param_shape = None
    prospect_mode = "unknown"
    if "param_mlp.9.weight" in sd:
        param_shape = tuple(sd["param_mlp.9.weight"].shape)
        out_dim = int(param_shape[0])
        if out_dim == 7:
            prospect_mode = "full"
        elif out_dim % 7 == 0:
            prospect_mode = f"segmented_{out_dim // 7}_segments"

    return {
        "encoder_mode": encoder_mode,
        "has_residual_mlp": bool(has_residual),
        "param_mlp_9_weight_shape": param_shape,
        "prospect_mode": prospect_mode,
    }


# -------------------------------------------------------------------------
# Config/model/dataset construction
# -------------------------------------------------------------------------

def apply_inference_model_config(cfg, args):
    """
    Force the intended inference architecture.

    Training discriminator is global. The discriminator itself is not used for
    inference, but this flag is written for traceability.
    """
    setattr(cfg, "DISCRIMINATOR_MODE", "global")
    setattr(cfg, "USE_SEGMENTED_PROSPECT", True)
    setattr(cfg, "USE_SEGMENT_RESIDUAL", True)
    setattr(cfg, "BAND_ENCODER_MODE", "separate")

    if args.species_filter is not None:
        if str(args.species_filter).strip().lower() in ["all", "any", "*", "none", ""]:
            setattr(cfg, "SPECIES_FILTER", None)
        else:
            setattr(cfg, "SPECIES_FILTER", args.species_filter)

    # Inference stability defaults. These only affect dataloader behavior.
    if args.max_patches_per_band is not None:
        setattr(cfg, "MAX_PATCHES_PER_BAND", int(args.max_patches_per_band))
    if args.min_patches_per_band is not None:
        setattr(cfg, "MIN_PATCHES", int(args.min_patches_per_band))

    return cfg


def build_generator(cfg, generator_module: str, device: torch.device):
    GeneratorClass = import_from_module(
        generator_module,
        "MultiSpectralPatchToProspectGenerator",
    )

    bands = get_cfg_value(cfg, ["BANDS"], DEFAULT_BANDS)

    kwargs = {
        "bands": bands,
        "base_features": get_cfg_value(cfg, ["BASE_FEATURES"], 8),
        "embed_dim": get_cfg_value(cfg, ["EMBED_DIM"], 64),
        "mins": get_cfg_value(cfg, ["PROSPECT_PARAM_MINS"], None),
        "maxs": get_cfg_value(cfg, ["PROSPECT_PARAM_MAXS"], None),
        "wavelength_min": get_cfg_value(cfg, ["WAVELENGTH_MIN", "wavelength_min"], 400.0),
        "wavelength_max": get_cfg_value(cfg, ["WAVELENGTH_MAX", "wavelength_max"], 2500.0),
        "wavelength_count": get_cfg_value(cfg, ["WAVELENGTH_COUNT", "wavelength_count"], 2101),
        "spectral_segments": get_cfg_value(
            cfg,
            ["SPECTRAL_SEGMENTS"],
            [(400.0, 700.0), (700.0, 800.0), (800.0, 1400.0), (1400.0, 2500.0)],
        ),
        "use_segmented_prospect": True,
        "use_segment_residual": True,
        "segment_residual_scale": get_cfg_value(cfg, ["SEGMENT_RESIDUAL_SCALE"], 0.05),
        "segment_blend_width": get_cfg_value(cfg, ["SEGMENT_BLEND_WIDTH"], 0),
        "patch_encoder_type": get_cfg_value(cfg, ["PATCH_ENCODER_TYPE"], "cnn"),
        "pooling_type": get_cfg_value(cfg, ["POOLING_TYPE"], "attention_stats"),
        "band_encoder_mode": "separate",
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
    norm_mode = get_cfg_value(cfg, ["IMAGE_NORMALIZATION_MODE"], "stage_band_robust_zscore")
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


def move_batch_bands_to_device(batch_bands, device, non_blocking=False):
    out = {}
    for band, tensors in batch_bands.items():
        out[band] = [t.to(device, non_blocking=non_blocking) for t in tensors]
    return out


# -------------------------------------------------------------------------
# CSV preparation
# -------------------------------------------------------------------------

def dummy_spectral_string(cfg) -> str:
    spectral_drop = int(get_cfg_value(cfg, ["SPECTRAL_DROP_FIRST_N"], 50))
    wl_count = int(get_cfg_value(cfg, ["WAVELENGTH_COUNT", "wavelength_count"], 2101))
    dummy_len = wl_count + max(0, spectral_drop)
    return "[" + ",".join(["0.0"] * dummy_len) + "]"


def prepare_csv_for_dataset(
    source_csv: str,
    cfg,
    default_stage: str,
    species_filter: Optional[str],
    temp_dir: Path,
    role: str,
) -> str:
    source_csv = expand_path(source_csv)
    df = pd.read_csv(source_csv)

    missing_bands = [b for b in DEFAULT_BANDS if b not in df.columns]
    if missing_bands:
        raise ValueError(
            f"{role} CSV is missing required image columns: {missing_bands}. "
            f"Expected columns include: {DEFAULT_BANDS}"
        )

    df = df.copy()
    if "__input_row_id" not in df.columns:
        df.insert(0, "__input_row_id", np.arange(len(df), dtype=int))

    if "Species" not in df.columns:
        df["Species"] = species_filter if species_filter not in [None, "", "all", "any", "*"] else "unknown"

    if "Stages" not in df.columns:
        df["Stages"] = canonical_stage_name(default_stage)

    df["Stages"] = df["Stages"].map(canonical_stage_name)

    if "spectral" not in df.columns:
        df["spectral"] = dummy_spectral_string(cfg)

    out = temp_dir / f"{role}_prepared_for_pix2spectral.csv"
    df.to_csv(out, index=False)
    return str(out)


# -------------------------------------------------------------------------
# Checkpoint selection
# -------------------------------------------------------------------------

def checkpoint_for_stage(args, stage: str) -> str:
    if args.checkpoint is not None:
        return expand_path(args.checkpoint)

    if args.checkpoint_template is not None:
        template = args.checkpoint_template
    else:
        template = (
            "{results_root}/{experiment_dir}/"
            "{experiment_prefix}_{stage}_gen_best.pth.tar"
        )

    path = template.format(
        results_root=str(Path(args.results_root).expanduser().resolve()),
        experiment_dir=args.experiment_dir,
        experiment_prefix=args.experiment_prefix,
        stage=stage,
    )
    return expand_path(path)


def choose_stats_stage(args, cfg, stage: str) -> str:
    if args.stats_source == "stage":
        return stage
    if args.stats_source == "all":
        return "all"

    norm_scope = get_cfg_value(cfg, ["IMAGE_NORMALIZATION_SCOPE"], "stage_band")
    if norm_scope == "stage_band":
        return stage
    return "all"


def determine_stages(args, prepared_input_csv: str) -> List[str]:
    if args.stage is not None:
        return [canonical_stage_name(args.stage)]

    if len(args.stages) == 1 and str(args.stages[0]).lower() == "auto":
        df = pd.read_csv(prepared_input_csv)
        return ordered_unique_stages(df["Stages"].tolist())

    return [canonical_stage_name(s) for s in args.stages]


# -------------------------------------------------------------------------
# Inference loop
# -------------------------------------------------------------------------

def run_inference_for_stage(
    args,
    cfg,
    stage: str,
    prepared_input_csv: str,
    prepared_stats_csv: str,
    device: torch.device,
    normalization_stats_cache: Dict[str, Any],
):
    dataset_module = args.dataset_module
    generator_module = args.generator_module
    img_dir = expand_path(args.img_dir)

    stats_stage = choose_stats_stage(args, cfg, stage)
    if stats_stage not in normalization_stats_cache:
        print(f"Computing normalization stats from stats CSV with stage='{stats_stage}'...")
        stats_dataset = build_dataset(
            cfg=cfg,
            dataset_module=dataset_module,
            csv_path=prepared_stats_csv,
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

    print(f"Building inference dataset for stage='{stage}'...")
    infer_dataset = build_dataset(
        cfg=cfg,
        dataset_module=dataset_module,
        csv_path=prepared_input_csv,
        img_dir=img_dir,
        stage=stage,
        normalization_stats=normalization_stats,
        compute_normalization_stats=False,
        cache_patches=args.cache_patches,
    )

    patch_collate_fn = import_from_module(dataset_module, "patch_collate_fn")
    loader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=patch_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = checkpoint_for_stage(args, stage)
    print(f"Loading generator checkpoint for stage='{stage}': {checkpoint_path}")

    ckpt_arch = inspect_checkpoint_architecture(checkpoint_path)
    print(f"Checkpoint architecture: {ckpt_arch}")
    if ckpt_arch["encoder_mode"] != "separate":
        raise RuntimeError(
            "Checkpoint does not appear to use separate band encoders. "
            f"Detected encoder_mode='{ckpt_arch['encoder_mode']}'. "
            "This inference script is intended for the corrected model with "
            "BAND_ENCODER_MODE='separate'."
        )
    if not ckpt_arch["has_residual_mlp"]:
        raise RuntimeError(
            "Checkpoint does not contain residual_mlp.* keys. "
            "Expected segmented + residual generator."
        )

    gen = build_generator(cfg, generator_module, device)
    load_generator_checkpoint(
        checkpoint_path=checkpoint_path,
        gen=gen,
        device=device,
        strict=not args.non_strict_load,
    )
    gen.eval()

    generated_rows = []
    params_rows = []
    spectra_list = []

    wavelengths = None
    wl_cols = None
    non_blocking = device.type == "cuda"

    sample_counter = 0

    with torch.no_grad():
        for batch_idx, (batch_bands, _dummy_y) in enumerate(loader):
            batch_bands = move_batch_bands_to_device(
                batch_bands,
                device,
                non_blocking=non_blocking,
            )

            with torch.amp.autocast("cuda", enabled=False):
                y_fake, p_params = gen.forward_batch_list(batch_bands)

            y_fake = y_fake.float().detach().cpu().numpy()
            p_params = p_params.float().detach().cpu().numpy()

            if y_fake.ndim != 2:
                y_fake = y_fake.reshape(y_fake.shape[0], -1)

            if wavelengths is None:
                wavelengths = make_wavelengths_from_config(cfg, fallback_count=y_fake.shape[1])
                wl_cols = wavelength_columns(wavelengths)

            for j in range(y_fake.shape[0]):
                dataset_idx = batch_idx * args.batch_size + j
                if dataset_idx >= len(infer_dataset.df):
                    continue

                df_row = infer_dataset.df.iloc[dataset_idx]
                pred = y_fake[j].reshape(-1)
                params = p_params[j]

                if not np.isfinite(pred).all():
                    raise FloatingPointError(
                        f"Non-finite generated spectrum at stage={stage}, "
                        f"batch={batch_idx}, item={j}"
                    )

                row_meta = {
                    "sample_global_index": sample_counter,
                    "input_row_id": int(df_row["__input_row_id"]) if "__input_row_id" in df_row else dataset_idx,
                    "stage": stage,
                    "species": df_row["Species"] if "Species" in df_row else "",
                    "checkpoint": checkpoint_path,
                    "generator_architecture": "segmented_prospect_residual_separate_encoders",
                    "training_discriminator_mode": "global",
                }

                for band in DEFAULT_BANDS:
                    row_meta[f"{band}_image"] = df_row[band] if band in df_row else ""
                    row_meta[f"{band}_basename"] = Path(str(df_row[band])).name if band in df_row else ""

                generated_rows.append(
                    {
                        **row_meta,
                        **{col: float(val) for col, val in zip(wl_cols, pred)},
                        "generated_spectrum_json": json.dumps(pred.astype(float).tolist()),
                    }
                )

                params_rows.append(
                    {
                        **row_meta,
                        "params_shape": str(tuple(params.shape)),
                        "params_json": json.dumps(np.asarray(params, dtype=float).tolist()),
                    }
                )

                spectra_list.append(pred.astype(np.float32))
                sample_counter += 1

    return generated_rows, params_rows, spectra_list, wavelengths


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reflectance spectra from multispectral image CSV using a trained pix2spectral generator."
    )

    parser.add_argument("--input-csv", required=True, help="CSV containing image filename columns.")
    parser.add_argument(
        "--stats-csv",
        default=None,
        help=(
            "CSV used to compute image normalization statistics. "
            "Recommended: the TRAIN CSV used during training. "
            "Default: input CSV."
        ),
    )
    parser.add_argument("--img-dir", required=True, help="Root directory for image paths.")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--config-module", default="config")
    parser.add_argument("--dataset-module", default="dataset")
    parser.add_argument("--generator-module", default="generator_model")

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Single generator checkpoint used for all rows. If omitted, use template/results-root.",
    )
    parser.add_argument("--results-root", default="~/Results/pix2spectral_groupB_loss_ablation_separate")
    parser.add_argument("--experiment-dir", default="L5_full_physics_informed_loss")
    parser.add_argument("--experiment-prefix", default="avocado")
    parser.add_argument(
        "--checkpoint-template",
        default=None,
        help=(
            "Optional template with placeholders {results_root}, {experiment_dir}, "
            "{experiment_prefix}, {stage}. Default is "
            "{results_root}/{experiment_dir}/{experiment_prefix}_{stage}_gen_best.pth.tar"
        ),
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        default=["auto"],
        help="Stages to process, or 'auto' to read unique stages from CSV.",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Force all inference to one stage/checkpoint. Useful with --checkpoint.",
    )
    parser.add_argument(
        "--stats-source",
        choices=["auto", "stage", "all"],
        default="auto",
        help="How to choose the stage used for computing normalization stats.",
    )
    parser.add_argument("--species-filter", default=None)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-patches", action="store_true")
    parser.add_argument("--non-strict-load", action="store_true")

    parser.add_argument("--max-patches-per-band", type=int, default=10)
    parser.add_argument("--min-patches-per-band", type=int, default=10)

    parser.add_argument("--write-long", action="store_true")
    parser.add_argument("--save-npy", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = ensure_dir(args.output_dir)
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="pix2spectral_infer_")
    temp_dir = Path(temp_dir_obj.name)

    cfg = importlib.import_module(args.config_module)
    cfg = apply_inference_model_config(cfg, args)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    input_csv = expand_path(args.input_csv)
    stats_csv = expand_path(args.stats_csv) if args.stats_csv is not None else input_csv

    default_stage = args.stage if args.stage is not None else "all"

    prepared_input_csv = prepare_csv_for_dataset(
        source_csv=input_csv,
        cfg=cfg,
        default_stage=default_stage,
        species_filter=args.species_filter,
        temp_dir=temp_dir,
        role="input",
    )
    prepared_stats_csv = prepare_csv_for_dataset(
        source_csv=stats_csv,
        cfg=cfg,
        default_stage=default_stage,
        species_filter=args.species_filter,
        temp_dir=temp_dir,
        role="stats",
    )

    stages = determine_stages(args, prepared_input_csv)

    print("=" * 80)
    print("pix2spectral inference")
    print("=" * 80)
    print(f"Input CSV:       {input_csv}")
    print(f"Stats CSV:       {stats_csv}")
    print(f"Image root:      {expand_path(args.img_dir)}")
    print(f"Output dir:      {output_dir}")
    print(f"Stages:          {stages}")
    print(f"Device:          {device}")
    print("Forced model architecture:")
    print("  Generator: segmented PROSPECT + residual + separate band encoders")
    print("  Training discriminator: global")
    print("=" * 80)

    normalization_stats_cache: Dict[str, Any] = {}
    all_generated_rows = []
    all_params_rows = []
    all_spectra = []
    wavelengths = None

    for stage in stages:
        rows, params_rows, spectra, stage_wavelengths = run_inference_for_stage(
            args=args,
            cfg=cfg,
            stage=stage,
            prepared_input_csv=prepared_input_csv,
            prepared_stats_csv=prepared_stats_csv,
            device=device,
            normalization_stats_cache=normalization_stats_cache,
        )

        all_generated_rows.extend(rows)
        all_params_rows.extend(params_rows)
        all_spectra.extend(spectra)

        if wavelengths is None and stage_wavelengths is not None:
            wavelengths = stage_wavelengths

    if not all_generated_rows:
        raise RuntimeError("No generated spectra were produced. Check CSV filters/stages.")

    generated_df = pd.DataFrame(all_generated_rows)
    params_df = pd.DataFrame(all_params_rows)

    generated_path = output_dir / "generated_spectra_wide.csv"
    params_path = output_dir / "prospect_parameters.csv"
    generated_df.to_csv(generated_path, index=False)
    params_df.to_csv(params_path, index=False)

    if args.write_long:
        if wavelengths is None:
            raise RuntimeError("Cannot write long output because wavelengths were not initialized.")

        long_rows = []
        wl_cols = wavelength_columns(wavelengths)
        meta_cols = [
            c for c in generated_df.columns
            if not c.startswith("wl_") and c != "generated_spectrum_json"
        ]

        for _, row in generated_df.iterrows():
            meta = {c: row[c] for c in meta_cols}
            for wl, wl_col in zip(wavelengths, wl_cols):
                long_rows.append(
                    {
                        **meta,
                        "wavelength": float(wl),
                        "generated_reflectance": float(row[wl_col]),
                    }
                )

        long_path = output_dir / "generated_spectra_long.csv"
        pd.DataFrame(long_rows).to_csv(long_path, index=False)
    else:
        long_path = None

    if args.save_npy:
        spectra_arr = np.stack(all_spectra, axis=0).astype(np.float32)
        npy_path = output_dir / "generated_spectra.npy"
        np.save(npy_path, spectra_arr)
    else:
        npy_path = None

    manifest = {
        "input_csv": input_csv,
        "stats_csv": stats_csv,
        "img_dir": expand_path(args.img_dir),
        "output_dir": str(output_dir),
        "stages": stages,
        "config_module": args.config_module,
        "dataset_module": args.dataset_module,
        "generator_module": args.generator_module,
        "checkpoint": expand_path(args.checkpoint) if args.checkpoint else None,
        "results_root": str(Path(args.results_root).expanduser().resolve()),
        "experiment_dir": args.experiment_dir,
        "experiment_prefix": args.experiment_prefix,
        "checkpoint_template": args.checkpoint_template,
        "forced_generator_architecture": "segmented_prospect_residual_separate_encoders",
        "training_discriminator_mode": "global",
        "normalization_stats_keys": list(normalization_stats_cache.keys()),
        "num_generated_samples": len(all_generated_rows),
        "outputs": {
            "generated_spectra_wide": str(generated_path),
            "prospect_parameters": str(params_path),
            "generated_spectra_long": str(long_path) if long_path is not None else None,
            "generated_spectra_npy": str(npy_path) if npy_path is not None else None,
        },
    }

    manifest_path = output_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("=" * 80)
    print("Inference finished")
    print("=" * 80)
    print(f"Generated spectra:   {generated_path}")
    print(f"PROSPECT parameters: {params_path}")
    if long_path is not None:
        print(f"Long spectra:        {long_path}")
    if npy_path is not None:
        print(f"NumPy spectra:       {npy_path}")
    print(f"Manifest:            {manifest_path}")
    print(f"Samples generated:   {len(all_generated_rows)}")
    print("=" * 80)

    temp_dir_obj.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
