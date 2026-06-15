#!/usr/bin/env python3
"""
Assign FMC_d values to generated spectra.

This script joins:

1) generated_spectra_wide.csv
   Required columns:
       stage, species, blue_basename, generated_spectrum_json

2) original data CSV, e.g. avocado_test.csv
   Required columns:
       Species, Stages, FMC_d, spectral, blue

Matching key:
       species + stage + blue image basename

Output columns:
       stage, species, blue_basename, generated_spectrum_json, FMC_d

Example
-------
python assign_fmc_to_generated_spectra.py \
    --generated-csv ~/Results/pix2spectral_inference/avocado_L5/generated_spectra_wide.csv \
    --original-csv ~/Code/pix2spectral/Data/dataset_splits_70_20_10/avocado_test.csv \
    --output-csv ~/Results/pix2spectral_inference/avocado_L5/generated_spectra_with_FMC_d.csv \
    --strict
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


GENERATED_REQUIRED_COLUMNS = [
    "stage",
    "species",
    "blue_basename",
    # "generated_spectrum_json",
    "params_json",
]

ORIGINAL_REQUIRED_COLUMNS = [
    "Species",
    "Stages",
    "FMC_d",
    "spectral",
    "blue",
]

OUTPUT_COLUMNS = [
    "stage",
    "species",
    "blue_basename",
    "params_json",
    # "generated_spectrum_json",
    "FMC_d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign FMC_d values from an original dataset CSV to generated "
            "spectra by matching species, stage, and blue image basename."
        )
    )

    parser.add_argument(
        "--generated-csv",
        required=True,
        help="Path to generated_spectra_wide.csv.",
    )
    parser.add_argument(
        "--original-csv",
        required=True,
        help="Path to original dataset CSV, e.g. avocado_test.csv.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "If enabled, stop with an error when any generated row cannot be "
            "matched to FMC_d in the original CSV."
        ),
    )
    parser.add_argument(
        "--case-sensitive-species",
        action="store_true",
        help="Use case-sensitive species matching. Default is case-insensitive.",
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help=(
            "Keep unmatched rows with FMC_d=NaN. This is the default when "
            "--strict is not used."
        ),
    )
    parser.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="Drop generated rows without a matching FMC_d.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help=(
            "Optional path to write a diagnostic report containing match status "
            "and normalized join keys."
        ),
    )
    parser.add_argument(
        "--validate-spectrum-json",
        action="store_true",
        help="Validate that generated_spectrum_json can be parsed as JSON.",
    )

    return parser.parse_args()


def expand_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def require_columns(df: pd.DataFrame, required: Iterable[str], csv_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def basename_any(path_like) -> str:
    """
    Extract basename robustly for Unix paths, Windows paths, or plain filenames.
    """
    if pd.isna(path_like):
        return ""

    text = str(path_like).strip()
    if text == "":
        return ""

    # First handle Windows-style backslashes, then POSIX paths.
    win_name = PureWindowsPath(text).name
    posix_name = PurePosixPath(win_name).name
    return posix_name.strip()


def normalize_species(value, case_sensitive: bool = False) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    if not case_sensitive:
        text = text.lower()

    return text


def normalize_stage(value) -> str:
    """
    Canonicalize dehydration stage labels for matching.

    Examples:
        "Stage 1", "stage_1", "s1", "1" -> "stage1"
        "fresh" -> "fresh"
        "dry" -> "dry"
    """
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("_", "")
    text = text.replace("-", "")
    text = re.sub(r"\s+", "", text)

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
    }

    return aliases.get(text, text)


def normalize_basename(value) -> str:
    return basename_any(value).strip().lower()


def validate_generated_spectrum_json(series: pd.Series) -> None:
    bad_rows = []

    for idx, value in series.items():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            arr = np.asarray(parsed, dtype=float)
            if arr.ndim != 1 or arr.size == 0:
                bad_rows.append((idx, "not a non-empty 1D array"))
            elif not np.isfinite(arr).all():
                bad_rows.append((idx, "contains non-finite values"))
        except Exception as exc:
            bad_rows.append((idx, str(exc)))

    if bad_rows:
        preview = "\n".join(f"  row={idx}: {reason}" for idx, reason in bad_rows[:20])
        extra = "" if len(bad_rows) <= 20 else f"\n  ... plus {len(bad_rows) - 20} more"
        raise ValueError(
            f"Some generated_spectrum_json values are invalid:\n{preview}{extra}"
        )


def prepare_generated_df(
    df: pd.DataFrame, case_sensitive_species: bool
) -> pd.DataFrame:
    require_columns(df, GENERATED_REQUIRED_COLUMNS, "generated CSV")

    out = df[GENERATED_REQUIRED_COLUMNS].copy()
    out["_generated_row_id"] = np.arange(len(out), dtype=int)

    out["_key_species"] = out["species"].map(
        lambda x: normalize_species(x, case_sensitive=case_sensitive_species)
    )
    out["_key_stage"] = out["stage"].map(normalize_stage)
    out["_key_blue_basename"] = out["blue_basename"].map(normalize_basename)

    empty_key = (
        (out["_key_species"] == "")
        | (out["_key_stage"] == "")
        | (out["_key_blue_basename"] == "")
    )
    if empty_key.any():
        rows = out.loc[
            empty_key,
            ["_generated_row_id", "stage", "species", "blue_basename"],
        ].head(20)
        raise ValueError(
            "Some generated rows have empty matching keys after normalization. "
            "First rows:\n"
            f"{rows.to_string(index=False)}"
        )

    return out


def prepare_original_df(df: pd.DataFrame, case_sensitive_species: bool) -> pd.DataFrame:
    require_columns(df, ORIGINAL_REQUIRED_COLUMNS, "original CSV")

    out = df[ORIGINAL_REQUIRED_COLUMNS].copy()
    out["_original_row_id"] = np.arange(len(out), dtype=int)

    out["blue_basename"] = out["blue"].map(basename_any)

    out["_key_species"] = out["Species"].map(
        lambda x: normalize_species(x, case_sensitive=case_sensitive_species)
    )
    out["_key_stage"] = out["Stages"].map(normalize_stage)
    out["_key_blue_basename"] = out["blue_basename"].map(normalize_basename)

    empty_key = (
        (out["_key_species"] == "")
        | (out["_key_stage"] == "")
        | (out["_key_blue_basename"] == "")
    )
    if empty_key.any():
        rows = out.loc[
            empty_key,
            ["_original_row_id", "Species", "Stages", "blue", "blue_basename"],
        ].head(20)
        raise ValueError(
            "Some original rows have empty matching keys after normalization. "
            "First rows:\n"
            f"{rows.to_string(index=False)}"
        )

    # Make sure FMC_d is numeric when possible.
    out["FMC_d"] = pd.to_numeric(out["FMC_d"], errors="coerce")

    if out["FMC_d"].isna().any():
        rows = out.loc[
            out["FMC_d"].isna(),
            ["_original_row_id", "Species", "Stages", "blue", "FMC_d"],
        ].head(20)
        raise ValueError(
            "Some original rows have missing or non-numeric FMC_d values. "
            "First rows:\n"
            f"{rows.to_string(index=False)}"
        )

    return out


def resolve_original_duplicates(original: pd.DataFrame) -> pd.DataFrame:
    """
    Original CSV should have a unique key:
        species + stage + blue_basename

    If duplicates exist with the same FMC_d, keep the first.
    If duplicates exist with conflicting FMC_d, stop with an error.
    """
    key_cols = ["_key_species", "_key_stage", "_key_blue_basename"]

    duplicate_mask = original.duplicated(key_cols, keep=False)
    if not duplicate_mask.any():
        return original

    dup = original.loc[duplicate_mask].copy()

    conflict_keys = []
    for key, group in dup.groupby(key_cols, dropna=False):
        unique_fmc = group["FMC_d"].dropna().unique()
        if len(unique_fmc) > 1:
            conflict_keys.append(
                (key, unique_fmc.tolist(), group["_original_row_id"].tolist())
            )

    if conflict_keys:
        lines = []
        for key, fmc_values, row_ids in conflict_keys[:20]:
            lines.append(
                f"  key={key}, FMC_d values={fmc_values}, original rows={row_ids}"
            )
        extra = (
            ""
            if len(conflict_keys) <= 20
            else f"\n  ... plus {len(conflict_keys) - 20} more"
        )
        raise ValueError(
            "Original CSV contains duplicate matching keys with conflicting FMC_d values:\n"
            f"{chr(10).join(lines)}{extra}"
        )

    print(
        "Warning: original CSV contains duplicate matching keys with identical "
        "FMC_d. Keeping the first occurrence."
    )

    return original.drop_duplicates(key_cols, keep="first").copy()


def assign_fmc(
    generated: pd.DataFrame,
    original: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["_key_species", "_key_stage", "_key_blue_basename"]

    original_lookup = original[
        key_cols + ["FMC_d", "_original_row_id", "Species", "Stages", "blue"]
    ].copy()

    merged = generated.merge(
        original_lookup,
        how="left",
        on=key_cols,
        validate="many_to_one",
        indicator=True,
    )

    output = merged[OUTPUT_COLUMNS].copy()
    report_cols = [
        "_generated_row_id",
        "_original_row_id",
        "_merge",
        "stage",
        "species",
        "blue_basename",
        "_key_species",
        "_key_stage",
        "_key_blue_basename",
        "FMC_d",
        "Species",
        "Stages",
        "blue",
    ]
    report = merged[report_cols].copy()

    return output, report


def print_match_summary(report: pd.DataFrame) -> None:
    total = len(report)
    matched = int((report["_merge"] == "both").sum())
    unmatched = total - matched

    print("=" * 80)
    print("FMC_d assignment summary")
    print("=" * 80)
    print(f"Generated rows: {total}")
    print(f"Matched rows:   {matched}")
    print(f"Unmatched rows: {unmatched}")

    if total > 0:
        print(f"Match rate:     {100.0 * matched / total:.2f}%")

    if unmatched > 0:
        print("")
        print("First unmatched rows:")
        cols = [
            "_generated_row_id",
            "stage",
            "species",
            "blue_basename",
            "_key_species",
            "_key_stage",
            "_key_blue_basename",
        ]
        print(
            report.loc[report["_merge"] != "both", cols].head(20).to_string(index=False)
        )

    print("=" * 80)


def main() -> int:
    args = parse_args()

    generated_csv = expand_path(args.generated_csv)
    original_csv = expand_path(args.original_csv)
    output_csv = expand_path(args.output_csv)

    if not generated_csv.exists():
        raise FileNotFoundError(f"Generated CSV not found: {generated_csv}")
    if not original_csv.exists():
        raise FileNotFoundError(f"Original CSV not found: {original_csv}")

    generated_raw = pd.read_csv(generated_csv)
    original_raw = pd.read_csv(original_csv)

    if args.validate_spectrum_json:
        validate_generated_spectrum_json(generated_raw["generated_spectrum_json"])

    generated = prepare_generated_df(
        generated_raw,
        case_sensitive_species=args.case_sensitive_species,
    )
    original = prepare_original_df(
        original_raw,
        case_sensitive_species=args.case_sensitive_species,
    )
    original = resolve_original_duplicates(original)

    output, report = assign_fmc(generated, original)

    unmatched_mask = report["_merge"] != "both"
    unmatched_count = int(unmatched_mask.sum())

    print_match_summary(report)

    if unmatched_count > 0 and args.strict:
        raise RuntimeError(
            f"{unmatched_count} generated rows could not be matched to FMC_d. "
            "Run without --strict to write unmatched rows with FMC_d=NaN, or "
            "inspect the report CSV using --report-csv."
        )

    if args.drop_unmatched:
        output = output.loc[~unmatched_mask.values].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    if args.report_csv is not None:
        report_csv = expand_path(args.report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_csv, index=False)
        print(f"Diagnostic report written to: {report_csv}")

    print(f"Output written to: {output_csv}")
    print(f"Output columns: {OUTPUT_COLUMNS}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
