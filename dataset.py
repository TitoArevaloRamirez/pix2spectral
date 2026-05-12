import ast
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


BAND_KEYS = ["blue", "green", "red", "nir", "red_edge"]


def normalize_stage_name(stage: str) -> str:
    """
    Normalize stage labels for robust filtering and grouping.

    Examples:
        "Stage 1"  -> "stage1"
        "stage_2"  -> "stage2"
        "Dry"      -> "dry"
    """
    s = str(stage).strip().lower()
    s = s.replace("_", "")
    s = s.replace("-", "")
    s = s.replace(" ", "")

    aliases = {
        "fresh": "fresh",
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
        "dried": "dry",
    }

    return aliases.get(s, s)


def get_spectral_data(data: pd.DataFrame, remove_first_bands: int = 50) -> np.ndarray:
    """
    Parse the CSV 'spectral' column into an array.

    Returns:
        spectra: [N, M]
    """
    if "spectral" not in data.columns:
        raise ValueError("CSV must contain a 'spectral' column.")

    spectral_strings = data["spectral"].to_list()
    spectra = [ast.literal_eval(s) for s in spectral_strings]
    spectra = np.asarray(spectra, dtype=np.float32)

    if spectra.ndim != 2:
        raise ValueError(f"Expected parsed spectra to be [N, M], got {spectra.shape}")

    if remove_first_bands > 0:
        if spectra.shape[1] <= remove_first_bands:
            raise ValueError(
                f"Cannot remove first {remove_first_bands} bands from spectra "
                f"with only {spectra.shape[1]} bands."
            )
        spectra = spectra[:, remove_first_bands:]

    return spectra.astype(np.float32)


def read_band_image_as_roi_patches(
    path: str,
    patch_h: int,
    patch_w: int,
    stride_h: Optional[int] = None,
    stride_w: Optional[int] = None,
    black_thr: float = 0.0,
) -> torch.Tensor:
    """
    Read a band image and return patches fully inside the non-black ROI.

    ROI rule:
        pixel > black_thr

    Kept patch rule:
        all pixels in the patch must be inside ROI.

    Returns:
        patches: Tensor [N, 1, patch_h, patch_w]
                 N can be 0.
    """
    if stride_h is None:
        stride_h = patch_h
    if stride_w is None:
        stride_w = patch_w

    if not os.path.exists(path):
        raise FileNotFoundError(f"Band image not found: {path}")

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    height, width = arr.shape

    if height < patch_h or width < patch_w:
        return torch.empty(0, 1, patch_h, patch_w, dtype=torch.float32)

    roi = arr > black_thr
    patches = []

    for top in range(0, height - patch_h + 1, stride_h):
        for left in range(0, width - patch_w + 1, stride_w):
            roi_patch = roi[top : top + patch_h, left : left + patch_w]

            if not roi_patch.all():
                continue

            patch = arr[top : top + patch_h, left : left + patch_w]
            patch = torch.from_numpy(patch).unsqueeze(0)
            patches.append(patch)

    if not patches:
        return torch.empty(0, 1, patch_h, patch_w, dtype=torch.float32)

    return torch.stack(patches, dim=0).float()


class SpectralOnlyCSVDataset(Dataset):
    """
    CSV-driven dataset that returns only spectral signatures.

    Required columns:
        spectral, Species, Stages

    Returns:
        spectrum: Tensor [M]
    """

    def __init__(
        self,
        csv_path: str,
        species: Optional[str] = None,
        stage: Optional[str] = None,
        remove_first_bands: int = 50,
    ):
        csv_path = os.path.expanduser(csv_path)
        df = pd.read_csv(csv_path)

        required_cols = ["spectral", "Species", "Stages"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError("Missing columns in CSV: " + ", ".join(missing))

        df["Species"] = df["Species"].astype(str).str.strip().str.lower()
        df["Stages"] = df["Stages"].map(normalize_stage_name)

        if species is not None:
            species = str(species).strip().lower()
            df = df[df["Species"] == species]

        if stage is not None:
            stage = normalize_stage_name(stage)
            if stage != "all":
                df = df[df["Stages"] == stage]

        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No rows left after filtering. Check species/stage values.")

        self.df = df
        self.spectral_np = get_spectral_data(df, remove_first_bands=remove_first_bands)
        self.spec = torch.from_numpy(self.spectral_np).float()

    def __len__(self) -> int:
        return self.spec.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.spec[index]


class MultiSpectralCSVPatchDataset(Dataset):
    """
    CSV-driven multispectral dataset with filtering by species and stage.

    Required CSV columns:
        blue, green, red, nir, red_edge, spectral, Species, Stages

    Returns:
        band_patches : dict {band_name: Tensor [N_i, 1, patch_h, patch_w]}
        spectrum     : Tensor [M]
        stage_label  : str
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: Optional[str] = None,
        species: Optional[str] = None,
        stage: Optional[str] = None,
        patch_h: int = 32,
        patch_w: int = 32,
        stride_h: Optional[int] = None,
        stride_w: Optional[int] = None,
        black_thr: float = 0.0,
        remove_first_bands: int = 50,
        print_distribution: bool = True,
    ):
        csv_path = os.path.expanduser(csv_path)
        df = pd.read_csv(csv_path)

        self.root_dir = os.path.expanduser(root_dir) if root_dir is not None else ""

        required_cols = [
            "blue",
            "green",
            "red",
            "nir",
            "red_edge",
            "spectral",
            "Species",
            "Stages",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError("Missing columns in CSV: " + ", ".join(missing))

        df["Species"] = df["Species"].astype(str).str.strip().str.lower()
        df["Stages"] = df["Stages"].map(normalize_stage_name)

        if species is not None:
            species = str(species).strip().lower()
            df = df[df["Species"] == species]

        if stage is not None:
            stage = str(stage).strip().lower()

            if stage == "all":
                print("Using all stages")

            elif stage.startswith("no_"):
                excluded_stage = normalize_stage_name(stage.replace("no_", "", 1))
                print(f"Using all stages except: {excluded_stage}")
                df = df[df["Stages"] != excluded_stage]

            else:
                stage = normalize_stage_name(stage)
                print(f"Using only stage: {stage}")
                df = df[df["Stages"] == stage]

        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No rows left after filtering. Check species/stage values.")

        self.df = df
        self.band_cols = BAND_KEYS.copy()

        self.spectral_np = get_spectral_data(
            self.df,
            remove_first_bands=remove_first_bands,
        ).astype(np.float32)

        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.stride_h = int(stride_h) if stride_h is not None else None
        self.stride_w = int(stride_w) if stride_w is not None else None
        self.black_thr = float(black_thr)

        if print_distribution:
            self.print_stage_distribution()

    def print_stage_distribution(self) -> None:
        unique, counts = np.unique(self.df["Stages"].to_numpy(), return_counts=True)

        print("\nDataset stage distribution")
        print("--------------------------")
        for stage, count in zip(unique, counts):
            print(f"{stage}: {count}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, fname: str) -> str:
        """
        Resolve image paths robustly.

        Handles:
            - absolute paths
            - relative paths
            - Windows-style paths in the CSV
            - bare filenames under root_dir
        """
        fname = str(fname).strip().replace("\\", os.sep)

        if os.path.exists(fname):
            return fname

        candidates = []

        if self.root_dir:
            candidates.append(os.path.join(self.root_dir, fname))
            candidates.append(os.path.join(self.root_dir, os.path.basename(fname)))

        candidates.append(os.path.basename(fname))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        # Return the most useful attempted path for the FileNotFoundError message.
        return candidates[0] if candidates else fname

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        band_patches = {}

        for band_name in self.band_cols:
            path = self._resolve_image_path(row[band_name])

            band_patches[band_name] = read_band_image_as_roi_patches(
                path=path,
                patch_h=self.patch_h,
                patch_w=self.patch_w,
                stride_h=self.stride_h,
                stride_w=self.stride_w,
                black_thr=self.black_thr,
            )

        spec = torch.from_numpy(self.spectral_np[index]).float()
        stage_label = str(row["Stages"])

        return band_patches, spec, stage_label


def patch_collate_fn(batch):
    """
    Collate function for MultiSpectralCSVPatchDataset.

    Input batch:
        list of (band_patches_dict, spectrum, stage_label)

    Returns:
        batch_bands : dict {band_name: list[Tensor [N_i, 1, ph, pw]]}
        batch_spec  : Tensor [B, M]
        batch_stage : list[str]
    """
    batch_bands = {k: [] for k in BAND_KEYS}
    specs = []
    stages = []

    for band_dict, spec, stage_label in batch:
        for band_name in BAND_KEYS:
            batch_bands[band_name].append(band_dict[band_name])

        specs.append(spec)
        stages.append(str(stage_label))

    batch_spec = torch.stack(specs, dim=0)

    return batch_bands, batch_spec, stages


if __name__ == "__main__":
    dataset = MultiSpectralCSVPatchDataset(
        csv_path="~/Code/pix2spectral/Data/Dataset_with_images.csv",
        root_dir="/media/usr3/Expansion/Data/EstradaDataset/Avocado/Multispectral Images/",
        species="Avocado",
        stage="all",
        patch_h=32,
        patch_w=32,
        stride_h=16,
        stride_w=16,
        black_thr=0.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=patch_collate_fn,
    )

    for band_dict, spec, stages in loader:
        print("spec shape:", spec.shape)
        print("stages:", stages)
        print("blue sample 0 patches:", band_dict["blue"][0].shape)
        break
