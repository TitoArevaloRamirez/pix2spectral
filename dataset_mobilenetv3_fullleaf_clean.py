"""
Clean full-leaf multispectral CSV dataset for MobileNetV3 cGAN training.

Each CSV row returns one complete resized/padded image for each multispectral band.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

FULLLEAF_BANDS = ["blue", "green", "red", "nir", "red_edge"]
DEFAULT_STAGE_TO_INDEX = {"fresh": 0, "stage1": 1, "stage2": 2, "stage3": 3, "dry": 4}


def save_normalization_stats(stats, path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_normalization_stats(path):
    return json.loads(Path(path).expanduser().read_text())


def parse_spectral_column(df: pd.DataFrame, spectral_col: str = "spectral", drop_first_n: int = 50) -> np.ndarray:
    spectra = []
    for value in df[spectral_col].tolist():
        text = str(value)
        try:
            arr = ast.literal_eval(text)
        except Exception:
            arr = json.loads(text)
        spectra.append(np.asarray(arr, dtype=np.float32).reshape(-1))
    out = np.stack(spectra, axis=0).astype(np.float32)
    if int(drop_first_n) > 0:
        if out.shape[1] <= int(drop_first_n):
            raise ValueError(f"Cannot drop {drop_first_n} values from spectral length {out.shape[1]}")
        out = out[:, int(drop_first_n):]
    return out


def resolve_fullleaf_image_path(filename: Any, image_root_dir: Optional[str]) -> str:
    """
    Resolve one band image path, preferring the preprocessed full-leaf directory.
    This prevents loading raw images when the CSV stores absolute raw-image paths.
    """
    text = str(filename).strip()
    if text == "" or text.lower() == "nan":
        raise ValueError("Empty image filename in CSV.")

    original = Path(text)
    candidates: List[Path] = []
    if image_root_dir is not None and str(image_root_dir) != "":
        root = Path(image_root_dir).expanduser()
        name = Path(PureWindowsPath(text).name).name
        stem = Path(name).stem
        candidates.extend([root / text, root / name, root / Path(text).name])
        for ext in [".tif", ".tiff", ".png", ".npy"]:
            candidates.append(root / f"{stem}{ext}")
        parts = Path(PureWindowsPath(text)).parts
        for k in range(1, min(len(parts), 6) + 1):
            rel = Path(*parts[-k:])
            candidates.extend([root / rel, (root / rel).with_suffix(".tif"), (root / rel).with_suffix(".png"), (root / rel).with_suffix(".npy")])
    if original.is_absolute():
        candidates.append(original)
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0] if candidates else original)


def read_single_band_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p).astype(np.float32)
    else:
        img = Image.open(p)
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[..., :3].astype(np.float32).mean(axis=2)
        arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(arr.max()) if arr.size else 0.0
    if mx > 1.5:
        arr = arr / (255.0 if mx <= 255.0 else 65535.0)
    return arr.astype(np.float32)


def resize_pad_if_needed(arr: np.ndarray, image_size: Optional[int]) -> np.ndarray:
    if image_size is None:
        return arr.astype(np.float32)
    image_size = int(image_size)
    if arr.shape == (image_size, image_size):
        return arr.astype(np.float32)
    h, w = arr.shape
    scale = float(image_size) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    img = Image.fromarray(arr.astype(np.float32), mode="F").resize((new_w, new_h), resample=Image.BILINEAR)
    resized = np.asarray(img).astype(np.float32)
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    top = (image_size - new_h) // 2
    left = (image_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


class FullLeafMultispectralCSVDataset(Dataset):
    """Return full images for all five bands and one spectral signature per CSV row."""
    def __init__(
        self,
        csv_path,
        image_root_dir=None,
        species_filter="all",
        stage_filter="all",
        spectral_drop_first_n: int = 50,
        image_size: Optional[int] = None,
        return_stage_index: bool = True,
        stage_to_index: Optional[Dict[str, int]] = None,
        cache_images: bool = False,
        **unused,
    ):
        self.csv_path = str(csv_path)
        self.image_root_dir = "" if image_root_dir is None else str(image_root_dir)
        self.image_size = None if image_size is None else int(image_size)
        self.return_stage_index = bool(return_stage_index)
        self.stage_to_index = dict(DEFAULT_STAGE_TO_INDEX if stage_to_index is None else stage_to_index)
        self.cache_images = bool(cache_images)
        self._image_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.normalization_stats = None

        df = pd.read_csv(csv_path)
        required = FULLLEAF_BANDS + ["spectral", "Species", "Stages"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(missing))
        df["Species"] = df["Species"].astype(str).str.strip().str.lower()
        df["Stages"] = df["Stages"].astype(str).str.strip().str.lower()

        if species_filter is not None and str(species_filter).strip().lower() not in ["all", "any", "*", ""]:
            df = df[df["Species"] == str(species_filter).strip().lower()]
        if stage_filter is not None and str(stage_filter).strip().lower() not in ["all", "any", "*", ""]:
            df = df[df["Stages"] == str(stage_filter).strip().lower()]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows left after filtering.")
        self.df = df
        self.spectral_np = parse_spectral_column(df, drop_first_n=spectral_drop_first_n)

    def __len__(self):
        return len(self.df)

    def _load_fullleaf_images(self, index: int) -> Dict[str, torch.Tensor]:
        if self.cache_images and index in self._image_cache:
            return {k: v.clone() for k, v in self._image_cache[index].items()}
        row = self.df.iloc[index]
        images: Dict[str, torch.Tensor] = {}
        for band in FULLLEAF_BANDS:
            path = resolve_fullleaf_image_path(row[band], self.image_root_dir)
            arr = resize_pad_if_needed(read_single_band_image(path), self.image_size)
            images[band] = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).float()
        if self.cache_images:
            self._image_cache[index] = {k: v.clone() for k, v in images.items()}
        return images

    def __getitem__(self, index):
        band_images = self._load_fullleaf_images(index)
        spectrum = torch.from_numpy(self.spectral_np[index]).float()
        if self.return_stage_index:
            stage_name = str(self.df.iloc[index]["Stages"]).strip().lower()
            stage_index = torch.tensor(int(self.stage_to_index.get(stage_name, -1)), dtype=torch.long)
            return band_images, spectrum, stage_index
        return band_images, spectrum


def fullleaf_collate_fn(batch):
    """Collate full-leaf image samples into dict-of-lists plus spectral tensor."""
    has_stage = len(batch[0]) == 3
    band_batch = {b: [] for b in FULLLEAF_BANDS}
    spectra = []
    stages = []
    for item in batch:
        if has_stage:
            band_images, spectrum, stage_index = item
            stages.append(stage_index)
        else:
            band_images, spectrum = item
        for band in FULLLEAF_BANDS:
            band_batch[band].append(band_images[band])
        spectra.append(spectrum)
    spectrum_batch = torch.stack(spectra, dim=0)
    if has_stage:
        return band_batch, spectrum_batch, torch.stack(stages, dim=0).long()
    return band_batch, spectrum_batch
