import os
import ast
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def get_spectral_data(data):
    """Extract spectral data from DataFrame."""
    x_data = data["spectral"]
    x_data = x_data.to_list()
    np_data = [ast.literal_eval(s) for s in x_data]
    np_data = np.array(np_data)
    # TODO: Document why first 50 wavelengths are removed
    # PROSPECT-D outputs 2101 wavelengths (400-2500nm)
    # This truncation needs verification
    np_data = np_data[:, 50:]
    x = np_data.reshape(np_data.shape[0], np_data.shape[1])
    return x


def read_band_image_as_roi_patches(path, patch_h, patch_w, stride_h=None, stride_w=None, black_thr=0.0):
    """
    Reads a (masked) band image and returns ONLY patches fully inside the ROI.

    ROI definition:
      - pixels with value > black_thr are considered "inside ROI"
      - patch is kept only if ALL pixels in the patch are > black_thr
        (so it contains no black/background pixels)

    Returns:
      patches: Tensor [N, 1, patch_h, patch_w]  (N can be 0)
    """
    if stride_h is None:
        stride_h = patch_h
    if stride_w is None:
        stride_w = patch_w

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)  # [H, W]

    H, W = arr.shape
    if H < patch_h or W < patch_w:
        # too small to extract any patch
        return torch.empty(0, 1, patch_h, patch_w, dtype=torch.float32)

    roi = arr > black_thr  # [H, W] boolean

    patches = []
    for top in range(0, H - patch_h + 1, stride_h):
        for left in range(0, W - patch_w + 1, stride_w):
            roi_patch = roi[top : top + patch_h, left : left + patch_w]
            if not roi_patch.all():
                continue  # reject if any black/background pixel exists

            p = arr[top : top + patch_h, left : left + patch_w]
            p = torch.from_numpy(p).unsqueeze(0)  # [1, patch_h, patch_w]
            patches.append(p)

    if len(patches) == 0:
        return torch.empty(0, 1, patch_h, patch_w, dtype=torch.float32)

    return torch.stack(patches, dim=0).float()  # [N, 1, patch_h, patch_w]


class MultiSpectralCSVPatchDataset(Dataset):
    """
    CSV-driven multispectral dataset (no transforms), with filtering by species and stage.

    Required CSV columns:
      blue, green, red, nir, red_edge, spectral, species, stage

    Returns:
      image   Tensor [5, H, W]
      spectrum Tensor [L]
    """
    def __init__(
        self,
        csv_path,
        root_dir=None,
        species=None,
        stage=None,
        patch_h=32,
        patch_w=32,
        stride_h=None,
        stride_w=None,
        black_thr=0.0,
    ):
        df = pd.read_csv(csv_path)
        self.root_dir = root_dir if root_dir is not None else ""

        required_cols = ["blue", "green", "red", "nir", "red_edge", "spectral", "Species", "Stages"]
        missing = [c for c in required_cols if c not in df.columns]
        if len(missing) > 0:
            raise ValueError("Missing columns in CSV: " + ", ".join(missing))

        # Normalize for robust matching (strip + lowercase)
        df["Species"] = df["Species"].astype(str).str.strip().str.lower()
        df["Stages"] = df["Stages"].astype(str).str.strip().str.lower()

        # FIXED: Changed != to == to actually filter FOR the desired species
        if species is not None:
            species = str(species).strip().lower()
            df = df[df["Species"] == species]
            print(f"Filtered to species '{species}': {len(df)} samples")

        # FIXED: Changed != to == to actually filter FOR the desired stage
        if stage is not None:
            stage = str(stage).strip().lower()
            df = df[df["Stages"] == stage]  # ✅ FIXED: was !=, now ==
            print(f"Filtered to stage '{stage}': {len(df)} samples")

        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows left after filtering. Check species/stage values.\n"
                f"Requested: species='{species}', stage='{stage}'"
            )

        self.df = df
        self.band_cols = ["blue", "green", "red", "nir", "red_edge"]

        # Parse spectra once (after filtering)
        self.spectral_np = get_spectral_data(self.df).astype(np.float32)

        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.stride_h = int(stride_h) if stride_h is not None else None
        self.stride_w = int(stride_w) if stride_w is not None else None
        self.black_thr = float(black_thr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        band_patches = {}
        for c in self.band_cols:
            fname = str(row[c])
            parts = fname.split('\\')
            path = os.path.join(self.root_dir, parts[1]) if self.root_dir else parts[1]

            band_patches[c] = read_band_image_as_roi_patches(
                path=path,
                patch_h=self.patch_h,
                patch_w=self.patch_w,
                stride_h=self.stride_h,
                stride_w=self.stride_w,
                black_thr=self.black_thr,
            )
        spec = torch.from_numpy(self.spectral_np[index]).float()  # [L]

        return band_patches, spec


def patch_collate_fn(batch):
    """
    Batch is a list of (band_patches_dict, spectrum).

    Because each sample can yield a different number of ROI patches per band,
    we keep patches as lists and stack spectra normally.

    Returns:
      batch_bands: dict {band_name: list of Tensor [N_i, 1, ph, pw]}
      batch_spec : Tensor [B, L]
    """
    band_keys = ["blue", "green", "red", "nir", "red_edge"]
    batch_bands = {k: [] for k in band_keys}
    specs = []

    for band_dict, spec in batch:
        for k in band_keys:
            batch_bands[k].append(band_dict[k])
        specs.append(spec)

    batch_spec = torch.stack(specs, dim=0)
    return batch_bands, batch_spec


if __name__ == "__main__":
    dataset = MultiSpectralCSVPatchDataset(
        csv_path="/Volumes/data/EstradaDataset/Dataset_with_images.csv",
        root_dir="/Volumes/data/EstradaDataset/DATASET/Avocado/Multispectral Images/",
        species="Avocado",
        stage="Fresh",
        patch_h=32,
        patch_w=32,
        stride_h=16,   # non-overlapping; set smaller for overlap
        stride_w=16,
        black_thr=0.0,
    )

    loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=patch_collate_fn,)

    for band_dict, spec in loader:
        print("spec shape:", spec.shape)  # [B, L]
        print("num b patches in sample0:", band_dict["blue"][0].shape[0])
        print("num g patches in sample0:", band_dict["green"][0].shape[0])
        print("num r patches in sample0:", band_dict["red"][0].shape[0])
        print("num nir patches in sample0:", band_dict["nir"][0].shape[0])
        print("num re patches in sample0:", band_dict["red_edge"][0].shape[0])
        print("blue patch tensor shape:", band_dict["blue"][0].shape)
        break
