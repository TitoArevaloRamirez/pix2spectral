# conditioner.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCNNEncoder(nn.Module):
    """
    Encodes individual grayscale patches [N, 1, H, W] into embeddings [N, D].
    """

    def __init__(self, emb_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Linear(128, emb_dim)

    def forward(self, patches):
        """
        patches: Tensor [N, 1, H, W]
        returns: Tensor [N, emb_dim]
        """
        h = self.net(patches)  # [N, 128, 1, 1]
        h = h.flatten(1)  # [N, 128]
        h = self.proj(h)  # [N, emb_dim]
        return h


class AttentionPool(nn.Module):
    """
    Pools variable-size patch embeddings [N, D] into one vector [D].
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, patch_embs):
        """
        patch_embs: Tensor [N, D]
        returns: Tensor [D]
        """
        weights = self.score(patch_embs)  # [N, 1]
        weights = torch.softmax(weights, dim=0)  # [N, 1]
        pooled = (weights * patch_embs).sum(dim=0)
        return pooled  # [D]


class MultiSpectralConditioner(nn.Module):
    """
    Converts variable-number, unaligned multispectral patch bags into a fixed
    conditioning vector for conditional flow matching.

    Expected input format matches your current dataset collate output:

        batch_bands = {
            "blue":     list of B tensors [N_i, 1, H, W],
            "green":    list of B tensors [N_i, 1, H, W],
            "red":      list of B tensors [N_i, 1, H, W],
            "nir":      list of B tensors [N_i, 1, H, W],
            "red_edge": list of B tensors [N_i, 1, H, W],
        }

    Returns:
        condition: Tensor [B, condition_dim]
    """

    def __init__(
        self,
        emb_dim=128,
        condition_dim=256,
        bands=("blue", "green", "red", "nir", "red_edge"),
        shared_patch_encoder=True,
        pooling="attention",
    ):
        super().__init__()

        self.bands = list(bands)
        self.emb_dim = emb_dim
        self.condition_dim = condition_dim
        self.pooling = pooling

        if shared_patch_encoder:
            shared_encoder = PatchCNNEncoder(emb_dim)
            self.patch_encoders = nn.ModuleDict(
                {band: shared_encoder for band in self.bands}
            )
        else:
            self.patch_encoders = nn.ModuleDict(
                {band: PatchCNNEncoder(emb_dim) for band in self.bands}
            )

        if pooling == "attention":
            self.poolers = nn.ModuleDict(
                {band: AttentionPool(emb_dim) for band in self.bands}
            )
        elif pooling == "mean":
            self.poolers = None
        else:
            raise ValueError("pooling must be 'attention' or 'mean'")

        self.empty_band_tokens = nn.ParameterDict(
            {band: nn.Parameter(torch.zeros(emb_dim)) for band in self.bands}
        )

        self.fusion = nn.Sequential(
            nn.Linear(len(self.bands) * emb_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def pool_band(self, band, patches):
        """
        patches: Tensor [N, 1, H, W]
        returns: Tensor [emb_dim]
        """
        if patches.numel() == 0 or patches.shape[0] == 0:
            return self.empty_band_tokens[band]

        patch_embs = self.patch_encoders[band](patches)

        if self.pooling == "mean":
            return patch_embs.mean(dim=0)

        return self.poolers[band](patch_embs)

    def forward(self, batch_bands):
        """
        batch_bands: dict of band -> list of B patch tensors
        returns: Tensor [B, condition_dim]
        """
        batch_size = len(batch_bands[self.bands[0]])
        device = next(self.parameters()).device

        sample_conditions = []

        for i in range(batch_size):
            band_reprs = []

            for band in self.bands:
                patches = batch_bands[band][i].to(device)
                band_repr = self.pool_band(band, patches)
                band_reprs.append(band_repr)

            sample_repr = torch.cat(band_reprs, dim=0)  # [5 * emb_dim]
            sample_conditions.append(sample_repr)

        sample_conditions = torch.stack(sample_conditions, dim=0)
        condition = self.fusion(sample_conditions)

        return condition
