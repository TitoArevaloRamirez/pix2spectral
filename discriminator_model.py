import torch
import torch.nn as nn


class CNNBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, use_bn=False):
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                padding_mode="reflect",
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SpectralPatchDiscriminator1D(nn.Module):
    """
    1D PatchGAN-style discriminator over wavelength.

    Inputs:
      x : real spectrum  [B, L]  or [B, 1, L]
      y : generated      [B, L]  or [B, 1, L]

    Output:
      patch logits       [B, 1, L']  (use BCEWithLogitsLoss)
    """
    def __init__(self, in_channels=1, features=(64, 128, 256, 512), use_bn=False):
        super().__init__()

        # Concatenate (x,y) along channel dim like pix2pix: [B, 2*in_channels, L]
        self.initial = nn.Sequential(
            nn.Conv1d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        c_in = features[0]
        for f in features[1:]:
            layers.append(
                CNNBlock1D(
                    c_in,
                    f,
                    stride=1 if f == features[-1] else 2,
                    use_bn=use_bn,
                )
            )
            c_in = f

        # final patch logits
        layers.append(
            nn.Conv1d(
                c_in,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # Accept [B, L] or [B, 1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if y.dim() == 2:
            y = y.unsqueeze(1)

        z = torch.cat([x, y], dim=1)  # [B, 2, L] if in_channels=1
        z = self.initial(z)
        return self.model(z)


class SpectralDiscriminator1D(nn.Module):
    """
    Single-input 1D discriminator (standard GAN):
    Input:
      s : spectrum [B, L] or [B, 1, L]
    Output:
      patch logits [B, 1, L']
    """
    def __init__(self, in_channels=1, features=(64, 128, 256, 512), use_bn=False):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        c_in = features[0]
        for f in features[1:]:
            layers.append(
                CNNBlock1D(
                    c_in,
                    f,
                    stride=1 if f == features[-1] else 2,
                    use_bn=use_bn,
                )
            )
            c_in = f

        layers.append(
            nn.Conv1d(
                c_in,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, s):
        if s.dim() == 2:
            s = s.unsqueeze(1)
        s = self.initial(s)
        return self.model(s)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    B = 1
    L = 2151  # e.g., 350..2500 nm with 1nm step

    real = torch.randn(B, L)
    fake = torch.randn(B, L)

    # Pairwise (pix2pix-like): compare real vs generated as a pair
    D_pair = SpectralPatchDiscriminator1D(in_channels=1, use_bn=False)
    logits_pair = D_pair(real, fake)
    print("pair logits:", logits_pair.shape)  # [B, 1, L']

    # Single-input: discriminate one spectrum at a time
    D_single = SpectralDiscriminator1D(in_channels=1, use_bn=True)
    logits_real = D_single(real)
    logits_fake = D_single(fake)
    print("single logits:", logits_real.shape, logits_fake.shape)

    # Loss tip:
    # criterion = nn.BCEWithLogitsLoss()
    # loss_real = criterion(logits_real, torch.ones_like(logits_real))
    # loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))
