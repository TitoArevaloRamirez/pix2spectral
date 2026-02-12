from pyPro4Sail import prospect_jacobian

import torch
import torch.nn as nn
import numpy as np



# ============================================================
# 1) Parameter bounding (physics-valid PROSPECT inputs)
# ============================================================

class ProspectDParameterBounds(nn.Module):
    """
    Maps raw network outputs to bounded PROSPECT-D parameters with sigmoid scaling.

    Order: (Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant)
    """
    def __init__(self, mins=None, maxs=None, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

        # Reasonable defaults (adjust to your dataset/protocol if needed)
        default_mins = torch.tensor([1.0,   0.0,  0.0,  0.0,  0.0001, 0.0001, 0.0], dtype=torch.float32)
        default_maxs = torch.tensor([3.5, 100.0, 30.0,  2.0,  0.0500, 0.0300, 30.0], dtype=torch.float32)

        if mins is None:
            mins = default_mins
        if maxs is None:
            maxs = default_maxs

        mins = torch.as_tensor(mins, dtype=torch.float32)
        maxs = torch.as_tensor(maxs, dtype=torch.float32)

        if mins.shape != (7,) or maxs.shape != (7,):
            raise ValueError("mins and maxs must be shape [7]")
        if torch.any(maxs <= mins):
            raise ValueError("Each max must be > min")

        self.register_buffer("mins", mins)
        self.register_buffer("maxs", maxs)

    def forward(self, raw_params):
        u = torch.sigmoid(raw_params)
        u = u * (1.0 - 2.0 * self.eps) + self.eps
        return self.mins + u * (self.maxs - self.mins)


# ============================================================
# 2) Differentiable PROSPECT-D using analytic Jacobian
# ============================================================

class ProspectDLayerAnalytic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params):
        """
        params: torch [B,7]
        returns: rho reflectance torch [B,L]
        """
        p_np = params.detach().cpu().numpy().astype(np.float64)
        B = p_np.shape[0]

        rho_list = []
        J_list = []

        for i in range(B):
            p = p_np[i]

            wl, rho, tau, Delta_rho, Delta_tau = prospect_jacobian.JacProspectD(
                float(p[0]), float(p[1]), float(p[2]),
                float(p[3]), float(p[4]), float(p[5]),
                float(p[6]),
            )

            rho = np.asarray(rho, dtype=np.float32).reshape(-1)  # [L]
            Delta_rho = np.asarray(Delta_rho, dtype=np.float32)

            # Delta_rho is typically [7, L] in the source; convert to J [L,7]
            if Delta_rho.ndim != 2:
                raise ValueError("Delta_rho must be 2D, got " + str(Delta_rho.shape))

            if Delta_rho.shape[0] == 7:
                J = Delta_rho.T
            elif Delta_rho.shape[1] == 7:
                J = Delta_rho
            else:
                raise ValueError("Unexpected Delta_rho shape: " + str(Delta_rho.shape))

            rho_list.append(rho)
            J_list.append(J)

        rho_np = np.stack(rho_list, axis=0)   # [B,L]
        J_np = np.stack(J_list, axis=0)       # [B,L,7]

        ctx.save_for_backward(torch.from_numpy(J_np))
        return torch.from_numpy(rho_np).to(params.device).type_as(params)

    @staticmethod
    def backward(ctx, grad_output):
        (J_t,) = ctx.saved_tensors
        J_t = J_t.to(grad_output.device).type_as(grad_output)  # [B,L,7]
        grad_params = torch.einsum("bl,blk->bk", grad_output, J_t)
        return grad_params


def prospectd_reflectance_torch(params):
    return ProspectDLayerAnalytic.apply(params)


class PhysicsInformedProspectHead(nn.Module):
    """
    raw_params [B,7] -> bounded pParams [B,7] -> y_fake [B,L] (reflectance), differentiable
    """
    def __init__(self, mins=None, maxs=None):
        super().__init__()
        self.bounds = ProspectDParameterBounds(mins=mins, maxs=maxs)

    def forward(self, raw_params):
        pParams = self.bounds(raw_params)
        y_fake = prospectd_reflectance_torch(pParams)
        return y_fake, pParams


# ============================================================
# 3) Minimal U-Net patch encoder for 32x32 patches (shared weights)
# ============================================================

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallUNetPatchEncoder(nn.Module):
    """
    U-Net style encoder for patch embeddings.
    Input:  [N, 1, 32, 32]
    Output: [N, embed_dim]
    """
    def __init__(self, in_channels=1, base_features=8, embed_dim=64):
        super().__init__()
        f = base_features

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(f),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.d1 = Down(f,   f * 2)
        self.d2 = Down(f*2, f * 4)
        self.d3 = Down(f*4, f * 8)
        self.d4 = Down(f*8, f * 8)
        self.d5 = Down(f*8, f * 8)

        # 1 -> 2 -> 4 -> 8 -> 16 -> 32
        self.u1 = Up(f*8,          f*8, use_dropout=False)
        self.u2 = Up(f*8 + f*8,    f*8, use_dropout=False)
        self.u3 = Up(f*8 + f*8,    f*4, use_dropout=False)
        self.u4 = Up(f*4 + f*4,    f*2, use_dropout=False)
        self.u5 = Up(f*2 + f*2,    f,   use_dropout=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(f + f, f, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Linear(f, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x0 = self.initial(x)  # [N,f,32,32]
        x1 = self.d1(x0)      # [N,2f,16,16]
        x2 = self.d2(x1)      # [N,4f,8,8]
        x3 = self.d3(x2)      # [N,8f,4,4]
        x4 = self.d4(x3)      # [N,8f,2,2]
        x5 = self.d5(x4)      # [N,8f,1,1]

        y1 = self.u1(x5)                       # [N,8f,2,2]
        y2 = self.u2(torch.cat([y1, x4], 1))    # [N,8f,4,4]
        y3 = self.u3(torch.cat([y2, x3], 1))    # [N,4f,8,8]
        y4 = self.u4(torch.cat([y3, x2], 1))    # [N,2f,16,16]
        y5 = self.u5(torch.cat([y4, x1], 1))    # [N,f,32,32]

        z = self.fuse(torch.cat([y5, x0], 1))   # [N,f,32,32]
        z = z.mean(dim=(2, 3))                  # [N,f]
        z = self.proj(z)                        # [N,embed_dim]
        return z


class MeanPool(nn.Module):
    def forward(self, E):
        if E.numel() == 0:
            return torch.zeros(E.shape[1], device=E.device, dtype=E.dtype)
        return E.mean(dim=0)


# ============================================================
# 4) Full generator: multispectral patches -> params -> PROSPECT spectrum
# ============================================================

class MultiSpectralPatchToProspectGenerator(nn.Module):
    """
    Input per sample:
      band_patches dict:
        "blue": [N1,1,32,32]
        "green": [N2,1,32,32]
        "red": [N3,1,32,32]
        "nir": [N4,1,32,32]
        "red_edge": [N5,1,32,32]

    Output per sample:
      y_fake: [L]
      pParams: [7]
    """
    def __init__(self, bands=None, base_features=8, embed_dim=64, mins=None, maxs=None):
        super().__init__()
        if bands is None:
            bands = ["blue", "green", "red", "nir", "red_edge"]
        self.bands = bands

        # Shared weights across bands
        self.patch_encoder = SmallUNetPatchEncoder(in_channels=1, base_features=base_features, embed_dim=embed_dim)
        self.pool = nn.ModuleDict({b: MeanPool() for b in self.bands})

        fused_dim = embed_dim * len(self.bands)
        self.param_mlp = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 7),
        )

        self.physics = PhysicsInformedProspectHead(mins=mins, maxs=maxs)

    def forward(self, band_patches):
        vecs = []
        for b in self.bands:
            P = band_patches[b]          # [N,1,32,32]
            E = self.patch_encoder(P)    # [N,D]
            v = self.pool[b](E)          # [D]
            vecs.append(v)

        fused = torch.cat(vecs, dim=0).unsqueeze(0)  # [1, 5D]
        raw_params = self.param_mlp(fused)           # [1,7]
        y_fake, pParams = self.physics(raw_params)   # y_fake [1,L], pParams [1,7]
        return y_fake.squeeze(0), pParams.squeeze(0)

    def forward_batch_list(self, batch_band_dict):
        """
        If your DataLoader collate_fn returns dict band -> list[tensor],
        each tensor is [N_i,1,32,32] and list length is batch size.

        Returns:
          y_fake: [B,L]
          pParams: [B,7]
        """
        B = len(batch_band_dict[self.bands[0]])
        y_list = []
        p_list = []

        for i in range(B):
            sample = {b: batch_band_dict[b][i] for b in self.bands}
            y_i, p_i = self.forward(sample)
            y_list.append(y_i)
            p_list.append(p_i)

        return torch.stack(y_list, dim=0), torch.stack(p_list, dim=0)


# ============================================================
# 5) Checking code (forward shapes + gradient flow)
# ============================================================

def _make_fake_sample(device):
    return {
        "blue": torch.randn(7, 1, 32, 32, device=device),
        "green": torch.randn(3, 1, 32, 32, device=device),
        "red": torch.randn(5, 1, 32, 32, device=device),
        "nir": torch.randn(9, 1, 32, 32, device=device),
        "red_edge": torch.randn(4, 1, 32, 32, device=device),
    }


def _make_fake_batch(device, B=2):
    batch = {b: [] for b in ["blue", "green", "red", "nir", "red_edge"]}
    for _ in range(B):
        s = _make_fake_sample(device)
        for b in batch.keys():
            batch[b].append(s[b])
    return batch


def check_generator(gen, device):
    gen.to(device)
    gen.train()

    # Single forward
    sample = _make_fake_sample(device)
    y_fake, pParams = gen(sample)

    print("Single forward OK")
    print("  y_fake shape:", tuple(y_fake.shape))
    print("  pParams shape:", tuple(pParams.shape))

    # Batch forward
    batch = _make_fake_batch(device, B=2)
    y_b, p_b = gen.forward_batch_list(batch)

    print("Batch forward OK")
    print("  y_b shape:", tuple(y_b.shape))
    print("  p_b shape:", tuple(p_b.shape))

    # Gradient flow check
    y_true = torch.rand_like(y_b)
    loss = torch.nn.functional.l1_loss(y_b, y_true)

    gen.zero_grad(set_to_none=True)
    loss.backward()

    has_grad = False
    max_grad = 0.0
    for name, p in gen.named_parameters():
        if p.grad is not None:
            has_grad = True
            g = float(p.grad.abs().max().detach().cpu())
            if g > max_grad:
                max_grad = g

    print("Backward OK")
    print("  has_grad:", has_grad)
    print("  max_grad:", max_grad)


if __name__ == "__main__":
    device = torch.device("cpu")

    gen = MultiSpectralPatchToProspectGenerator(
        bands=["blue", "green", "red", "nir", "red_edge"],
        base_features=8,
        embed_dim=64,
    )

    check_generator(gen, device)

