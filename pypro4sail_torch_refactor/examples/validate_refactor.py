"""Smoke test for the PyTorch refactor."""

import torch
import pypro4sail_torch as p4s
from pypro4sail_torch import prospect, four_sail


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    wl, rho, tau = prospect.prospectd(1.5, 40.0, 8.0, 0.0, 0.01, 0.009, 1.0, device=device, dtype=dtype)
    assert torch.isfinite(rho).all()
    assert torch.isfinite(tau).all()
    lidf = four_sail.calc_lidf_campbell(57.0, device=device, dtype=dtype)
    out = four_sail.foursail(3.0, 0.01, lidf, 30.0, 10.0, 0.0, rho, tau, torch.ones_like(rho) * 0.2)
    assert torch.isfinite(out[17]).all()
    Cab = torch.tensor(40.0, device=device, dtype=dtype, requires_grad=True)
    _, canopy = p4s.run(1.5, Cab, 8.0, 0.0, 0.01, 0.009, 1.0, 3.0, 0.01, 30.0, 180.0, 10.0, 180.0, 57.0, device=device, dtype=dtype)
    canopy[100:200].mean().backward()
    assert Cab.grad is not None and torch.isfinite(Cab.grad)
    print(f"ok on {device}; canopy shape={tuple(canopy.shape)}; dloss/dCab={Cab.grad.item():.6g}")


if __name__ == "__main__":
    main()
