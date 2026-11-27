"""Colour-space and dynamic-range helpers.

All HDR tensors are *linear* radiance. "Exposed" means multiplied by the per-sequence
auto-exposure scalar so that middle grey sits near 0.18.

Domains the network can work in (experiment E2):
  linear : exposed linear HDR (unbounded, highlights dominate L1)
  mulaw  : mu-law compressed HDR, log(1 + mu*x) / log(1 + mu)
  ldr    : tone-mapped + sRGB-encoded image, i.e. tone-map *before* upscaling
"""
import math

import torch

MU = 5000.0
_LOG1P_MU = math.log1p(MU)


def aces(x: torch.Tensor) -> torch.Tensor:
    """Narkowicz's fitted ACES filmic curve (exposed linear -> display linear, [0,1])."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = x.clamp(min=0)
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0, 1)


def srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def srgb_eotf(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def tonemap_srgb(hdr_exposed: torch.Tensor) -> torch.Tensor:
    """Exposed linear HDR -> display-referred sRGB in [0,1]."""
    return srgb_oetf(aces(hdr_exposed))


def to_domain(hdr_exposed: torch.Tensor, domain: str) -> torch.Tensor:
    if domain == "linear":
        return hdr_exposed
    if domain == "mulaw":
        return torch.log1p(MU * hdr_exposed.clamp(min=0)) / _LOG1P_MU
    if domain == "ldr":
        return tonemap_srgb(hdr_exposed)
    raise ValueError(domain)


def from_domain(y: torch.Tensor, domain: str) -> torch.Tensor:
    """Inverse of to_domain for HDR domains (returns exposed linear HDR)."""
    if domain == "linear":
        return y.clamp(min=0)
    if domain == "mulaw":
        return torch.expm1(y.clamp(min=0) * _LOG1P_MU) / MU
    raise ValueError(f"{domain} has no HDR inverse")


def domain_to_srgb(y: torch.Tensor, domain: str) -> torch.Tensor:
    """Network output in `domain` -> final display sRGB in [0,1] (what metrics see)."""
    if domain == "ldr":
        return y.clamp(0, 1)
    return tonemap_srgb(from_domain(y, domain))


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def srgb_to_lab(srgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] (N,3,H,W) -> CIELAB (D65)."""
    lin = srgb_eotf(srgb)
    m = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=srgb.dtype, device=srgb.device,
    )
    xyz = torch.einsum("ij,njhw->nihw", m, lin)
    white = torch.tensor([0.95047, 1.0, 1.08883], dtype=srgb.dtype, device=srgb.device).view(1, 3, 1, 1)
    t = xyz / white
    eps = 216 / 24389
    kappa = 24389 / 27
    f = torch.where(t > eps, t.clamp(min=1e-8) ** (1 / 3), (kappa * t + 16) / 116)
    L = 116 * f[:, 1:2] - 16
    a = 500 * (f[:, 0:1] - f[:, 1:2])
    b = 200 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], 1)
