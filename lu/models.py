"""Upscalers: non-learned baselines and two tiny CNNs, all x2.

Every learned model predicts a residual on top of a bilinear upsample of the LR colour, so an
untrained network starts as bilinear upscaling and training only has to learn the detail.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img, mv, depth_prev=None, pz=None, tol=0.05):
    """Backward-warp the previous frame `img` to the current frame.

    mv: (N,2,H,W) pixel offset from current pixel to its position in the previous frame.
    Returns the warped image and a validity mask (in bounds and, if depths are given,
    not disoccluded: the reprojected depth must match the previous depth buffer).
    """
    N, _, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    u = xs + 0.5 + mv[:, 0]
    v = ys + 0.5 + mv[:, 1]
    grid = torch.stack([2 * u / W - 1, 2 * v / H - 1], -1)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    mask = ((grid.abs() <= 1).all(-1, keepdim=True)).permute(0, 3, 1, 2).to(img.dtype)
    if depth_prev is not None and pz is not None:
        d = F.grid_sample(depth_prev, grid, mode="nearest", padding_mode="border", align_corners=False)
        mask = mask * ((d - pz).abs() < tol * pz.abs().clamp(min=1e-3)).to(img.dtype)
    return out, mask


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class Bicubic(nn.Module):
    learned = False

    def forward(self, x):
        return F.interpolate(x[:, :3], scale_factor=2, mode="bicubic", align_corners=False)


class CASUpscale(nn.Module):
    """Bicubic + contrast-adaptive sharpening (CAS-style, in the spirit of FSR 1's RCAS pass).

    Sharpening strength adapts to local contrast: flat or already high-contrast areas are
    sharpened less, which limits halos. Intended for display-referred [0,1] input (domain=ldr).
    """
    learned = False

    def __init__(self, sharpness: float = 0.6):
        super().__init__()
        self.peak = -1.0 / (8.0 + (5.0 - 8.0) * sharpness)

    def forward(self, x):
        up = F.interpolate(x[:, :3], scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
        p = F.pad(up, (1, 1, 1, 1), mode="replicate")
        n, s = p[:, :, :-2, 1:-1], p[:, :, 2:, 1:-1]
        w_, e = p[:, :, 1:-1, :-2], p[:, :, 1:-1, 2:]
        win = F.unfold(p, 3).view(up.shape[0], 3, 9, *up.shape[2:])
        mn, mx = win.min(2).values, win.max(2).values
        amp = (torch.minimum(mn, 1 - mx) / mx.clamp(min=1e-4)).clamp(0, 1).sqrt()
        w = amp * self.peak
        return ((n + s + w_ + e) * w + up) / (1 + 4 * w)


class ESPCN(nn.Module):
    """Shi et al. 2016 sub-pixel CNN, shrunk; ~10k parameters for RGB input."""
    learned = True

    def __init__(self, cin=3, width=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, width, 5, padding=2), nn.Tanh(),
            nn.Conv2d(width, width // 2, 3, padding=1), nn.Tanh(),
            nn.Conv2d(width // 2, 3 * 4, 3, padding=1),
        )

    def forward(self, x):
        return upsample(x[:, :3]) + F.pixel_shuffle(self.body(x), 2)


class LiteUNet(nn.Module):
    """Two-scale U-Net at LR resolution with a pixel-shuffle x2 head.

    width=16 -> ~40k params; all convs are 3x3 so it maps cleanly to mobile NPUs / INT8.
    """
    learned = True

    def __init__(self, cin=3, width=16):
        super().__init__()
        w = width
        self.head = nn.Sequential(nn.Conv2d(cin, w, 3, padding=1), nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(nn.Conv2d(2 * w, w * 4, 3, padding=1), nn.PixelShuffle(2))
        self.fuse = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True))
        self.tail = nn.Sequential(nn.Conv2d(w, 3 * 4, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, x):
        h = self.head(x)
        d = self.down(h)
        u = self.up(d)[:, :, : h.shape[2], : h.shape[3]]
        f = self.fuse(h + u)
        return upsample(x[:, :3]) + self.tail(f)


def build_model(cfg, cin):
    name = cfg["model"]
    if name == "bicubic":
        return Bicubic()
    if name == "cas":
        return CASUpscale(cfg.get("sharpness", 0.6))
    if name == "espcn":
        return ESPCN(cin, cfg.get("width", 32))
    if name == "liteunet":
        return LiteUNet(cin, cfg.get("width", 16))
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, x):
    """Multiply-accumulates of all Conv2d layers for input x."""
    total = [0]

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        total[0] += out.numel() // out.shape[0] * k * mod.in_channels // mod.groups

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total[0]
