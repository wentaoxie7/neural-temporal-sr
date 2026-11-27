"""Image-quality metrics and image-processing analysis. Inputs are display sRGB in [0,1]."""
import torch
import torch.nn.functional as F

from .color import luminance, srgb_to_lab


def psnr(x, y):
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return 10 * torch.log10(1 / mse)


def _gauss1d(win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win) - win // 2) ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).float()


_G = _gauss1d()


def _blur(x):
    """Separable 11x11 Gaussian (sigma 1.5): two 1-D passes, far cheaper on CPU than one 2-D conv."""
    ch = x.shape[1]
    g = _G.to(x.dtype)
    x = F.conv2d(x, g.view(1, 1, 1, -1).expand(ch, 1, 1, 11), padding=(0, 5), groups=ch)
    return F.conv2d(x, g.view(1, 1, -1, 1).expand(ch, 1, 11, 1), padding=(5, 0), groups=ch)


def ssim_map(x, y, data_range=1.0):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    n = x.shape[0]
    mu_x, mu_y, exx, eyy, exy = _blur(torch.cat([x, y, x * x, y * y, x * y], 0)).split(n, 0)
    sxx = exx - mu_x ** 2
    syy = eyy - mu_y ** 2
    sxy = exy - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def ssim(x, y):
    return ssim_map(x, y)[:, :, 5:-5, 5:-5].mean(dim=(1, 2, 3))


def delta_e76(x, y):
    """Mean CIE76 colour difference (Delta E*ab) in CIELAB."""
    return (srgb_to_lab(x) - srgb_to_lab(y)).norm(dim=1).mean(dim=(1, 2))


_LAP = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
_SOBEL_X = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)


def _local_std(lum, k=7):
    mu = F.avg_pool2d(lum, k, 1, k // 2, count_include_pad=False)
    var = F.avg_pool2d(lum * lum, k, 1, k // 2, count_include_pad=False) - mu ** 2
    return var.clamp(min=0).sqrt()


def image_analysis(out, gt):
    """Contrast, sharpness, dynamic-range and edge/flat breakdown of the error (batch means)."""
    lo, lg = luminance(out), luminance(gt)
    err = (out - gt).abs().mean(1, keepdim=True)
    lap_o = F.conv2d(lo, _LAP, padding=1).abs().mean()
    lap_g = F.conv2d(lg, _LAP, padding=1).abs().mean()
    gx = F.conv2d(lg, _SOBEL_X, padding=1)
    gy = F.conv2d(lg, _SOBEL_X.transpose(2, 3), padding=1)
    grad = (gx ** 2 + gy ** 2).sqrt()
    edge_thr = torch.quantile(grad.flatten()[:: max(1, grad.numel() // 200000)], 0.9)
    flat_thr = torch.quantile(grad.flatten()[:: max(1, grad.numel() // 200000)], 0.5)

    def masked(m):
        m = m.float()
        return (err * m).sum().item() / m.sum().clamp(min=1).item()

    return {
        "contrast_ratio": (_local_std(lo).mean() / _local_std(lg).mean().clamp(min=1e-6)).item(),
        "global_contrast_ratio": (lo.std() / lg.std().clamp(min=1e-6)).item(),
        "sharpness_ratio": (lap_o / lap_g.clamp(min=1e-6)).item(),
        "mae_shadows": masked(lg < 0.15),
        "mae_midtones": masked((lg >= 0.15) & (lg <= 0.75)),
        "mae_highlights": masked(lg > 0.75),
        "clipped_highlights_frac": ((lo > 0.98).float().mean() - (lg > 0.98).float().mean()).item(),
        "mae_edges": masked(grad >= edge_thr),
        "mae_flat": masked(grad <= flat_thr),
    }


def temporal_flicker(out_t, out_prev_warped, gt_t, gt_prev_warped, mask):
    """Temporal error: frame-to-frame change of the output vs. that of the ground truth.

    Zero means the output changes over time exactly like the reference (no extra flicker or
    ghosting); computed only where the reprojection is valid.
    """
    d = ((out_t - out_prev_warped) - (gt_t - gt_prev_warped)).abs().mean(1, keepdim=True)
    m = mask.float()
    return ((d * m).sum() / m.sum().clamp(min=1)).item()


def warping_error(out_t, out_prev_warped, mask):
    d = (out_t - out_prev_warped).abs().mean(1, keepdim=True)
    m = mask.float()
    return ((d * m).sum() / m.sum().clamp(min=1)).item()
