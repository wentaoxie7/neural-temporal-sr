"""Training losses (experiment E4). All computed in the network's working domain."""
import torch

from .metrics import ssim_map


def l1(x, y):
    return (x - y).abs().mean()


def ssim_loss(x, y):
    return 1 - ssim_map(x, y)[:, :, 5:-5, 5:-5].mean()


def fft_loss(x, y):
    """L1 between Fourier amplitude spectra: penalises lost high-frequency detail (blur)."""
    fx = torch.fft.rfft2(x, norm="ortho").abs()
    fy = torch.fft.rfft2(y, norm="ortho").abs()
    return (fx - fy).abs().mean()


def temporal_loss(out_t, out_prev_w, gt_t, gt_prev_w, mask):
    """Match the frame-to-frame change of the output to that of the ground truth."""
    d = ((out_t - out_prev_w) - (gt_t - gt_prev_w)).abs()
    return (d * mask).sum() / (mask.sum() * d.shape[1]).clamp(min=1)


def spatial_loss(out, gt, weights):
    loss = weights.get("l1", 1.0) * l1(out, gt)
    if weights.get("ssim"):
        loss = loss + weights["ssim"] * ssim_loss(out, gt)
    if weights.get("fft"):
        loss = loss + weights["fft"] * fft_loss(out, gt)
    return loss
