"""PyTorch port of distortion/perceptual_losses.py fidelity terms + a faithful
port of tf.image.ssim_multiscale (the MS-SSIM used by distortion_l1_msssim).

All functions take channels-LAST tensors [B,H,W,C] in [0,255] to mirror TF.
Per-sample [B] outputs (the loss machinery reduces over batch + adds gamma*rate).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _fspecial_gauss(size: int, sigma: float, device, dtype):
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)  # [size,size]


def _reducer(x, kernel):  # x [B,C,H,W], depthwise VALID gaussian
    C = x.shape[1]
    k = kernel.expand(C, 1, kernel.shape[-2], kernel.shape[-1])
    return F.conv2d(x, k, stride=1, padding=0, groups=C)


def _ssim_per_channel(x, y, max_val, filter_size, filter_sigma, k1, k2):
    """x,y: [B,C,H,W]. Returns (ssim[B,C], cs[B,C]) — matches tf _ssim_per_channel."""
    kernel = _fspecial_gauss(filter_size, filter_sigma, x.device, x.dtype)
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    mean0 = _reducer(x, kernel); mean1 = _reducer(y, kernel)
    num0 = mean0 * mean1 * 2.0
    den0 = mean0 ** 2 + mean1 ** 2
    luminance = (num0 + c1) / (den0 + c1)
    num1 = _reducer(x * y, kernel) * 2.0
    den1 = _reducer(x ** 2 + y ** 2, kernel)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    ssim_val = (luminance * cs).mean(dim=(-2, -1))  # over spatial -> [B,C]
    cs = cs.mean(dim=(-2, -1))
    return ssim_val, cs


def ssim_multiscale_tf(img1, img2, max_val=255.0, filter_size=7, filter_sigma=1.5,
                       k1=0.01, k2=0.03, power_factors=_MSSSIM_WEIGHTS):
    """Port of tf.image.ssim_multiscale. img1,img2 [B,H,W,C] in [0,max_val].
    Returns [B] (mean over channels). Assumes even spatial dims (our 128/256 crops)."""
    x = img1.permute(0, 3, 1, 2).contiguous()
    y = img2.permute(0, 3, 1, 2).contiguous()
    mcs = []
    ssim_last = None
    for k in range(len(power_factors)):
        if k > 0:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)  # VALID 2x2
            y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)
        ssim_k, cs_k = _ssim_per_channel(x, y, max_val, filter_size, filter_sigma, k1, k2)
        mcs.append(torch.relu(cs_k))
        ssim_last = torch.relu(ssim_k)
    mcs.pop()  # drop last scale's cs; use ssim there instead
    stack = torch.stack(mcs + [ssim_last], dim=-1)  # [B,C,5]
    pf = torch.tensor(power_factors, dtype=x.dtype, device=x.device)
    ms = torch.prod(stack ** pf, dim=-1)  # [B,C]
    return ms.mean(dim=-1)  # [B]


# ---- distortion fidelity terms (per-sample [B]) ----------------------------
def distortion_mse01(gt, pred, lambda_l2=1.0):
    return lambda_l2 * ((gt - pred) / 255.0).pow(2).mean(dim=(1, 2, 3))


def distortion_mae01(gt, pred, lambda_l2=1.0):
    return lambda_l2 * ((gt - pred) / 255.0).abs().mean(dim=(1, 2, 3))


def distortion_l1_msssim(gt, pred, lambda_l2=1.0, alpha=0.2, beta=0.8):
    l1_01 = ((gt - pred) / 255.0).abs().mean(dim=(1, 2, 3))         # [B]
    msssim = ssim_multiscale_tf(gt, pred, max_val=255.0, filter_size=7)  # [B]
    lf = alpha * l1_01 + beta * (1.0 - msssim)
    hwc = float(np.prod(pred.shape[1:]))  # norm-cancellation (matches TF)
    return lambda_l2 * lf * hwc
