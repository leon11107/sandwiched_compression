"""DPP total loss: L = gamma*L_P + lambda*L_R + L_F  (paper eq 6/7/8, Sec 3.7).

- L_F  = alpha*L1(Y) + beta*(1 - MS-SSIM(Y))  on LUMA (alpha=0.2, beta=0.8). Fidelity
         anchor (coeff 1) — dominant, caps metric-hacking.
- L_P  = -predicted MOS (NR-NIMA-on-Koniq) on the decoded-Y+lossless-UV->RGB pred.
- L_R  = bits-per-pixel of the codec (normalized rate; lambda shifts the RD point).
gamma ~ 0.01, lambda in [0.001, 0.01] (per paper/supp).
"""
from __future__ import annotations
import torch
from torch_port.preproc import tf_rgb_to_yuv
from torch_port.losses import ssim_multiscale_tf


def fidelity_luma(gt_bhwc, pred_bhwc, alpha=0.2, beta=0.8):
    """L_F on luminance, per-sample [B]. gt/pred [B,H,W,3] in [0,255]."""
    Yg = tf_rgb_to_yuv(gt_bhwc)[..., 0:1]
    Yp = tf_rgb_to_yuv(pred_bhwc)[..., 0:1]
    l1 = ((Yg - Yp) / 255.0).abs().mean(dim=(1, 2, 3))                 # [B]
    msssim = ssim_multiscale_tf(Yg, Yp, max_val=255.0, filter_size=7)  # [B], on 1ch Y
    return alpha * l1 + beta * (1.0 - msssim)


def dpp_loss(gt_bhwc, pred_bhwc, rate_bits, perceptual, gamma=0.01, lam=0.01,
             alpha=0.2, beta=0.8):
    """Returns (total[B], components dict). rate_bits [B] = codec real bits per sample."""
    H, W = pred_bhwc.shape[1], pred_bhwc.shape[2]
    LF = fidelity_luma(gt_bhwc, pred_bhwc, alpha, beta)   # [B]
    LP = perceptual.loss(pred_bhwc)                        # [B] = -MOS
    LR = rate_bits / float(H * W)                          # [B] bpp
    total = LF + gamma * LP + lam * LR
    comp = {"LF": LF.mean(), "LP": LP.mean(), "LR_bpp": LR.mean(),
            "MOS": -LP.mean(), "total": total.mean()}
    return total, comp
