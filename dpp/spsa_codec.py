"""Zeroth-order (SPSA) gradient through the REAL JPEG codec — option (b).

The codec is non-differentiable; estimate g = d(loss_real)/d(codec_input) via SPSA
(Rademacher luma perturbations, real codec value evals), then inject it into the
preprocessor's autograd as a surrogate gradient. loss_real = DPP loss computed on the
REAL codec output (L_F luma + gamma*L_P[NIMA] + lambda*L_R[real bits]).

This module: (i) loss_real value fn, (ii) spsa_grad estimator, (iii) a validate-first
DESCENT test (does stepping along -g_hat actually reduce the real loss?).
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspace/sandwiched_compression")
import numpy as np
import torch
from torch_port.codec import encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
from torch_port.losses import ssim_multiscale_tf


def _restore(dec, orig):
    d = tf_rgb_to_yuv(dec); o = tf_rgb_to_yuv(orig)
    return tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))


@torch.no_grad()
def loss_real(z, x, q, nima, gamma=0.01, lam=0.005, alpha=0.2, beta=0.8):
    """Per-image DPP loss on the REAL codec. z,x [B,H,W,3] 0..255. Returns [B]."""
    dev = z.device
    d, bits = encode_decode_with_jpeg(z.detach().cpu().numpy(), float(q), False, False)
    dec = _restore(torch.from_numpy(d).to(dev).float(), x)
    Yx = tf_rgb_to_yuv(x)[..., 0:1]; Yd = tf_rgb_to_yuv(dec)[..., 0:1]
    l1 = ((Yx - Yd) / 255.0).abs().mean(dim=(1, 2, 3))
    ms = ssim_multiscale_tf(Yx, Yd, max_val=255.0, filter_size=7)
    LF = alpha * l1 + beta * (1.0 - ms)
    mos = nima.mos(dec)  # value (higher=better)
    LP = -mos
    H, W = x.shape[1], x.shape[2]
    LR = torch.from_numpy(bits).to(dev).float() / (H * W)
    return LF + gamma * LP + lam * LR


@torch.no_grad()
def spsa_grad(z, x, q, nima, K=32, c=2.0, **lw):
    """SPSA estimate of d(loss_real)/dz. Rademacher LUMA perturbations (broadcast to
    RGB = pure-luma, matching codec_luma_only). Returns g_hat [B,H,W,3]."""
    B, H, W, _ = z.shape
    g = torch.zeros_like(z)
    for _ in range(K):
        d = (torch.randint(0, 2, (B, H, W, 1), device=z.device).float() * 2 - 1).repeat(1, 1, 1, 3)
        lp = loss_real(z + c * d, x, q, nima, **lw)
        lm = loss_real(z - c * d, x, q, nima, **lw)
        coef = ((lp - lm) / (2 * c)).view(B, 1, 1, 1)
        g += coef * d
    return g / K


if __name__ == "__main__":
    # validate-first: does -g_hat reduce the real loss? sweep K and step alpha.
    import argparse, glob, sys
    from PIL import Image
    sys.path.insert(0, "/workspace/sandwiched_compression")
    from dpp.perceptual import NimaMOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--q", type=float, default=32.0)
    ap.add_argument("--Ks", default="8,32,64"); ap.add_argument("--alphas", default="1,4,16")
    a = ap.parse_args()
    dev = "cuda"
    nima = NimaMOS("nima-koniq", device=dev)
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).to(dev)[None]
            for p in sorted(glob.glob("/workspace/sandwiched_compression/dpp/data/val/*.png"))[:a.n]]
    base = torch.stack([loss_real(im, im, a.q, nima)[0] for im in imgs]).mean()
    print(f"baseline real loss (z=x, identity preproc) = {float(base):.5f}")
    # SPSA-convergence: cosine between independent K-sample estimates (high => low variance)
    for K in [int(k) for k in a.Ks.split(",")]:
        coss = []
        for im in imgs:
            g1 = spsa_grad(im, im, a.q, nima, K=K).reshape(-1).cpu().numpy()
            g2 = spsa_grad(im, im, a.q, nima, K=K).reshape(-1).cpu().numpy()
            coss.append(float(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-30)))
        # descent test: step along -g_hat (normalized), measure real-loss change
        for al in [float(x) for x in a.alphas.split(",")]:
            dl = []
            for im in imgs:
                g = spsa_grad(im, im, a.q, nima, K=K)
                gn = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-9)
                l0 = loss_real(im, im, a.q, nima); l1 = loss_real(im - al * gn * 100, im, a.q, nima)
                dl.append(float((l1 - l0).mean()))
            print(f"  K={K:3d} selfcos={np.mean(coss):+.3f} | step alpha={al:>4g}: dLoss={np.mean(dl):+.5f} "
                  f"({'DESCENDS' if np.mean(dl) < 0 else 'no'})")
