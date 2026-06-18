"""Differentiable virtual codec for the DPP reproduction (115100C.pdf S3.2-3.4).

Pieces:
  - soft block-matching prediction (Eq.1-3): MAE similarity over an M-search
    window, softmax relaxation of the argmin one-hot. INTER: reference =
    previous precoded frame. INTRA: reference = current precoded frame with a
    checkerboard DC-mask (declared implementation choice: the paper "masks the
    block being queried"; per-block masks don't vectorize, so blocks of one
    checkerboard colour are predicted from a frame whose same-colour blocks
    are reduced to their DC -> no AC self-leak, neighbours keep detail).
  - H.264 4x4 integer core transform (orthonormal scaling) on the residual,
    Qstep quantization with additive uniform noise (training) / rounding
    (eval); Qstep(QP) = base6[QP%6] * 2^floor(QP/6), QP sampled per step.
  - rate: compressai EntropyBottleneck over the 16 subband channels of
    y = DCT(residual)/Qstep (single density marginalizes over QP, as in the
    paper's QP-randomized training).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

BASE6 = [0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125]


def qstep_of_qp(qp):
    return BASE6[qp % 6] * (2.0 ** (qp // 6))


# ---- H.264 4x4 integer core transform (orthonormal version) ----------------
_C4 = np.array([[1, 1, 1, 1], [2, 1, -1, -2], [1, -1, -1, 1], [1, -2, 2, -1]],
               np.float32)
_S4 = np.diag(1.0 / np.sqrt((_C4 ** 2).sum(1))).astype(np.float32)
_T4 = (_S4 @ _C4)  # orthonormal: T4 @ T4.T == I


def t4(dev):
    return torch.from_numpy(_T4).to(dev)


def dct4(x):
    """[B,1,H,W] -> subbands [B,16,H/4,W/4] (H,W multiples of 4)."""
    B, _, H, W = x.shape
    T = t4(x.device)
    u = F.unfold(x, 4, stride=4)  # [B,16,L]
    blocks = u.transpose(1, 2).reshape(-1, 4, 4)
    c = T @ blocks @ T.transpose(0, 1)
    c = c.reshape(B, -1, 16).transpose(1, 2)
    return c.reshape(B, 16, H // 4, W // 4)


def idct4(c):
    B, _, h, w = c.shape
    T = t4(c.device)
    blocks = c.reshape(B, 16, -1).transpose(1, 2).reshape(-1, 4, 4)
    x = T.transpose(0, 1) @ blocks @ T
    u = x.reshape(B, -1, 16).transpose(1, 2)
    return F.fold(u, (h * 4, w * 4), 4, stride=4)


# ---- soft block matching (Eq.1-3) -------------------------------------------
def dc_checker_mask(x, k):
    """Replace each kxk block by its mean for blocks of checkerboard colour 0/1.
    -> (masked0, masked1): masked_c has colour-c blocks DC'd, others intact."""
    B, _, H, W = x.shape
    means = F.avg_pool2d(x, k)
    up = F.interpolate(means, scale_factor=k, mode="nearest")
    gy = torch.arange(H // k, device=x.device).view(-1, 1)
    gx = torch.arange(W // k, device=x.device).view(1, -1)
    checker = ((gy + gx) % 2).float()[None, None]
    checker = F.interpolate(checker, scale_factor=k, mode="nearest")
    return x * checker + up * (1 - checker), x * (1 - checker) + up * checker


def soft_block_pred(cur, ref, k, m=24, tau=1.0, chunk=None):
    """Soft block-matching prediction of `cur` from `ref` (Eq.1-3).
    cur, ref: [B,1,H,W] (H,W multiples of k). Returns pred [B,1,H,W].
    Search: all kxk patches of ref whose top-left lies within +-m/2 of the
    query block's top-left. tau = per-pixel MAE softmax temperature.
    Chunks are gradient-checkpointed (recomputed in backward) — otherwise the
    [B,k^2,L,(m+1)^2] gather tensor lives until backward and OOMs."""
    from torch.utils.checkpoint import checkpoint
    B, _, H, W = cur.shape
    r = m // 2
    if chunk is None:
        chunk = {4: 4096, 8: 1024}.get(k, 256)
    refp = F.pad(ref, (r, r, r, r), mode="replicate")
    # all stride-1 kxk patches of padded ref, fp16 (the K=16 fp32 version is
    # 2.2GB and its grad another 2.2GB -> OOM): [B,k*k,(H+m-k+1)*(W+m-k+1)]
    pat = F.unfold(refp, k, stride=1).half()
    pw = W + 2 * r - k + 1
    blocks = F.unfold(cur, k, stride=k).half()  # [B,k*k,L]
    L = blocks.shape[-1]
    nbx = W // k
    dy, dx = torch.meshgrid(torch.arange(m + 1, device=cur.device),
                            torch.arange(m + 1, device=cur.device),
                            indexing="ij")
    dgrid = (dy.reshape(-1) * pw + dx.reshape(-1))  # [(m+1)^2]

    def do_chunk(pat_, blocks_, idx):
        # fp16 internals: prediction error ~0.06/255, absorbed by the residual
        with torch.autocast("cuda", dtype=torch.float16):
            by, bx = idx // nbx, idx % nbx
            base = (by * k) * pw + bx * k  # query top-left in padded coords
            cand = base[:, None] + dgrid[None]  # [l,(m+1)^2]
            g = pat_[:, :, cand.reshape(-1)].reshape(B, k * k, len(idx), -1)
            q = blocks_[:, :, idx].unsqueeze(-1)  # [B,kk,l,1]
            eps = (g - q).abs().mean(1)  # [B,l,(m+1)^2] per-pixel MAE
            wgt = torch.softmax(-eps.float() / tau, dim=-1)
            out = torch.einsum("bkln,bln->bkl", g, wgt.half())
        return out.float()

    outs = []
    for c0 in range(0, L, chunk):
        idx = torch.arange(c0, min(c0 + chunk, L), device=cur.device)
        if pat.requires_grad or blocks.requires_grad:
            outs.append(checkpoint(do_chunk, pat, blocks, idx,
                                   use_reentrant=False))
        else:
            outs.append(do_chunk(pat, blocks, idx))
    return F.fold(torch.cat(outs, dim=2), (H, W), k, stride=k)


def intra_pred(p, k, m=24, tau=1.0):
    """Intra soft prediction with checkerboard DC self-masking."""
    m0, m1 = dc_checker_mask(p, k)
    B, _, H, W = p.shape
    gy = torch.arange(H // k, device=p.device).view(-1, 1)
    gx = torch.arange(W // k, device=p.device).view(1, -1)
    checker = ((gy + gx) % 2).float()[None, None]
    cmask = F.interpolate(checker, scale_factor=k, mode="nearest")
    # m0 has EVEN (checker==0) blocks DC-masked -> use it to predict THOSE
    pred0 = soft_block_pred(p, m0, k, m, tau)
    pred1 = soft_block_pred(p, m1, k, m, tau)  # m1 masks ODD blocks
    return pred0 * (1 - cmask) + pred1 * cmask


class VirtualCodec(torch.nn.Module):
    """residual -> DCT4 -> /Qstep -> EntropyBottleneck (noise+rate) -> IDCT4."""

    def __init__(self):
        super().__init__()
        from compressai.entropy_models import EntropyBottleneck
        self.eb = EntropyBottleneck(16)

    def forward(self, resid, qstep):
        y = dct4(resid) / qstep
        y_hat, lik = self.eb(y)
        rate_bpp = (-torch.log2(lik)).sum(dim=(1, 2, 3)) / \
            (resid.shape[-1] * resid.shape[-2])  # bits per residual pixel
        r_hat = idct4(y_hat * qstep)
        return r_hat, rate_bpp


if __name__ == "__main__":  # self-tests
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(2, 1, 64, 64, device=dev) * 255
    assert (idct4(dct4(x)) - x).abs().max() < 1e-3, "DCT4 roundtrip"
    p = torch.rand(1, 1, 128, 128, device=dev) * 255
    pr = intra_pred(p, k=8)
    res_e = (p - pr).abs().mean()
    base_e = (p - p.mean()).abs().mean()
    print(f"intra resid MAE {res_e:.2f} vs DC-only {base_e:.2f} (rand frame)")
    # natural-image-like: smooth gradient should predict well
    g = torch.linspace(0, 255, 128, device=dev).view(1, 1, 1, -1).expand(1, 1, 128, -1).contiguous()
    g = g + torch.linspace(0, 60, 128, device=dev).view(1, 1, -1, 1)
    pr = intra_pred(g, k=8)
    print(f"gradient-image intra resid MAE {(g - pr).abs().mean():.3f}")
    vc = VirtualCodec().to(dev)
    r = (x - 128)
    for qp in (10, 25, 40):
        rh, bpp = vc(r, qstep_of_qp(qp))
        print(f"QP{qp} qstep {qstep_of_qp(qp):.2f}: resid-recon MAE "
              f"{(rh-r).abs().mean():.2f} rate {bpp.mean():.2f} bpp")
    print("self-tests done")
