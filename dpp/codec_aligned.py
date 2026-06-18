"""Aligned JPEG codec — training codec == eval_v2 deployment codec (AUDIT fix #1).

Forward  = REAL PIL JPEG: Annex-K scaled tables via `quality`, 4:2:0, optimize=True
           (byte-identical to dpp/eval_v2.py jpeg_rt).
Backward = differentiable luma proxy: 8x8 DCT + PER-SUBBAND qvec (the actual luma
           quantization table PIL uses at that quality, natural raster order, read
           from the encoded stream itself) + noise-injection (or round) quant + iDCT,
           STE'd to the real decoded luma. Chroma comes from the real decode,
           detached (preproc is luma-only -> no gradient is lost).

Rate     = real bits (PIL byte count) as the VALUE; the differentiable rate proxy is
           divisively-normalized coeffs c/qvec fed to a (pretrained, frozen)
           FactorizedEntropy — exposed via `luma_coeffs_norm()` for the trainer.

Replaces flat-qtable 4:4:4 EncodeDecodeIntraTorch for DPP Phase 1. torch-env.
Self-test: python dpp/codec_aligned.py  (forward equivalence, proxy-real luma
agreement, gradient flow).
"""
from __future__ import annotations
import io
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.codec import JpegProxyTorch, make_noise_injection_round

_TABLE_CACHE: dict[int, np.ndarray] = {}


def pil_luma_table(quality: int) -> np.ndarray:
    """Exact luma quant table PIL/libjpeg uses at `quality` (natural raster, [64])."""
    q = int(quality)
    if q not in _TABLE_CACHE:
        buf = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buf, format="jpeg", quality=q,
                                        subsampling="4:2:0")
        _TABLE_CACHE[q] = np.array(Image.open(buf).quantization[0], np.float32)
    return _TABLE_CACHE[q]


def jpeg_rt_batch(x: np.ndarray, quality: int):
    """eval_v2-identical real JPEG round-trip. x [B,H,W,3] f32 0..255 ->
    (decoded [B,H,W,3] f32, bits [B])."""
    dec = np.zeros_like(x)
    bits = np.zeros(x.shape[0], np.float32)
    for i in range(x.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(np.rint(np.clip(x[i], 0, 255)).astype(np.uint8)).save(
            buf, format="jpeg", quality=int(quality), subsampling="4:2:0",
            optimize=True)
        dec[i] = np.asarray(Image.open(buf).convert("RGB"), np.float32)
        bits[i] = 8 * len(buf.getbuffer())
    return dec, bits


class AlignedJpegCodec(nn.Module):
    """[B,H,W,3] channels-last 0..255 -> (decoded same shape, real_bits [B]).
    decoded luma is STE (real forward value, proxy backward); chroma real, detached."""

    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.jp = JpegProxyTorch(convert_to_yuv=True, clip_to_image_max=True,
                                 device=device, dtype=dtype)
        self.device_, self.dtype_ = device, dtype

    def qvec(self, quality: int) -> torch.Tensor:
        """per-subband luma steps [1,1,1,64] (natural raster, matches _forward_dct_2d)."""
        return torch.from_numpy(pil_luma_table(quality)).to(
            self.device_, self.dtype_).view(1, 1, 1, 64)

    def _round_fn(self, mode, generator):
        if mode == "noise":
            return make_noise_injection_round(generator)
        return lambda t: t + (torch.round(t) - t).detach()  # ste round

    def luma_coeffs_norm(self, x, quality: int, generator=None, add_noise=True):
        """Divisively-normalized luma DCT coeffs c/qvec (+U(-.5,.5) noise) for the
        frozen entropy rate term. -> [B,64,h,w]"""
        y = self.jp._rgb_to_yuv(x)[..., 0:1]
        c = self.jp._forward_dct_2d(y) / self.qvec(quality)
        if add_noise:
            c = c + (torch.rand(c.shape, device=c.device, generator=generator) - 0.5)
        return c.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, quality: int, generator=None, quant_mode="noise"):
        # integer-pixel emulation on the proxy path (same role as torch_port codec)
        rf = self._round_fn(quant_mode, generator)
        x_int = rf(x)
        # --- proxy luma (differentiable) ---
        yuv = self.jp._rgb_to_yuv(x_int)
        y = yuv[..., 0:1]
        c = self.jp._forward_dct_2d(y)
        qv = self.qvec(quality)
        cq = rf(c / qv) * qv
        y_proxy = self.jp._inverse_dct_2d(cq)
        y_proxy = torch.clamp(y_proxy, 0.0, 255.0)
        # --- real codec (value + bits) ---
        dec_np, bits_np = jpeg_rt_batch(x.detach().cpu().numpy(), quality)
        dec_real = torch.from_numpy(dec_np).to(x.device, x.dtype)
        yuv_real = self.jp._rgb_to_yuv(dec_real)
        # --- STE luma + real chroma ---
        y_ste = y_proxy + (yuv_real[..., 0:1] - y_proxy).detach()
        dec = self.jp._yuv_to_rgb(torch.cat([y_ste, yuv_real[..., 1:3].detach()], -1))
        dec = torch.clamp(dec, 0.0, 255.0)
        bits = torch.from_numpy(bits_np).to(x.device, x.dtype)
        return dec, bits


def _psnr(a, b):
    m = float(((a.double() - b.double()) ** 2).mean())
    return 10 * np.log10(255.0 ** 2 / m) if m > 0 else 99.0


def main():
    """Self-test / validation."""
    import glob
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cod = AlignedJpegCodec(device=dev)
    paths = sorted(glob.glob("/workspace/sandwiched_compression/dpp/data/val50/*.png"))[:6]
    sys.path.insert(0, "/workspace/sandwiched_compression/dpp")
    from eval_v2 import load16, jpeg_rt as eval_jpeg_rt

    imgs = [load16(p) for p in paths]
    print("== (a) real-forward equivalence vs eval_v2.jpeg_rt (must be exact) ==")
    for q in [5, 12, 32]:
        d_ev, b_ev = eval_jpeg_rt(imgs[0], q)
        d_us, b_us = jpeg_rt_batch(imgs[0][None], q)
        print(f"  q={q:3d} bytes equal={int(b_ev)==int(b_us[0])} "
              f"maxpix={np.abs(d_ev - d_us[0]).max():.1f}")

    print("== (b) proxy-luma vs real-luma decode agreement (round mode) ==")
    for q in [5, 8, 12, 20, 32]:
        ag, agn = [], []
        for im in imgs:
            x = torch.from_numpy(im[None]).to(dev)
            with torch.no_grad():
                # proxy luma only (no STE) vs real luma
                y = cod.jp._rgb_to_yuv(x)[..., 0:1]
                c = cod.jp._forward_dct_2d(y)
                qv = cod.qvec(q)
                yp = torch.clamp(cod.jp._inverse_dct_2d(torch.round(c / qv) * qv), 0, 255)
                dec_np, _ = jpeg_rt_batch(im[None], q)
                yr = cod.jp._rgb_to_yuv(torch.from_numpy(dec_np).to(dev))[..., 0:1]
            ag.append(_psnr(yp, yr))
            # also: noise-mode proxy (training backward path) vs real
            gen = torch.Generator(device=dev); gen.manual_seed(0)
            with torch.no_grad():
                dec, _ = cod(x, q, generator=gen, quant_mode="noise")
            agn.append(_psnr(cod.jp._rgb_to_yuv(dec)[..., 0:1], yr))
        print(f"  q={q:3d} PSNR(proxy_round_Y, real_Y)={np.mean(ag):.2f}dB "
              f"| STE output Y == real Y: {np.mean(agn):.1f}dB (inf=exact)")

    print("== (c) gradient flow ==")
    x = torch.from_numpy(imgs[0][None]).to(dev).requires_grad_(True)
    gen = torch.Generator(device=dev); gen.manual_seed(1)
    dec, bits = cod(x, 12, generator=gen)
    loss = ((dec - x.detach()) ** 2).mean()
    g = torch.autograd.grad(loss, x)[0]
    print(f"  d(dist)/dx: finite={bool(torch.isfinite(g).all())} norm={float(g.norm()):.4e}")
    from dpp.entropy import FactorizedEntropy
    eb = FactorizedEntropy(64).to(dev)
    x2 = torch.from_numpy(imgs[0][None]).to(dev).requires_grad_(True)
    cq = cod.luma_coeffs_norm(x2, 12, generator=gen)
    g2 = torch.autograd.grad(eb.bits(cq).sum(), x2)[0]
    print(f"  d(bits)/dx: finite={bool(torch.isfinite(g2).all())} norm={float(g2.norm()):.4e}")
    print(f"  bits value positive: {bool((bits > 0).all())}")


if __name__ == "__main__":
    main()
