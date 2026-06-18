"""Pretrain + freeze the Ballé factorized entropy prior on NATURAL-image DCT
sub-band statistics. This makes the rate term an HONEST, fixed estimator of
real-JPEG compressibility: the preproc must genuinely reduce DCT-coefficient
entropy (simplify) to lower it, instead of co-adapting a learnable entropy head
to absorb whatever (rate-costly) detail it adds. Fit = minimize bits of
noise-quantized coeffs/q over a qstep range; freeze; save state_dict.

torch-env. NO TF import.
"""
import argparse, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp.entropy import FactorizedEntropy
from dpp.train import CropDataset
from torch_port.codec import JpegProxyTorch


def coeffs_over_q(jp, x, q, dev, gen):
    """x:[B,H,W,3] in [0,255] -> noise-quantized luma DCT coeffs/q, [B,64,h,w]."""
    y = jp._rgb_to_yuv(x)[..., 0:1]
    c = jp._forward_dct_2d(y)                       # [B,h,w,64]
    cq = c / q + (torch.rand(c.shape, device=dev, generator=gen) - 0.5)
    return cq.permute(0, 3, 1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/sandwiched_compression/dpp/runs/prior/entropy.pt")
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/train_big")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qstep-lo", type=float, default=16.0); ap.add_argument("--qstep-hi", type=float, default=128.0)
    a = ap.parse_args()
    assert torch.cuda.is_available(); dev = "cuda"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    qrng = np.random.default_rng(7)
    jp = JpegProxyTorch(convert_to_yuv=True, clip_to_image_max=True, device=dev)
    eb = FactorizedEntropy(64).to(dev)
    eb.eval()   # noise is added MANUALLY (coeffs_over_q); keep internal-noise OFF so
                # pretrain and frozen-train see identical single U(-0.5,0.5). eval() does
                # NOT freeze params (Adam still updates them) — only disables internal noise.
    opt = torch.optim.Adam(eb.parameters(), lr=a.lr)
    ds = CropDataset(a.img_dir, a.crop, a.steps * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)

    t0 = time.time(); run = 0.0; n = 0; npix = a.crop * a.crop
    for it, batch in enumerate(dl):
        x = batch.float().to(dev)
        q = float(qrng.uniform(a.qstep_lo, a.qstep_hi))
        gen = torch.Generator(device=dev); gen.manual_seed(int(qrng.integers(1 << 30)))
        cq = coeffs_over_q(jp, x, q, dev, gen)
        bpp = (eb.bits(cq) / npix).mean()
        opt.zero_grad(); bpp.backward(); opt.step()
        run += float(bpp); n += 1
        if (it + 1) % 500 == 0:
            print(f"step {it+1}/{a.steps}  bpp={run/n:.4f}  ({time.time()-t0:.0f}s)", flush=True); run = 0.0; n = 0
        if it + 1 >= a.steps: break

    # validation: fixed-q sanity — natural < pre-emphasized(sharpened) should cost more
    eb.eval()
    with torch.no_grad():
        xb = next(iter(dl)).float().to(dev)
        gen = torch.Generator(device=dev); gen.manual_seed(123)
        for q in (32.0, 64.0, 96.0):
            cq = coeffs_over_q(jp, xb, q, dev, gen)
            nat = float((eb.bits(cq) / npix).mean())
            # sharpened: amplify high-freq -> should cost MORE bits under natural prior
            y = jp._rgb_to_yuv(xb)[..., 0:1]; c = jp._forward_dct_2d(y)
            sharp = c.clone(); sharp[..., 1:] *= 1.5
            cqs = (sharp / q + (torch.rand(c.shape, device=dev, generator=gen) - 0.5)).permute(0, 3, 1, 2).contiguous()
            shp = float((eb.bits(cqs) / npix).mean())
            print(f"[val q={q:g}] natural bpp={nat:.4f}  sharpened(HFx1.5) bpp={shp:.4f}  "
                  f"({'OK penalizes HF' if shp > nat else 'WARN: no HF penalty'})")
    torch.save({"entropy": eb.state_dict(), "qlo": a.qstep_lo, "qhi": a.qstep_hi}, a.out)
    print(f"saved frozen prior -> {a.out}")


if __name__ == "__main__":
    main()
