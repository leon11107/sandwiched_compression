"""Foundational eval-methodology diagnostic (run BEFORE any more training).

Establishes whether the VMAF_NEG BD-rate metric is even usable in our regime:
 (1) BD-rate noise floor: base-vs-base must be ~0; per-image BD spread = noise band.
 (2) Quality regime: full RD curve over a WIDE qstep range -> is base VMAF_NEG
     saturated (no headroom) at our qsteps? where is there real separation room?
 (3) A trained model's full-range RD curve vs base.

No training. torch-env. Deterministic.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
sys.path.insert(0, "/workspace/sandwiched_compression/experiments/m2_lowres_repro")
from dpp.model import DPPModel
from torch_port.codec import encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
from distortion import vmaf_metric as vm
from compute_dpp_bd_inline import bd_rate
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


def restore_chroma(dec, orig):
    d = tf_rgb_to_yuv(torch.from_numpy(dec[None]).float())
    o = tf_rgb_to_yuv(torch.from_numpy(orig[None]).float())
    return np.clip(tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))[0].numpy(), 0, 255)


def rgb_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


def bd(rb, qb, rm, qm):
    r = bd_rate(rb, qb, rm, qm)
    return float(r[0] if isinstance(r, (tuple, list)) else r)


def encode_set(imgs, q, preproc=None, dev="cpu"):
    """Return per-image (bpp, psnr, decoded_uint8, ref_uint8) at qstep q."""
    bpp, psnr, decs, refs = [], [], [], []
    for im in imgs:
        src = im
        if preproc is not None:
            with torch.no_grad():
                src = np.clip(preproc(torch.from_numpy(im[None]).float().to(dev))[0].cpu().numpy(), 0, 255)
        d, b = encode_decode_with_jpeg(src[None], q, False, False)
        d = restore_chroma(d[0], im)
        bpp.append(b[0] / (im.shape[0] * im.shape[1])); psnr.append(rgb_psnr(d, im))
        decs.append(np.rint(d).astype(np.uint8)); refs.append(np.rint(im).astype(np.uint8))
    return np.array(bpp), np.array(psnr), decs, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val")
    ap.add_argument("--qsteps", default="16,24,32,48,64,96,128,160")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qsteps = [float(x) for x in a.qsteps.split(",")]
    imgs = [np.asarray(Image.open(p).convert("RGB"), np.float32)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))]
    print(f"{len(imgs)} val imgs; qsteps={qsteps}")

    preproc = None
    if a.model:
        m = DPPModel(ch=a.ch, codec_forward_mode="proxy", device=dev)
        ckpt = torch.load(a.model, map_location=dev)
        m.preproc.load_state_dict(ckpt["preproc"] if "preproc" in ckpt else ckpt) if isinstance(ckpt, dict) and "preproc" in ckpt else m.load_state_dict(ckpt)
        m.eval(); preproc = m.preproc
        print(f"loaded model {a.model}")

    # ---- baseline RD over wide range (per-image arrays kept) ----
    print("\n== BASELINE RD (wide qstep range) ==")
    base = {}  # q -> (bpp[N], psnr[N], vmafN[N])
    for q in qsteps:
        bpp, psnr, decs, refs = encode_set(imgs, q, None, dev)
        vmn = np.array([s["vmaf_neg"] for s in vm.vmaf_scores(refs, decs)])
        base[q] = (bpp, psnr, vmn)
        print(f"q={q:6g}  bpp={bpp.mean():.3f}  psnr={psnr.mean():.2f}  "
              f"vmafN={vmn.mean():.2f} (min {vmn.min():.1f}, max {vmn.max():.1f}, std {vmn.std():.2f})")

    # ---- (1) BD-rate noise floor ----
    print("\n== NOISE FLOOR ==")
    mb = [base[q][0].mean() for q in qsteps]
    pv = [base[q][2].mean() for q in qsteps]
    print(f"base-vs-base BD (sanity, must be ~0): VMAF_NEG={bd(mb, pv, mb, pv):+.3f}%")
    # per-image leave-one-out spread of the baseline curve's own bpp/vmaf (curve stability)
    N = len(imgs)
    loo = []
    for i in range(N):
        idx = [j for j in range(N) if j != i]
        b2 = [base[q][0][idx].mean() for q in qsteps]
        v2 = [base[q][2][idx].mean() for q in qsteps]
        loo.append(bd(mb, pv, b2, v2))
    loo = np.array(loo)
    print(f"leave-one-out BD spread (curve sampling noise): mean={loo.mean():+.2f}%  std={loo.std():.2f}%  range=[{loo.min():+.2f},{loo.max():+.2f}]")

    # ---- (3) model RD + BD over wide range ----
    if preproc is not None:
        print("\n== MODEL RD (wide qstep range) ==")
        mod = {}
        for q in qsteps:
            bpp, psnr, decs, refs = encode_set(imgs, q, preproc, dev)
            vmn = np.array([s["vmaf_neg"] for s in vm.vmaf_scores(refs, decs)])
            mod[q] = (bpp, psnr, vmn)
            db = base[q][2].mean()
            print(f"q={q:6g}  bpp={bpp.mean():.3f}  psnr={psnr.mean():.2f}  "
                  f"vmafN={vmn.mean():.2f}  (base vmafN {db:.2f}, delta {vmn.mean()-db:+.2f})")
        mmb = [mod[q][0].mean() for q in qsteps]
        mmv = [mod[q][2].mean() for q in qsteps]
        mmp = [mod[q][1].mean() for q in qsteps]
        print(f"\nFULL-RANGE BD: PSNR={bd(mb, [base[q][1].mean() for q in qsteps], mmb, mmp):+.2f}%  "
              f"VMAF_NEG={bd(mb, pv, mmb, mmv):+.2f}%")


if __name__ == "__main__":
    main()
