"""Definitive VMAF_NEG BD-rate with BOOTSTRAP confidence interval (the statistically
correct noise floor). BD-rate (equal-quality rate saving) is the RIGHT metric; the
per-qstep VMAF delta is NOT (it compares different rate points). We compute per-image
RD points over a wide NON-saturated qstep range, then bootstrap over images to get a
CI. A model has a REAL win only if its CI is comfortably below 0.

torch-env. Compares base vs N models; base encoded once.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
sys.path.insert(0, "/workspace/sandwiched_compression/experiments/m2_lowres_repro")
from dpp.model import DPPModel
from dpp.diag_floor import encode_set, restore_chroma, rgb_psnr  # reuse
from distortion import vmaf_metric as vm
from compute_dpp_bd_inline import bd_rate
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


def bd(rb, qb, rm, qm):
    try:
        r = bd_rate(list(rb), list(qb), list(rm), list(qm))
        v = float(r[0] if isinstance(r, (tuple, list)) else r)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def per_image_curves(imgs, qsteps, preproc, dev):
    """-> bpp[Q,N], vmaf[Q,N]."""
    Q, N = len(qsteps), len(imgs)
    bpp = np.zeros((Q, N)); vmn = np.zeros((Q, N))
    for qi, q in enumerate(qsteps):
        b, p, decs, refs = encode_set(imgs, q, preproc, dev)
        scores = vm.vmaf_scores(refs, decs)
        bpp[qi] = b; vmn[qi] = np.array([s["vmaf_neg"] for s in scores])
    return bpp, vmn


def boot_bd(bpp_b, vmn_b, bpp_m, vmn_m, B, seed):
    """bootstrap over images (axis=1). returns array of BD-rate VMAF_NEG."""
    Q, N = bpp_b.shape
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        rb = bpp_b[:, idx].mean(1); qb = vmn_b[:, idx].mean(1)
        rm = bpp_m[:, idx].mean(1); qm = vmn_m[:, idx].mean(1)
        out.append(bd(rb, qb, rm, qm))
    return np.array(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val50")
    ap.add_argument("--qsteps", default="32,48,64,96,128")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qsteps = [float(x) for x in a.qsteps.split(",")]
    imgs = [np.asarray(Image.open(p).convert("RGB"), np.float32)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))]
    print(f"{len(imgs)} imgs; qsteps={qsteps}; boot={a.boot}")

    bpp_b, vmn_b = per_image_curves(imgs, qsteps, None, dev)
    # noise-floor sanity: base vs base (split bootstrap) -> distribution centered 0
    floor = boot_bd(bpp_b, vmn_b, bpp_b, vmn_b, a.boot, 1)
    print(f"[base-vs-base] BD VMAF_NEG mean={np.nanmean(floor):+.2f}% (must be ~0)")

    for spec in a.models:
        name, path = spec.split("=", 1)
        m = DPPModel(ch=a.ch, codec_forward_mode="proxy", device=dev)
        ck = torch.load(path, map_location=dev)
        (m.preproc.load_state_dict(ck["preproc"]) if isinstance(ck, dict) and "preproc" in ck
         else m.load_state_dict(ck))
        m.eval()
        bpp_m, vmn_m = per_image_curves(imgs, qsteps, m.preproc, dev)
        pt = bd(bpp_b.mean(1), vmn_b.mean(1), bpp_m.mean(1), vmn_m.mean(1))
        bs = boot_bd(bpp_b, vmn_b, bpp_m, vmn_m, a.boot, 42)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        nanfrac = np.mean(~np.isfinite(bs))
        win = "REAL WIN" if hi < 0 else ("real loss" if lo > 0 else "within noise")
        print(f"[{name:14s}] BD VMAF_NEG point={pt:+.2f}%  boot mean={np.nanmean(bs):+.2f}%  "
              f"95%CI=[{lo:+.2f},{hi:+.2f}]  nan={nanfrac:.0%}  -> {win}")


if __name__ == "__main__":
    main()
