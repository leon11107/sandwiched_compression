"""DPP eval: load a trained DPP model, compute RD curve (baseline vs preprocessed)
over qsteps with real JPEG + codec_luma_only restore, metrics = rgb_psnr, bpp,
VMAF_NEG; report BD-rate. torch-env (vmaf_metric is TF-free subprocess)."""
from __future__ import annotations
import argparse, glob, json, os, sys
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


def bd(rate_b, q_b, rate_m, q_m):
    r = bd_rate(rate_b, q_b, rate_m, q_m)
    return float(r[0] if isinstance(r, (tuple, list)) else r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val")
    ap.add_argument("--qsteps", default="16,24,32,48,64")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qsteps = [float(x) for x in a.qsteps.split(",")]
    imgs = [np.asarray(Image.open(p).convert("RGB"), np.float32)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))]
    m = DPPModel(ch=a.ch, codec_forward_mode="proxy", device=dev)
    ckpt = torch.load(a.model, map_location=dev)
    if isinstance(ckpt, dict) and "preproc" in ckpt:   # balle checkpoint {preproc, entropy}
        m.preproc.load_state_dict(ckpt["preproc"])
    else:
        m.load_state_dict(ckpt)
    m.eval()
    print(f"loaded {a.model}; {len(imgs)} eval imgs; scaler={float(m.preproc.scaler):.4f}")

    rows = {"baseline": [], "model": []}
    for q in qsteps:
        bpp_b, ps_b, bpp_m, ps_m = [], [], [], []
        dec_b_all, dec_m_all, refs = [], [], []
        for im in imgs:
            db, bb = encode_decode_with_jpeg(im[None], q, False, False)
            db = restore_chroma(db[0], im)
            bpp_b.append(bb[0] / (im.shape[0] * im.shape[1])); ps_b.append(rgb_psnr(db, im))
            with torch.no_grad():
                pre = np.clip(m.preproc(torch.from_numpy(im[None]).float().to(dev))[0].cpu().numpy(), 0, 255)
            dm, bm = encode_decode_with_jpeg(pre[None], q, False, False)
            dm = restore_chroma(dm[0], im)
            bpp_m.append(bm[0] / (im.shape[0] * im.shape[1])); ps_m.append(rgb_psnr(dm, im))
            refs.append(np.rint(im).astype(np.uint8))
            dec_b_all.append(np.rint(db).astype(np.uint8)); dec_m_all.append(np.rint(dm).astype(np.uint8))
        vb = vm.vmaf_scores(refs, dec_b_all); vmn_b = float(np.mean([s["vmaf_neg"] for s in vb]))
        vmm = vm.vmaf_scores(refs, dec_m_all); vmn_m = float(np.mean([s["vmaf_neg"] for s in vmm]))
        rows["baseline"].append({"qstep": q, "bpp": float(np.mean(bpp_b)), "rgb_psnr": float(np.mean(ps_b)), "vmaf_neg": vmn_b})
        rows["model"].append({"qstep": q, "bpp": float(np.mean(bpp_m)), "rgb_psnr": float(np.mean(ps_m)), "vmaf_neg": vmn_m})
        print(f"q={q:g} base bpp={np.mean(bpp_b):.3f} psnr={np.mean(ps_b):.2f} vmafN={vmn_b:.2f} | "
              f"model bpp={np.mean(bpp_m):.3f} psnr={np.mean(ps_m):.2f} vmafN={vmn_m:.2f}")
    bdr_p = bd([r["bpp"] for r in rows["baseline"]], [r["rgb_psnr"] for r in rows["baseline"]],
               [r["bpp"] for r in rows["model"]], [r["rgb_psnr"] for r in rows["model"]])
    bdr_v = bd([r["bpp"] for r in rows["baseline"]], [r["vmaf_neg"] for r in rows["baseline"]],
               [r["bpp"] for r in rows["model"]], [r["vmaf_neg"] for r in rows["model"]])
    drop = max(b["rgb_psnr"] - mm["rgb_psnr"] for b, mm in zip(rows["baseline"], rows["model"]))
    print(f"=== BD-rate (neg=win): PSNR={bdr_p:+.2f}%  VMAF_NEG={bdr_v:+.2f}%  | max PSNR drop={drop:+.2f}dB")
    if a.out:
        json.dump({"rows": rows, "bd_psnr": bdr_p, "bd_vmaf_neg": bdr_v, "psnr_drop": drop},
                  open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
