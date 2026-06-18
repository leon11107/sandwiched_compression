"""DPP eval v2 — corrected protocol (Phase 0 of the JPEG-intra re-baseline).

Fixes vs eval.py (the saturated/non-standard protocol):
  1. REAL JPEG anchor: PIL quality scale (Annex-K scaled tables), 4:2:0, full
     lossy decode for BOTH arms (no lossless-chroma restore, no flat qtable).
     This is the deployable claim: "preproc -> standard JPEG encoder".
  2. Operating range moved DOWN (quality points span VMAF_NEG ~55-92, the DPP
     paper's range) instead of qstep 16-64 (VMAF_NEG 97.8+, saturated).
  3. VMAF/VMAF_NEG measured after bicubic upscale to 1080p-equivalent area
     (paper protocol: "all lower resolutions are upscaled ... to 1080p prior
     to quality measurements"). Identical treatment for both arms.
  4. MS-SSIM (luma, native res) added — L_F trains 0.8*(1-MS-SSIM) but it was
     never evaluated. PSNR kept as the fidelity guard.
  5. BD-rate per checkpoint AND convex-hull envelope across checkpoints
     (paper computes BD over the hull of multiple-lambda models).

Images are center-cropped to multiples of 16 (4:2:0 MCU) so JPEG never pads
and MS-SSIM's 5-scale avg-pool sees even dims. torch-env; VMAF is the
from-source CLI (CPU subprocess).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import io
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
sys.path.insert(0, "/workspace/sandwiched_compression/experiments/m2_lowres_repro")
from dpp.model import DPPModel
from torch_port.preproc import tf_rgb_to_yuv
from torch_port.losses import ssim_multiscale_tf
from distortion import vmaf_metric as vm
from compute_dpp_bd_inline import bd_rate
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

TARGET_AREA = 1920 * 1080  # paper: upscale to 1080p before VMAF


def load16(path):
    """RGB float32, center-cropped to multiple-of-16 dims."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    w16, h16 = (w // 16) * 16, (h // 16) * 16
    l, t = (w - w16) // 2, (h - h16) // 2
    return np.asarray(im.crop((l, t, l + w16, t + h16)), np.float32)


def jpeg_rt(img, quality):
    """Real JPEG round-trip: standard scaled tables, 4:2:0. -> (decoded f32, bits)."""
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8)).save(
        buf, format="jpeg", quality=int(quality), subsampling="4:2:0", optimize=True)
    dec = np.asarray(Image.open(buf).convert("RGB"), np.float32)
    return dec, 8 * len(buf.getbuffer())


def up1080(img_u8):
    """Bicubic upscale to 1080p-equivalent AREA (even dims), identical for both arms."""
    h, w = img_u8.shape[:2]
    s = np.sqrt(TARGET_AREA / (w * h))
    if s <= 1.0:
        return img_u8
    nw, nh = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    return np.asarray(Image.fromarray(img_u8).resize((nw, nh), Image.BICUBIC))


def rgb_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


@torch.no_grad()
def msssim_luma(ref, dec, dev):
    """Standard MS-SSIM (filter 11, 5 scales) on luma, native res. -> float"""
    r = torch.from_numpy(ref[None]).float().to(dev)
    d = torch.from_numpy(dec[None]).float().to(dev)
    ry = tf_rgb_to_yuv(r)[..., 0:1]
    dy = tf_rgb_to_yuv(d)[..., 0:1]
    return float(ssim_multiscale_tf(ry, dy, max_val=255.0, filter_size=11)[0])


def eval_arm(imgs, pre_imgs, quality, dev):
    """One (arm, quality) condition over all images. pre_imgs: list of encoder
    inputs (baseline: originals; model: preprocessed). Metrics vs ORIGINALS."""
    bpp, psnr, msss, decs = [], [], [], []
    for orig, src in zip(imgs, pre_imgs):
        dec, bits = jpeg_rt(src, quality)
        bpp.append(bits / (orig.shape[0] * orig.shape[1]))
        psnr.append(rgb_psnr(dec, orig))
        msss.append(msssim_luma(orig, dec, dev))
        decs.append(np.rint(dec).astype(np.uint8))
    return bpp, psnr, msss, decs


def vmaf_cond(refs_up, decs):
    decs_up = [up1080(d) for d in decs]
    sc = vm.vmaf_scores(refs_up, decs_up)
    return ([s["vmaf"] for s in sc], [s["vmaf_neg"] for s in sc])


def pareto(points):
    """Pareto envelope: keep points not dominated by any other (<=bpp AND >=quality)."""
    pts = sorted(set(points))
    return [(b, q) for b, q in pts
            if not any(pb <= b and pq >= q and (pb, pq) != (b, q) for pb, pq in pts)]


def bd(base_rows, model_rows, key):
    r = bd_rate([x["bpp"] for x in base_rows], [x[key] for x in base_rows],
                [x["bpp"] for x in model_rows], [x[key] for x in model_rows])
    return float(r[0] if isinstance(r, (tuple, list)) else r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="name=ckpt_path pairs")
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val50")
    ap.add_argument("--qualities", default="5,8,12,20,32,50")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (calibration)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vmaf-workers", type=int, default=4)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = "cuda"
    Q = [int(x) for x in a.qualities.split(",")]
    paths = sorted(glob.glob(os.path.join(a.img_dir, "*.png")))
    if a.limit:
        paths = paths[: a.limit]
    imgs = [load16(p) for p in paths]
    refs_up = [up1080(np.rint(im).astype(np.uint8)) for im in imgs]
    print(f"{len(imgs)} imgs from {a.img_dir}; qualities={Q}", flush=True)

    arms = {"baseline": imgs}
    for spec in a.models:
        name, path = spec.split("=", 1)
        m = DPPModel(ch=64, codec_forward_mode="proxy", device=dev)
        ck = torch.load(path, map_location=dev)
        (m.preproc.load_state_dict(ck["preproc"]) if isinstance(ck, dict) and "preproc" in ck
         else m.load_state_dict(ck))
        m.eval()
        with torch.no_grad():
            pre = [np.clip(m.preproc(torch.from_numpy(im[None]).float().to(dev))[0]
                           .cpu().numpy(), 0, 255) for im in imgs]
        arms[name] = pre
        del m
        print(f"preprocessed: {name} (scaler n/a) <- {path}", flush=True)

    # JPEG round-trips + pixel metrics (GPU msssim) serial; VMAF calls threaded.
    results = {arm: [] for arm in arms}
    vmaf_jobs = []
    for arm, srcs in arms.items():
        for q in Q:
            bpp, psnr, msss, decs = eval_arm(imgs, srcs, q, dev)
            row = {"quality": q, "bpp": float(np.mean(bpp)), "rgb_psnr": float(np.mean(psnr)),
                   "msssim": float(np.mean(msss)),
                   "per_img": {"bpp": bpp, "rgb_psnr": psnr, "msssim": msss}}
            results[arm].append(row)
            vmaf_jobs.append((arm, row, decs))
            print(f"[{arm} q={q}] bpp={row['bpp']:.4f} psnr={row['rgb_psnr']:.2f} "
                  f"msssim={row['msssim']:.5f}", flush=True)

    # VMAF grouped PER IMAGE: y4m requires equal-size frames; batching mixed-size
    # images warps everything to frame-0 dims (bug found 2026-06-10 — invalidated
    # earlier cross-image-batched VMAF numbers; PSNR/MS-SSIM were never affected).
    for job in vmaf_jobs:
        arm, row, decs = job
        row["per_img"]["vmaf"] = [None] * len(imgs)
        row["per_img"]["vmaf_neg"] = [None] * len(imgs)

    def run_vmaf_img(i):
        dists = [up1080(job[2][i]) for job in vmaf_jobs]
        sc = vm.vmaf_scores([refs_up[i]] * len(dists), dists)
        for (arm, row, _), s in zip(vmaf_jobs, sc):
            row["per_img"]["vmaf"][i] = s["vmaf"]
            row["per_img"]["vmaf_neg"][i] = s["vmaf_neg"]
        print(f"[vmaf img {i+1}/{len(imgs)}] done", flush=True)
    with ThreadPoolExecutor(a.vmaf_workers) as ex:
        list(ex.map(run_vmaf_img, range(len(imgs))))
    for arm, row, _ in vmaf_jobs:
        row["vmaf"] = float(np.mean(row["per_img"]["vmaf"]))
        row["vmaf_neg"] = float(np.mean(row["per_img"]["vmaf_neg"]))
        print(f"[vmaf {arm} q={row['quality']}] vmaf={row['vmaf']:.2f} "
              f"vmaf_neg={row['vmaf_neg']:.2f} @ {row['bpp']:.4f}bpp", flush=True)

    base = results["baseline"]
    summary = {}
    for arm in arms:
        if arm == "baseline":
            continue
        rows = results[arm]
        summary[arm] = {
            "bd_psnr": bd(base, rows, "rgb_psnr"),
            "bd_msssim": bd(base, rows, "msssim"),
            "bd_vmaf": bd(base, rows, "vmaf"),
            "bd_vmaf_neg": bd(base, rows, "vmaf_neg"),
            "max_psnr_drop": max(b["rgb_psnr"] - m["rgb_psnr"] for b, m in zip(base, rows)),
        }
        s = summary[arm]
        print(f"=== {arm}: BD-rate%% (neg=win) PSNR={s['bd_psnr']:+.2f} "
              f"MSSSIM={s['bd_msssim']:+.2f} VMAF={s['bd_vmaf']:+.2f} "
              f"VMAF_NEG={s['bd_vmaf_neg']:+.2f} | maxPSNRdrop={s['max_psnr_drop']:+.2f}dB",
              flush=True)

    # convex-hull envelope across ALL model checkpoints (paper-style multi-lambda hull)
    if len(arms) > 2:
        env = {}
        for key in ["rgb_psnr", "msssim", "vmaf", "vmaf_neg"]:
            pts = [(r["bpp"], r[key]) for arm in arms if arm != "baseline"
                   for r in results[arm] if key in r]
            hull = pareto(pts)
            if len(hull) >= 2:
                hb, hq = zip(*hull)
                r = bd_rate([x["bpp"] for x in base], [x[key] for x in base], list(hb), list(hq))
                env[f"bd_{key}"] = float(r[0] if isinstance(r, (tuple, list)) else r)
        summary["ENVELOPE(all models)"] = env
        print(f"=== ENVELOPE(all): " + " ".join(f"{k}={v:+.2f}" for k, v in env.items()),
              flush=True)

    if a.out:
        json.dump({"qualities": Q, "results": results, "summary": summary},
                  open(a.out, "w"), indent=2)
        print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
