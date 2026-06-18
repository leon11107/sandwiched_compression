"""Final val50 evaluation of the image-adaptive s-predictor (line A) under the
corrected protocol (real JPEG 4:2:0 Annex-K, per-image VMAF after 1080p-area
upscale, qualities 5..32).

Arms:
  baseline : plain JPEG
  s_by_q   : FIXED mean oracle s (val50-run means; q<=8 -> s8 else s20) — the
             deployable reference that scored BD-VMAF_NEG -2.35 [-3.00,-1.78]
  adaptive : SPredictor (q<=8 -> predict(img,8) else predict(img,20))
  cascade  : C_l05 smoothing net THEN s_by_q pre-emphasis (--cascade-ckpt) —
             tests whether the MS-SSIM win and the VMAF_NEG win stack

Metrics per (arm, q, image): bpp, RGB-PSNR, MS-SSIM(luma), VMAF, VMAF_NEG.
Summary: BD-rate per metric (mean curves) + bootstrap CI over images
(VMAF_NEG, MS-SSIM) + best arm per 0.1bpp bin + RD plot.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
sys.path.insert(0, "/workspace/sandwiched_compression/experiments/m2_lowres_repro")
from distortion import vmaf_metric as vm
from dpp.oracle_vmafneg import apply_s, block_dct, jpeg_rt, load16, luma, up1080
from dpp.eval_v2 import msssim_luma, rgb_psnr
from dpp.bd_ci import bd, boot_bd
from dpp.spred import SPredictor

VAL_DIR = "/workspace/sandwiched_compression/dpp/data/val50"
ORACLE_JSON = "/workspace/sandwiched_compression/dpp/runs/oracle_vmafneg.json"
RUNS = "/workspace/sandwiched_compression/dpp/runs"
QUALITIES = [5, 8, 12, 20, 32]


def mean_oracle_s():
    d = json.load(open(ORACLE_JSON))
    return {q: np.array([r["s"] for r in d if r["q"] == q]).mean(0) for q in (8, 20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spred", default=os.path.join(RUNS, "spred_model.npz"))
    ap.add_argument("--out", default=os.path.join(RUNS, "eval_spred.json"))
    ap.add_argument("--vmaf-workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--cascade-ckpt",
                    default=os.path.join(RUNS, "v2_C_l05", "model.pt"),
                    help="smoothing-net ckpt for the cascade arm ('' disables)")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = "cuda"
    sp = SPredictor.load(a.spred)
    ms = mean_oracle_s()
    paths = sorted(glob.glob(os.path.join(VAL_DIR, "*.png")))
    imgs = [load16(p) for p in paths]
    N = len(imgs)
    print(f"{N} val imgs; qualities={QUALITIES}; spred={a.spred} "
          f"(kind={sp.d['kind']})", flush=True)

    # per-image precompute: luma DCT + per-arm s vectors per quality
    pre = []
    for im in imgs:
        y = luma(im); c = block_dct(y)
        s_ad = {8: sp.predict(im, 8), 20: sp.predict(im, 20)}
        pre.append((y, c, s_ad))

    arms = ["baseline", "s_by_q", "adaptive"]
    casc = []  # per-image (net_out, luma, dct) for the cascade arm
    if a.cascade_ckpt:
        from dpp.model import DPPModel
        m = DPPModel(ch=64, codec_forward_mode="proxy", device=dev)
        ck = torch.load(a.cascade_ckpt, map_location=dev)
        (m.preproc.load_state_dict(ck["preproc"])
         if isinstance(ck, dict) and "preproc" in ck else m.load_state_dict(ck))
        m.eval()
        with torch.no_grad():
            for im in imgs:
                out = np.clip(m.preproc(torch.from_numpy(im[None]).float()
                                        .to(dev))[0].cpu().numpy(), 0, 255)
                yc = luma(out)
                casc.append((out, yc, block_dct(yc)))
        del m
        arms.append("cascade")
        print(f"cascade net loaded <- {a.cascade_ckpt}", flush=True)
    res = {arm: {q: {"bpp": np.zeros(N), "rgb_psnr": np.zeros(N),
                     "msssim": np.zeros(N), "vmaf": np.zeros(N),
                     "vmaf_neg": np.zeros(N)} for q in QUALITIES} for arm in arms}

    def src_for(arm, i, q):
        if arm == "baseline":
            return imgs[i]
        if arm == "cascade":
            out, yc, cc = casc[i]
            return apply_s(out, cc, yc, ms[8] if q <= 8 else ms[20])
        y, c, s_ad = pre[i]
        s = (ms[8] if q <= 8 else ms[20]) if arm == "s_by_q" else s_ad[8 if q <= 8 else 20]
        return apply_s(imgs[i], c, y, s)

    # pixel metrics + decodes (serial; GPU msssim), then per-image VMAF batched
    decs = {}
    for arm in arms:
        for q in QUALITIES:
            r = res[arm][q]
            for i, im in enumerate(imgs):
                dec, bits = jpeg_rt(src_for(arm, i, q), q)
                r["bpp"][i] = bits / (im.shape[0] * im.shape[1])
                r["rgb_psnr"][i] = rgb_psnr(dec, im)
                r["msssim"][i] = msssim_luma(im, dec, dev)
                decs[(arm, q, i)] = np.rint(dec).astype(np.uint8)
            print(f"[{arm} q={q}] bpp={r['bpp'].mean():.4f} "
                  f"psnr={r['rgb_psnr'].mean():.2f} msssim={r['msssim'].mean():.5f}",
                  flush=True)

    conds = [(arm, q) for arm in arms for q in QUALITIES]

    def run_vmaf_img(i):
        ref = up1080(np.rint(imgs[i]).astype(np.uint8))
        dists = [up1080(decs[(arm, q, i)]) for arm, q in conds]
        sc = vm.vmaf_scores([ref] * len(dists), dists)
        for (arm, q), s in zip(conds, sc):
            res[arm][q]["vmaf"][i] = s["vmaf"]
            res[arm][q]["vmaf_neg"][i] = s["vmaf_neg"]
        print(f"[vmaf img {i+1}/{N}] done", flush=True)

    with ThreadPoolExecutor(a.vmaf_workers) as ex:
        list(ex.map(run_vmaf_img, range(N)))

    # ---- summary ----
    METRICS = ["rgb_psnr", "msssim", "vmaf", "vmaf_neg"]
    curves = {arm: {m: np.array([res[arm][q][m].mean() for q in QUALITIES])
                    for m in METRICS + ["bpp"]} for arm in arms}
    summary = {}
    for arm in arms[1:]:
        s = {f"bd_{m}": bd(curves["baseline"]["bpp"], curves["baseline"][m],
                           curves[arm]["bpp"], curves[arm][m]) for m in METRICS}
        # bootstrap CI over images for vmaf_neg + msssim
        for m in ("vmaf_neg", "msssim"):
            bb = np.stack([res["baseline"][q]["bpp"] for q in QUALITIES])
            qb = np.stack([res["baseline"][q][m] for q in QUALITIES])
            bm = np.stack([res[arm][q]["bpp"] for q in QUALITIES])
            qm = np.stack([res[arm][q][m] for q in QUALITIES])
            bs = boot_bd(bb, qb, bm, qm, a.boot, 42)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            s[f"bd_{m}_ci"] = [float(lo), float(hi)]
        s["max_psnr_drop"] = float(max(
            res["baseline"][q]["rgb_psnr"].mean() - res[arm][q]["rgb_psnr"].mean()
            for q in QUALITIES))
        summary[arm] = s
        print(f"=== {arm}: BD%% (neg=win) PSNR={s['bd_rgb_psnr']:+.2f} "
              f"MSSSIM={s['bd_msssim']:+.2f} CI{s['bd_msssim_ci']} "
              f"VMAF={s['bd_vmaf']:+.2f} VMAF_NEG={s['bd_vmaf_neg']:+.2f} "
              f"CI{s['bd_vmaf_neg_ci']} | maxPSNRdrop={s['max_psnr_drop']:.2f}dB",
              flush=True)

    # best arm per 0.1bpp bin (iso-bpp interpolated VMAF_NEG / MS-SSIM deltas)
    bins = {}
    bb0 = curves["baseline"]["bpp"]
    for lo_b in (0.1, 0.2, 0.3, 0.4):
        mid = lo_b + 0.05
        if mid < bb0.min() or mid > bb0.max():
            continue
        row = {}
        for m in ("vmaf_neg", "msssim"):
            base_v = float(np.interp(mid, bb0, curves["baseline"][m]))
            for arm in arms[1:]:
                o = np.argsort(curves[arm]["bpp"])
                row[f"{arm}.{m}"] = float(np.interp(
                    mid, curves[arm]["bpp"][o], curves[arm][m][o])) - base_v
        bins[f"[{lo_b:.1f},{lo_b+0.1:.1f})"] = row
    print("per-0.1bpp-bin deltas vs baseline:", flush=True)
    for b, row in bins.items():
        print(f"  {b} " + "  ".join(f"{k}={v:+.4f}" for k, v in row.items()), flush=True)

    json.dump({"qualities": QUALITIES, "summary": summary, "bins": bins,
               "curves": {arm: {m: curves[arm][m].tolist() for m in curves[arm]}
                          for arm in arms},
               "per_img": {arm: {str(q): {m: res[arm][q][m].tolist()
                                          for m in res[arm][q]}
                                 for q in QUALITIES} for arm in arms}},
              open(a.out, "w"), indent=2)
    print(f"saved {a.out}", flush=True)

    # RD plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, m in zip(axes, ("vmaf_neg", "msssim", "rgb_psnr")):
        for arm, sty in zip(arms, ("k-o", "b-s", "r-^", "g-d")):
            ax.plot(curves[arm]["bpp"], curves[arm][m], sty, label=arm, ms=4)
        ax.set_xlabel("bpp"); ax.set_ylabel(m); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("val50 corrected protocol: adaptive s-prediction vs fixed s vs baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "rd_spred.png"), dpi=130)
    print(f"saved {os.path.join(RUNS, 'rd_spred.png')}", flush=True)


if __name__ == "__main__":
    main()
