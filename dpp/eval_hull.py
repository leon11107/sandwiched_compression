"""Convex-hull ("dynamic optimizer") evaluation on val50, DPO-paper style
(reference/DPP/115100C.pdf S4.2) but restricted by default to DEPLOYABLE
freedoms at FIXED resolution: per-image, per-rate selection among
{baseline, C_l05, C_l075, C_l10, s_by_q, spred} x a dense JPEG quality ladder.
Encoder-side selection by a full-reference score is deployable (the encoder
has the source), so this is a legitimate single system, not metric peeking.

Selection score: S_alpha = alpha * VMAF_NEG + (1 - alpha) * 100 * MS-SSIM,
swept over alpha to trace the achievable (BD-VMAF_NEG, BD-MSSSIM) frontier.
Goal check: any alpha with BD-VMAF_NEG <= -5 AND BD-MSSSIM <= -2 (the AND bar).

--scales adds the DPO-style resolution ladder (Lanczos down -> preproc ->
JPEG -> bicubic back to source resolution; bits normalized by SOURCE pixels).
That arm is a labeled REFERENCE only (user: lowering resolution is protocol
advantage, not acceptable in deployment).

BD methodology matches the paper: per-image BD-rate of the selection hull vs
the baseline-only ladder, averaged over images, bootstrap CI over images.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
sys.path.insert(0, "/workspace/sandwiched_compression/experiments/m2_lowres_repro")
from distortion import vmaf_metric as vm
from dpp.oracle_vmafneg import apply_s, block_dct, jpeg_rt, load16, luma, up1080
from dpp.eval_v2 import msssim_luma, rgb_psnr
from dpp.bd_ci import bd
from dpp.spred import SPredictor

VAL_DIR = "/workspace/sandwiched_compression/dpp/data/val50"
RUNS = "/workspace/sandwiched_compression/dpp/runs"
ORACLE_JSON = os.path.join(RUNS, "oracle_vmafneg.json")
QUALITIES = [3, 5, 8, 12, 16, 20, 26, 32, 40]
NET_SYS = {"C_l05": "v2_C_l05", "C_l075": "v2_C_l075", "C_l10": "v2_C_l10"}
GPU_LOCK = threading.Lock()


def mean_oracle_s():
    d = json.load(open(ORACLE_JSON))
    return {q: np.array([r["s"] for r in d if r["q"] == q]).mean(0) for q in (8, 20)}


def resize(img, w, h, method):
    return np.asarray(Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8))
                      .resize((w, h), method), np.float32)


def pareto_hull(bpp, score):
    """Indices of the monotone Pareto front, then convex-hull pruned
    (paper: 'monotonically-increasing points in the convex hull')."""
    o = np.argsort(bpp)
    front = []
    best = -np.inf
    for i in o:
        if score[i] > best + 1e-12:
            front.append(i); best = score[i]
    if len(front) <= 2:
        return front
    h = []
    for i in front:  # upper concave envelope in (bpp, score)
        while len(h) >= 2:
            (x1, y1), (x2, y2) = (bpp[h[-2]], score[h[-2]]), (bpp[h[-1]], score[h[-1]])
            if (x2 - x1) * (score[i] - y1) - (y2 - y1) * (bpp[i] - x1) >= 0:
                h.pop()
            else:
                break
        h.append(i)
    return h


def run_img(args):
    i, path, net_outs, sp, ms, scales, dev = args
    im = load16(path)
    H0, W0 = im.shape[:2]
    rows = []  # (sys, scale, q, bpp, msssim, psnr) + decs for vmaf
    decs, keys = [], []
    for sc in scales:
        h, w = (H0, W0) if sc == 1.0 else (int(round(H0 * sc)) // 2 * 2,
                                           int(round(W0 * sc)) // 2 * 2)
        src_b = im if sc == 1.0 else resize(im, w, h, Image.LANCZOS)
        # per-system source at this scale (band systems depend on q-bucket)
        srcs = {"baseline": {None: src_b}}
        for name in NET_SYS:
            srcs[name] = {None: net_outs[(name, sc)]}
        y = luma(src_b); c = block_dct(y)
        for qb in (8, 20):
            srcs.setdefault("s_by_q", {})[qb] = apply_s(src_b, c, y, ms[qb])
            srcs.setdefault("spred", {})[qb] = apply_s(src_b, c, y, sp.predict(src_b, qb))
        # mild-smoothing blends of C_l05 (full-strength C_l05 is NEG +52: too
        # aggressive; DPO-style win needs VMAF-neutral mild smoothing) and the
        # mild-smooth + band pre-emphasis combo (weak cascade)
        for b in (0.3, 0.5):
            mixed = b * net_outs[("C_l05", sc)] + (1 - b) * src_b
            srcs[f"C_l05_b{int(b*100)}"] = {None: mixed}
            ym = luma(mixed); cm = block_dct(ym)
            srcs[f"C_l05_b{int(b*100)}_sband"] = {
                qb: apply_s(mixed, cm, ym, sp.predict(mixed, qb)) for qb in (8, 20)}
        for sys_name, byqb in srcs.items():
            for q in QUALITIES:
                src = byqb[None] if None in byqb else byqb[8 if q <= 8 else 20]
                dec, bits = jpeg_rt(src, q)
                if sc != 1.0:
                    dec = resize(dec, W0, H0, Image.BICUBIC)
                with GPU_LOCK:
                    mss = msssim_luma(im, dec, dev)
                rows.append({"sys": sys_name, "scale": sc, "q": q,
                             "bpp": bits / (H0 * W0), "msssim": float(mss),
                             "psnr": rgb_psnr(dec, im)})
                decs.append(up1080(np.rint(dec).astype(np.uint8)))
                keys.append(len(rows) - 1)
    ref_up = up1080(np.rint(im).astype(np.uint8))
    scores = vm.vmaf_scores([ref_up] * len(decs), decs)
    for k, s in zip(keys, scores):
        rows[k]["vmaf"] = float(s["vmaf"]); rows[k]["vmaf_neg"] = float(s["vmaf_neg"])
    print(f"[img {i}] {len(rows)} candidates done", flush=True)
    return i, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="1", help="comma list, e.g. 1,0.6667,0.5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alphas", default="0,0.25,0.4,0.5,0.6,0.75,1")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(RUNS, "eval_hull.json"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = "cuda"
    scales = [float(x) for x in a.scales.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    paths = sorted(glob.glob(os.path.join(VAL_DIR, "*.png")))
    if a.limit:
        paths = paths[: a.limit]
    sp = SPredictor.load(os.path.join(RUNS, "spred_model.npz"))
    ms = mean_oracle_s()

    # precompute net preproc outputs (GPU) per (system, image, scale)
    from dpp.model import DPPModel
    net_outs_all = {}
    for name, run in NET_SYS.items():
        m = DPPModel(ch=64, codec_forward_mode="proxy", device=dev)
        ck = torch.load(os.path.join(RUNS, run, "model.pt"), map_location=dev)
        (m.preproc.load_state_dict(ck["preproc"])
         if isinstance(ck, dict) and "preproc" in ck else m.load_state_dict(ck))
        m.eval()
        with torch.no_grad():
            for pi, p in enumerate(paths):
                im = load16(p)
                H0, W0 = im.shape[:2]
                for sc in scales:
                    src = im if sc == 1.0 else resize(
                        im, int(round(W0 * sc)) // 2 * 2,
                        int(round(H0 * sc)) // 2 * 2, Image.LANCZOS)
                    out = np.clip(m.preproc(torch.from_numpy(src[None]).float()
                                            .to(dev))[0].cpu().numpy(), 0, 255)
                    net_outs_all[(name, pi, sc)] = out.astype(np.float32)
        del m
        print(f"net {name} preproc outputs cached", flush=True)

    jobs = [(i, p, {(n, sc): net_outs_all[(n, i, sc)] for n in NET_SYS
                    for sc in scales}, sp, ms, scales, dev)
            for i, p in enumerate(paths)]
    allrows = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, rows in ex.map(run_img, jobs):
            allrows[i] = rows
    N = len(paths)

    # per-image BDs for a candidate-subset + alpha selection
    def per_image_bd(sel_fn):
        bd_neg, bd_mss = [], []
        for i in range(N):
            rows = allrows[i]
            base = [r for r in rows if r["sys"] == "baseline" and r["scale"] == 1.0]
            bb = np.array([r["bpp"] for r in base])
            o = np.argsort(bb)
            bneg = np.array([r["vmaf_neg"] for r in base])
            bmss = np.array([r["msssim"] for r in base])
            cand = sel_fn(rows)
            cb = np.array([r["bpp"] for r in cand])
            cneg = np.array([r["vmaf_neg"] for r in cand])
            cmss = np.array([r["msssim"] for r in cand])
            bd_neg.append(bd(bb[o], bneg[o], cb, cneg))
            bd_mss.append(bd(bb[o], bmss[o], cb, cmss))
        return np.array(bd_neg, float), np.array(bd_mss, float)

    def boot_mean(x, B, seed):
        x = x[np.isfinite(x)]
        rng = np.random.default_rng(seed)
        m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    out = {"scales": scales, "qualities": QUALITIES, "alphas": alphas,
           "n_imgs": N, "pure": {}, "hull": {}}

    # pure single-system ladders (fixed res) for in-protocol comparability
    for sname in ["s_by_q", "spred", "C_l05_b30", "C_l05_b50",
                  "C_l05_b30_sband", "C_l05_b50_sband"] + list(NET_SYS):
        def sel(rows, sname=sname):
            sub = [r for r in rows if r["sys"] == sname and r["scale"] == 1.0]
            return sorted(sub, key=lambda r: r["bpp"])
        bn, bm = per_image_bd(sel)
        out["pure"][sname] = {
            "bd_vmaf_neg": float(np.nanmean(bn)), "bd_msssim": float(np.nanmean(bm)),
            "n_ok": int(np.isfinite(bn).sum())}
        print(f"PURE {sname}: BD-NEG {np.nanmean(bn):+.2f} BD-MSSSIM "
              f"{np.nanmean(bm):+.2f} (n={np.isfinite(bn).sum()})", flush=True)

    # selection hull per alpha
    for al in alphas:
        def sel(rows, al=al):
            sub = rows  # all systems, all scales in this run
            b = np.array([r["bpp"] for r in sub])
            s = al * np.array([r["vmaf_neg"] for r in sub]) + \
                (1 - al) * 100.0 * np.array([r["msssim"] for r in sub])
            idx = pareto_hull(b, s)
            return [sub[j] for j in idx]
        bn, bm = per_image_bd(sel)
        ci_n = boot_mean(bn, a.boot, 42)
        ci_m = boot_mean(bm, a.boot, 43)
        out["hull"][al] = {
            "bd_vmaf_neg": float(np.nanmean(bn)), "bd_vmaf_neg_ci": ci_n,
            "bd_msssim": float(np.nanmean(bm)), "bd_msssim_ci": ci_m,
            "n_ok": int(np.isfinite(bn).sum()),
            "per_img_bd_neg": bn.tolist(), "per_img_bd_mss": bm.tolist()}
        hit = np.nanmean(bn) <= -5 and np.nanmean(bm) <= -2
        print(f"ALPHA {al:.2f}: BD-NEG {np.nanmean(bn):+.2f} {ci_n} | BD-MSSSIM "
              f"{np.nanmean(bm):+.2f} {ci_m} | AND-bar {'HIT' if hit else 'miss'}",
              flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    # raw rows for later re-analysis (selection rules can be re-run offline)
    json.dump({str(i): allrows[i] for i in range(N)},
              open(a.out.replace(".json", "_rows.json"), "w"))
    print(f"saved {a.out} (+_rows.json)", flush=True)

    # frontier plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    xs = [out["hull"][al]["bd_msssim"] for al in alphas]
    ys = [out["hull"][al]["bd_vmaf_neg"] for al in alphas]
    ax.plot(xs, ys, "b-o", ms=4)
    for al, x, y in zip(alphas, xs, ys):
        ax.annotate(f"a={al:g}", (x, y), fontsize=8)
    for nm, d in out["pure"].items():
        ax.plot(d["bd_msssim"], d["bd_vmaf_neg"], "r^", ms=6)
        ax.annotate(nm, (d["bd_msssim"], d["bd_vmaf_neg"]), fontsize=8, color="r")
    ax.axhline(-5, color="g", ls="--", lw=1); ax.axvline(-2, color="g", ls="--", lw=1)
    ax.set_xlabel("BD-rate MS-SSIM % (neg=win)")
    ax.set_ylabel("BD-rate VMAF_NEG % (neg=win)")
    ax.grid(alpha=0.3)
    ax.set_title(f"val50 selection-hull frontier (scales={a.scales})")
    fig.tight_layout()
    png = a.out.replace(".json", ".png")
    fig.savefig(png, dpi=130)
    print(f"saved {png}", flush=True)


if __name__ == "__main__":
    main()
