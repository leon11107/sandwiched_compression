"""Oracle s-target generation for the image-adaptive predictor (line A).

Per TRAIN image (train_big sample, zero overlap with val50) and q in {8,20}:
  warm-started (mu,lambda)-ES directly on the real VMAF_NEG binary over the
  64-dim DCT-band luma pre-scaling family (same as oracle_vmafneg.py), seeded
  at the known-good region (mean oracle s8/s20 from oracle_vmafneg.json) so a
  small budget (gens~12, pop~24) suffices.

Differences vs oracle_vmafneg.py (the val50 proof-of-existence run):
  1. Baseline curve is recomputed HONESTLY in-job (per-image, equal-size vmaf
     call over Q_BASE encodes) — the old run read per-image curves from
     eval_v2_full.json, which the y4m mixed-size bug had corrupted. kappa
     (iso-rate scalarization slope) and the verdict both use the honest curve.
  2. Incremental jsonl output + resume (skip (img,q) pairs already done).

Output rows: {img, path, q, s, best_vmaf_neg, best_bpp, bpp_base, base_curve,
              kappa, iso_bpp_delta, gen}. torch-free, multiprocessing.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/workspace/sandwiched_compression")
from distortion import vmaf_metric as vm
from dpp.oracle_vmafneg import (apply_s, block_dct, jpeg_rt, load16, luma,
                                seeds, up1080, vmaf_neg_batch)

TRAIN_DIR = "/workspace/sandwiched_compression/dpp/data/train_big"
ORACLE_JSON = "/workspace/sandwiched_compression/dpp/runs/oracle_vmafneg.json"
OUT = "/workspace/sandwiched_compression/dpp/runs/oracle_targets.jsonl"
Q_BASE = [5, 8, 12, 20, 32]  # honest per-image baseline curve points


def mean_oracle_s():
    d = json.load(open(ORACLE_JSON))
    out = {}
    for q in (8, 20):
        out[q] = np.array([r["s"] for r in d if r["q"] == q]).mean(0)
    return out


def run_job(args):
    img_idx, path, q, mean_s, gens, lam_pop, threads, seed = args
    rng = np.random.default_rng(seed)
    rgb = load16(path)
    y = luma(rgb); coeffs = block_dct(y)
    npix = rgb.shape[0] * rgb.shape[1]
    with tempfile.TemporaryDirectory() as td:
        ref_up = up1080(np.rint(rgb).astype(np.uint8))

        # 1) honest per-image baseline curve (one small equal-size vmaf call)
        bdecs, bbpp = [], []
        for bq in Q_BASE:
            dec, bits = jpeg_rt(rgb, bq)
            bdecs.append(up1080(np.rint(dec).astype(np.uint8)))
            bbpp.append(bits / npix)
        rdirb = os.path.join(td, "rb"); os.makedirs(rdirb)
        ry4mb = os.path.join(td, "refb.y4m")
        vm._png_seq_to_y4m([ref_up] * len(Q_BASE), rdirb, ry4mb)
        bneg = vmaf_neg_batch(ry4mb, bdecs, td, threads=threads)
        qi = Q_BASE.index(q)
        bpp0, neg0 = bbpp[qi], bneg[qi]
        # local slope (vmaf_neg pts per 1% bpp) bracketing q: (q5,q12) or (q12,q32)
        lo_i, hi_i = (0, 2) if q <= 8 else (2, 4)
        kappa = (bneg[hi_i] - bneg[lo_i]) / max(
            (bbpp[hi_i] - bbpp[lo_i]) / bbpp[lo_i] * 100.0, 1e-6)

        # 2) warm-started ES
        pop = [np.ones(64), mean_s[8].copy(), mean_s[20].copy()] + seeds()[1:]
        while len(pop) < lam_pop:
            base = mean_s[q]
            pop.append(np.clip(base + rng.normal(0, 0.08, 64), 0.3, 1.5))
        pop = [np.clip(p, 0.3, 1.5) for p in pop[:lam_pop]]
        rdir = os.path.join(td, "r"); os.makedirs(rdir)
        ry4m = os.path.join(td, "ref.y4m")
        vm._png_seq_to_y4m([ref_up] * lam_pop, rdir, ry4m)

        sigma = 0.08
        best = {"fit": -1e9}
        for g in range(gens):
            decs, bpps = [], []
            for s in pop:
                dec, bits = jpeg_rt(apply_s(rgb, coeffs, y, s), q)
                decs.append(up1080(np.rint(dec).astype(np.uint8)))
                bpps.append(bits / npix)
            vs = vmaf_neg_batch(ry4m, decs, td, threads=threads)
            fits = [v - kappa * (b - bpp0) / bpp0 * 100.0 for v, b in zip(vs, bpps)]
            order = np.argsort(fits)[::-1]
            if fits[order[0]] > best["fit"]:
                i = order[0]
                best = {"fit": float(fits[i]), "vmaf_neg": float(vs[i]),
                        "bpp": float(bpps[i]), "s": pop[i].tolist(), "gen": g}
            elites = [pop[i] for i in order[:6]]
            newpop = list(elites)
            while len(newpop) < lam_pop:
                e = elites[rng.integers(len(elites))]
                if rng.random() < 0.25:
                    m = e.copy(); ii = rng.integers(0, 64, 8)
                    m[ii] += rng.normal(0, sigma * 2, 8)
                else:
                    m = e + rng.normal(0, sigma, 64)
                newpop.append(np.clip(m, 0.3, 1.5))
            pop = newpop
            sigma *= 0.92
    bb = np.array(bbpp); vb = np.array(bneg)
    o = np.argsort(bb)
    iso = float(np.interp(best["bpp"], bb[o], vb[o]))
    res = {"img": img_idx, "path": path, "q": q, "s": best["s"],
           "best_vmaf_neg": best["vmaf_neg"], "best_bpp": best["bpp"],
           "bpp_base": float(bpp0), "vmafneg_base": float(neg0),
           "base_curve": {"q": Q_BASE, "bpp": bbpp, "vmaf_neg": bneg},
           "kappa": float(kappa), "iso_bpp_delta": best["vmaf_neg"] - iso,
           "gen": best["gen"]}
    print(f"[{os.path.basename(path)} q{q}] base {neg0:.2f}@{bpp0:.4f} -> "
          f"{best['vmaf_neg']:.2f}@{best['bpp']:.4f} | iso {res['iso_bpp_delta']:+.2f} "
          f"(gen {best['gen']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-imgs", type=int, default=200)
    ap.add_argument("--qualities", default="8,20")
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--vmaf-threads", type=int, default=3)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
    stride = max(1, len(paths) // a.n_imgs)
    paths = paths[::stride][: a.n_imgs]
    mean_s = mean_oracle_s()

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["path"], r["q"]))
            except json.JSONDecodeError:
                pass
    jobs = [(i, p, q, mean_s, a.gens, a.pop, a.vmaf_threads, 7000 + 31 * i + q)
            for q in [int(x) for x in a.qualities.split(",")]
            for i, p in enumerate(paths) if (p, q) not in done]
    print(f"{len(jobs)} jobs (skipped {len(done)} done); imgs={len(paths)} "
          f"gens={a.gens} pop={a.pop} workers={a.workers}", flush=True)
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        n = 0
        for res in pool.imap_unordered(run_job, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"--- {n}/{len(jobs)} jobs done ---", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
