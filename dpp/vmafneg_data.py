"""Training-data generation for the differentiable VMAF_NEG surrogate (line S).

Per train_big image, ONE batched NEG-only vmaf call scoring a diverse dist set
(recipes chosen to span the behaviours the surrogate must rank correctly):
  plain     q in {5,8,12,20,32}   : straight JPEG
  mean_s    q in {5,8,12,20,32}   : fixed mean-oracle band pre-emphasis (s_by_q)
  rand_s    q in {5,8,12,20,32}   : random band scaling ~ clip(1+N(0,0.12))
  lowpass   q in {5,8,12,20,32}   : classic rc>=6 -> 0.6
  contrast  q=20                  : 1.3x contrast + 8 pre-boost (NEG-gaming probe)
  hfboost   q=20                  : strong high-band boost s=1.4 (over-emphasis probe)
=> 22 rows/image. Rows carry the full recipe (s vector) so dists are exactly
reconstructable on the fly during surrogate training (16-aligned crops commute
with blockwise JPEG + apply_s). Resume-safe jsonl output.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/workspace/sandwiched_compression")
from distortion import vmaf_metric as vm
from dpp.oracle_vmafneg import (apply_s, block_dct, jpeg_rt, load16, luma,
                                up1080, vmaf_neg_batch)
from dpp.oracle_targets import mean_oracle_s

TRAIN_DIR = "/workspace/sandwiched_compression/dpp/data/train_big"
OUT = "/workspace/sandwiched_compression/dpp/runs/vmafneg_data.jsonl"
QS = [5, 8, 12, 20, 32]
RC = (np.arange(64) // 8 + np.arange(64) % 8)


def recipes(rng, mean_s):
    """-> list of (kind, q, s or None, contrast or None)."""
    out = []
    lp = np.ones(64); lp[RC >= 6] = 0.6
    hb = np.ones(64); hb[RC >= 4] = 1.4
    for q in QS:
        out.append(("plain", q, None, None))
        out.append(("mean_s", q, (mean_s[8] if q <= 8 else mean_s[20]).copy(), None))
        out.append(("rand_s", q, np.clip(1 + rng.normal(0, 0.12, 64), 0.3, 1.5), None))
        out.append(("lowpass", q, lp.copy(), None))
    out.append(("contrast", 20, None, (1.3, 8.0)))
    out.append(("hfboost", 20, hb.copy(), None))
    return out


def make_dist(rgb, y, coeffs, kind, q, s, contrast):
    if kind == "contrast":
        mean = rgb.mean(axis=(0, 1), keepdims=True)
        src = np.clip((rgb - mean) * contrast[0] + mean + contrast[1], 0, 255)
    elif s is not None:
        src = apply_s(rgb, coeffs, y, np.asarray(s))
    else:
        src = rgb
    return jpeg_rt(src, q)


def run_img(args):
    path, mean_s, seed = args
    rng = np.random.default_rng(seed)
    rgb = load16(path)
    y = luma(rgb); coeffs = block_dct(y)
    npix = rgb.shape[0] * rgb.shape[1]
    recs = recipes(rng, mean_s)
    decs, bpps = [], []
    for kind, q, s, con in recs:
        dec, bits = make_dist(rgb, y, coeffs, kind, q, s, con)
        decs.append(up1080(np.rint(dec).astype(np.uint8)))
        bpps.append(bits / npix)
    ref_up = up1080(np.rint(rgb).astype(np.uint8))
    with tempfile.TemporaryDirectory() as td:
        rdir = os.path.join(td, "r"); os.makedirs(rdir)
        ry4m = os.path.join(td, "ref.y4m")
        vm._png_seq_to_y4m([ref_up] * len(recs), rdir, ry4m)
        vs = vmaf_neg_batch(ry4m, decs, td, threads=2)
    rows = []
    for (kind, q, s, con), b, v in zip(recs, bpps, vs):
        rows.append({"path": path, "kind": kind, "q": q,
                     "s": (np.asarray(s).round(4).tolist() if s is not None else None),
                     "contrast": con, "bpp": float(b), "vmaf_neg": float(v)})
    p20 = next(r for r in rows if r["kind"] == "plain" and r["q"] == 20)
    print(f"[{os.path.basename(path)}] plain q20 NEG {p20['vmaf_neg']:.1f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
    if a.limit:
        paths = paths[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["path"])
            except json.JSONDecodeError:
                pass
    mean_s = mean_oracle_s()
    jobs = [(p, mean_s, 5000 + i) for i, p in enumerate(paths) if p not in done]
    print(f"{len(jobs)} imgs to score (skipped {len(done)})", flush=True)
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        n = 0
        for rows in pool.imap_unordered(run_img, jobs):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"--- {n}/{len(jobs)} imgs done ---", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
