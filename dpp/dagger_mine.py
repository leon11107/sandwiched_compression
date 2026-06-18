"""DAgger round for the VMAF_NEG surrogate: the v1/v2 surrogate-guided ES
Goodharts the ensemble (pred_gain +5.4 NEG pts vs realized -0.03; corr(pred,
real) = -0.23) because ES-visited s vectors are off the training distribution
(plain/mean_s/rand_s(0.12)/lowpass). Fix: label the states the CURRENT policy
visits with the real expert and retrain.

Phase "es"   (GPU, torch): run CropES on train images; per (img, q) emit the
             final pick, two mid-shortlist candidates and one wide rand_s.
Phase "score" (CPU, torch-free): one batched real NEG call per image over all
             its mined candidates -> rows in vmafneg_data schema, appendable
             to surrogate training (kind es_pick / es_mid / rand_big).

Mining EXCLUDES the surrogate holdout tail (keeps holdout honest) and the
post-DAgger validation slice sorted(targets)[40:80] (keeps capture honest).
"""
from __future__ import annotations
import argparse, glob, json, os, sys, tempfile
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/workspace/sandwiched_compression")

RUNS = "/workspace/sandwiched_compression/dpp/runs"
TRAIN_DIR = "/workspace/sandwiched_compression/dpp/data/train_big"
TARGETS = os.path.join(RUNS, "oracle_targets.jsonl")
CANDS = os.path.join(RUNS, "dagger_cands.jsonl")
OUT = os.path.join(RUNS, "vmafneg_data_es.jsonl")


def mining_paths(n_imgs):
    paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
    n_ho = max(1, int(len(paths) * 0.1))
    ho_tail = paths[-n_ho:]  # surrogate holdout (split by path in the trainer)
    tpaths = sorted({json.loads(l)["path"] for l in open(TARGETS)})
    # [0:40]: v2-validation picks already converted to rows (real v stored);
    # [40:80]: fresh post-DAgger capture-validation slice
    excl = set(tpaths[:80])
    tr_pool = [p for p in paths[:-n_ho] if p not in excl]
    # ~1/8 of the budget goes to holdout imgs: never trained on, but gives the
    # trainer's holdout metric coverage of ES-mined pairs (es_rank_acc)
    n_hm = max(1, n_imgs // 8)
    n_tr = n_imgs - n_hm
    sel = tr_pool[:: max(1, len(tr_pool) // n_tr)][:n_tr]
    sel += ho_tail[:: max(1, len(ho_tail) // n_hm)][:n_hm]
    return sel


def phase_es(a):
    import torch
    from dpp.surrogate_oracle import CropES, global_kappa, load_net, CROP
    from dpp.oracle_vmafneg import block_dct, load16, luma
    from dpp.oracle_targets import mean_oracle_s
    from dpp.spred import SPredictor
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    nets = [load_net(p, dev) for p in a.ckpt.split(",")]
    sp = SPredictor.load(os.path.join(RUNS, "spred_model.npz"))
    ms = mean_oracle_s()
    es = CropES(nets, dev, global_kappa())
    si, sn = [int(x) for x in a.shard.split("/")]
    paths = mining_paths(a.n_imgs)[si::sn]
    done = set()
    if os.path.exists(a.cands):
        for line in open(a.cands):
            try:
                r = json.loads(line); done.add((r["path"], r["q"], r["kind"]))
            except json.JSONDecodeError:
                pass
    with open(a.cands, "a") as f:
        for p in paths:
            img = load16(p)
            if min(img.shape[:2]) < CROP:
                continue
            rng = np.random.default_rng(hash(p) % 2**31 ^ 7)
            for q in (8, 20):
                if (p, q, "es_pick") in done:
                    continue
                seeds = [sp.predict(img, q), ms[8 if q <= 8 else 20], np.ones(64)]
                best = es.run(img, q, seeds, rng)
                rows = [("es_pick", best["s"])]
                pool = [c for j, c in enumerate(es.last_cands)
                        if j >= es.last_nseeds
                        and not np.array_equal(c, best["s"])]
                for j in rng.choice(len(pool), min(2, len(pool)), replace=False):
                    rows.append(("es_mid", pool[j]))
                rows.append(("rand_big",
                             np.clip(1 + rng.normal(0, 0.18, 64), 0.3, 1.5)))
                for kind, s in rows:
                    f.write(json.dumps({"path": p, "q": q, "kind": kind,
                                        "s": np.round(s, 4).tolist()}) + "\n")
                f.flush()
                print(f"[{os.path.basename(p)} q{q}] mined {len(rows)} "
                      f"(pred {best['pred_gain']:+.2f})", flush=True)
    print("SHARD DONE", flush=True)


def score_img(args):
    path, rows = args
    from distortion import vmaf_metric as vm
    from dpp.oracle_vmafneg import (apply_s, block_dct, jpeg_rt, load16, luma,
                                    up1080, vmaf_neg_batch)
    rgb = load16(path)
    y = luma(rgb); coeffs = block_dct(y)
    npix = rgb.shape[0] * rgb.shape[1]
    decs, bpps = [], []
    for r in rows:
        dec, bits = jpeg_rt(apply_s(rgb, coeffs, y, np.asarray(r["s"])), r["q"])
        decs.append(up1080(np.rint(dec).astype(np.uint8)))
        bpps.append(bits / npix)
    ref_up = up1080(np.rint(rgb).astype(np.uint8))
    with tempfile.TemporaryDirectory() as td:
        rdir = os.path.join(td, "r"); os.makedirs(rdir)
        ry4m = os.path.join(td, "ref.y4m")
        vm._png_seq_to_y4m([ref_up] * len(rows), rdir, ry4m)
        vs = vmaf_neg_batch(ry4m, decs, td, threads=2)
    out = []
    for r, b, v in zip(rows, bpps, vs):
        out.append({"path": path, "kind": r["kind"], "q": r["q"], "s": r["s"],
                    "contrast": None, "bpp": float(b), "vmaf_neg": float(v)})
    print(f"[{os.path.basename(path)}] {len(out)} rows scored", flush=True)
    return out


def phase_score(a):
    byimg = {}
    for line in open(a.cands):
        r = json.loads(line)
        byimg.setdefault(r["path"], []).append(r)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["path"])
            except json.JSONDecodeError:
                pass
    jobs = sorted((p, rows) for p, rows in byimg.items() if p not in done)
    print(f"{len(jobs)} imgs to score (skipped {len(done)})", flush=True)
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        n = 0
        for rows in pool.imap_unordered(score_img, jobs):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                print(f"--- {n}/{len(jobs)} imgs done ---", flush=True)
    print("ALL DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["es", "score"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-imgs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--ckpt", default=",".join(
        os.path.join(RUNS, f"surrogate_{v}.pt") for v in ("base", "wide", "rank")))
    ap.add_argument("--cands", default=CANDS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    (phase_es if a.phase == "es" else phase_score)(a)


if __name__ == "__main__":
    main()
