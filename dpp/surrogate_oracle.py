"""Surrogate-oracle: per-image ES over the 64-dim DCT-band luma pre-scaling s,
with the trained VMAF_NEG surrogate as a FAST fitness (GPU) instead of the real
vmaf binary (CPU, ~6 min/img). ~4 s/(img,q) => deep per-image search at scale.

Fitness construction (v2, anti-Goodhart — v1 with fixed crops + per-image
2-encode kappa let the ES climb surrogate errors: best-at-last-gen, 13% capture):
  - ENSEMBLE fitness: mean over the base/wide/rank nets (adversarial directions
    are net-specific; the ensemble suppresses single-net exploitation).
  - FRESH K crops re-sampled every generation: a candidate must win across crop
    re-draws, so crop-overfit and brittle surrogate exploits don't survive.
  - GLOBAL kappa per q: median real NEG slope per 1% bpp over the 203 stored
    honest base curves, /100 into surrogate (sigmoid) units — removes the noisy
    per-image 2-point estimate.
  - FINAL selection on a larger fresh crop set over a shortlist of per-gen
    winners + the warm seeds — the chosen s can never look worse than the
    real-validated seeds to the surrogate's honest (held-out-crop) opinion.

Modes:
  --validate: run on images that HAVE real-ES targets (oracle_targets.jsonl),
    then score the surrogate-chosen s with ONE real NEG vmaf call per image
    (both q at once) against the stored honest per-image baseline curve.
    Report real iso-bpp delta vs the real-ES upper bound. Gate for scale-up.
  default: scale over train_big -> surrogate_targets.jsonl for distillation.
"""
from __future__ import annotations
import argparse, glob, io, json, os, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from distortion import vmaf_metric as vm
from dpp.oracle_vmafneg import (apply_s, block_dct, block_idct, jpeg_rt, load16,
                                luma, up1080, vmaf_neg_batch)
from dpp.oracle_targets import mean_oracle_s
from dpp.vmafneg_surrogate import Net, CROP
from dpp.spred import SPredictor

RUNS = "/workspace/sandwiched_compression/dpp/runs"
TRAIN_DIR = "/workspace/sandwiched_compression/dpp/data/train_big"
TARGETS = os.path.join(RUNS, "oracle_targets.jsonl")


def global_kappa():
    """Median real NEG slope per 1% bpp around q over the stored honest
    per-image base curves (same bracketing as the real oracle), /100 into
    surrogate (sigmoid) units."""
    ks = {8: [], 20: []}
    for line in open(TARGETS):
        r = json.loads(line)
        if r["q"] not in ks:
            continue
        b, v = r["base_curve"]["bpp"], r["base_curve"]["vmaf_neg"]
        lo_i, hi_i = (0, 2) if r["q"] <= 8 else (2, 4)
        ks[r["q"]].append((v[hi_i] - v[lo_i]) /
                          max((b[hi_i] - b[lo_i]) / b[lo_i] * 100.0, 1e-6))
    return {q: float(np.median(k)) / 100.0 for q, k in ks.items()}


def jpeg_rt_crop(img, quality):
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8)).save(
        buf, format="jpeg", quality=int(quality), subsampling="4:2:0")
    return np.asarray(Image.open(buf).convert("RGB"), np.float32), 8 * len(buf.getbuffer())


class CropES:
    """Per-image ES; ensemble fitness on freshly re-sampled crops per gen."""

    def __init__(self, nets, dev, kappa_by_q, k_crops=8, pop=32, gens=16,
                 threads=8, k_final=16):
        self.nets, self.dev, self.kq = nets, dev, kappa_by_q
        self.K, self.pop, self.gens, self.KF = k_crops, pop, gens, k_final
        self.ex = ThreadPoolExecutor(threads)

    def _crops(self, img, rng, k):
        H, W = img.shape[:2]
        out = []
        for _ in range(k):
            oy = 16 * rng.integers(0, (H - CROP) // 16 + 1)
            ox = 16 * rng.integers(0, (W - CROP) // 16 + 1)
            out.append(img[oy:oy + CROP, ox:ox + CROP])
        return out

    @torch.no_grad()
    def _score(self, crops, dists):
        """crops list of K refs; dists list of (cand,K) arrays -> [C] means."""
        refs_t = (torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2))
                  .to(self.dev) / 255.0).float()
        d = torch.from_numpy(np.stack([np.stack(c) for c in dists])  # [C,K,H,W,3]
                             .transpose(0, 1, 4, 2, 3)).to(self.dev) / 255.0
        C, K = d.shape[:2]
        r = refs_t.unsqueeze(0).expand(C, -1, -1, -1, -1).reshape(C * K, *refs_t.shape[1:])
        dd = d.reshape(C * K, *d.shape[2:]).float()
        ps = []
        for c0 in range(0, C * K, 128):
            with torch.autocast("cuda"):
                p = sum(net(r[c0:c0 + 128], dd[c0:c0 + 128])
                        for net in self.nets) / len(self.nets)
            ps.append(p.float())
        return torch.cat(ps).view(C, K).mean(1).cpu().numpy()

    def _eval(self, crops, cands, q):
        """Encode all cands on all crops; -> (ensemble scores, summed bits)."""
        cy = [luma(c) for c in crops]
        cc = [block_dct(y) for y in cy]
        K = len(crops)

        def enc_one(args):
            ci, s = args
            return jpeg_rt_crop(apply_s(crops[ci], cc[ci], cy[ci], s), q)

        jobs = [(ci, s) for s in cands for ci in range(K)]
        res = list(self.ex.map(enc_one, jobs))
        dists = [[res[i * K + ci][0] for ci in range(K)]
                 for i in range(len(cands))]
        bits = np.array([sum(res[i * K + ci][1] for ci in range(K))
                         for i in range(len(cands))], float)
        return self._score(crops, dists), bits

    def run(self, img, q, seeds, rng):
        kappa, one = self.kq[q], np.ones(64)
        shortlist = [np.clip(s, 0.3, 1.5) for s in seeds]  # seeds always kept
        pop = list(shortlist[: self.pop])
        while len(pop) < self.pop:
            base = seeds[rng.integers(len(seeds))]
            pop.append(np.clip(base + rng.normal(0, 0.05, 64), 0.3, 1.5))
        sigma = 0.05
        for g in range(self.gens):
            crops = self._crops(img, rng, self.K)
            scores, bits = self._eval(crops, [one] + pop, q)
            fits = scores[1:] - kappa * (bits[1:] - bits[0]) / bits[0] * 100.0
            order = np.argsort(fits)[::-1]
            shortlist += [pop[order[0]].copy(), pop[order[1]].copy()]
            elites = [pop[i] for i in order[:6]]
            newpop = list(elites)
            while len(newpop) < self.pop:
                e = elites[rng.integers(len(elites))]
                if rng.random() < 0.25:
                    m = e.copy(); ii = rng.integers(0, 64, 8)
                    m[ii] += rng.normal(0, sigma * 2, 8)
                else:
                    m = e + rng.normal(0, sigma, 64)
                newpop.append(np.clip(m, 0.3, 1.5))
            pop = newpop
            sigma *= 0.93

        # final selection: larger fresh crop set, dedup'd shortlist (seeds first)
        cands, seen = [], set()
        for s in shortlist:
            k = tuple(np.round(s, 3))
            if k not in seen:
                seen.add(k); cands.append(s)
        crops = self._crops(img, rng, self.KF)
        scores, bits = self._eval(crops, [one] + cands, q)
        fits = scores[1:] - kappa * (bits[1:] - bits[0]) / bits[0] * 100.0
        i = int(np.argmax(fits))
        seed_fit = float(fits[: len(seeds)].max())
        # stash final shortlist for hard-example mining (DAgger)
        self.last_cands, self.last_fits, self.last_nseeds = cands, fits, len(seeds)
        return {"s": cands[i], "fit": float(fits[i]),
                "score": float(scores[1 + i]), "score_plain": float(scores[0]),
                "dbpp_pct": float((bits[1 + i] - bits[0]) / bits[0] * 100.0),
                "pred_gain": float((fits[i] - scores[0]) * 100.0),
                "seed_gap": float((fits[i] - seed_fit) * 100.0),
                "picked_seed": bool(i < len(seeds)), "kappa": kappa}


def load_net(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    net = Net(ck["width"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=",".join(
        os.path.join(RUNS, f"surrogate_{v}.pt") for v in ("base", "wide", "rank")),
        help="comma-separated ckpts -> ensemble fitness")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--validate", type=int, default=0,
                    help="N imgs with real-ES targets: compare vs real oracle")
    ap.add_argument("--val-offset", type=int, default=0,
                    help="skip first M target imgs (fresh slice after DAgger)")
    ap.add_argument("--n-imgs", type=int, default=600)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=16)
    ap.add_argument("--k-crops", type=int, default=8)
    ap.add_argument("--k-final", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(RUNS, "surrogate_targets.jsonl"))
    ap.add_argument("--shard", default="0/1", help="i/n image sharding")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    nets = [load_net(p, dev) for p in a.ckpt.split(",")]
    sp = SPredictor.load(os.path.join(RUNS, "spred_model.npz"))
    ms = mean_oracle_s()
    kq = global_kappa()
    print(f"{len(nets)}-net ensemble; global kappa (surrogate units) "
          f"q8={kq[8]:.5f} q20={kq[20]:.5f}", flush=True)
    es = CropES(nets, dev, kq, k_crops=a.k_crops, pop=a.pop, gens=a.gens,
                k_final=a.k_final)
    si, sn = [int(x) for x in a.shard.split("/")]

    if a.validate:
        real = {}
        for line in open(TARGETS):
            r = json.loads(line)
            real.setdefault(r["path"], {})[r["q"]] = r
        vpaths = sorted(real)[a.val_offset: a.val_offset + a.validate][si::sn]
        rows_out = []
        for p in vpaths:
            img = load16(p)
            if min(img.shape[:2]) < CROP or set(real[p]) != {8, 20}:
                continue
            rng = np.random.default_rng(hash(p) % 2**31)
            y = luma(img); c = block_dct(y)
            decs, info = [], []
            for q in (8, 20):
                seeds = [sp.predict(img, q), ms[8 if q <= 8 else 20], np.ones(64)]
                t0 = time.time()
                best = es.run(img, q, seeds, rng)
                dt = time.time() - t0
                dec, bits = jpeg_rt(apply_s(img, c, y, best["s"]), q)
                decs.append(up1080(np.rint(dec).astype(np.uint8)))
                info.append((q, best, bits / (img.shape[0] * img.shape[1]), dt))
            ref_up = up1080(np.rint(img).astype(np.uint8))
            with tempfile.TemporaryDirectory() as td:
                rdir = os.path.join(td, "r"); os.makedirs(rdir)
                ry4m = os.path.join(td, "ref.y4m")
                vm._png_seq_to_y4m([ref_up] * len(decs), rdir, ry4m)
                vs = vmaf_neg_batch(ry4m, decs, td, threads=4)
            for (q, best, bpp, dt), v in zip(info, vs):
                rr = real[p][q]
                bc = rr["base_curve"]
                bb, vb = np.array(bc["bpp"]), np.array(bc["vmaf_neg"])
                o = np.argsort(bb)
                iso = v - float(np.interp(bpp, bb[o], vb[o]))
                rows_out.append({"path": p, "q": q, "iso_real_es": rr["iso_bpp_delta"],
                                 "iso_surr_es": iso, "s": best["s"].tolist(),
                                 "v": float(v), "bpp": float(bpp),
                                 "pred_gain": best["pred_gain"],
                                 "seed_gap": best["seed_gap"],
                                 "picked_seed": best["picked_seed"], "dt": dt})
                print(f"[{os.path.basename(p)} q{q}] surrogate-ES iso {iso:+.2f} "
                      f"(pred {best['pred_gain']:+.2f}, seed_gap "
                      f"{best['seed_gap']:+.2f}{', SEED' if best['picked_seed'] else ''})"
                      f" vs real-ES {rr['iso_bpp_delta']:+.2f} ({dt:.1f}s)", flush=True)
        sr = np.array([r["iso_surr_es"] for r in rows_out])
        rl = np.array([r["iso_real_es"] for r in rows_out])
        print(f"\nVALIDATION ({len(rows_out)} rows): surrogate-ES mean {sr.mean():+.3f} "
              f"vs real-ES mean {rl.mean():+.3f} -> capture {sr.mean()/rl.mean()*100:.0f}%"
              f" | surr>0: {(sr>0).sum()}/{len(sr)}", flush=True)
        sfx = f"_{si}" if sn > 1 else ""
        json.dump(rows_out, open(os.path.join(
            RUNS, f"surrogate_oracle_val{sfx}.json"), "w"), indent=2)
        return

    # scale mode: shard over train_big
    paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
    stride = max(1, len(paths) // a.n_imgs)
    paths = paths[::stride][: a.n_imgs][si::sn]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r["path"], r["q"]))
            except json.JSONDecodeError:
                pass
    with open(a.out, "a") as f:
        for p in paths:
            img = load16(p)
            if min(img.shape[:2]) < CROP:
                continue
            rng = np.random.default_rng(hash(p) % 2**31)
            for q in (8, 20):
                if (p, q) in done:
                    continue
                seeds = [sp.predict(img, q), ms[8 if q <= 8 else 20], np.ones(64)]
                t0 = time.time()
                best = es.run(img, q, seeds, rng)
                f.write(json.dumps({"path": p, "q": q, "s": best["s"].tolist(),
                                    "score": best["score"],
                                    "score_plain": best["score_plain"],
                                    "dbpp_pct": best["dbpp_pct"],
                                    "pred_gain": best["pred_gain"],
                                    "picked_seed": best["picked_seed"]}) + "\n")
                f.flush()
                print(f"[{os.path.basename(p)} q{q}] dscore "
                      f"{best['score']-best['score_plain']:+.4f} dbpp {best['dbpp_pct']:+.1f}% "
                      f"({time.time()-t0:.1f}s)", flush=True)
    print("SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
