"""ORACLE upper-bound experiment: can ANY luma preprocessor gain VMAF_NEG on JPEG
intra? Per-image black-box optimization DIRECTLY on the real VMAF_NEG binary —
no network, no proxy, no surrogate ("god's eye view").

Search family: per-image 64-dim DCT-subband luma pre-scaling s in [0.3,1.5]^64,
  Y' = blockIDCT(s * blockDCT(Y)); RGB' = RGB + (Y'-Y) (BT.601 full-range: a pure
  luma shift adds equally to R,G,B). Superset of classical pre-filtering (low-pass,
  band-shaping, unsharp).
Codec/protocol: EXACTLY eval_v2 (real JPEG 4:2:0 Annex-K via PIL quality, VMAF_NEG
  after 1080p-area bicubic upscale, val50 center-crop /16).
Optimizer: (mu,lambda)-ES, seeded with identity + classic filter shapes; one
  NEG-only vmaf call per generation (all candidates batched into one y4m).
Fitness: vmaf_neg - kappa_img * (bpp-bpp_base)/bpp_base*100 (kappa = per-image
  local slope of the baseline RD curve = iso-rate scalarization).
Verdict metric: iso-bpp delta of the best candidate vs the per-image baseline
  curve (from eval_v2_full.json per-image data).

Either outcome is decisive: ~0 => ceiling claim upgraded from induction to
oracle-bound (within this family); >0 => ceiling REFUTED + distillation targets.
torch-free workers (numpy + PIL + vmaf subprocess), multiprocessing.
"""
from __future__ import annotations
import argparse, glob, io, json, os, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, "/workspace/sandwiched_compression")
from distortion import vmaf_metric as vm

VAL_DIR = "/workspace/sandwiched_compression/dpp/data/val50"
BASE_JSON = "/workspace/sandwiched_compression/dpp/runs/eval_v2_full.json"
TARGET_AREA = 1920 * 1080
_D = None  # 8x8 orthonormal DCT-II matrix


def dct_mat():
    global _D
    if _D is None:
        k, n = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        D = np.cos(np.pi / 8 * (n + 0.5) * k) * np.sqrt(2.0 / 8)
        D[0] /= np.sqrt(2.0)
        _D = D.astype(np.float64)
    return _D


def load16(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    w16, h16 = (w // 16) * 16, (h // 16) * 16
    l, t = (w - w16) // 2, (h - h16) // 2
    return np.asarray(im.crop((l, t, l + w16, t + h16)), np.float32)


def luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def block_dct(y):
    """y [H,W] -> coeffs [H/8, W/8, 8, 8]"""
    H, W = y.shape
    b = y.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    D = dct_mat()
    return np.einsum("ij,abjk,lk->abil", D, b, D)


def block_idct(c):
    D = dct_mat()
    b = np.einsum("ji,abjk,kl->abil", D, c, D)
    H8, W8 = b.shape[:2]
    return b.transpose(0, 2, 1, 3).reshape(H8 * 8, W8 * 8)


def apply_s(rgb, coeffs, y, s):
    """pre-scale luma DCT bands by s[64]; luma-only edit on RGB."""
    c2 = coeffs * s.reshape(1, 1, 8, 8)
    y2 = block_idct(c2)
    out = rgb + (y2 - y)[..., None]
    return np.clip(out, 0, 255)


def jpeg_rt(img, quality):
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8)).save(
        buf, format="jpeg", quality=int(quality), subsampling="4:2:0", optimize=True)
    return np.asarray(Image.open(buf).convert("RGB"), np.float32), 8 * len(buf.getbuffer())


def up1080(u8):
    h, w = u8.shape[:2]
    sc = np.sqrt(TARGET_AREA / (w * h))
    if sc <= 1.0:
        return u8
    nw, nh = int(round(w * sc / 2)) * 2, int(round(h * sc / 2)) * 2
    return np.asarray(Image.fromarray(u8).resize((nw, nh), Image.BICUBIC))


def vmaf_neg_batch(ref_y4m, dists_up, td, threads=4):
    ddir = os.path.join(td, "d")
    os.makedirs(ddir, exist_ok=True)
    for f in glob.glob(os.path.join(ddir, "*.png")):
        os.remove(f)
    dy4m = os.path.join(td, "dist.y4m")
    vm._png_seq_to_y4m(dists_up, ddir, dy4m)
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = vm._LD
    out_json = os.path.join(td, "neg.json")
    cmd = [vm._VMAF_BIN, "--reference", ref_y4m, "--distorted", dy4m,
           "--model", "path=" + vm._MODEL_NEG, "--threads", str(threads),
           "--output", out_json, "--json"]
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.loads(open(out_json).read())
    out = []
    for fr in d["frames"]:
        m = fr["metrics"]
        out.append(float(m[next(x for x in m if x.startswith("vmaf"))]))
    return out


def seeds():
    """identity + classic shapes (natural-raster index r*8+c)."""
    idx = np.arange(64); r, c = idx // 8, idx % 8; rc = r + c
    S = [np.ones(64)]
    for cut, lo in [(4, 0.5), (6, 0.5), (8, 0.5), (4, 0.75), (6, 0.75)]:
        s = np.ones(64); s[rc >= cut] = lo; S.append(s)
    s = np.ones(64); s[rc >= 5] = 1.2; S.append(s)        # mild HF boost (anti-seed)
    s = np.full(64, 0.9); s[0] = 1.0; S.append(s)         # global soften
    s = np.ones(64) - 0.06 * rc; S.append(np.clip(s, 0.3, 1.5))  # linear rolloff
    return S


def run_job(args):
    img_idx, path, q, base_curve, kappa, gens, lam_pop, seed = args
    rng = np.random.default_rng(seed)
    rgb = load16(path)
    y = luma(rgb); coeffs = block_dct(y)
    npix = rgb.shape[0] * rgb.shape[1]
    _, bits0 = jpeg_rt(rgb, q)
    bpp0 = bits0 / npix
    with tempfile.TemporaryDirectory() as td:
        # ref y4m built once (repeated frames = candidate count, fixed pop size)
        ref_up = up1080(np.rint(rgb).astype(np.uint8))
        rdir = os.path.join(td, "r"); os.makedirs(rdir)
        ry4m = os.path.join(td, "ref.y4m")

        pop = seeds()
        while len(pop) < lam_pop:
            pop.append(np.clip(1.0 + rng.normal(0, 0.1, 64), 0.3, 1.5))
        pop = [np.clip(p, 0.3, 1.5) for p in pop[:lam_pop]]
        vm._png_seq_to_y4m([ref_up] * lam_pop, rdir, ry4m)

        sigma = 0.12
        best = {"fit": -1e9}
        for g in range(gens):
            decs, bpps = [], []
            for s in pop:
                dec, bits = jpeg_rt(apply_s(rgb, coeffs, y, s), q)
                decs.append(up1080(np.rint(dec).astype(np.uint8)))
                bpps.append(bits / npix)
            vs = vmaf_neg_batch(ry4m, decs, td)
            fits = [v - kappa * (b - bpp0) / bpp0 * 100.0 for v, b in zip(vs, bpps)]
            order = np.argsort(fits)[::-1]
            if fits[order[0]] > best["fit"]:
                i = order[0]
                best = {"fit": float(fits[i]), "vmaf_neg": float(vs[i]),
                        "bpp": float(bpps[i]), "s": pop[i].tolist(), "gen": g}
            elites = [pop[i] for i in order[:8]]
            newpop = list(elites)
            while len(newpop) < lam_pop:
                e = elites[rng.integers(len(elites))]
                if rng.random() < 0.25:   # sparse: perturb 8 random bands
                    m = e.copy(); ii = rng.integers(0, 64, 8)
                    m[ii] += rng.normal(0, sigma * 2, 8)
                else:
                    m = e + rng.normal(0, sigma, 64)
                newpop.append(np.clip(m, 0.3, 1.5))
            pop = newpop
            sigma *= 0.93
    # verdict: iso-bpp delta vs per-image baseline curve
    bb, vb = base_curve
    iso = float(np.interp(best["bpp"], bb, vb))
    # baseline self-point: vmaf_neg of unmodified encode at q (from curve at bpp0)
    iso0 = float(np.interp(bpp0, bb, vb))
    res = {"img": img_idx, "q": q, "bpp_base": float(bpp0), "vmafneg_base@q": iso0,
           "best_vmaf_neg": best["vmaf_neg"], "best_bpp": best["bpp"],
           "iso_bpp_delta": best["vmaf_neg"] - iso, "fit": best["fit"],
           "gen": best["gen"], "s": best["s"]}
    print(f"[img{img_idx:02d} q{q}] base {iso0:.2f}@{bpp0:.4f} -> oracle "
          f"{best['vmaf_neg']:.2f}@{best['bpp']:.4f} | iso-bpp delta = "
          f"{res['iso_bpp_delta']:+.2f} (gen {best['gen']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-imgs", type=int, default=10)
    ap.add_argument("--qualities", default="8,20")
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="/workspace/sandwiched_compression/dpp/runs/oracle_vmafneg.json")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(VAL_DIR, "*.png")))[: a.n_imgs]
    base = json.load(open(BASE_JSON))["results"]["baseline"]
    jobs = []
    for q in [int(x) for x in a.qualities.split(",")]:
        for i, p in enumerate(paths):
            bb = np.array([r["per_img"]["bpp"][i] for r in base])
            vb = np.array([r["per_img"]["vmaf_neg"][i] for r in base])
            o = np.argsort(bb); bb, vb = bb[o], vb[o]
            # per-image local slope (vmaf_neg pts per 1% bpp) around the q point
            qs = [r["quality"] for r in base]
            lo_i, hi_i = (0, 2) if q <= 8 else (2, 4)   # (q5,q12) or (q12,q32)
            blo, bhi = [r["per_img"]["bpp"][i] for r in (base[lo_i], base[hi_i])]
            vlo, vhi = [r["per_img"]["vmaf_neg"][i] for r in (base[lo_i], base[hi_i])]
            kappa = (vhi - vlo) / max((bhi - blo) / blo * 100.0, 1e-6)
            jobs.append((i, paths[i], q, (bb, vb), float(kappa), a.gens, a.pop,
                         1000 + 97 * i + q))
    print(f"{len(jobs)} oracle jobs (imgs={len(paths)}, gens={a.gens}, pop={a.pop})",
          flush=True)
    with Pool(a.workers) as pool:
        results = pool.map(run_job, jobs)
    deltas = np.array([r["iso_bpp_delta"] for r in results])
    print("\n=== ORACLE VERDICT (iso-bpp VMAF_NEG delta, >0 = preprocessing CAN win) ===")
    for q in sorted(set(r["q"] for r in results)):
        d = np.array([r["iso_bpp_delta"] for r in results if r["q"] == q])
        print(f"  q={q}: mean {d.mean():+.2f}  median {np.median(d):+.2f}  "
              f"min {d.min():+.2f}  max {d.max():+.2f}  positive {int((d > 0.5).sum())}/{len(d)}")
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
