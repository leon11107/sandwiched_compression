"""S0 of the DPP reproduction (reference/DPP/115100C.pdf S4.2): baseline
x264 evaluation machinery on XIPH clips + the protocol-bonus question.

Per clip: encode a ladder of 8 resolutions x 7 CRF with the paper recipe
(x264 slow, -tune ssim, refs 5, g/keyint 150, sc_threshold 0, lanczos down),
decode + bicubic upscale to 1080p, score VMAF + AH-VMAF (NEG model == our
VMAF_NEG) + SSIM (float_ssim) against the 150-frame 1080p source in ONE vmaf
run (two --model + feature). Then per metric: fixed-1080p CRF curve vs the
full-ladder convex hull -> BD-rate = how much of the paper's gain protocol
freedom alone provides (no precoder). Paper quality clamps: 40<=VMAF<=96,
88<=SSIM(x100)<=99 applied to both arms before BD.

All bulky data lives in /dev/shm/dppv (tmpfs); results -> dpp/runs/.
Resume-safe per (clip, res, crf) score json.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp.bd_ci import bd

WORK = "/dev/shm/dppv"
RUNS = "/workspace/sandwiched_compression/dpp/runs"
XIPH = "https://media.xiph.org/video/derf/y4m"
CLIPS = ["blue_sky_1080p25", "pedestrian_area_1080p25", "riverbed_1080p25",
         "rush_hour_1080p25", "sunflower_1080p25", "tractor_1080p25"]
FRAMES = 150
HEIGHTS = [1080, 720, 540, 432, 360, 288, 216, 144]
CRFS = [18, 22, 26, 30, 34, 38, 42]
MODEL_STD = "/workspace/sandwiched_compression/reference/vmaf_models/vmaf_v0.6.1.json"
MODEL_NEG = "/workspace/sandwiched_compression/reference/vmaf_models/vmaf_v0.6.1neg.json"
VMAF_LD = "/usr/local/lib/x86_64-linux-gnu"


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, **kw)


def ensure_src(clip):
    dst = f"{WORK}/src/{clip}.y4m"
    if not os.path.exists(dst):
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{XIPH}/{clip}.y4m",
            "-frames:v", str(FRAMES), "-pix_fmt", "yuv420p", dst])
        print(f"[src] {clip} downloaded", flush=True)
    return dst


def enc_path(clip, h, crf):
    return f"{WORK}/enc/{clip}_{h}_{crf}.mp4"


def encode(args):
    clip, h, crf = args
    out = enc_path(clip, h, crf)
    if os.path.exists(out):
        return
    w = {1080: 1920, 720: 1280, 540: 960, 432: 768, 360: 640, 288: 512,
         216: 384, 144: 256}[h]
    vf = f"scale={w}:{h}:flags=lanczos" if h != 1080 else "null"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{WORK}/src/{clip}.y4m",
        "-vf", vf, "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
        "-crf", str(crf), "-refs", "5", "-g", "150", "-keyint_min", "150",
        "-sc_threshold", "0", "-tune", "ssim", "-x264opts", "ssim=1", out])
    print(f"[enc] {clip} {h}p crf{crf}", flush=True)


def kbps(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-show_entries", "packet=size", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    bits = sum(int(x) for x in r.stdout.split() if x) * 8
    return bits / (FRAMES / 25.0) / 1000.0


def score(args):
    clip, h, crf = args
    sj = f"{WORK}/score/{clip}_{h}_{crf}.json"
    if os.path.exists(sj):
        return
    enc = enc_path(clip, h, crf)
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        dist = os.path.join(td, "d.y4m")
        vf = ("scale=1920:1080:flags=bicubic" if h != 1080 else "null")
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-vf", vf,
            "-pix_fmt", "yuv420p", dist])
        env = dict(os.environ); env["LD_LIBRARY_PATH"] = VMAF_LD
        base = ["vmaf", "--reference", f"{WORK}/src/{clip}.y4m",
                "--distorted", dist, "--threads", "4", "--json"]
        oj1, oj2 = os.path.join(td, "v1.json"), os.path.join(td, "v2.json")
        sh(base + ["--model", "path=" + MODEL_STD, "--feature", "float_ssim",
                   "--output", oj1], env=env)
        sh(base + ["--model", "path=" + MODEL_NEG, "--output", oj2], env=env)
        p1 = json.loads(open(oj1).read())["pooled_metrics"]
        p2 = json.loads(open(oj2).read())["pooled_metrics"]
    row = {"clip": clip, "h": h, "crf": crf, "kbps": kbps(enc),
           "vmaf": p1["vmaf"]["mean"],
           "vmaf_neg": p2[next(k for k in p2 if k.startswith("vmaf"))]["mean"],
           "ssim": p1["float_ssim"]["mean"] * 100.0}
    json.dump(row, open(sj, "w"))
    print(f"[score] {clip} {h}p crf{crf}: vmaf {row['vmaf']:.1f} "
          f"neg {row['vmaf_neg']:.1f} ssim {row['ssim']:.2f} "
          f"{row['kbps']:.0f}kbps", flush=True)


CLAMP = {"vmaf": (40, 96), "vmaf_neg": (40, 96), "ssim": (88, 99)}


def front(points):
    """monotone Pareto front (kbps asc, score strictly asc), then convex hull."""
    pts = sorted(points)
    f, best = [], -np.inf
    for r, s in pts:
        if s > best + 1e-9:
            f.append((r, s)); best = s
    if len(f) <= 2:
        return f
    h = []
    for p in f:
        while len(h) >= 2 and \
                (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - \
                (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= 0:
            h.pop()
        h.append(p)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--heights", default=",".join(map(str, HEIGHTS)))
    ap.add_argument("--crfs", default=",".join(map(str, CRFS)))
    ap.add_argument("--enc-workers", type=int, default=24)
    ap.add_argument("--score-workers", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(RUNS, "s0_baseline_hull.json"))
    a = ap.parse_args()
    clips = a.clips.split(",")
    heights = [int(x) for x in a.heights.split(",")]
    crfs = [int(x) for x in a.crfs.split(",")]
    for d in ("src", "enc", "score"):
        os.makedirs(f"{WORK}/{d}", exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(ensure_src, clips))
    conds = [(c, h, q) for c in clips for h in heights for q in crfs]
    with ThreadPoolExecutor(a.enc_workers) as ex:
        list(ex.map(encode, conds))
    print(f"encodes done ({len(conds)})", flush=True)
    with ThreadPoolExecutor(a.score_workers) as ex:
        list(ex.map(score, conds))
    print("scoring done", flush=True)

    rows = [json.load(open(f"{WORK}/score/{c}_{h}_{q}.json"))
            for c, h, q in conds]
    out = {"rows": rows, "bd_hull_vs_1080": {}}
    for met in ("vmaf", "vmaf_neg", "ssim"):
        lo, hi = CLAMP[met]
        bds = {}
        for c in clips:
            cr = [r for r in rows if r["clip"] == c]
            fixed = front([(r["kbps"], r[met]) for r in cr if r["h"] == 1080
                           and lo <= r[met] <= hi])
            hull = front([(r["kbps"], r[met]) for r in cr
                          if lo <= r[met] <= hi])
            if len(fixed) >= 4 and len(hull) >= 4:
                bds[c] = bd([p[0] for p in fixed], [p[1] for p in fixed],
                            [p[0] for p in hull], [p[1] for p in hull])
        out["bd_hull_vs_1080"][met] = bds
        v = [x for x in bds.values() if np.isfinite(x)]
        print(f"PROTOCOL BONUS {met}: ladder-hull vs fixed-1080p BD-rate "
              f"mean {np.mean(v):+.2f}% (per-clip "
              + " ".join(f"{k.split('_1080')[0]}:{x:+.1f}" for k, x in bds.items())
              + ")", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
