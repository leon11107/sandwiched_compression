"""S3: hull-vs-hull BD against paper Table 1 (115100C.pdf).

Per precoder ckpt: precode each clip's LUMA at full 1080p (replace Y, keep
UV), then run the S0 ladder (8 res x 7 CRF, x264 slow recipe) on the precoded
clip, score against the ORIGINAL reference, and compare per-clip convex hulls:
  BD( baseline-hull -> DPO-hull )  per metric (VMAF / AH-VMAF / SSIM),
with the paper quality clamps. The DPO hull pools BOTH model variants
(enh0m=lam0.01, enh3m=lam0.001) exactly as the paper does; baseline rows come
from the S0 run (same /dev/shm/dppv/score cache).
Paper Table 1 averages (x264 slow): SSIM -4.67 / AH-VMAF -12.27 / VMAF -25.08.
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp_repro.s0_hull import (CLAMP, CLIPS, CRFS, FRAMES, HEIGHTS, MODEL_NEG,
                               MODEL_STD, VMAF_LD, WORK, front, kbps, sh)
from dpp_repro.s1_train import Precoder
from dpp.bd_ci import bd

RUNS = "/workspace/sandwiched_compression/dpp/runs"


def precode_clip(clip, pre, dev, tag):
    """Original y4m -> precoded-luma y4m (UV untouched)."""
    out = f"{WORK}/src/{clip}__{tag}.y4m"
    if os.path.exists(out):
        return out
    from dpp_repro.y4m import read_y4m, y4m_header
    src = f"{WORK}/src/{clip}.y4m"
    # write the container ourselves (ffmpeg rawvideo->y4m range-converts)
    with open(out, "wb") as f, torch.no_grad():
        f.write(y4m_header(src))
        for y_, u_, v_ in read_y4m(src):
            y = torch.from_numpy(y_.astype(np.float32)).to(dev)[None, None]
            p = pre(y).clamp(0, 255).round().byte().cpu().numpy()[0, 0]
            f.write(b"FRAME\n")
            f.write(p.tobytes()); f.write(u_.tobytes()); f.write(v_.tobytes())
    print(f"[precode] {clip} ({tag})", flush=True)
    return out


def encode(args):
    src, clip, h, crf, tag = args
    out = f"{WORK}/enc/{clip}__{tag}_{h}_{crf}.mp4"
    if os.path.exists(out):
        return
    w = {1080: 1920, 720: 1280, 540: 960, 432: 768, 360: 640, 288: 512,
         216: 384, 144: 256}[h]
    vf = f"scale={w}:{h}:flags=lanczos" if h != 1080 else "null"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf,
        "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
        "-crf", str(crf), "-refs", "5", "-g", "150", "-keyint_min", "150",
        "-sc_threshold", "0", "-tune", "ssim", "-x264opts", "ssim=1", out])


def score(args):
    import tempfile
    clip, h, crf, tag = args
    sj = f"{WORK}/score/{clip}__{tag}_{h}_{crf}.json"
    if os.path.exists(sj):
        return
    enc = f"{WORK}/enc/{clip}__{tag}_{h}_{crf}.mp4"
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        dist = os.path.join(td, "d.y4m")
        vf = "scale=1920:1080:flags=bicubic" if h != 1080 else "null"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-vf", vf,
            "-pix_fmt", "yuv420p", dist])
        env = dict(os.environ); env["LD_LIBRARY_PATH"] = VMAF_LD
        base = ["vmaf", "--reference", f"{WORK}/src/{clip}.y4m",  # ORIGINAL ref
                "--distorted", dist, "--threads", "4", "--json"]
        oj1, oj2 = os.path.join(td, "v1.json"), os.path.join(td, "v2.json")
        sh(base + ["--model", "path=" + MODEL_STD, "--feature", "float_ssim",
                   "--output", oj1], env=env)
        sh(base + ["--model", "path=" + MODEL_NEG, "--output", oj2], env=env)
        p1 = json.loads(open(oj1).read())["pooled_metrics"]
        p2 = json.loads(open(oj2).read())["pooled_metrics"]
    row = {"clip": clip, "h": h, "crf": crf, "tag": tag, "kbps": kbps(enc),
           "vmaf": p1["vmaf"]["mean"],
           "vmaf_neg": p2[next(k for k in p2 if k.startswith("vmaf"))]["mean"],
           "ssim": p1["float_ssim"]["mean"] * 100.0}
    json.dump(row, open(sj, "w"))
    print(f"[score] {clip} {tag} {h}p crf{crf}: neg {row['vmaf_neg']:.1f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="tag=path comma list, e.g. enh0m=runs/s2_lam0.01/model.pt")
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--heights", default=",".join(map(str, HEIGHTS)))
    ap.add_argument("--crfs", default=",".join(map(str, CRFS)))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--enc-workers", type=int, default=20)
    ap.add_argument("--score-workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(RUNS, "s3_eval.json"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    clips = a.clips.split(",")
    heights = [int(x) for x in a.heights.split(",")]
    crfs = [int(x) for x in a.crfs.split(",")]
    tags = []
    for spec in a.ckpts.split(","):
        tag, path = spec.split("=")
        pre = Precoder().to(dev).eval()
        pre.load_state_dict(torch.load(path, map_location=dev)["pre"])
        for c in clips:
            precode_clip(c, pre, dev, tag)
        tags.append(tag)

    conds = [(c, h, q, t) for c in clips for h in heights for q in crfs
             for t in tags]
    with ThreadPoolExecutor(a.enc_workers) as ex:
        list(ex.map(encode, [(f"{WORK}/src/{c}__{t}.y4m", c, h, q, t)
                             for c, h, q, t in conds]))
    print("encodes done", flush=True)
    with ThreadPoolExecutor(a.score_workers) as ex:
        list(ex.map(score, conds))
    print("scoring done", flush=True)

    rows = [json.load(open(f"{WORK}/score/{c}__{t}_{h}_{q}.json"))
            for c, h, q, t in conds]
    base_rows = [json.load(open(f"{WORK}/score/{c}_{h}_{q}.json"))
                 for c in clips for h in heights for q in crfs]
    out = {"tags": tags, "bd": {}, "bd_per_tag": {}}
    for met in ("ssim", "vmaf_neg", "vmaf"):
        lo, hi = CLAMP[met]
        bds = {}
        for c in clips:
            bh = front([(r["kbps"], r[met]) for r in base_rows
                        if r["clip"] == c and lo <= r[met] <= hi])
            dh = front([(r["kbps"], r[met]) for r in rows
                        if r["clip"] == c and lo <= r[met] <= hi])
            if len(bh) >= 4 and len(dh) >= 4:
                bds[c] = bd([p[0] for p in bh], [p[1] for p in bh],
                            [p[0] for p in dh], [p[1] for p in dh])
        v = [x for x in bds.values() if np.isfinite(x)]
        out["bd"][met] = {"mean": float(np.mean(v)), "per_clip": bds}
        print(f"S3 {met}: DPO-hull vs baseline-hull BD mean {np.mean(v):+.2f}% "
              + " ".join(f"{k.split('_1080')[0]}:{x:+.1f}" for k, x in bds.items()),
              flush=True)
        for t in tags:  # single-model arms (no pooling)
            tb = []
            for c in clips:
                bh = front([(r["kbps"], r[met]) for r in base_rows
                            if r["clip"] == c and lo <= r[met] <= hi])
                dh = front([(r["kbps"], r[met]) for r in rows
                            if r["clip"] == c and r["tag"] == t
                            and lo <= r[met] <= hi])
                if len(bh) >= 4 and len(dh) >= 4:
                    tb.append(bd([p[0] for p in bh], [p[1] for p in bh],
                                 [p[0] for p in dh], [p[1] for p in dh]))
            out["bd_per_tag"].setdefault(t, {})[met] = float(np.nanmean(tb))
            print(f"  tag {t}: {met} {np.nanmean(tb):+.2f}%", flush=True)
    print("paper Table1 slow: ssim -4.67 / vmaf_neg(AH) -12.27 / vmaf -25.08",
          flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
