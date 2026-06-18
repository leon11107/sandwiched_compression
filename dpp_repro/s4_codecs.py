"""S4: cross-codec BD check (x265 / AV1) on held-out clips, reusing the
already-precoded sources (clip__g05.y4m, clip__g2.y4m in /dev/shm/dppv/src).
Paper's AV1 arm used aomenc 2-pass target-bitrate; we use single-pass CRF
(declared deviation). x265 is our extra (not in the paper).
Reports pooled (g05+g2, paper-style 2-model hull) + per-tag BDs per metric.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp_repro.s0_hull import (CLAMP, FRAMES, MODEL_NEG, MODEL_STD, VMAF_LD,
                               WORK, front, kbps, sh)
from dpp.bd_ci import bd

RUNS = "/workspace/sandwiched_compression/dpp/runs"
HELD = ["aspen_1080p", "red_kayak_1080p", "west_wind_easy_1080p",
        "controlled_burn_1080p"]
WH = {1080: 1920, 720: 1280, 540: 960, 432: 768, 360: 640, 288: 512,
      216: 384, 144: 256}
CRFS = {"x265": [18, 22, 26, 30, 34, 38, 42],
        "av1": [20, 26, 32, 38, 44, 50, 56]}


def encode(args):
    codec, clip, arm, h, crf = args
    out = f"{WORK}/enc/{clip}__{arm}_{codec}_{h}_{crf}.mp4"
    if os.path.exists(out):
        return
    src = f"{WORK}/src/{clip}.y4m" if arm == "base" else \
        f"{WORK}/src/{clip}__{arm}.y4m"
    vf = f"scale={WH[h]}:{h}:flags=lanczos" if h != 1080 else "null"
    if codec == "x265":
        enc = ["-c:v", "libx265", "-preset", "slow", "-crf", str(crf),
               "-tune", "ssim", "-x265-params",
               "keyint=150:min-keyint=150:scenecut=0"]
    else:
        enc = ["-c:v", "libaom-av1", "-cpu-used", "5", "-crf", str(crf),
               "-b:v", "0", "-g", "150", "-threads", "4", "-row-mt", "1"]
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf] + enc + [out])
    print(f"[enc] {codec} {clip} {arm} {h}p crf{crf}", flush=True)


def score(args):
    codec, clip, arm, h, crf = args
    sj = f"{WORK}/score/{clip}__{arm}_{codec}_{h}_{crf}.json"
    if os.path.exists(sj):
        return
    enc = f"{WORK}/enc/{clip}__{arm}_{codec}_{h}_{crf}.mp4"
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        dist = os.path.join(td, "d.y4m")
        vf = "scale=1920:1080:flags=bicubic" if h != 1080 else "null"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-vf", vf,
            "-pix_fmt", "yuv420p", dist])
        env = dict(os.environ); env["LD_LIBRARY_PATH"] = VMAF_LD
        b = ["vmaf", "--reference", f"{WORK}/src/{clip}.y4m", "--distorted",
             dist, "--threads", "4", "--json"]
        o1, o2 = os.path.join(td, "1.json"), os.path.join(td, "2.json")
        sh(b + ["--model", "path=" + MODEL_STD, "--feature", "float_ssim",
                "--output", o1], env=env)
        sh(b + ["--model", "path=" + MODEL_NEG, "--output", o2], env=env)
        p1 = json.loads(open(o1).read())["pooled_metrics"]
        p2 = json.loads(open(o2).read())["pooled_metrics"]
    json.dump({"clip": clip, "arm": arm, "codec": codec, "h": h, "crf": crf,
               "kbps": kbps(enc), "vmaf": p1["vmaf"]["mean"],
               "vmaf_neg": p2[next(k for k in p2 if k.startswith("vmaf"))]["mean"],
               "ssim": p1["float_ssim"]["mean"] * 100.0}, open(sj, "w"))
    print(f"[score] {codec} {clip} {arm} {h}p crf{crf}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", choices=["x265", "av1"], required=True)
    ap.add_argument("--enc-workers", type=int, default=10)
    ap.add_argument("--score-workers", type=int, default=6)
    a = ap.parse_args()
    conds = [(a.codec, c, arm, h, q) for c in HELD
             for arm in ("base", "g05", "g2") for h in WH for q in CRFS[a.codec]]
    with ThreadPoolExecutor(a.enc_workers) as ex:
        list(ex.map(encode, conds))
    print("encodes done", flush=True)
    with ThreadPoolExecutor(a.score_workers) as ex:
        list(ex.map(score, conds))
    print("scoring done", flush=True)

    rows = [json.load(open(f"{WORK}/score/{c}__{arm}_{a.codec}_{h}_{q}.json"))
            for _, c, arm, h, q in conds]
    out = {"codec": a.codec, "bd": {}}
    for met in ("ssim", "vmaf_neg", "vmaf"):
        lo, hi = CLAMP[met]
        for arms, label in ((("g05", "g2"), "pooled"), (("g05",), "g05"),
                            (("g2",), "g2")):
            bds = []
            for c in HELD:
                bh = front([(r["kbps"], r[met]) for r in rows
                            if r["clip"] == c and r["arm"] == "base"
                            and lo <= r[met] <= hi])
                dh = front([(r["kbps"], r[met]) for r in rows
                            if r["clip"] == c and r["arm"] in arms
                            and lo <= r[met] <= hi])
                if len(bh) >= 4 and len(dh) >= 4:
                    bds.append(bd([p[0] for p in bh], [p[1] for p in bh],
                                  [p[0] for p in dh], [p[1] for p in dh]))
            out["bd"].setdefault(met, {})[label] = float(np.nanmean(bds))
            print(f"S4 {a.codec} {met} [{label}]: {np.nanmean(bds):+.2f}%",
                  flush=True)
    json.dump(out, open(os.path.join(RUNS, f"s4_{a.codec}.json"), "w"), indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
