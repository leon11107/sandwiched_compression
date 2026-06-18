"""Part 2 first-cut: the preprocessor's BD-rate ceiling under the joint
constraint (iso-rate PSNR drop <= 0.5 dB AND VMAF_NEG gain AND MS-SSIM gain).

Sweeps the EDIT-STRENGTH axis (a base precoder's edit scaled 0.5x..2x) on the
held-out clips' hull, scoring 4 metrics (PSNR, MS-SSIM, VMAF_NEG, VMAF) so we
can see, as the precoder edits harder: how the perceptual gains rise and where
the PSNR cost crosses 0.5 dB. The best both-perceptual-win point still inside
the PSNR budget = the first-cut ceiling for this edit family. (A full per-content
oracle, if warranted, comes next.)

Reuses the S0/S3 x264 ladder + adds psnr & float_ms_ssim libvmaf features.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp_repro.s0_hull import (CLAMP, FRAMES, MODEL_NEG, MODEL_STD, VMAF_LD,
                               WORK, front, kbps, sh)
from dpp_repro.y4m import read_y4m, y4m_header
from dpp_repro.s1_train import Precoder
from dpp.bd_ci import bd

RUNS = "/workspace/sandwiched_compression/dpp/runs"
HELD = ["aspen_1080p", "red_kayak_1080p", "west_wind_easy_1080p", "controlled_burn_1080p"]
HEIGHTS = [1080, 720, 540, 432, 360, 288, 216, 144]
WH = {1080: 1920, 720: 1280, 540: 960, 432: 768, 360: 640, 288: 512, 216: 384, 144: 256}
CRFS = [18, 22, 26, 30, 34, 38, 42]
CLAMP2 = dict(CLAMP, psnr=(20, 60), ms_ssim=(0.80, 0.999))


def precode(clip, name, dev, precoders):
    """Write a precoded-luma y4m (replace Y, keep UV). name=identity|model:scale."""
    out = f"{WORK}/src/{clip}__{name.replace(':','_')}.y4m"
    if os.path.exists(out) or name == "identity":
        return out if name != "identity" else f"{WORK}/src/{clip}.y4m"
    mname, scale = (name.split(":") + ["1"])[:2]
    pre = precoders[mname]; s = float(scale)
    src = f"{WORK}/src/{clip}.y4m"
    with open(out, "wb") as f, torch.no_grad():
        f.write(y4m_header(src))
        for y, u, v in read_y4m(src):
            t = torch.from_numpy(y.astype(np.float32)).to(dev)[None, None]
            edit = pre(t) - t
            p = (t + s * edit).clamp(0, 255).round().byte()[0, 0].cpu().numpy()
            f.write(b"FRAME\n"); f.write(p.tobytes()); f.write(u.tobytes()); f.write(v.tobytes())
    print(f"[precode] {clip} {name}", flush=True)
    return out


def enc_score(args):
    src, clip, name, h, crf = args
    tag = name.replace(":", "_")
    sj = f"{WORK}/score/{clip}__{tag}_{h}_{crf}.json"
    if os.path.exists(sj):
        return
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        mp4 = os.path.join(td, "e.mp4"); dec = os.path.join(td, "d.y4m")
        vf = f"scale={WH[h]}:{h}:flags=lanczos" if h != 1080 else "null"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf,
            "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
            "-crf", str(crf), "-refs", "5", "-g", "150", "-keyint_min", "150",
            "-sc_threshold", "0", "-tune", "ssim", "-x264opts", "ssim=1", mp4])
        kb = kbps(mp4)
        vfu = "scale=1920:1080:flags=bicubic" if h != 1080 else "null"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vfu,
            "-pix_fmt", "yuv420p", dec])
        env = dict(os.environ); env["LD_LIBRARY_PATH"] = VMAF_LD
        base = ["vmaf", "--reference", f"{WORK}/src/{clip}.y4m", "--distorted",
                dec, "--threads", "4", "--json"]
        o1, o2 = os.path.join(td, "1.json"), os.path.join(td, "2.json")
        sh(base + ["--model", "path=" + MODEL_STD, "--feature", "float_ssim",
                   "--feature", "psnr", "--feature", "float_ms_ssim",
                   "--output", o1], env=env)
        sh(base + ["--model", "path=" + MODEL_NEG, "--output", o2], env=env)
        p1 = json.loads(open(o1).read())["pooled_metrics"]
        p2 = json.loads(open(o2).read())["pooled_metrics"]

    def g(pm, pref):
        return pm[next(k for k in pm if k.startswith(pref))]["mean"]
    json.dump({"clip": clip, "name": name, "h": h, "crf": crf, "kbps": kb,
               "vmaf": g(p1, "vmaf"), "ssim": g(p1, "float_ssim") * 100.0,
               "ms_ssim": g(p1, "float_ms_ssim"), "psnr": g(p1, "psnr"),
               "vmaf_neg": g(p2, "vmaf")}, open(sj, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="g2", help="base precoder ckpt key")
    ap.add_argument("--scales", default="0.5,1,1.5,2")
    ap.add_argument("--extra", default="g0.5_big,g0,g5",
                    help="extra single-strength candidates (model keys)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--enc-workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(RUNS, "s5_ceiling.json"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    keys = {a.base, *[e for e in a.extra.split(",") if e]}
    ck = {"g2": "s2_lam0.01_g2_big", "g0.5_big": "s2_lam0.01_g0.5_big",
          "g0": "s2_lam0.01_g0", "g5": "s2_lam0.01_g5",
          "g0.5": "s2_lam0.01_g0.5", "g2_big": "s2_lam0.01_g2_big"}
    precoders = {}
    for k in keys:
        pre = Precoder().to(dev).eval()
        pre.load_state_dict(torch.load(os.path.join(RUNS, ck[k], "model.pt"),
                                       map_location=dev)["pre"])
        precoders[k] = pre
    cands = ["identity"] + [f"{a.base}:{s}" for s in a.scales.split(",")] + \
            [e for e in a.extra.split(",") if e]

    srcs = {}
    for clip in HELD:
        for name in cands:
            srcs[(clip, name)] = precode(clip, name, dev, precoders)
    conds = [(srcs[(c, n)], c, n, h, q) for c in HELD for n in cands
             for h in HEIGHTS for q in CRFS]
    with ThreadPoolExecutor(a.enc_workers) as ex:
        list(ex.map(enc_score, conds))
    print("scoring done", flush=True)

    rows = {}
    for c, n in [(c, n) for c in HELD for n in cands]:
        tag = n.replace(":", "_")
        rows[(c, n)] = [json.load(open(f"{WORK}/score/{c}__{tag}_{h}_{q}.json"))
                        for h in HEIGHTS for q in CRFS]

    out = {"base": a.base, "cands": cands, "bd": {}, "psnr_drop": {}}
    METRICS = ["psnr", "ms_ssim", "vmaf_neg", "vmaf", "ssim"]
    for name in cands:
        out["bd"][name] = {}
        for met in METRICS:
            lo, hi = CLAMP2[met]
            bds = []
            for c in HELD:
                br = rows[(c, "identity")]; cr = rows[(c, name)]
                bh = front([(r["kbps"], r[met]) for r in br if lo <= r[met] <= hi])
                dh = front([(r["kbps"], r[met]) for r in cr if lo <= r[met] <= hi])
                if len(bh) >= 4 and len(dh) >= 4:
                    bds.append(bd([p[0] for p in bh], [p[1] for p in bh],
                                  [p[0] for p in dh], [p[1] for p in dh]))
            out["bd"][name][met] = float(np.nanmean(bds))
        # iso-rate PSNR drop (dB) in the USABLE quality band (cand VMAF in [40,96]):
        # at each cand op, baseline psnr interpolated at the cand's bitrate.
        drops = []
        for c in HELD:
            br = rows[(c, "identity")]; cr = rows[(c, name)]
            bb = np.array([r["kbps"] for r in br]); bp = np.array([r["psnr"] for r in br])
            o = np.argsort(bb)
            for r in cr:
                if 40 <= r["vmaf"] <= 96:
                    drops.append(float(np.interp(r["kbps"], bb[o], bp[o])) - r["psnr"])
        out["psnr_drop"][name] = float(np.nanmax(drops)) if drops else float("nan")
        out.setdefault("psnr_drop_med", {})[name] = (
            float(np.nanmedian(drops)) if drops else float("nan"))
        b = out["bd"][name]
        ok = (out["psnr_drop"][name] <= 0.5 and b["vmaf_neg"] < 0 and b["ms_ssim"] < 0)
        print(f"{name:12s} PSNRbd {b['psnr']:+6.2f} (drop max {out['psnr_drop'][name]:.2f} "
              f"med {out['psnr_drop_med'][name]:.2f}dB) MSSSIM {b['ms_ssim']:+6.2f} "
              f"AHVMAF {b['vmaf_neg']:+6.2f} VMAF {b['vmaf']:+6.2f}"
              f"  {'<= CONSTRAINT-OK' if ok else ''}", flush=True)

    # first-cut ceiling = best (vmaf_neg, ms_ssim) among constraint-satisfying cands
    feasible = [n for n in cands if out["psnr_drop"][n] <= 0.5
                and out["bd"][n]["vmaf_neg"] < 0 and out["bd"][n]["ms_ssim"] < 0]
    if feasible:
        best = min(feasible, key=lambda n: out["bd"][n]["vmaf_neg"] + out["bd"][n]["ms_ssim"])
        print(f"\nFIRST-CUT CEILING (PSNRdrop<=0.5 & both perceptual win): {best} -> "
              f"AHVMAF {out['bd'][best]['vmaf_neg']:+.2f} MSSSIM {out['bd'][best]['ms_ssim']:+.2f} "
              f"VMAF {out['bd'][best]['vmaf']:+.2f} (PSNR maxdrop {out['psnr_drop'][best]:.2f}dB)",
              flush=True)
    else:
        print("\nNO candidate satisfies PSNRdrop<=0.5 & both perceptual win", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
