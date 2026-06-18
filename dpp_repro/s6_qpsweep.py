"""Fixed-QP eval (NO resolution ladder): trained preprocessor vs no-preprocessor,
at native 1080p, constant-QP {22,27,32,37}, on x264 / x265 / AV1.

Arms: baseline (original) | g2_big | g0.5_big (precoded sources reused from
/dev/shm if present, else generated from the ckpt). Per (codec, arm, QP):
bitrate + PSNR / SSIM / MS-SSIM / VMAF / VMAF_NEG (libvmaf, native res, no
scaling). BD-rate per (codec, arm) vs that codec's baseline over the 4 QP points.
QP note: x264/x265 use constant-QP (-qp); libaom-AV1 has a different quantizer
scale, so it uses constant-quality CRF at the same numeric values (the
preproc-vs-baseline comparison is within-codec, so the scale choice is fair).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp_repro.y4m import read_y4m, y4m_header
from dpp_repro.s1_train import Precoder
from dpp.bd_ci import bd

WORK = "/dev/shm/dppv"
RUNS = "/workspace/sandwiched_compression/dpp/runs"
MODEL_STD = "/workspace/sandwiched_compression/reference/vmaf_models/vmaf_v0.6.1.json"
MODEL_NEG = "/workspace/sandwiched_compression/reference/vmaf_models/vmaf_v0.6.1neg.json"
VMAF_LD = "/usr/local/lib/x86_64-linux-gnu"
HELD = ["aspen_1080p", "red_kayak_1080p", "west_wind_easy_1080p", "controlled_burn_1080p"]
QPS = [22, 27, 32, 37]
# arm -> precoded-source suffix (None = baseline/original); ckpt for on-the-fly gen
ARMS = {"baseline": None, "g2_big": "g2_1", "g0.5_big": "g0.5_big"}
ARM_CKPT = {"g2_big": "s2_lam0.01_g2_big", "g0.5_big": "s2_lam0.01_g0.5_big"}


def sh(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)


def ensure_precoded(clip, arm, dev):
    if ARMS[arm] is None:
        return f"{WORK}/src/{clip}.y4m"
    out = f"{WORK}/src/{clip}__{ARMS[arm]}.y4m"
    if os.path.exists(out):
        return out
    pre = Precoder().to(dev).eval()
    pre.load_state_dict(torch.load(os.path.join(RUNS, ARM_CKPT[arm], "model.pt"),
                                   map_location=dev)["pre"])
    src = f"{WORK}/src/{clip}.y4m"
    with open(out, "wb") as f, torch.no_grad():
        f.write(y4m_header(src))
        for y, u, v in read_y4m(src):
            t = torch.from_numpy(y.astype(np.float32)).to(dev)[None, None]
            p = pre(t).clamp(0, 255).round().byte()[0, 0].cpu().numpy()
            f.write(b"FRAME\n"); f.write(p.tobytes()); f.write(u.tobytes()); f.write(v.tobytes())
    print(f"[precode] {clip} {arm}", flush=True)
    return out


def enc_cmd(codec, qp, out):
    common = ["-g", "150", "-keyint_min", "150"]
    if codec == "x264":
        return ["-c:v", "libx264", "-preset", "slow", "-qp", str(qp),
                "-tune", "ssim", "-sc_threshold", "0"] + common + [out]
    if codec == "x265":
        return ["-c:v", "libx265", "-preset", "slow",
                "-x265-params", f"qp={qp}:keyint=150:min-keyint=150:scenecut=0:tune=ssim",
                out]
    return ["-c:v", "libaom-av1", "-cpu-used", "5", "-crf", str(qp), "-b:v", "0",
            "-threads", "4", "-row-mt", "1"] + common + [out]


def kbps(path, nframes, fps=25.0):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-show_entries", "packet=size", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return sum(int(x) for x in r.stdout.split() if x) * 8 / (nframes / fps) / 1000.0


def enc_score(args):
    src, ref, clip, arm, codec, qp, nframes = args
    sj = f"{WORK}/score/qpsweep_{clip}_{arm}_{codec}_{qp}.json"
    if os.path.exists(sj):
        return
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        ext = "webm" if codec == "av1" else "mp4"
        enc = os.path.join(td, f"e.{ext}"); dec = os.path.join(td, "d.y4m")
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src] + enc_cmd(codec, qp, enc))
        kb = kbps(enc, nframes)
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-pix_fmt", "yuv420p", dec])
        env = dict(os.environ); env["LD_LIBRARY_PATH"] = VMAF_LD
        base = ["vmaf", "--reference", ref, "--distorted", dec, "--threads", "4", "--json"]
        o1, o2 = os.path.join(td, "1.json"), os.path.join(td, "2.json")
        sh(base + ["--model", "path=" + MODEL_STD, "--feature", "float_ssim",
                   "--feature", "psnr", "--feature", "float_ms_ssim", "--output", o1])
        sh(base + ["--model", "path=" + MODEL_NEG, "--output", o2])
        p1 = json.loads(open(o1).read())["pooled_metrics"]
        p2 = json.loads(open(o2).read())["pooled_metrics"]
    g = lambda pm, pref: pm[next(k for k in pm if k.startswith(pref))]["mean"]
    json.dump({"clip": clip, "arm": arm, "codec": codec, "qp": qp, "kbps": kb,
               "vmaf": g(p1, "vmaf"), "ssim": g(p1, "float_ssim") * 100.0,
               "ms_ssim": g(p1, "float_ms_ssim"), "psnr": g(p1, "psnr"),
               "vmaf_neg": g(p2, "vmaf")}, open(sj, "w"))
    print(f"[{codec} {clip} {arm} qp{qp}] {kb:.0f}kbps psnr{g(p1,'psnr'):.1f} "
          f"vmafneg{g(p2,'vmaf'):.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--codecs", default="x264,x265,av1")
    ap.add_argument("--out", default=os.path.join(RUNS, "s6_qpsweep.json"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    codecs = a.codecs.split(",")
    srcs, nfr = {}, {}
    for clip in HELD:
        nfr[clip] = sum(1 for _ in read_y4m(f"{WORK}/src/{clip}.y4m"))
        for arm in ARMS:
            srcs[(clip, arm)] = ensure_precoded(clip, arm, dev)
    conds = [(srcs[(c, arm)], f"{WORK}/src/{c}.y4m", c, arm, codec, qp, nfr[c])
             for c in HELD for arm in ARMS for codec in codecs for qp in QPS]
    print(f"{len(conds)} encodes ({len(HELD)} clips x {len(ARMS)} arms x "
          f"{len(codecs)} codecs x {len(QPS)} QP)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(enc_score, conds))
    print("scoring done", flush=True)

    rows = {}
    for c in HELD:
        for arm in ARMS:
            for codec in codecs:
                rows[(c, arm, codec)] = [
                    json.load(open(f"{WORK}/score/qpsweep_{c}_{arm}_{codec}_{q}.json"))
                    for q in QPS]
    METRICS = ["psnr", "ssim", "ms_ssim", "vmaf_neg", "vmaf"]
    out = {"qps": QPS, "codecs": codecs, "arms": list(ARMS), "bd": {}, "raw": {}}
    for codec in codecs:
        out["bd"][codec] = {}
        for arm in ARMS:
            if arm == "baseline":
                continue
            per = {}
            for met in METRICS:
                bds = []
                for c in HELD:
                    br = rows[(c, "baseline", codec)]; cr = rows[(c, arm, codec)]
                    bds.append(bd([r["kbps"] for r in br], [r[met] for r in br],
                                  [r["kbps"] for r in cr], [r[met] for r in cr]))
                per[met] = float(np.nanmean(bds))
            out["bd"][codec][arm] = per
            print(f"[{codec} {arm}] BD-rate %% (neg=win): "
                  f"PSNR {per['psnr']:+.2f}  SSIM {per['ssim']:+.2f}  "
                  f"MS-SSIM {per['ms_ssim']:+.2f}  VMAF_NEG {per['vmaf_neg']:+.2f}  "
                  f"VMAF {per['vmaf']:+.2f}", flush=True)
    # raw per-QP means across clips (for the curves)
    for codec in codecs:
        for arm in ARMS:
            out["raw"][f"{codec}/{arm}"] = {
                str(q): {m: float(np.mean([rows[(c, arm, codec)][i][m] for c in HELD]))
                         for m in METRICS + ["kbps"]}
                for i, q in enumerate(QPS)}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
