"""Multi-clip A/B panels: for each held-out clip, auto-pick the highest-detail
crop (max Laplacian energy) and show original | no-preproc | g2_big-preproc
decoded at a chosen QP, with PSNR + VMAF_NEG labels. Shows the preprocessor
effect across diverse test patterns (foliage / water+people / structure / smoke)."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndi
sys.path.insert(0, "/workspace/sandwiched_compression")

WORK = "/dev/shm/dppv"; OUT = "/workspace/sandwiched_compression/dpp_repro"
NEG = "/workspace/sandwiched_compression/reference/vmaf_models/vmaf_v0.6.1neg.json"
CLIPS = ["aspen_1080p", "red_kayak_1080p", "west_wind_easy_1080p", "controlled_burn_1080p"]
S = 360


def sh(c):
    subprocess.run(c, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def dec0(src, qp, codec="x264"):
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        e = f"{td}/e.mp4"; p = f"{td}/f.png"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
            "-preset", "slow", "-qp", str(qp), "-tune", "ssim", "-sc_threshold", "0",
            "-g", "150", e])
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", e, "-frames:v", "1", p])
        return np.asarray(Image.open(p).convert("RGB"))


def orig0(src):
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-frames:v", "1", f"{WORK}/_o.png"])
    return np.asarray(Image.open(f"{WORK}/_o.png").convert("RGB"))


def best_crop(rgb):
    """top-left of the S×S window with the most high-freq detail."""
    g = rgb.mean(2)
    lap = np.abs(ndi.laplace(g))
    H, W = g.shape
    integ = lap.cumsum(0).cumsum(1)
    best, by, bx = -1, 0, 0
    for y in range(0, H - S, 60):
        for x in range(0, W - S, 60):
            tot = (integ[y+S, x+S] - integ[y, x+S] - integ[y+S, x] + integ[y, x])
            if tot > best:
                best, by, bx = tot, y, x
    return by, bx


def metrics(ref_src, dist_rgb):
    ref = orig0(ref_src).astype(np.float32)
    psnr = 10 * np.log10(255**2 / ((ref - dist_rgb.astype(np.float32))**2).mean())
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        Image.fromarray(dist_rgb).save(f"{td}/d.png")
        rd = f"{td}/r.y4m"; dd = f"{td}/d.y4m"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", ref_src, "-frames:v", "1",
            "-pix_fmt", "yuv420p", rd])
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{td}/d.png", "-pix_fmt", "yuv420p", dd])
        oj = f"{td}/v.json"; env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/usr/local/lib/x86_64-linux-gnu"
        subprocess.run(["vmaf", "--reference", rd, "--distorted", dd, "--model",
                        "path=" + NEG, "--output", oj, "--json"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        pm = json.loads(open(oj).read())["pooled_metrics"]
        return psnr, pm[next(k for k in pm if k.startswith("vmaf"))]["mean"]


def main(qp=37):
    fig, axes = plt.subplots(len(CLIPS), 3, figsize=(13.5, 4.4 * len(CLIPS)))
    for ri, clip in enumerate(CLIPS):
        base_src = f"{WORK}/src/{clip}.y4m"; pre_src = f"{WORK}/src/{clip}__g2_1.y4m"
        o = orig0(base_src)
        by, bx = best_crop(o)
        b = dec0(base_src, qp); p = dec0(pre_src, qp)
        pb, nb = metrics(base_src, b); pp, npg = metrics(base_src, p)
        cr = lambda im: im[by:by+S, bx:bx+S]
        cols = [("original", cr(o)),
                (f"no-preproc qp{qp}\nPSNR {pb:.1f}  VMAF_NEG {nb:.1f}", cr(b)),
                (f"g2_big preproc qp{qp}\nPSNR {pp:.1f} ({pp-pb:+.1f})  "
                 f"VMAF_NEG {npg:.1f} ({npg-nb:+.1f})", cr(p))]
        for ci, (t, im) in enumerate(cols):
            ax = axes[ri][ci]; ax.imshow(im); ax.axis("off")
            ax.set_title((clip.replace("_1080p", "") + " — " + t) if ci == 0 else t,
                         fontsize=10)
        print(f"[{clip}] crop@({bx},{by}) base PSNR{pb:.1f}/NEG{nb:.1f} "
              f"pre PSNR{pp:.1f}/NEG{npg:.1f}", flush=True)
    fig.suptitle(f"A/B across test patterns — x264 qp{qp}, highest-detail crop per clip",
                 fontsize=13)
    fig.tight_layout()
    out = f"{OUT}/fig_ab_multiclip.png"; fig.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
