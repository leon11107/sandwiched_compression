"""Visualize the fixed-QP eval: (1) RD curves (rate vs metric, baseline vs
preprocessor, per codec), (2) A/B image panels (original | baseline-decoded |
preprocessor-decoded crops at a chosen QP) so the perception-distortion
tradeoff is visible (sharper detail, higher VMAF, lower PSNR)."""
from __future__ import annotations
import io, json, os, subprocess, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp_repro.y4m import read_y4m

WORK = "/dev/shm/dppv"; OUT = "/workspace/sandwiched_compression/dpp_repro"
RUNS = "/workspace/sandwiched_compression/dpp/runs"


def sh(c):
    subprocess.run(c, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---- 1. RD curves -----------------------------------------------------------
def rd_curves():
    d = json.load(open(os.path.join(RUNS, "s6_qpsweep.json")))
    codecs = d["codecs"]; qps = d["qps"]
    mets = [("vmaf_neg", "VMAF_NEG (higher=better)"),
            ("psnr", "PSNR dB (higher=better)"),
            ("ms_ssim", "MS-SSIM (higher=better)")]
    arms = [("baseline", "k-o", "no preproc"), ("g2_big", "r-s", "g2_big"),
            ("g0.5_big", "b-^", "g0.5_big")]
    fig, axes = plt.subplots(len(mets), len(codecs), figsize=(13, 10))
    for ri, (mk, mlabel) in enumerate(mets):
        for ci, codec in enumerate(codecs):
            ax = axes[ri][ci]
            for arm, sty, lab in arms:
                raw = d["raw"][f"{codec}/{arm}"]
                x = [raw[str(q)]["kbps"] for q in qps]
                y = [raw[str(q)][mk] for q in qps]
                ax.plot(x, y, sty, ms=5, label=lab)
            ax.set_xscale("log")
            if ri == 0:
                ax.set_title(codec, fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            if ri == len(mets) - 1:
                ax.set_xlabel("bitrate kbps (log)", fontsize=9)
            ax.grid(alpha=0.3)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Fixed-QP {22,27,32,37}, native 1080p — RD curves (4-clip mean)",
                 fontsize=13)
    fig.tight_layout()
    p = f"{OUT}/fig_qpsweep_rd.png"; fig.savefig(p, dpi=120); print("saved", p)


# ---- 2. A/B image panel -----------------------------------------------------
def decode_frame0_rgb(src_y4m, codec, qp):
    """encode src at QP, decode frame 0 -> RGB uint8 [H,W,3]."""
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        enc = os.path.join(td, "e.mp4"); png = os.path.join(td, "f.png")
        if codec == "x264":
            cmd = ["-c:v", "libx264", "-preset", "slow", "-qp", str(qp),
                   "-tune", "ssim", "-sc_threshold", "0", "-g", "150"]
        else:
            cmd = ["-c:v", "libx265", "-preset", "slow",
                   "-x265-params", f"qp={qp}:keyint=150:tune=ssim"]
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src_y4m] + cmd + [enc])
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-frames:v", "1", png])
        from PIL import Image
        return np.asarray(Image.open(png).convert("RGB"))


def frame0_metrics(ref_y4m, dist_rgb):
    """whole-frame-0 PSNR(RGB) + VMAF_NEG(luma binary) of a decoded RGB frame."""
    from PIL import Image
    MODEL_NEG = "/workspace/sandwiched_compression/reference/vmaf_models/vmaf_v0.6.1neg.json"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", ref_y4m, "-frames:v", "1",
        f"{WORK}/_ref0.png"])
    ref = np.asarray(Image.open(f"{WORK}/_ref0.png").convert("RGB")).astype(np.float32)
    mse = ((ref - dist_rgb.astype(np.float32)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        Image.fromarray(dist_rgb).save(f"{td}/d.png")
        rd = f"{td}/r.y4m"; dd = f"{td}/d.y4m"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", ref_y4m, "-frames:v", "1",
            "-pix_fmt", "yuv420p", rd])
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{td}/d.png",
            "-pix_fmt", "yuv420p", dd])
        oj = f"{td}/v.json"; env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/usr/local/lib/x86_64-linux-gnu"
        subprocess.run(["vmaf", "--reference", rd, "--distorted", dd, "--model",
                        "path=" + MODEL_NEG, "--output", oj, "--json"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        pm = json.loads(open(oj).read())["pooled_metrics"]
        neg = pm[next(k for k in pm if k.startswith("vmaf"))]["mean"]
    return psnr, neg


def ab_panel(clip="aspen_1080p", qp=37, codec="x264", crop=(560, 980, 360)):
    """crop=(y,x,size). Original | baseline-decoded | preproc-decoded."""
    from PIL import Image
    base_src = f"{WORK}/src/{clip}.y4m"
    pre_src = f"{WORK}/src/{clip}__g2_1.y4m"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", base_src, "-frames:v", "1",
        f"{WORK}/orig.png"])
    orig = np.asarray(Image.open(f"{WORK}/orig.png").convert("RGB"))
    base = decode_frame0_rgb(base_src, codec, qp)
    pre = decode_frame0_rgb(pre_src, codec, qp)
    pb, nb = frame0_metrics(base_src, base)
    pp, npg = frame0_metrics(base_src, pre)
    y, x, s = crop
    cr = lambda im: im[y:y + s, x:x + s]
    panels = [("original", cr(orig)),
              (f"no-preproc {codec} qp{qp}\nPSNR {pb:.1f}dB  VMAF_NEG {nb:.1f}", cr(base)),
              (f"g2_big preproc {codec} qp{qp}\nPSNR {pp:.1f}dB ({pp-pb:+.1f})  "
               f"VMAF_NEG {npg:.1f} ({npg-nb:+.1f})", cr(pre))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (t, im) in zip(axes, panels):
        ax.imshow(im); ax.set_title(t, fontsize=11); ax.axis("off")
    fig.suptitle(f"A/B crop — {clip}, {crop[2]}x{crop[2]} @ ({crop[1]},{crop[0]})  "
                 f"(preproc trades PSNR for sharper detail / higher VMAF)", fontsize=12)
    fig.tight_layout()
    p = f"{OUT}/fig_qpsweep_ab.png"; fig.savefig(p, dpi=130); print("saved", p)
    # also a 2x zoom inset of the center 140px
    z = s // 2 - 70
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (t, im) in zip(axes2, panels):
        ax.imshow(im[z:z + 140, z:z + 140]); ax.set_title("2x zoom: " + t, fontsize=10)
        ax.axis("off")
    fig2.tight_layout()
    p2 = f"{OUT}/fig_qpsweep_ab_zoom.png"; fig2.savefig(p2, dpi=130); print("saved", p2)


if __name__ == "__main__":
    rd_curves()
    ab_panel()
