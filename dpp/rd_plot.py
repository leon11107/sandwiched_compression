"""RD-curve plot (hard rule): VMAF_NEG vs bpp, baseline vs DPP model(s), on val50.
Also prints best operating point (max VMAF_NEG gain at equal bpp) per 0.1-bpp bin.
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/workspace/sandwiched_compression")
sys.path.insert(0, "/workspace/sandwiched_compression/experiments/m2_lowres_repro")
from dpp.model import DPPModel
from dpp.bd_ci import per_image_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val50")
    ap.add_argument("--qsteps", default="32,48,64,96,128")
    ap.add_argument("--out", default="/workspace/sandwiched_compression/dpp/runs/rd_cotrained.png")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qsteps = [float(x) for x in a.qsteps.split(",")]
    imgs = [np.asarray(Image.open(p).convert("RGB"), np.float32)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))]
    bpp_b, vmn_b = per_image_curves(imgs, qsteps, None, dev)
    plt.figure(figsize=(7, 5))
    plt.plot(bpp_b.mean(1), vmn_b.mean(1), "k-o", lw=2, label="baseline JPEG")
    for spec in a.models:
        name, path = spec.split("=", 1)
        m = DPPModel(ch=64, codec_forward_mode="proxy", device=dev)
        ck = torch.load(path, map_location=dev)
        (m.preproc.load_state_dict(ck["preproc"]) if isinstance(ck, dict) and "preproc" in ck
         else m.load_state_dict(ck)); m.eval()
        bpp_m, vmn_m = per_image_curves(imgs, qsteps, m.preproc, dev)
        plt.plot(bpp_m.mean(1), vmn_m.mean(1), "-s", label=name)
    plt.xlabel("bpp"); plt.ylabel("VMAF_NEG"); plt.title("DPP (co-trained Ballé) vs baseline — val50")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(a.out, dpi=110)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
