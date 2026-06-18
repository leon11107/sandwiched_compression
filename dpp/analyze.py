"""Root-cause analysis of the DPP runs (NO retrain). For a trained model:
 (1) PROXY vs REAL codec PSNR/bpp (baseline vs model) -> is it a domain gap or an
     objective problem?
 (2) preproc residual stats (magnitude + high-freq energy) -> what is it DOING?
 (3) where do the extra bits come from (HF energy of the preprocessed luma)?
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp.model import DPPModel
from torch_port.codec import encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb


def restore(dec, orig):
    d = tf_rgb_to_yuv(torch.from_numpy(dec[None]).float()); o = tf_rgb_to_yuv(torch.from_numpy(orig[None]).float())
    return np.clip(tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))[0].numpy(), 0, 255)

def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / m)

def hf_energy(y):  # high-freq energy of a luma plane = y - blur(y)
    t = torch.from_numpy(y[None, None]).float()
    k = torch.ones(1, 1, 5, 5) / 25.0
    blur = torch.nn.functional.conv2d(t, k, padding=2)
    return float(((t - blur) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val")
    ap.add_argument("--qsteps", default="16,32,64"); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = [np.asarray(Image.open(p).convert("RGB"), np.float32)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))[:a.n]]
    m = DPPModel(ch=a.ch, codec_forward_mode="proxy", device=dev)
    m.load_state_dict(torch.load(a.model, map_location=dev)); m.eval()
    print(f"model={a.model} scaler={float(m.preproc.scaler):.4f}  ({len(imgs)} imgs)")

    # (2)/(3) residual + HF stats (preproc output vs input, luma)
    res_abs, hf_in, hf_pre = [], [], []
    for im in imgs:
        with torch.no_grad():
            pre = m.preproc(torch.from_numpy(im[None]).float().to(dev))[0].cpu().numpy()
        yin = tf_rgb_to_yuv(torch.from_numpy(im[None]).float())[0, ..., 0].numpy()
        ypre = tf_rgb_to_yuv(torch.from_numpy(pre[None]).float())[0, ..., 0].numpy()
        res_abs.append(np.mean(np.abs(ypre - yin)))
        hf_in.append(hf_energy(yin)); hf_pre.append(hf_energy(ypre))
    print(f"[residual] mean|ΔY|={np.mean(res_abs):.3f}  HF_energy in={np.mean(hf_in):.1f} "
          f"pre={np.mean(hf_pre):.1f} ({'SHARPEN +' if np.mean(hf_pre)>np.mean(hf_in) else 'SMOOTH -'}"
          f"{100*(np.mean(hf_pre)-np.mean(hf_in))/np.mean(hf_in):+.1f}%)")

    # (1) PROXY (eval, hard-round diff JPEG) vs REAL JPEG, baseline vs model
    print(f"{'qstep':>6} | {'arm':>8} | {'PROXY psnr/bpp':>16} | {'REAL psnr/bpp':>16}")
    for q in [float(x) for x in a.qsteps.split(",")]:
        for arm in ("baseline", "model"):
            pp, pb, rp, rb = [], [], [], []
            for im in imgs:
                x = torch.from_numpy(im[None]).float().to(dev)
                if arm == "model":
                    with torch.no_grad():
                        out = m(x, input_qstep=q)               # PROXY codec (eval hard round)
                    proxy_pred = np.clip(out["prediction"][0].cpu().numpy(), 0, 255)
                    proxy_rate = float(out["rate"][0])
                    with torch.no_grad():
                        pre = np.clip(m.preproc(x)[0].cpu().numpy(), 0, 255)
                else:
                    # baseline: no preproc -> proxy codec on the raw image
                    with torch.no_grad():
                        out = DPPModel(ch=a.ch, codec_forward_mode="proxy", device=dev).__class__.forward  # placeholder
                    # compute proxy on raw image via the codec directly
                    dec_p, rate_p = m.codec(x, input_qstep=q)
                    # codec_luma_only restore for baseline too (chroma lossless)
                    yd = tf_rgb_to_yuv(dec_p); yi = tf_rgb_to_yuv(x)
                    proxy_pred = np.clip(tf_yuv_to_rgb(torch.cat([yd[..., 0:1], yi[..., 1:3]], -1))[0].detach().cpu().numpy(), 0, 255)
                    proxy_rate = float(rate_p[0]); pre = im
                pp.append(psnr(proxy_pred, im)); pb.append(proxy_rate / (im.shape[0] * im.shape[1]))
                dr, br = encode_decode_with_jpeg(pre[None], q, False, False)
                dr = restore(dr[0], im); rp.append(psnr(dr, im)); rb.append(br[0] / (im.shape[0] * im.shape[1]))
            print(f"{q:>6g} | {arm:>8} | {np.mean(pp):>7.2f}/{np.mean(pb):>5.3f}    | {np.mean(rp):>7.2f}/{np.mean(rb):>5.3f}")


if __name__ == "__main__":
    main()
