"""Step 3: which gradient PATH is broken? Measure proxy<->real gradient alignment
SEPARATELY for the 3 loss terms that all flow through the codec backward:
  distortion = MSE(restore(codec(z)), x)
  rate       = bits  (proxy: log_nonzero backward; real: PIL byte count FD)
  perceptual = NIMA(restore(codec(z)))   (DPP L_P path)
cosine(d_proxy autodiff, d_real finite-diff) over K luma directions. No preproc.
If only distortion is broken but rate/perceptual align -> fix is targeted; if all
broken -> the codec backward itself is the problem (need a real-aligned codec grad).
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.codec import EncodeDecodeIntraTorch, encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
from dpp.perceptual import NimaMOS


def restore(dec, orig):
    d = tf_rgb_to_yuv(dec); o = tf_rgb_to_yuv(orig)
    return tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))

def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val")
    ap.add_argument("--qsteps", default="16,32"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--dirs", type=int, default=16); ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--quantizer", default="straight_through")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).to(dev)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))[:a.n]]
    rng = np.random.default_rng(7)
    nima = NimaMOS("nima-koniq", device=dev)

    def real_terms(z, xb, q):  # returns (dist, bits, mos) on the REAL codec
        d, b = encode_decode_with_jpeg(z.detach().cpu().numpy(), q, False, False)
        dec = restore(torch.from_numpy(d).to(dev).float(), xb)
        dist = float(((dec - xb) ** 2).mean())
        mos = float(nima.mos(dec))
        return dist, float(b[0]), mos

    for q in [float(x) for x in a.qsteps.split(",")]:
        cod = EncodeDecodeIntraTorch(qstep_init=q, train_qstep=False, min_qstep=1.0,
            quantizer_mode=a.quantizer, rate_proxy_mode="log_nonzero", codec_forward_mode="proxy",
            output_clip_mode="hard", convert_to_yuv=True, device=dev)
        cd, cr, cp = [], [], []
        for x in imgs:
            xb = x[None]; H, W = x.shape[:2]
            # proxy autodiff grads for the 3 terms wrt codec input z
            z = xb.clone().requires_grad_(True)
            dec_t, rate_t = cod(z, input_qstep=q)
            dec_r = restore(dec_t, xb)
            Ld = ((dec_r - xb) ** 2).mean()
            Lr = rate_t.sum()
            Lp = nima.mos(dec_r).sum()
            gd = torch.autograd.grad(Ld, z, retain_graph=True)[0][0]
            gr = torch.autograd.grad(Lr, z, retain_graph=True)[0][0]
            gp = torch.autograd.grad(Lp, z)[0][0]
            us = [torch.from_numpy(np.repeat((rng.standard_normal((H, W, 1)) * a.eps).astype(np.float32), 3, -1)).to(dev)
                  for _ in range(a.dirs)]
            dpx_d, dpx_r, dpx_p, drl_d, drl_r, drl_p = [], [], [], [], [], []
            for u in us:
                dpx_d.append(float((gd * u).sum())); dpx_r.append(float((gr * u).sum())); dpx_p.append(float((gp * u).sum()))
                dp = real_terms(xb + u[None], xb, q); dm = real_terms(xb - u[None], xb, q)
                drl_d.append((dp[0] - dm[0]) / 2); drl_r.append((dp[1] - dm[1]) / 2); drl_p.append((dp[2] - dm[2]) / 2)
            cd.append(cos(dpx_d, drl_d)); cr.append(cos(dpx_r, drl_r)); cp.append(cos(dpx_p, drl_p))
        print(f"q{q:>3g} [{a.quantizer}]  distortion={np.mean(cd):+.3f}  rate={np.mean(cr):+.3f}  "
              f"perceptual(NIMA)={np.mean(cp):+.3f}   (cosine proxy vs real, n={len(imgs)})")


if __name__ == "__main__":
    main()
