"""Proxy<->real distortion-gradient alignment, RIGOROUS: per quantizer-mode +
a real-vs-real self-consistency control (proves the finite-diff is reliable, so a
negative proxy-real cosine is real, not noise). No preproc. torch-env."""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.codec import EncodeDecodeIntraTorch, encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb


def restore(dec, orig):
    d = tf_rgb_to_yuv(dec); o = tf_rgb_to_yuv(orig)
    return tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val")
    ap.add_argument("--qsteps", default="16,32"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--dirs", type=int, default=24); ap.add_argument("--eps", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).to(dev)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))[:a.n]]
    rng = np.random.default_rng(7)
    modes = ["straight_through", "noise_injection", "polynomial", "ste_polynomial"]

    def real(x, q):
        d, b = encode_decode_with_jpeg(x.detach().cpu().numpy(), q, False, False)
        return torch.from_numpy(d).to(dev).float()

    def real_dist(z, xb, q):  # MSE(real_codec(z) restored, xb)
        return float(((restore(real(z, q), xb) - xb) ** 2).mean())

    def cos(a_, b_):
        a_, b_ = np.array(a_), np.array(b_)
        return float(np.sum(a_ * b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-30))

    for q in [float(x) for x in a.qsteps.split(",")]:
        print(f"== qstep {q:g} ==")
        # precompute per-image directions + real FD (shared across proxy modes)
        per_img = []
        for x in imgs:
            xb = x[None]; H, W = x.shape[:2]
            us = [torch.from_numpy(np.repeat((rng.standard_normal((H, W, 1)) * a.eps).astype(np.float32), 3, -1)).to(dev)
                  for _ in range(a.dirs)]
            drl = [(real_dist(xb + u[None], xb, q) - real_dist(xb - u[None], xb, q)) / 2.0 for u in us]
            # self-consistency: a SECOND independent real FD with fresh dirs
            us2 = [torch.from_numpy(np.repeat((rng.standard_normal((H, W, 1)) * a.eps).astype(np.float32), 3, -1)).to(dev)
                   for _ in range(a.dirs)]
            drl2a = [(real_dist(xb + u[None], xb, q) - real_dist(xb - u[None], xb, q)) / 2.0 for u in us2]
            drl2b = [(real_dist(xb + u[None], xb, q) - real_dist(xb - u[None], xb, q)) / 2.0 for u in us2]
            per_img.append((xb, us, np.array(drl), cos(drl2a, drl2b)))
        sc = np.mean([p[3] for p in per_img])
        print(f"  [control] real-vs-real FD self-consistency cosine = {sc:+.3f} (should be ~+1)")
        for mode in modes:
            cod = EncodeDecodeIntraTorch(qstep_init=q, train_qstep=False, min_qstep=1.0,
                quantizer_mode=mode, rate_proxy_mode="log_nonzero", codec_forward_mode="proxy",
                output_clip_mode="hard", convert_to_yuv=True, device=dev)
            coss = []
            for xb, us, drl, _ in per_img:
                z = xb.clone().requires_grad_(True)
                dp = restore(cod(z, input_qstep=q)[0], xb)
                L = ((dp - xb) ** 2).mean()
                g = torch.autograd.grad(L, z)[0][0]
                dpx = [float((g * u).sum()) for u in us]
                coss.append(cos(dpx, drl))
            print(f"  {mode:18s} distortion-grad cosine(proxy,real) = {np.mean(coss):+.3f}")


if __name__ == "__main__":
    main()
