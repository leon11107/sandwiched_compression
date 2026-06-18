"""Does the OFFICIAL Diff-JPEG (Reich et al. WACV2024, ste=True) give a distortion
gradient that ALIGNS with the real PIL codec? Compares its proxy->real cosine vs our
flat-DCT proxies (straight_through/noise). luma-restore chroma on both for a clean
luma comparison. If DiffJPEG aligns (cos>0, ideally >0.3) -> the fix; else hard-codec
gradient is fundamental. torch-env."""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression/reference/Diff-JPEG")
sys.path.insert(0, "/workspace/sandwiched_compression")
from diff_jpeg import DiffJPEGCoding
from torch_port.codec import EncodeDecodeIntraTorch, encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb


def restore(dec, orig):  # dec,orig [1,H,W,3] 0..255; keep dec Y, orig chroma
    d = tf_rgb_to_yuv(dec); o = tf_rgb_to_yuv(orig)
    return tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))

def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qsteps", default="16,32"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--dirs", type=int, default=20); ap.add_argument("--eps", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).to(dev)
            for p in sorted(glob.glob("/workspace/sandwiched_compression/dpp/data/val/*.png"))[:a.n]]
    rng = np.random.default_rng(7)
    dj = DiffJPEGCoding(ste=True).to(dev)

    def real_dist(z, xb, q):
        d, _ = encode_decode_with_jpeg(z.detach().cpu().numpy(), q, False, False)
        return float(((restore(torch.from_numpy(d).to(dev).float(), xb) - xb) ** 2).mean())

    def diffjpeg_dec(z_bhwc, q):  # z [1,H,W,3] 0..255 -> decoded [1,H,W,3]
        zc = z_bhwc.permute(0, 3, 1, 2).contiguous()
        qt = torch.full((8, 8), float(q), device=dev)
        out = dj(zc, torch.tensor([50.0], device=dev), quantization_table_y=qt, quantization_table_c=qt)
        return out.permute(0, 2, 3, 1).contiguous()

    for q in [float(x) for x in a.qsteps.split(",")]:
        st = EncodeDecodeIntraTorch(qstep_init=q, train_qstep=False, min_qstep=1.0,
            quantizer_mode="straight_through", rate_proxy_mode="log_nonzero", codec_forward_mode="proxy",
            output_clip_mode="hard", convert_to_yuv=True, device=dev)
        c_dj, c_st = [], []
        for x in imgs:
            xb = x[None]; H, W = x.shape[:2]
            us = [torch.from_numpy(np.repeat((rng.standard_normal((H, W, 1)) * a.eps).astype(np.float32), 3, -1)).to(dev)
                  for _ in range(a.dirs)]
            drl = [(real_dist(xb + u[None], xb, q) - real_dist(xb - u[None], xb, q)) / 2 for u in us]
            # DiffJPEG grad
            z = xb.clone().requires_grad_(True)
            Ldj = ((restore(diffjpeg_dec(z, q), xb) - xb) ** 2).mean()
            gdj = torch.autograd.grad(Ldj, z)[0][0]
            # straight_through proxy grad (reference)
            z2 = xb.clone().requires_grad_(True)
            Lst = ((restore(st(z2, input_qstep=q)[0], xb) - xb) ** 2).mean()
            gst = torch.autograd.grad(Lst, z2)[0][0]
            c_dj.append(cos([float((gdj * u).sum()) for u in us], drl))
            c_st.append(cos([float((gst * u).sum()) for u in us], drl))
        print(f"q{q:>3g}: distortion-grad cosine vs real  ->  DiffJPEG(ste)={np.mean(c_dj):+.3f}   "
              f"flatDCT-straight_through={np.mean(c_st):+.3f}")


if __name__ == "__main__":
    main()
