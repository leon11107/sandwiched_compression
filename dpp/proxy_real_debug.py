"""Debug the FOUNDATION (no preprocessor): does the differentiable proxy codec
represent the real JPEG codec? Measures proxy<->real domain shift in
 (1) FORWARD  : PSNR/bpp of proxy vs real + decoded-agreement PSNR(proxy,real)
 (2) GRADIENT : d(distortion)/d(codec_input) proxy(autodiff) vs real(finite-diff)
                cosine + sign — the quantity training actually descends
 (3) TRANSFER : optimize a per-image delta to minimize PROXY distortion, then check
                whether REAL distortion also drops (= can proxy-training transfer?)
All on raw images, identity preprocessor. torch-env.
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.codec import EncodeDecodeIntraTorch, encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb


def restore(dec, orig):  # codec_luma_only: keep decoded Y, original chroma
    d = tf_rgb_to_yuv(dec); o = tf_rgb_to_yuv(orig)
    return tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))

def psnr(a, b):
    m = float(((a.double() - b.double()) ** 2).mean())
    return 10 * np.log10(255.0 ** 2 / m) if m > 0 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val")
    ap.add_argument("--qsteps", default="16,32,64"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dirs", type=int, default=16); ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).to(dev)
            for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))[:a.n]]
    qsteps = [float(x) for x in a.qsteps.split(",")]
    rng = np.random.default_rng(7)

    def proxy_codec(q):
        return EncodeDecodeIntraTorch(qstep_init=q, train_qstep=False, min_qstep=1.0,
            quantizer_mode="straight_through", rate_proxy_mode="log_nonzero",
            codec_forward_mode="proxy", output_clip_mode="hard",
            convert_to_yuv=True, device=dev)

    def real(x, q):  # x [1,H,W,3] -> decoded [1,H,W,3] + bits
        d, b = encode_decode_with_jpeg(x.detach().cpu().numpy(), q, False, False)
        return torch.from_numpy(d).to(dev).float(), float(b[0])

    # (1) FORWARD codec comparison ------------------------------------------------
    print("== (1) FORWARD: proxy vs real (no preproc) ==")
    print(f"{'q':>4} | {'PSNR_proxy':>10} {'PSNR_real':>9} {'dPSNR':>6} | "
          f"{'bpp_proxy':>9} {'bpp_real':>8} | {'decoded agree':>13}")
    for q in qsteps:
        cod = proxy_codec(q)
        pp, pr, bp, br, agree = [], [], [], [], []
        for x in imgs:
            xb = x[None]; H, W = x.shape[:2]
            with torch.no_grad():
                dp, ratep = cod(xb, input_qstep=q); dp = restore(dp, xb)
            drr, bits = real(xb, q); drr = restore(drr, xb)
            pp.append(psnr(dp, xb)); pr.append(psnr(drr, xb))
            bp.append(float(ratep[0]) / (H * W)); br.append(bits / (H * W))
            agree.append(psnr(dp, drr))  # how close proxy-decoded is to real-decoded
        print(f"{q:>4g} | {np.mean(pp):>10.2f} {np.mean(pr):>9.2f} {np.mean(pp)-np.mean(pr):>+6.2f} | "
              f"{np.mean(bp):>9.3f} {np.mean(br):>8.3f} | {np.mean(agree):>10.2f} dB")

    # (2) GRADIENT alignment: d(MSE(codec(z),x))/dz, proxy autodiff vs real FD -----
    print("== (2) GRADIENT: distortion grad proxy(autodiff) vs real(finite-diff) ==")
    for q in qsteps:
        cod = proxy_codec(q)
        coss, signs = [], []
        for x in imgs:
            xb = x[None]
            z = xb.clone().requires_grad_(True)
            dp, _ = cod(z, input_qstep=q); dp = restore(dp, xb)
            Lp = ((dp - xb) ** 2).mean()
            g = torch.autograd.grad(Lp, z)[0][0]  # [H,W,3]
            H, W = x.shape[:2]
            dpx, drl = [], []
            for _ in range(a.dirs):
                u = torch.from_numpy((rng.standard_normal((H, W, 1)) * a.eps).astype(np.float32)).to(dev)
                u = u.repeat(1, 1, 3)  # luma-ish perturbation
                dpx.append(float((g * u).sum()))
                dpv, _ = real(xb + u[None], q); dpv = restore(dpv, xb)
                dmv, _ = real(xb - u[None], q); dmv = restore(dmv, xb)
                Lp_p = float(((dpv - xb) ** 2).mean()); Lp_m = float(((dmv - xb) ** 2).mean())
                drl.append((Lp_p - Lp_m) / 2.0)
            dpx, drl = np.array(dpx), np.array(drl)
            coss.append(float(np.sum(dpx * drl) / (np.linalg.norm(dpx) * np.linalg.norm(drl) + 1e-30)))
            signs.append(float(np.mean(np.sign(dpx) == np.sign(drl))))
        print(f"  q{q:>3g}: distortion-grad cosine(proxy,real)={np.mean(coss):+.3f}  sign-match={np.mean(signs):.2f}")

    # (3) TRANSFER: optimize delta through PROXY, measure REAL distortion change ----
    print("== (3) TRANSFER: optimize delta via PROXY distortion -> does REAL drop? ==")
    for q in qsteps:
        cod = proxy_codec(q)
        d_proxy0, d_proxy1, d_real0, d_real1 = [], [], [], []
        for x in imgs:
            xb = x[None]
            delta = torch.zeros_like(xb, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=0.5)
            with torch.no_grad():
                dp0, _ = cod(xb, input_qstep=q); dp0 = restore(dp0, xb)
                dr0, _ = real(xb, q); dr0 = restore(dr0, xb)
            for _ in range(a.steps):
                z = xb + delta
                dp, _ = cod(z, input_qstep=q); dp = restore(dp, xb)
                L = ((dp - xb) ** 2).mean()      # minimize PROXY distortion (pre-emphasis)
                opt.zero_grad(); L.backward(); opt.step()
            with torch.no_grad():
                z = xb + delta
                dp1, _ = cod(z, input_qstep=q); dp1 = restore(dp1, xb)
                dr1, _ = real(z, q); dr1 = restore(dr1, xb)
            d_proxy0.append(psnr(dp0, xb)); d_proxy1.append(psnr(dp1, xb))
            d_real0.append(psnr(dr0, xb)); d_real1.append(psnr(dr1, xb))
        dpx = np.mean(d_proxy1) - np.mean(d_proxy0)
        drx = np.mean(d_real1) - np.mean(d_real0)
        print(f"  q{q:>3g}: PROXY PSNR {np.mean(d_proxy0):.2f}->{np.mean(d_proxy1):.2f} ({dpx:+.2f}) | "
              f"REAL PSNR {np.mean(d_real0):.2f}->{np.mean(d_real1):.2f} ({drx:+.2f})  "
              f"=> {'TRANSFERS' if drx > 0.05 else 'DOES NOT TRANSFER (proxy-opt hurts/ignores real)'}")


if __name__ == "__main__":
    main()
