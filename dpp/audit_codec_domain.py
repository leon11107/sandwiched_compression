"""Domain-shift audit: TRAINING codec vs CORRECTED-EVAL codec (eval_v2 protocol).

Training codec  = flat-qtable 4:4:4 PIL JPEG + lossless-chroma restore (what every
                  d-series checkpoint was trained against, qstep 12-64).
Eval codec      = Annex-K scaled tables (PIL quality), 4:2:0, full lossy (eval_v2).

Quantifies, on val50 subset:
 (1) RANGE MAP   : RD tables of both arms + the effective Annex-K luma-table step
                   per quality (libjpeg scaling law) vs flat qstep -> regime overlap.
 (2) MATCHED-BPP : per eval quality point, find flat qstep with closest bpp ->
                   PSNR(dec_train, dec_eval) + per-DCT-subband luma error spectra
                   (where do the artifact domains differ: HF vs LF).
 (3) CHROMA ISO  : Annex-K luma-only (lossless chroma restore, 4:4:4) vs full 4:2:0
                   -> how much of the shift is chroma handling vs quant table shape.
torch-env. Eval-only, no training.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import io
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.codec import encode_decode_with_jpeg, JpegProxyTorch
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
from torch_port.losses import ssim_multiscale_tf

QT_LUMA = np.array([
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55, 14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62, 18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99], np.float64)


def annexk_step(quality):
    """libjpeg quality scaling -> effective luma table (clamped 1..255)."""
    q = max(1, min(100, int(quality)))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    tbl = np.clip(np.floor((QT_LUMA * scale + 50) / 100), 1, 255)
    return tbl


def load16(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    w16, h16 = (w // 16) * 16, (h // 16) * 16
    l, t = (w - w16) // 2, (h - h16) // 2
    return np.asarray(im.crop((l, t, l + w16, t + h16)), np.float32)


def restore_chroma_np(dec, orig):
    d = tf_rgb_to_yuv(torch.from_numpy(dec[None]).float())
    o = tf_rgb_to_yuv(torch.from_numpy(orig[None]).float())
    return np.clip(tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))[0].numpy(), 0, 255)


def jpeg_annexk(img, quality, use_420=True):
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8)).save(
        buf, format="jpeg", quality=int(quality),
        subsampling="4:2:0" if use_420 else "4:4:4", optimize=True)
    dec = np.asarray(Image.open(buf).convert("RGB"), np.float32)
    return dec, 8 * len(buf.getbuffer())


def codec_train(img, qstep):
    """flat-table 4:4:4 + lossless-chroma restore (training arm)."""
    dec, bits = encode_decode_with_jpeg(img[None], qstep, False, False)
    return restore_chroma_np(dec[0], img), float(bits[0])


def rgb_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


@torch.no_grad()
def msssim_luma(ref, dec, dev):
    r = torch.from_numpy(ref[None]).float().to(dev)
    d = torch.from_numpy(dec[None]).float().to(dev)
    return float(ssim_multiscale_tf(tf_rgb_to_yuv(r)[..., 0:1], tf_rgb_to_yuv(d)[..., 0:1],
                                    max_val=255.0, filter_size=11)[0])


@torch.no_grad()
def err_spectrum(orig, dec, jp, dev):
    """per-DCT-subband luma error energy [64], natural raster order."""
    e = torch.from_numpy((dec - orig)[None]).float().to(dev)
    y = jp._rgb_to_yuv(e)[..., 0:1]            # error in Y (linear transform => ok)
    c = jp._forward_dct_2d(y)                  # [1,h,w,64]
    return (c ** 2).mean(dim=(0, 1, 2)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/val50")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--qualities", default="5,8,12,20,32")
    ap.add_argument("--qsteps", default="12,24,32,48,64,96,128")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jp = JpegProxyTorch(convert_to_yuv=True, clip_to_image_max=True, device=dev)
    imgs = [load16(p) for p in sorted(glob.glob(os.path.join(a.img_dir, "*.png")))[: a.n]]
    Q = [int(x) for x in a.qualities.split(",")]
    S = [float(x) for x in a.qsteps.split(",")]
    npix = [im.shape[0] * im.shape[1] for im in imgs]

    # ---------- (1) RANGE MAP ----------
    print("== (1) RANGE MAP: training arm (flat qstep, 4:4:4, lossless chroma) ==")
    train_pts = {}
    for s in S:
        rows = [codec_train(im, s) for im in imgs]
        bpp = float(np.mean([b / p for (_, b), p in zip(rows, npix)]))
        ps = float(np.mean([rgb_psnr(d, im) for (d, _), im in zip(rows, imgs)]))
        ms = float(np.mean([msssim_luma(im, d, dev) for (d, _), im in zip(rows, imgs)]))
        train_pts[s] = bpp
        print(f"  qstep={s:6.0f}  bpp={bpp:.4f}  psnr={ps:.2f}  msssim={ms:.5f}")
    print("== (1) eval arm (Annex-K quality, 4:2:0 full lossy) ==")
    print("   [effective Annex-K luma steps: DC/AC-mean/AC-max per quality]")
    eval_pts = {}
    for q in Q:
        tbl = annexk_step(q)
        rows = [jpeg_annexk(im, q) for im in imgs]
        bpp = float(np.mean([b / p for (_, b), p in zip(rows, npix)]))
        ps = float(np.mean([rgb_psnr(d, im) for (d, _), im in zip(rows, imgs)]))
        ms = float(np.mean([msssim_luma(im, d, dev) for (d, _), im in zip(rows, imgs)]))
        eval_pts[q] = bpp
        print(f"  quality={q:3d}  bpp={bpp:.4f}  psnr={ps:.2f}  msssim={ms:.5f}"
              f"   | table: DC={tbl[0]:.0f} ACmean={tbl[1:].mean():.0f} ACmax={tbl.max():.0f}")
    tr_lo, tr_hi = min(train_pts.values()), max(train_pts.values())
    ev_lo, ev_hi = min(eval_pts.values()), max(eval_pts.values())
    print(f"  -> training bpp range (qstep 12-64): [{train_pts[64.0]:.3f}, {train_pts[12.0]:.3f}]"
          f" | eval bpp range: [{ev_lo:.3f}, {ev_hi:.3f}]")

    # ---------- (2) MATCHED-BPP decoded agreement + error spectra ----------
    print("== (2) MATCHED-BPP: eval-q vs closest flat-qstep (per-image match) ==")
    fine = [float(s) for s in [8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 200, 255]]
    for q in Q:
        agree, spec_t, spec_e, used = [], [], [], []
        for im, p in zip(imgs, npix):
            de, be = jpeg_annexk(im, q)
            target = be / p
            best, bd, bs = None, None, 1e9
            for s in fine:
                dt, bt = codec_train(im, s)
                if abs(bt / p - target) < bs:
                    bs, best, bd = abs(bt / p - target), s, dt
            agree.append(rgb_psnr(bd, de))
            spec_t.append(err_spectrum(im, bd, jp, dev))
            spec_e.append(err_spectrum(im, de, jp, dev))
            used.append(best)
        st, se = np.mean(spec_t, 0), np.mean(spec_e, 0)
        # zigzag-ish split: band index r*8+c, LF = r+c<=2 (6 bands), HF = r+c>=8
        idx = np.arange(64); r, c = idx // 8, idx % 8
        lf, hf = (r + c) <= 2, (r + c) >= 8
        print(f"  eval q={q:3d} ~ flat qstep median={np.median(used):.0f} "
              f"| PSNR(train_dec, eval_dec)={np.mean(agree):.2f}dB "
              f"| errE ratio train/eval: LF={st[lf].sum()/max(se[lf].sum(),1e-9):.2f} "
              f"HF={st[hf].sum()/max(se[hf].sum(),1e-9):.2f} all={st.sum()/se.sum():.2f}")

    # ---------- (3) CHROMA ISOLATION ----------
    print("== (3) CHROMA ISO: Annex-K 4:4:4+lossless-chroma vs full 4:2:0 ==")
    for q in Q:
        b420, b444l, ms420, ms444l, ps420, ps444l = [], [], [], [], [], []
        for im, p in zip(imgs, npix):
            d1, b1 = jpeg_annexk(im, q, use_420=True)
            d2, b2 = jpeg_annexk(im, q, use_420=False)
            d2 = restore_chroma_np(d2, im)
            b420.append(b1 / p); b444l.append(b2 / p)
            ps420.append(rgb_psnr(d1, im)); ps444l.append(rgb_psnr(d2, im))
            ms420.append(msssim_luma(im, d1, dev)); ms444l.append(msssim_luma(im, d2, dev))
        print(f"  q={q:3d} 420full: bpp={np.mean(b420):.4f} psnr={np.mean(ps420):.2f} "
              f"ms={np.mean(ms420):.5f} | 444+losslessC: bpp={np.mean(b444l):.4f} "
              f"psnr={np.mean(ps444l):.2f} ms={np.mean(ms444l):.5f}")


if __name__ == "__main__":
    main()
