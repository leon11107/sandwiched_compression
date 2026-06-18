"""DPP with the FAITHFUL rate engine: Balle factorized-prior entropy on DCT sub-bands
(co-trained with the preprocessor) = the rate-distortion mechanism the earlier runs
skipped (they used log_nonzero, rate grad ~+0.1). loss = L_F + gamma*L_P + lambda*L_R[Balle].
Rate path is through the (linear, clean-gradient) DCT + differentiable entropy model;
decode (for L_F/L_P) is through the codec. Co-trains preproc + entropy params jointly.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace/sandwiched_compression")
import pyiqa
from dpp.preproc_dpp import DPPPreproc
from dpp.entropy import FactorizedEntropy
from dpp.train import CropDataset, diag
from torch_port.codec import EncodeDecodeIntraTorch, ste_round, make_noise_injection_round
from torch_port.codec import JpegProxyTorch
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
from torch_port.losses import ssim_multiscale_tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/train")
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.01)    # perceptual (NIMA)
    ap.add_argument("--lam", type=float, default=0.02)      # rate (Balle bpp)
    ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--perceptual", default="nima-koniq")
    ap.add_argument("--qstep-lo", type=float, default=12.0); ap.add_argument("--qstep-hi", type=float, default=64.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--entropy-ckpt", default=None, help="if set: load + FREEZE this prior (honest rate); optimizer skips entropy")
    # DPP-12 #4: lambda anneal (hold-then-cosine-decay) to lock the mid-run MOS peak at low bpp.
    ap.add_argument("--lam-hi", type=float, default=None); ap.add_argument("--lam-lo", type=float, default=None)
    ap.add_argument("--hold-frac", type=float, default=0.4)
    # DPP-12 #1: VIF (VMAF-core FR feature) perceptual term on luma, additive to NIMA.
    ap.add_argument("--vif-gamma", type=float, default=0.0)
    # DPP-13 #2: shape the rate-proxy quantization by the JPEG luma Q-table (mean-preserving)
    # so the co-trained Balle rate matches the real codec's frequency-dependent bit allocation,
    # instead of a flat q for all 64 sub-bands. (Entropy model is ALREADY per-subband/per-channel,
    # so divisive-norm is redundant; the real gap is flat-q vs the codec's Q-table.)
    ap.add_argument("--qtable", default="flat", choices=["flat", "jpeg_luma"])
    a = ap.parse_args()
    assert torch.cuda.is_available(); dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True); qrng = np.random.default_rng(20260609)

    preproc = DPPPreproc(ch=a.ch, scaler_init=0.0).to(dev)
    entropy = FactorizedEntropy(64).to(dev)                 # 64 DCT sub-bands
    codec = EncodeDecodeIntraTorch(qstep_init=32.0, train_qstep=False, min_qstep=1.0,
        quantizer_mode="noise_injection", rate_proxy_mode="log_nonzero",
        codec_forward_mode="real_ste", output_clip_mode="hard", convert_to_yuv=True, device=dev)
    jp = JpegProxyTorch(convert_to_yuv=True, clip_to_image_max=True, device=dev)  # for DCT sub-bands
    nima = pyiqa.create_metric(a.perceptual, as_loss=True, device=dev) if a.gamma > 0 else None
    vif = pyiqa.create_metric("vif", as_loss=True, device=dev) if a.vif_gamma > 0 else None
    anneal = (a.lam_hi is not None and a.lam_lo is not None)
    frozen_prior = a.entropy_ckpt is not None
    if frozen_prior:
        ck = torch.load(a.entropy_ckpt, map_location=dev); entropy.load_state_dict(ck["entropy"])
        for p in entropy.parameters(): p.requires_grad_(False)
        entropy.eval()                                  # fixed honest rate estimator
        opt = torch.optim.Adam(preproc.parameters(), lr=a.lr)
    else:
        opt = torch.optim.Adam(list(preproc.parameters()) + list(entropy.parameters()), lr=a.lr)
        entropy.train()
    preproc.train()

    ds = CropDataset(a.img_dir, a.crop, a.steps_per_epoch * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    diag_imgs = ds.imgs[:6]
    log = open(os.path.join(a.out_dir, "train.log"), "w")
    def emit(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    emit(f"[balle] Balle-rate DPP: gamma={a.gamma} lam={a.lam} perceptual={a.perceptual} ch={a.ch} ep={a.epochs}")

    # standard JPEG (Annex K) luma quant table, natural raster (r*8+c) order to match
    # _forward_dct_2d's flattening. mean-preserving relative shape: qvec = q * QT/mean(QT).
    QT_LUMA = torch.tensor([
        16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55, 14,13,16,24,40,57,69,56,
        14,17,22,29,51,87,80,62, 18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
        49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99], dtype=torch.float32, device=dev)
    QT_REL = (QT_LUMA / QT_LUMA.mean()) if a.qtable == "jpeg_luma" else torch.ones(64, device=dev)
    QT_REL = QT_REL.view(1, 1, 1, 64)                       # broadcast over [B,h,w,64]

    def balle_bpp(bottleneck, q, gen):
        """DCT(Y of preproc out) -> per-subband(Q-table) noise-quantize -> Balle bits -> bpp [B]."""
        yuv = jp._rgb_to_yuv(bottleneck); y = yuv[..., 0:1]
        coeffs = jp._forward_dct_2d(y)                      # [B, H/8, W/8, 64]
        qvec = (q * QT_REL).clamp(min=1.0)                  # per-subband step (flat if qtable=flat)
        qd = (torch.rand(coeffs.shape, device=dev, generator=gen) - 0.5)  # noise quant (width 1)
        cq = coeffs / qvec + qd                             # quantized (noise) indices
        cq = cq.permute(0, 3, 1, 2).contiguous()           # [B,64,H/8,W/8]
        return entropy.bits(cq), bottleneck.shape[1] * bottleneck.shape[2]

    import math
    total_steps = a.epochs * len(dl); gstep = 0
    def lam_at(t):  # hold-then-cosine-decay; constant a.lam if anneal off
        if not anneal: return a.lam
        if t < a.hold_frac: return a.lam_hi
        u = (t - a.hold_frac) / max(1e-9, 1 - a.hold_frac)
        return a.lam_lo + 0.5 * (a.lam_hi - a.lam_lo) * (1 + math.cos(math.pi * u))

    metrics = []
    for ep in range(a.epochs):
        t0 = time.time(); acc = {"LF": 0, "MOS": 0, "VIF": 0, "bpp": 0, "total": 0, "lam": 0}; gn = 0.0; nb = 0
        for batch in dl:
            lam_t = lam_at(gstep / total_steps); gstep += 1
            x = batch.float().to(dev)
            q = float(qrng.uniform(a.qstep_lo, a.qstep_hi))
            gen = torch.Generator(device=dev); gen.manual_seed(int(qrng.integers(1 << 30)))
            bottleneck = preproc(x)
            dec, _ = codec(bottleneck, input_qstep=q, generator=gen)
            yuv_dec = tf_rgb_to_yuv(dec); yuv_in = tf_rgb_to_yuv(x)
            pred = tf_yuv_to_rgb(torch.cat([yuv_dec[..., 0:1], yuv_in[..., 1:3]], -1))
            # L_F (luma)
            Yx = tf_rgb_to_yuv(x)[..., 0:1]; Yp = tf_rgb_to_yuv(pred)[..., 0:1]
            l1 = ((Yx - Yp) / 255.0).abs().mean(dim=(1, 2, 3))
            ms = ssim_multiscale_tf(Yx, Yp, max_val=255.0, filter_size=7)
            LF = (a.alpha * l1 + a.beta * (1 - ms)).mean()
            # L_R (Balle bpp)
            bits, npix = balle_bpp(bottleneck, q, gen); bpp = (bits / npix).mean()
            # L_P (NIMA, small gamma)
            if nima is not None:
                p01 = (pred / 255).clamp(0, 1).permute(0, 3, 1, 2).contiguous()
                mos_val = nima(p01)            # scalar, higher=better
                LPterm = -mos_val               # minimize -> maximize MOS (L_P = -MOS)
            else:
                mos_val = torch.tensor(0.0, device=dev); LPterm = torch.tensor(0.0, device=dev)
            # L_VIF (VMAF-core FR feature, on luma, repeated to 3ch; maximize -> -VIF)
            if vif is not None:
                Yx1 = (Yx / 255.0).clamp(0, 1).permute(0, 3, 1, 2).repeat(1, 3, 1, 1).contiguous()
                Yp1 = (Yp / 255.0).clamp(0, 1).permute(0, 3, 1, 2).repeat(1, 3, 1, 1).contiguous()
                vif_val = vif(Yp1, Yx1).clamp(0.0, 1.0)   # clamp: VIF log-ratio is singular on
                LVIF = -vif_val                            # flat blocks (->1e8); clamp zeros OOR grad (guard)
            else:
                vif_val = torch.tensor(0.0, device=dev); LVIF = torch.tensor(0.0, device=dev)
            total = LF + a.gamma * LPterm + a.vif_gamma * LVIF + lam_t * bpp
            if not torch.isfinite(total): continue
            opt.zero_grad(); total.backward()
            g = torch.nn.utils.clip_grad_norm_(list(preproc.parameters()) + list(entropy.parameters()), a.grad_clip)
            if not torch.isfinite(g): opt.zero_grad(); continue
            opt.step()
            acc["LF"] += float(LF); acc["MOS"] += float(mos_val); acc["VIF"] += float(vif_val)
            acc["bpp"] += float(bpp); acc["total"] += float(total); acc["lam"] += lam_t
            gn += float(g); nb += 1
        for k in acc: acc[k] /= max(nb, 1)
        # diag uses the real codec eval pipeline (reuse train.diag with a tiny model shim)
        class _M:  # diag expects .preproc and .eval()/.train()
            pass
        shim = _M(); shim.preproc = preproc
        shim.eval = lambda: preproc.eval(); shim.train = lambda: preproc.train()
        pb, bb, pm, bm = diag(shim, diag_imgs, 32.0, dev)
        emit(f"ep{ep:3d}/{a.epochs} LF={acc['LF']:.4f} MOS={acc['MOS']:.4f} VIF={acc['VIF']:.4f} "
             f"BalleBpp={acc['bpp']:.3f} lam={acc['lam']:.4f} total={acc['total']:.4f} gnorm={gn/max(nb,1):.3f} "
             f"scaler={float(preproc.scaler):.4f} | [diag q32] base PSNR={pb:.2f}@{bb:.3f} "
             f"model PSNR={pm:.2f}@{bm:.3f} ({time.time()-t0:.0f}s)")
        metrics.append({"ep": ep, **{k: float(v) for k, v in acc.items()},
                        "scaler": float(preproc.scaler), "diag_model_psnr": float(pm), "diag_model_bpp": float(bm)})
    torch.save({"preproc": preproc.state_dict(), "entropy": entropy.state_dict()}, os.path.join(a.out_dir, "model.pt"))
    json.dump(metrics, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    emit(f"[balle] done -> {a.out_dir}/model.pt")


if __name__ == "__main__":
    main()
