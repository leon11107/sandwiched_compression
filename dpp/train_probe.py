"""Instrumented PROBE trainer — full loss/grad dashboard for architecture debugging.

Replicates the champion recipe (train_balle.py: L = L_F + gamma*(-NIMA) + lam*BalleBpp,
co-trained FactorizedEntropy, real_ste codec, noise_injection quantizer, QP-marginalized)
but prints EVERYTHING needed to see what the optimization is actually doing:

per --log-every steps:
  L1_raw, (1-MSSSIM)_raw, L_F, MOS, gamma*L_P, BalleBpp, lam*bpp, total
  + term shares of |total|, sampled qstep, scaler value
per --grad-every steps (separate backward per term, preproc params only):
  ||g_LF||, ||g_rate||(unweighted), ||g_perc||, weighted norms,
  pairwise cos(g_LF,g_rate), cos(g_LF,g_perc), cos(g_rate,g_perc),
  per-term d(term)/d(scaler)  (ReZero gate: sign = wants MORE(+)/LESS(-) preprocessing)
  rate calibration: BalleBpp vs REAL bits bpp (codec real_ste exact) ratio same batch
  residual stats: mean|res|, max|res|, luma HF-energy ratio out/in (DCT r+c>=8)
per epoch (dual-codec diag, 6 fixed images):
  TRAINING codec (flat qstep 32, 4:4:4, lossless chroma) PSNR/bpp base vs model
  EVAL codec     (Annex-K q=12, 4:2:0 full lossy)        PSNR/bpp/MS-SSIM base vs model
Writes steps.jsonl + metrics.json. torch-env, GPU-guarded. SHORT probe (default 3 ep).
"""
import argparse, json, os, sys, time
import io
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
sys.path.insert(0, "/workspace/sandwiched_compression")
import pyiqa
from dpp.preproc_dpp import DPPPreproc
from dpp.entropy import FactorizedEntropy
from dpp.train import CropDataset, diag
from torch_port.codec import EncodeDecodeIntraTorch, JpegProxyTorch
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
from torch_port.losses import ssim_multiscale_tf


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def jpeg_annexk(img, quality):
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8)).save(
        buf, format="jpeg", quality=int(quality), subsampling="4:2:0", optimize=True)
    return np.asarray(Image.open(buf).convert("RGB"), np.float32), 8 * len(buf.getbuffer())


@torch.no_grad()
def eval_codec_diag(preproc, imgs, quality, dev):
    """EVAL-protocol diag: Annex-K 4:2:0 full lossy, metrics vs original."""
    out = {}
    for arm in ["base", "model"]:
        ps, ms, bpp = [], [], []
        for im in imgs:
            src = im if arm == "base" else np.clip(
                preproc(torch.from_numpy(im[None]).float().to(dev))[0].cpu().numpy(), 0, 255)
            dec, bits = jpeg_annexk(src, quality)
            mse = np.mean((dec.astype(np.float64) - im.astype(np.float64)) ** 2)
            ps.append(10 * np.log10(255.0 ** 2 / mse))
            r = torch.from_numpy(im[None]).float().to(dev)
            d = torch.from_numpy(dec[None]).float().to(dev)
            ms.append(float(ssim_multiscale_tf(tf_rgb_to_yuv(r)[..., 0:1],
                                               tf_rgb_to_yuv(d)[..., 0:1],
                                               max_val=255.0, filter_size=11)[0]))
            bpp.append(bits / (im.shape[0] * im.shape[1]))
        out[arm] = (float(np.mean(ps)), float(np.mean(bpp)), float(np.mean(ms)))
    return out


@torch.no_grad()
def hf_ratio(jp, x, y):
    """luma HF (r+c>=8) energy ratio y/x."""
    def hf(img):
        c = jp._forward_dct_2d(jp._rgb_to_yuv(img)[..., 0:1])
        idx = torch.arange(64, device=img.device); r, cc = idx // 8, idx % 8
        return float((c[..., (r + cc) >= 8] ** 2).mean())
    return hf(y) / max(hf(x), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/train")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--qstep-lo", type=float, default=12.0); ap.add_argument("--qstep-hi", type=float, default=64.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--grad-every", type=int, default=25)
    ap.add_argument("--eval-quality", type=int, default=12)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    qrng = np.random.default_rng(20260610)

    preproc = DPPPreproc(ch=a.ch, scaler_init=0.0).to(dev)
    entropy = FactorizedEntropy(64).to(dev)
    codec = EncodeDecodeIntraTorch(qstep_init=32.0, train_qstep=False, min_qstep=1.0,
        quantizer_mode="noise_injection", rate_proxy_mode="log_nonzero",
        codec_forward_mode="real_ste", output_clip_mode="hard", convert_to_yuv=True, device=dev)
    jp = JpegProxyTorch(convert_to_yuv=True, clip_to_image_max=True, device=dev)
    nima = pyiqa.create_metric("nima-koniq", as_loss=True, device=dev)
    opt = torch.optim.Adam(list(preproc.parameters()) + list(entropy.parameters()), lr=a.lr)
    preproc.train(); entropy.train()
    pparams = list(preproc.parameters())

    ds = CropDataset(a.img_dir, a.crop, a.steps_per_epoch * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    diag_imgs = ds.imgs[:6]
    slog = open(os.path.join(a.out_dir, "steps.jsonl"), "w")
    def emit(s): print(s, flush=True)
    emit(f"[probe] champion recipe instrumented: gamma={a.gamma} lam={a.lam} "
         f"ch={a.ch} ep={a.epochs} q~U[{a.qstep_lo},{a.qstep_hi}] eval_q={a.eval_quality}")

    def balle_bpp(bottleneck, q, gen):
        yuv = jp._rgb_to_yuv(bottleneck); y = yuv[..., 0:1]
        coeffs = jp._forward_dct_2d(y)
        qd = (torch.rand(coeffs.shape, device=dev, generator=gen) - 0.5)
        cq = (coeffs / q + qd).permute(0, 3, 1, 2).contiguous()
        return entropy.bits(cq), bottleneck.shape[1] * bottleneck.shape[2]

    gstep = 0; metrics = []
    for ep in range(a.epochs):
        t0 = time.time()
        for batch in dl:
            x = batch.float().to(dev)
            q = float(qrng.uniform(a.qstep_lo, a.qstep_hi))
            gen = torch.Generator(device=dev); gen.manual_seed(int(qrng.integers(1 << 30)))
            bottleneck = preproc(x)
            dec, real_rate = codec(bottleneck, input_qstep=q, generator=gen)
            yuv_dec = tf_rgb_to_yuv(dec); yuv_in = tf_rgb_to_yuv(x)
            pred = tf_yuv_to_rgb(torch.cat([yuv_dec[..., 0:1], yuv_in[..., 1:3]], -1))
            Yx = yuv_in[..., 0:1]; Yp = tf_rgb_to_yuv(pred)[..., 0:1]
            l1 = ((Yx - Yp) / 255.0).abs().mean()
            msl = (1 - ssim_multiscale_tf(Yx, Yp, max_val=255.0, filter_size=7)).mean()
            LF = a.alpha * l1 + a.beta * msl
            bits, npix = balle_bpp(bottleneck, q, gen); bpp = (bits / npix).mean()
            p01 = (pred / 255).clamp(0, 1).permute(0, 3, 1, 2).contiguous()
            mos = nima(p01); LP = -mos
            total = LF + a.gamma * LP + a.lam * bpp
            if not torch.isfinite(total):
                emit(f"  [NaN-guard] step {gstep}: total not finite, skipped"); continue

            rec = {"step": gstep, "ep": ep, "q": q,
                   "L1_raw": float(l1), "msssim_loss_raw": float(msl), "LF": float(LF),
                   "MOS": float(mos), "gLP_w": float(a.gamma * (-mos)),
                   "balle_bpp": float(bpp), "lam_bpp_w": float(a.lam * bpp),
                   "total": float(total), "scaler": float(preproc.scaler)}
            tot_abs = abs(rec["LF"]) + abs(rec["gLP_w"]) + abs(rec["lam_bpp_w"])
            rec["share_LF"] = abs(rec["LF"]) / tot_abs
            rec["share_perc"] = abs(rec["gLP_w"]) / tot_abs
            rec["share_rate"] = abs(rec["lam_bpp_w"]) / tot_abs

            if gstep % a.grad_every == 0:
                # per-term grads on PREPROC only (entropy excluded -> pure signal to net)
                gLF = flat(torch.autograd.grad(LF, pparams, retain_graph=True, allow_unused=True))
                gR = flat(torch.autograd.grad(bpp, pparams, retain_graph=True, allow_unused=True))
                gP = flat(torch.autograd.grad(LP, pparams, retain_graph=True, allow_unused=True))
                sLF = torch.autograd.grad(LF, preproc.scaler, retain_graph=True, allow_unused=True)[0]
                sR = torch.autograd.grad(bpp, preproc.scaler, retain_graph=True, allow_unused=True)[0]
                sP = torch.autograd.grad(LP, preproc.scaler, retain_graph=True, allow_unused=True)[0]
                # rate calibration: balle proxy vs real bits (same batch, fwd values)
                real_bpp = float((real_rate / npix).mean())
                rec.update({
                    "gnorm_LF": float(gLF.norm()), "gnorm_rate_raw": float(gR.norm()),
                    "gnorm_perc_raw": float(gP.norm()),
                    "gnorm_rate_w": float(a.lam * gR.norm()), "gnorm_perc_w": float(a.gamma * gP.norm()),
                    "cos_LF_rate": cos(gLF, gR), "cos_LF_perc": cos(gLF, gP),
                    "cos_rate_perc": cos(gR, gP),
                    "dscaler_LF": float(sLF) if sLF is not None else 0.0,
                    "dscaler_rate": float(sR) if sR is not None else 0.0,
                    "dscaler_perc": float(sP) if sP is not None else 0.0,
                    "balle_vs_real_ratio": float(bpp) / max(real_bpp, 1e-9),
                    "real_bpp": real_bpp,
                    "res_mean": float((bottleneck - x).abs().mean()),
                    "res_max": float((bottleneck - x).abs().max()),
                    "hf_out_in": hf_ratio(jp, x, bottleneck)})
                emit(f"  [grad s{gstep:4d}] |gLF|={rec['gnorm_LF']:.3e} "
                     f"|gR|raw={rec['gnorm_rate_raw']:.3e} w={rec['gnorm_rate_w']:.3e} "
                     f"|gP|raw={rec['gnorm_perc_raw']:.3e} w={rec['gnorm_perc_w']:.3e} | "
                     f"cos(LF,R)={rec['cos_LF_rate']:+.2f} cos(LF,P)={rec['cos_LF_perc']:+.2f} "
                     f"cos(R,P)={rec['cos_rate_perc']:+.2f} | "
                     f"dScaler LF={rec['dscaler_LF']:+.2e} R={rec['dscaler_rate']:+.2e} "
                     f"P={rec['dscaler_perc']:+.2e} | balle/real={rec['balle_vs_real_ratio']:.2f} "
                     f"(real {rec['real_bpp']:.3f}bpp) | res mean/max={rec['res_mean']:.2f}/"
                     f"{rec['res_max']:.1f} hfO/I={rec['hf_out_in']:.3f}")

            opt.zero_grad(); total.backward()
            g_pre = torch.nn.utils.clip_grad_norm_(pparams, a.grad_clip)
            g_ent = torch.nn.utils.clip_grad_norm_(entropy.parameters(), a.grad_clip)
            rec["gnorm_total_preclip"] = float(g_pre); rec["gnorm_entropy"] = float(g_ent)
            if not torch.isfinite(g_pre): opt.zero_grad(); continue
            opt.step()

            if gstep % a.log_every == 0:
                emit(f"[s{gstep:4d} ep{ep} q={q:5.1f}] L1={rec['L1_raw']:.4f} "
                     f"(1-MS)={rec['msssim_loss_raw']:.4f} LF={rec['LF']:.4f} MOS={rec['MOS']:.3f} "
                     f"bpp={rec['balle_bpp']:.3f} | total={rec['total']:.4f} "
                     f"shares LF/P/R={rec['share_LF']:.2f}/{rec['share_perc']:.2f}/{rec['share_rate']:.2f} "
                     f"| scaler={rec['scaler']:+.4f} gnorm={rec['gnorm_total_preclip']:.3f}")
            slog.write(json.dumps(rec) + "\n"); slog.flush()
            gstep += 1

        # dual-codec epoch diag
        class _M: pass
        shim = _M(); shim.preproc = preproc
        shim.eval = lambda: preproc.eval(); shim.train = lambda: preproc.train()
        pb, bb, pm, bm = diag(shim, diag_imgs, 32.0, dev)
        preproc.eval(); ev = eval_codec_diag(preproc, diag_imgs, a.eval_quality, dev); preproc.train()
        emit(f"== ep{ep} done ({time.time()-t0:.0f}s) ==")
        emit(f"   TRAIN-codec diag (flat q32, lossless chroma): base {pb:.2f}dB@{bb:.3f} "
             f"-> model {pm:.2f}dB@{bm:.3f}")
        emit(f"   EVAL-codec  diag (Annex-K q{a.eval_quality}, 4:2:0): "
             f"base {ev['base'][0]:.2f}dB@{ev['base'][1]:.3f} ms={ev['base'][2]:.5f} "
             f"-> model {ev['model'][0]:.2f}dB@{ev['model'][1]:.3f} ms={ev['model'][2]:.5f}")
        metrics.append({"ep": ep, "train_diag": [float(v) for v in (pb, bb, pm, bm)],
                        "eval_diag": {k: [float(x) for x in v] for k, v in ev.items()}})
    torch.save({"preproc": preproc.state_dict(), "entropy": entropy.state_dict()},
               os.path.join(a.out_dir, "model.pt"))
    json.dump(metrics, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    emit(f"[probe] done -> {a.out_dir}")


if __name__ == "__main__":
    main()
