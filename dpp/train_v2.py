"""DPP Phase-1 trainer — aligned codec + honest rate + dual perceptual, fully
instrumented (implements all fixes from runs/AUDIT_REPORT.md).

  codec   : AlignedJpegCodec == eval_v2 deployment codec (Annex-K 4:2:0 real fwd,
            per-subband-qvec noise-injection luma proxy bwd), quality ~ U{--q-lo..--q-hi}
  rate    : pretrained FROZEN FactorizedEntropy on divisively-normalized luma coeffs
            (--entropy-ckpt REQUIRED; no co-training conflation)
  loss    : L = L_F + gamma*(-NIMA) + vif_gamma*(1-VIF_luma) + lam*bpp_est
            L_F = alpha*L1 + beta*(1-MS-SSIM) on luma, pred = decoded-Y + lossless UV
  dashboard (train_probe.py heritage): per-term values/shares, per-term grad norms +
            dScaler tug-of-war + pairwise cosines every --grad-every, est/real rate
            ratio, residual stats, luma HF out/in, per-epoch EVAL-codec diag at two
            qualities (must MOVE, else the run is dead — containment monitor).

torch-env, GPU-guarded. steps.jsonl + metrics.json + model.pt in --out-dir.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace/sandwiched_compression")
import pyiqa
from dpp.preproc_dpp import DPPPreproc
from dpp.entropy import FactorizedEntropy
from dpp.train import CropDataset
from dpp.codec_aligned import AlignedJpegCodec, jpeg_rt_batch
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
from torch_port.losses import ssim_multiscale_tf


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@torch.no_grad()
def eval_codec_diag(preproc, imgs, quality, dev):
    out = {}
    for arm in ["base", "model"]:
        ps, ms, bpp = [], [], []
        for im in imgs:
            src = im if arm == "base" else np.clip(
                preproc(torch.from_numpy(im[None]).float().to(dev))[0].cpu().numpy(), 0, 255)
            dec, bits = jpeg_rt_batch(src[None], quality)
            dec = dec[0]
            mse = np.mean((dec.astype(np.float64) - im.astype(np.float64)) ** 2)
            ps.append(10 * np.log10(255.0 ** 2 / mse))
            r = torch.from_numpy(im[None]).float().to(dev)
            d = torch.from_numpy(dec[None]).float().to(dev)
            ms.append(float(ssim_multiscale_tf(tf_rgb_to_yuv(r)[..., 0:1],
                                               tf_rgb_to_yuv(d)[..., 0:1],
                                               max_val=255.0, filter_size=11)[0]))
            bpp.append(float(bits[0]) / (im.shape[0] * im.shape[1]))
        out[arm] = (float(np.mean(ps)), float(np.mean(bpp)), float(np.mean(ms)))
    return out


@torch.no_grad()
def hf_ratio(cod, x, y):
    def hf(img):
        c = cod.jp._forward_dct_2d(cod.jp._rgb_to_yuv(img)[..., 0:1])
        idx = torch.arange(64, device=img.device); r, cc = idx // 8, idx % 8
        return float((c[..., (r + cc) >= 8] ** 2).mean())
    return hf(y) / max(hf(x), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--entropy-ckpt", required=True)
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/train_big")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.0, help="NIMA weight")
    ap.add_argument("--vif-gamma", type=float, default=0.0, help="(1-VIF) luma weight")
    ap.add_argument("--lam", type=float, default=0.1, help="rate weight (honest est bpp units)")
    ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--q-lo", type=int, default=5); ap.add_argument("--q-hi", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--grad-every", type=int, default=50)
    ap.add_argument("--diag-qualities", default="8,20")
    ap.add_argument("--seed", type=int, default=20260610)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    qrng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    preproc = DPPPreproc(ch=a.ch, scaler_init=0.0).to(dev)
    cod = AlignedJpegCodec(device=dev)
    entropy = FactorizedEntropy(64).to(dev)
    ck = torch.load(a.entropy_ckpt, map_location=dev)
    entropy.load_state_dict(ck["entropy"])
    for p in entropy.parameters():
        p.requires_grad_(False)
    entropy.eval()
    # per-quality calibration: bpp_est/k(q) ~= REAL luma bpp (lam in real-bpp units)
    calib = {int(k): float(v) for k, v in ck.get("calib_luma", {}).items()}
    kq = lambda q: calib.get(int(q), 1.0)
    nima = pyiqa.create_metric("nima-koniq", as_loss=True, device=dev) if a.gamma > 0 else None
    vif = pyiqa.create_metric("vif", as_loss=True, device=dev) if a.vif_gamma > 0 else None
    opt = torch.optim.Adam(preproc.parameters(), lr=a.lr)
    preproc.train()
    pparams = list(preproc.parameters())

    ds = CropDataset(a.img_dir, a.crop, a.steps_per_epoch * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    diag_imgs = [im.astype(np.float32) for im in ds.imgs[:6]]  # ds caches uint8
    diag_q = [int(x) for x in a.diag_qualities.split(",")]
    slog = open(os.path.join(a.out_dir, "steps.jsonl"), "w")
    def emit(s): print(s, flush=True)
    emit(f"[v2] aligned-codec DPP: gamma={a.gamma} vif_gamma={a.vif_gamma} lam={a.lam} "
         f"q~U[{a.q_lo},{a.q_hi}] ch={a.ch} ep={a.epochs} prior={a.entropy_ckpt} "
         f"calib_luma={'yes(' + str(len(calib)) + 'q)' if calib else 'NO'}")

    gstep = 0; metrics = []; npix_crop = a.crop * a.crop
    for ep in range(a.epochs):
        t0 = time.time(); acc = {}
        for batch in dl:
            x = batch.float().to(dev)
            q = int(qrng.integers(a.q_lo, a.q_hi + 1))
            gen = torch.Generator(device=dev); gen.manual_seed(int(qrng.integers(1 << 30)))
            bottleneck = preproc(x)
            dec, real_bits = cod(bottleneck, q, generator=gen)
            yuv_dec = tf_rgb_to_yuv(dec); yuv_in = tf_rgb_to_yuv(x)
            pred = tf_yuv_to_rgb(torch.cat([yuv_dec[..., 0:1], yuv_in[..., 1:3]], -1))
            Yx = yuv_in[..., 0:1]; Yp = tf_rgb_to_yuv(pred)[..., 0:1]
            # L_F
            l1 = ((Yx - Yp) / 255.0).abs().mean()
            msl = (1 - ssim_multiscale_tf(Yx, Yp, max_val=255.0, filter_size=7)).mean()
            LF = a.alpha * l1 + a.beta * msl
            # honest rate (frozen prior, divisive norm, calibrated to real-luma-bpp units)
            cq = cod.luma_coeffs_norm(bottleneck, q, generator=gen)
            bpp_est = (entropy.bits(cq) / npix_crop).mean() / kq(q)
            # perceptual terms
            if nima is not None:
                p01 = (pred / 255).clamp(0, 1).permute(0, 3, 1, 2).contiguous()
                mos = nima(p01); LP = -mos
            else:
                mos = torch.tensor(0.0, device=dev); LP = torch.tensor(0.0, device=dev)
            if vif is not None:
                Yx3 = (Yx / 255.0).clamp(0, 1).permute(0, 3, 1, 2).repeat(1, 3, 1, 1).contiguous()
                Yp3 = (Yp / 255.0).clamp(0, 1).permute(0, 3, 1, 2).repeat(1, 3, 1, 1).contiguous()
                vif_val = vif(Yp3, Yx3).clamp(0.0, 1.0)  # clamp: VIF singular on flat blocks
                LV = 1.0 - vif_val
            else:
                vif_val = torch.tensor(0.0, device=dev); LV = torch.tensor(0.0, device=dev)
            total = LF + a.gamma * LP + a.vif_gamma * LV + a.lam * bpp_est
            if not torch.isfinite(total):
                emit(f"  [NaN-guard] step {gstep} skipped"); continue

            rec = {"step": gstep, "ep": ep, "q": q, "L1_raw": float(l1),
                   "msssim_loss_raw": float(msl), "LF": float(LF), "MOS": float(mos),
                   "VIF": float(vif_val), "bpp_est": float(bpp_est),
                   "w_perc": float(a.gamma * (-mos)), "w_vif": float(a.vif_gamma * LV),
                   "w_rate": float(a.lam * bpp_est), "total": float(total),
                   "scaler": float(preproc.scaler)}
            tot_abs = sum(abs(rec[k]) for k in ["LF", "w_perc", "w_vif", "w_rate"]) + 1e-12
            for k, s in [("LF", "sh_LF"), ("w_perc", "sh_P"), ("w_vif", "sh_V"), ("w_rate", "sh_R")]:
                rec[s] = abs(rec[k]) / tot_abs

            if gstep % a.grad_every == 0:
                terms = {"LF": LF, "rate": bpp_est}
                if nima is not None: terms["nima"] = LP
                if vif is not None: terms["vif"] = LV
                gs, dsc = {}, {}
                for name, t in terms.items():
                    gs[name] = flat(torch.autograd.grad(t, pparams, retain_graph=True,
                                                        allow_unused=True))
                    sg = torch.autograd.grad(t, preproc.scaler, retain_graph=True,
                                             allow_unused=True)[0]
                    dsc[name] = float(sg) if sg is not None else 0.0
                w = {"LF": 1.0, "rate": a.lam, "nima": a.gamma, "vif": a.vif_gamma}
                real_bpp = float((real_bits / npix_crop).mean())
                rec.update({f"gnorm_{k}_w": float(w[k] * v.norm()) for k, v in gs.items()})
                rec.update({f"dscaler_{k}": v for k, v in dsc.items()})
                names = list(gs)
                rec.update({f"cos_{m}_{n}": cos(gs[m], gs[n])
                            for i, m in enumerate(names) for n in names[i + 1:]})
                rec.update({"est_real_ratio": float(bpp_est) / max(real_bpp, 1e-9),
                            "real_bpp": real_bpp,
                            "res_mean": float((bottleneck - x).abs().mean()),
                            "res_max": float((bottleneck - x).abs().max()),
                            "hf_out_in": hf_ratio(cod, x, bottleneck)})
                emit("  [grad s%4d] %s | dScaler %s | cos %s | est/real=%.2f (real %.3f) "
                     "| res %.2f/%.1f hfO/I=%.3f" % (
                         gstep,
                         " ".join(f"|g{k}|w={rec[f'gnorm_{k}_w']:.2e}" for k in names),
                         " ".join(f"{k}={dsc[k]:+.2e}" for k in names),
                         " ".join(f"{m[:2]}-{n[:2]}={rec[f'cos_{m}_{n}']:+.2f}"
                                  for i, m in enumerate(names) for n in names[i + 1:]),
                         rec["est_real_ratio"], real_bpp, rec["res_mean"], rec["res_max"],
                         rec["hf_out_in"]))

            opt.zero_grad(); total.backward()
            g = torch.nn.utils.clip_grad_norm_(pparams, a.grad_clip)
            rec["gnorm_total"] = float(g)
            if not torch.isfinite(g):
                opt.zero_grad(); continue
            opt.step()
            if gstep % a.log_every == 0:
                emit(f"[s{gstep:5d} ep{ep:2d} q={q:2d}] L1={rec['L1_raw']:.4f} "
                     f"(1-MS)={rec['msssim_loss_raw']:.4f} VIF={rec['VIF']:.3f} "
                     f"MOS={rec['MOS']:.3f} bppE={rec['bpp_est']:.3f} | tot={rec['total']:.4f} "
                     f"sh LF/P/V/R={rec['sh_LF']:.2f}/{rec['sh_P']:.2f}/{rec['sh_V']:.2f}/"
                     f"{rec['sh_R']:.2f} | scl={rec['scaler']:+.4f} g={rec['gnorm_total']:.3f}")
            slog.write(json.dumps(rec) + "\n"); slog.flush()
            for k in ["LF", "VIF", "MOS", "bpp_est", "total"]:
                acc[k] = acc.get(k, 0.0) + rec[k if k != "bpp_est" else "bpp_est"]
            gstep += 1
        nb = max(1, a.steps_per_epoch)
        preproc.eval()
        diags = {q: eval_codec_diag(preproc, diag_imgs, q, dev) for q in diag_q}
        preproc.train()
        dstr = " | ".join(
            f"q{q}: base {d['base'][0]:.2f}dB@{d['base'][1]:.3f} ms={d['base'][2]:.5f} -> "
            f"model {d['model'][0]:.2f}dB@{d['model'][1]:.3f} ms={d['model'][2]:.5f}"
            for q, d in diags.items())
        emit(f"== ep{ep:2d} ({time.time()-t0:.0f}s) LF={acc['LF']/nb:.4f} VIF={acc['VIF']/nb:.3f} "
             f"MOS={acc['MOS']/nb:.3f} bppE={acc['bpp_est']/nb:.3f} scl={float(preproc.scaler):+.4f}")
        emit(f"   EVAL-diag {dstr}")
        metrics.append({"ep": ep, **{k: float(v / nb) for k, v in acc.items()},
                        "scaler": float(preproc.scaler),
                        "diag": {str(q): {k: [float(x) for x in v] for k, v in d.items()}
                                 for q, d in diags.items()}})
        torch.save({"preproc": preproc.state_dict()}, os.path.join(a.out_dir, "model.pt"))
        json.dump(metrics, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    emit(f"[v2] done -> {a.out_dir}")


if __name__ == "__main__":
    main()
