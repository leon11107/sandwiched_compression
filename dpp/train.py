"""DPP training loop (torch, GPU). Built on the TF-equivalence-validated torch_port
codec. DPP-faithful: dilated-conv luma preproc, gamma*L_P[NIMA] + lambda*L_R + L_F,
noise-injection quant, QP-marginalization (random qstep per step). Default training
codec = differentiable proxy (DPP virtual codec); --codec-forward real_ste optional.
"""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp.model import DPPModel
from dpp.perceptual import NimaMOS
from dpp.loss import dpp_loss
from torch_port.codec import encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb


class CropDataset(Dataset):
    def __init__(self, img_dir, crop=128, length=2000):
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert self.paths, f"no images in {img_dir}"
        # uint8 cache (4x less RAM than f32; 8 concurrent runs on train_big = ~106GB
        # in f32, OOM-risk on the 94GB box). __getitem__ converts the CROP to f32.
        self.imgs = [np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in self.paths]
        self.crop = crop; self.length = length
        self.rng = np.random.default_rng(0)

    def __len__(self): return self.length

    def __getitem__(self, i):
        im = self.imgs[i % len(self.imgs)]
        H, W = im.shape[:2]; c = self.crop
        top = np.random.randint(0, H - c + 1); left = np.random.randint(0, W - c + 1)
        patch = im[top:top + c, left:left + c, :].astype(np.float32)
        if np.random.rand() < 0.5: patch = patch[:, ::-1, :].copy()
        if np.random.rand() < 0.5: patch = patch[::-1, :, :].copy()
        return torch.from_numpy(np.ascontiguousarray(patch))  # [H,W,3] 0..255


def real_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


@torch.no_grad()
def diag(model, imgs, qstep, dev):
    """Real-JPEG PSNR/bpp at fixed qstep: preprocessed vs baseline (no preproc)."""
    model.eval()
    def restore(dec, orig):
        d = tf_rgb_to_yuv(torch.from_numpy(dec[None]).float()); o = tf_rgb_to_yuv(torch.from_numpy(orig[None]).float())
        return np.clip(tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))[0].numpy(), 0, 255)
    pm, pb, bm, bb = [], [], [], []
    for im in imgs:
        x = torch.from_numpy(im[None]).float().to(dev)
        pre = np.clip(model.preproc(x)[0].cpu().numpy(), 0, 255)
        dm, bits_m = encode_decode_with_jpeg(pre[None], qstep, False, False)
        db, bits_b = encode_decode_with_jpeg(im[None], qstep, False, False)
        dm = restore(dm[0], im); db = restore(db[0], im)
        pm.append(real_psnr(dm, im)); bm.append(bits_m[0] / (im.shape[0] * im.shape[1]))
        pb.append(real_psnr(db, im)); bb.append(bits_b[0] / (im.shape[0] * im.shape[1]))
    model.train()
    return np.mean(pb), np.mean(bb), np.mean(pm), np.mean(bm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/train")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.01)   # perceptual coeff
    ap.add_argument("--lam", type=float, default=0.005)    # rate coeff
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--qstep-lo", type=float, default=12.0)
    ap.add_argument("--qstep-hi", type=float, default=64.0)
    ap.add_argument("--perceptual", default="nima-koniq")
    ap.add_argument("--codec-forward", default="proxy", choices=["proxy", "real_ste"])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU required (gpu hard rule)"
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    dev = a.device
    qrng = np.random.default_rng(20260607)

    model = DPPModel(ch=a.ch, scaler_init=0.0, quantizer_mode="noise_injection",
                     codec_forward_mode=a.codec_forward, device=dev)
    model.train()
    nima = NimaMOS(a.perceptual, device=dev)
    opt = torch.optim.Adam(model.preproc.parameters(), lr=a.lr)
    ds = CropDataset(a.img_dir, a.crop, a.steps_per_epoch * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    diag_imgs = ds.imgs[:6]
    log = open(os.path.join(a.out_dir, "train.log"), "w")
    def emit(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    emit(f"[dpp-train] {a.perceptual} codec={a.codec_forward} gamma={a.gamma} lam={a.lam} "
         f"qstep[{a.qstep_lo},{a.qstep_hi}] ch={a.ch} batch={a.batch} ep={a.epochs}")

    metrics = []
    for ep in range(a.epochs):
        t0 = time.time(); accum = {"LF": 0, "LP": 0, "LR_bpp": 0, "MOS": 0, "total": 0}
        gnorm_acc = 0.0; nb = 0
        for batch in dl:
            x = batch.float().to(dev)
            q = float(qrng.uniform(a.qstep_lo, a.qstep_hi))  # QP-marginalization
            gen = torch.Generator(device=dev); gen.manual_seed(int(qrng.integers(1 << 30)))
            out = model(x, input_qstep=q, generator=gen)
            total, comp = dpp_loss(x, out["prediction"], out["rate"], nima, a.gamma, a.lam)
            opt.zero_grad(); total.mean().backward()
            gn = torch.nn.utils.clip_grad_norm_(model.preproc.parameters(), a.grad_clip)
            opt.step()
            for k in accum: accum[k] += float(comp[k])
            gnorm_acc += float(gn); nb += 1
        for k in accum: accum[k] /= nb
        pb, bb, pm, bm = diag(model, diag_imgs, 32.0, dev)
        emit(f"ep{ep:3d}/{a.epochs} LF={accum['LF']:.4f} LP={accum['LP']:.4f} "
             f"MOS={accum['MOS']:.4f} LR={accum['LR_bpp']:.3f}bpp total={accum['total']:.4f} "
             f"gnorm={gnorm_acc/nb:.3f} scaler={float(model.preproc.scaler):.4f} | "
             f"[diag q32] base PSNR={pb:.2f}@{bb:.3f} model PSNR={pm:.2f}@{bm:.3f} "
             f"({time.time()-t0:.0f}s)")
        metrics.append({"ep": ep, **{k: float(v) for k, v in accum.items()},
                        "scaler": float(model.preproc.scaler),
                        "diag_base_psnr": float(pb), "diag_base_bpp": float(bb),
                        "diag_model_psnr": float(pm), "diag_model_bpp": float(bm)})
    torch.save(model.state_dict(), os.path.join(a.out_dir, "model.pt"))
    json.dump(metrics, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    emit(f"[dpp-train] done -> {a.out_dir}/model.pt")


if __name__ == "__main__":
    main()
