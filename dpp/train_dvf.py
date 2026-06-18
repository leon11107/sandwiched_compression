"""(d)+(f) training: VMAF-aligned VIF objective + clean preproc-output anchor.

loss = w_vif*(1 - VIF(decoded, orig))            # (f) maximize VIF (VMAF core, FR, non-gameable)
     + w_anchor*||pre(x) - x||_1 [0,1]           # (d) clean fidelity anchor (NO codec grad -> no collapse)
     + lam * bpp                                  # rate
VIF gradient still flows through the codec backward (W1, weak ~+0.1) but is VMAF-ALIGNED,
so its push helps VMAF (unlike NIMA which gamed). The clean anchor contains the preproc.
"""
import argparse, glob, json, os, sys, time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace/sandwiched_compression")
import pyiqa
from dpp.model import DPPModel
from dpp.train import CropDataset, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-vif", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=0.005)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--qstep-lo", type=float, default=12.0); ap.add_argument("--qstep-hi", type=float, default=64.0)
    ap.add_argument("--codec-forward", default="proxy", choices=["proxy", "real_ste"])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    a = ap.parse_args()
    assert torch.cuda.is_available(); dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    qrng = np.random.default_rng(20260608)
    model = DPPModel(ch=a.ch, scaler_init=0.0, quantizer_mode="noise_injection",
                     codec_forward_mode=a.codec_forward, device=dev); model.train()
    vif = pyiqa.create_metric("vif", as_loss=True, device=dev)
    opt = torch.optim.Adam(model.preproc.parameters(), lr=a.lr)
    ds = CropDataset("/workspace/sandwiched_compression/dpp/data/train", a.crop, a.steps_per_epoch * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    diag_imgs = ds.imgs[:6]
    log = open(os.path.join(a.out_dir, "train.log"), "w")
    def emit(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    emit(f"[dvf] VIF obj + clean anchor: w_vif={a.w_vif} w_anchor={a.w_anchor} lam={a.lam} "
         f"codec={a.codec_forward} ch={a.ch} batch={a.batch} ep={a.epochs}")
    metrics = []
    for ep in range(a.epochs):
        t0 = time.time(); acc = {"vif": 0, "vifloss": 0, "anchor": 0, "bpp": 0, "total": 0}; gn = 0.0; nb = 0
        for batch in dl:
            x = batch.float().to(dev)
            q = float(qrng.uniform(a.qstep_lo, a.qstep_hi))
            gen = torch.Generator(device=dev); gen.manual_seed(int(qrng.integers(1 << 30)))
            out = model(x, input_qstep=q, generator=gen)
            pred = out["prediction"]; H, W = x.shape[1], x.shape[2]
            pred01 = (pred / 255.0).clamp(0, 1).permute(0, 3, 1, 2).contiguous()
            x01 = (x / 255.0).clamp(0, 1).permute(0, 3, 1, 2).contiguous()
            vif_val = vif(pred01, x01)                         # scalar mean, higher=better
            vif_loss = 1.0 - vif_val.clamp(max=1.0)            # cap at 1.0: no reward for VIF>1 (enhancement gaming)
            anchor = ((out["bottleneck"] - x) / 255.0).abs().mean()   # (d) clean, on preproc output
            bpp = (out["rate"] / (H * W)).mean()
            total = a.w_vif * vif_loss + a.w_anchor * anchor + a.lam * bpp
            if not torch.isfinite(total):
                continue  # NaN guard: skip degenerate step (VIF NaN on extreme inputs)
            opt.zero_grad(); total.backward()
            g = torch.nn.utils.clip_grad_norm_(model.preproc.parameters(), a.grad_clip)
            if not torch.isfinite(g):
                opt.zero_grad(); continue
            opt.step()
            acc["vif"] += float(vif_val); acc["vifloss"] += float(vif_loss); acc["anchor"] += float(anchor)
            acc["bpp"] += float(bpp); acc["total"] += float(total); gn += float(g); nb += 1
        for k in acc: acc[k] /= nb
        pb, bb, pm, bm = diag(model, diag_imgs, 32.0, dev)
        emit(f"ep{ep:3d}/{a.epochs} VIF={acc['vif']:.4f} anchor={acc['anchor']:.4f} bpp={acc['bpp']:.3f} "
             f"total={acc['total']:.4f} gnorm={gn/nb:.3f} scaler={float(model.preproc.scaler):.4f} | "
             f"[diag q32] base PSNR={pb:.2f}@{bb:.3f} model PSNR={pm:.2f}@{bm:.3f} ({time.time()-t0:.0f}s)")
        metrics.append({"ep": ep, **{k: float(v) for k, v in acc.items()},
                        "scaler": float(model.preproc.scaler), "diag_model_psnr": float(pm), "diag_model_bpp": float(bm)})
    torch.save(model.state_dict(), os.path.join(a.out_dir, "model.pt"))
    json.dump(metrics, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    emit(f"[dvf] done -> {a.out_dir}/model.pt")


if __name__ == "__main__":
    main()
