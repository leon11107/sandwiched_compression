"""Pretrain + FREEZE the factorized entropy prior on the ALIGNED codec's
divisively-normalized luma DCT statistics (c/qvec(quality), quality~U{lo..hi}),
so the Phase-1 rate term is an honest, FIXED estimator in the deployment domain
(AUDIT fix #2). Reports a real-bits calibration table at the end.
torch-env. python dpp/pretrain_prior_aligned.py
"""
import argparse, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp.entropy import FactorizedEntropy
from dpp.train import CropDataset
from dpp.codec_aligned import AlignedJpegCodec, jpeg_rt_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/sandwiched_compression/dpp/runs/prior_aligned/entropy.pt")
    ap.add_argument("--img-dir", default="/workspace/sandwiched_compression/dpp/data/train_big")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--q-lo", type=int, default=5); ap.add_argument("--q-hi", type=int, default=32)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = "cuda"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    qrng = np.random.default_rng(11)
    cod = AlignedJpegCodec(device=dev)
    eb = FactorizedEntropy(64).to(dev)
    eb.eval()  # noise added manually in luma_coeffs_norm; eval() only disables internal noise
    opt = torch.optim.Adam(eb.parameters(), lr=a.lr)
    ds = CropDataset(a.img_dir, a.crop, a.steps * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)

    t0 = time.time(); run = 0.0; n = 0; npix = a.crop * a.crop
    for it, batch in enumerate(dl):
        x = batch.float().to(dev)
        q = int(qrng.integers(a.q_lo, a.q_hi + 1))
        gen = torch.Generator(device=dev); gen.manual_seed(int(qrng.integers(1 << 30)))
        with torch.no_grad():
            cq = cod.luma_coeffs_norm(x, q, generator=gen)
        bpp = (eb.bits(cq) / npix).mean()
        opt.zero_grad(); bpp.backward(); opt.step()
        run += float(bpp); n += 1
        if (it + 1) % 250 == 0:
            print(f"  step {it+1}/{a.steps} est_luma_bpp={run/n:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
            run = 0.0; n = 0
    for p in eb.parameters():
        p.requires_grad_(False)
    torch.save({"entropy": eb.state_dict(), "q_lo": a.q_lo, "q_hi": a.q_hi}, a.out)
    print(f"saved {a.out}", flush=True)

    # calibration: est luma bpp vs REAL total bpp on full val images, per quality
    import glob
    from PIL import Image
    paths = sorted(glob.glob("/workspace/sandwiched_compression/dpp/data/val50/*.png"))[:10]
    from dpp.eval_v2 import load16
    imgs = [load16(p) for p in paths]
    print("== calibration: frozen-prior est luma bpp vs real total bpp ==", flush=True)
    for q in [5, 8, 12, 20, 32]:
        est, real = [], []
        for im in imgs:
            x = torch.from_numpy(im[None]).float().to(dev)
            gen = torch.Generator(device=dev); gen.manual_seed(0)
            with torch.no_grad():
                cq = cod.luma_coeffs_norm(x, q, generator=gen)
                e = float(eb.bits(cq)[0]) / (im.shape[0] * im.shape[1])
            _, b = jpeg_rt_batch(im[None], q)
            est.append(e); real.append(float(b[0]) / (im.shape[0] * im.shape[1]))
        print(f"  q={q:3d} est_luma={np.mean(est):.4f} real_total={np.mean(real):.4f} "
              f"ratio={np.mean(est)/np.mean(real):.3f}", flush=True)


if __name__ == "__main__":
    main()
