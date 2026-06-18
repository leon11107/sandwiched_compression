"""S1: DPP precoder training, intra-only virtual codec (115100C.pdf S3-S4).

Faithful pieces: luma-only dilated-CNN precoder Conv(3,16,{1,1,2,4,8}) +
Conv(3,1,1) with PReLU (paper S4.1; + declared global residual skip with
zero-init last conv for identity start); soft block-matching INTRA prediction
(random K in {4,8,16}, M=24) on the precoded frame; H.264 4x4 integer DCT of
the residual; QP-randomized Qstep with uniform-noise quantization; Balle
factorized rate over 16 subbands; L = gamma*L_P + lambda*L_R + L_F with
L_F = 0.2*L1 + 0.8*(1-MS-SSIM), gamma=0.1, lambda in {0.01 (enh0m),
0.001 (enh3m)}; Adam 1e-4, 40k steps, x0.1 decay at 20k; 512x512 crops.

DECLARED SUBSTITUTION: perceptual L_P uses our faithful differentiable
VMAF-NEG (maximize NEG(x, p_hat); MAE 0.016 vs binary) instead of the
proprietary iSIZE NR-IQA MOS model. Training data: public XIPH frames
(/dev/shm/dppv/src) + CLIC train_big stills (intra-only stage).
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp_repro.virtual_codec import (VirtualCodec, intra_pred, qstep_of_qp,
                                     soft_block_pred)
from torch_port.losses import ssim_multiscale_tf

RUNS = "/workspace/sandwiched_compression/dpp/runs"
SRC = "/dev/shm/dppv/src"
STILLS = "/workspace/sandwiched_compression/dpp/data/train_big"
CROP = 512


class Precoder(nn.Module):
    def __init__(self):
        super().__init__()
        spec = [(1, 16, 1), (16, 16, 1), (16, 16, 2), (16, 16, 4), (16, 16, 8)]
        layers = []
        for cin, cout, d in spec:
            layers += [nn.Conv2d(cin, cout, 3, 1, d, dilation=d), nn.PReLU(cout)]
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(16, 1, 3, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):  # identity at init (zero-init out + global skip)
        return x + self.out(self.body(x))


def load_frames():
    """y4m luma frames (1-in-5) + consecutive PAIRS (for inter) + stills."""
    from dpp_repro.y4m import read_y4m
    # NEVER train on the held-out evaluation clips (the gamma-ablation runs
    # of 06-12 were contaminated this way — disclosed; effect measured ~0
    # for this model class, but excluded strictly from here on)
    HOLDOUT = ("aspen_1080p", "red_kayak_1080p", "west_wind_easy_1080p",
               "controlled_burn_1080p")
    frames, pairs = [], []
    for f in sorted(glob.glob(f"{SRC}/*.y4m")):
        b = os.path.basename(f)
        if "__" in b or any(b.startswith(h) for h in HOLDOUT):
            continue  # skip precoded variants + eval clips
        clip = [y for y, _, _ in read_y4m(f)]
        frames += clip[::5]
        pairs += [(clip[i], clip[i + 1]) for i in range(0, len(clip) - 1, 2)]
    from PIL import Image
    stills = []
    for p in sorted(glob.glob(f"{STILLS}/*.png"))[::2]:
        im = np.asarray(Image.open(p).convert("RGB"), np.float32)
        y = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
        y = 16.0 + y * (219.0 / 255.0)  # limited range, same as y4m Y planes
        if min(y.shape) >= CROP:
            stills.append(np.rint(y).astype(np.uint8))
    print(f"frames {len(frames)} pairs {len(pairs)} stills {len(stills)}",
          flush=True)
    return frames, pairs, stills


def batch(frames, stills, bs, rng, dev):
    out = []
    for _ in range(bs):
        pool = frames if rng.random() < 0.5 else stills
        a = pool[rng.integers(len(pool))]
        oy = rng.integers(0, a.shape[0] - CROP + 1)
        ox = rng.integers(0, a.shape[1] - CROP + 1)
        out.append(a[oy:oy + CROP, ox:ox + CROP])
    x = torch.from_numpy(np.stack(out)).float().to(dev)[:, None]
    return x


def batch_pairs(pairs, bs, rng, dev):
    prev, cur = [], []
    for _ in range(bs):
        a, b = pairs[rng.integers(len(pairs))]
        oy = rng.integers(0, a.shape[0] - CROP + 1)
        ox = rng.integers(0, a.shape[1] - CROP + 1)
        prev.append(a[oy:oy + CROP, ox:ox + CROP])
        cur.append(b[oy:oy + CROP, ox:ox + CROP])
    t = lambda L: torch.from_numpy(np.stack(L)).float().to(dev)[:, None]
    return t(prev), t(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True, help="0.01 / 0.001")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--inter", action="store_true",
                    help="S2: inter prediction from previous precoded frame "
                         "(paper regime, 99:1 inter:intra)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    name = f"{'s2' if a.inter else 's1'}_lam{a.lam:g}" + \
        (f"_g{a.gamma:g}" if a.gamma != 0.1 else "") + a.tag
    os.makedirs(f"{RUNS}/{name}", exist_ok=True)

    from vmaf_torch import VMAF
    vt = VMAF(NEG=True).to(dev)
    for p_ in vt.parameters():
        p_.requires_grad_(False)
    pre = Precoder().to(dev)
    vc = VirtualCodec().to(dev)
    params = list(pre.parameters()) + \
        [p for n, p in vc.named_parameters() if not n.endswith("quantiles")]
    aux = [p for n, p in vc.named_parameters() if n.endswith("quantiles")]
    opt = torch.optim.Adam(params, lr=1e-4)
    opt_aux = torch.optim.Adam(aux, lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, a.steps // 2, 0.1)
    frames, pairs, stills = load_frames()
    rng = np.random.default_rng(7 + a.gpu)

    logs = {"lp": [], "lr": [], "lf": [], "neg": [], "mss": [], "dbits": []}
    t0 = time.time()
    for s in range(a.steps):
        k = int(rng.choice([4, 8, 16]))
        use_inter = a.inter and rng.random() >= 0.01  # paper: 1-in-100 intra
        if use_inter:
            x_prev, x = batch_pairs(pairs, a.batch, rng, dev)
            p = pre(x)
            p_prev = pre(x_prev)  # shared weights, open-loop (paper Fig.1)
            pred = soft_block_pred(p, p_prev, k, m=24, tau=1.0)
        else:
            x = batch(frames, stills, a.batch, rng, dev)
            p = pre(x)
            pred = intra_pred(p, k, m=24, tau=1.0)
        qp = int(rng.integers(12, 43))
        r_hat, rate = vc(p - pred, qstep_of_qp(qp))
        p_hat = torch.clamp(pred + r_hat, 0, 255)
        l1 = (x - p_hat).abs().mean() / 255.0
        mss = ssim_multiscale_tf(x.permute(0, 2, 3, 1), p_hat.permute(0, 2, 3, 1),
                                 max_val=255.0, filter_size=11).mean()
        lf = 0.2 * l1 + 0.8 * (1.0 - mss)
        # binary clips the SVR at 100; without the clamp lp goes negative and
        # the gradient chases meaningless >100 extrapolation
        if a.gamma > 0:
            neg = vt(x, p_hat).clamp(max=100.0).mean()
        else:  # gamma=0 ablation: log NEG without paying its backward
            with torch.no_grad():
                neg = vt(x, p_hat).clamp(max=100.0).mean()
        lp = (100.0 - neg) / 100.0
        lrr = rate.mean()
        loss = a.gamma * lp + a.lam * lrr + lf
        opt.zero_grad(); opt_aux.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        vc.eb.loss().backward(); opt_aux.step()
        logs["lp"].append(float(lp)); logs["lr"].append(float(lrr))
        logs["lf"].append(float(lf)); logs["neg"].append(float(neg))
        logs["mss"].append(float(mss))
        logs["dbits"].append(float((p - x).abs().mean()))
        if (s + 1) % 200 == 0:
            print(f"[{name} s{s+1}] lp {np.mean(logs['lp']):.4f} "
                  f"rate {np.mean(logs['lr']):.3f}bpp lf {np.mean(logs['lf']):.4f} "
                  f"NEG {np.mean(logs['neg']):.2f} MSS {np.mean(logs['mss']):.4f} "
                  f"|p-x| {np.mean(logs['dbits']):.2f} "
                  f"({(time.time()-t0)/(s+1):.2f}s/it)", flush=True)
            logs = {k_: [] for k_ in logs}
        if (s + 1) % 5000 == 0 or (s + 1) == a.steps:
            torch.save({"pre": pre.state_dict(), "vc": vc.state_dict(),
                        "step": s + 1},
                       f"{RUNS}/{name}/model.pt")
            print(f"[{name}] ckpt saved @ {s+1}", flush=True)
    print(f"[{name}] DONE", flush=True)


if __name__ == "__main__":
    main()
