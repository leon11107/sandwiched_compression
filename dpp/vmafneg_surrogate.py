"""Differentiable VMAF_NEG surrogate (line S): CNN scoring (ref, dist) pairs,
trained on vmafneg_data.jsonl with WEAK labels (global per-image NEG score
applied to 16-aligned 256-crops; blockwise JPEG + apply_s commute with aligned
cropping, so crop dists are exactly the global dists restricted to the crop).

What the surrogate must get right to be useful as a fast oracle:
  RANKING within a (image, q) group (ES compares candidates of the same image
  at the same q) — so the loss is MSE + in-batch pairwise hinge on same-group
  pairs, and model selection is by holdout pairwise ranking accuracy + the
  mean_s-vs-plain direction check (the signal the real oracle exploits).

Variants (one GPU each): base / wide (1.5x ch) / rank (4x rank weight).
"""
from __future__ import annotations
import argparse, glob, io, json, os, sys, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp.oracle_vmafneg import load16, luma, block_dct, apply_s

RUNS = "/workspace/sandwiched_compression/dpp/runs"
DATA_FILES = [os.path.join(RUNS, "vmafneg_data.jsonl"),
              os.path.join(RUNS, "vmafneg_data_es.jsonl")]  # DAgger-mined
CROP = 256


def jpeg_dec(img, quality):
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(img, 0, 255)).astype(np.uint8)).save(
        buf, format="jpeg", quality=int(quality), subsampling="4:2:0")
    return np.asarray(Image.open(buf).convert("RGB"), np.float32)


def make_dist_crop(crop, kind, q, s, contrast, gmean):
    """Reconstruct the recipe dist restricted to a 16-aligned crop."""
    if kind == "contrast":
        src = np.clip((crop - gmean) * contrast[0] + gmean + contrast[1], 0, 255)
    elif s is not None:
        y = luma(crop)
        src = apply_s(crop, block_dct(y), y, np.asarray(s, np.float64))
    else:
        src = crop
    return jpeg_dec(src, q)


class GroupPairs(torch.utils.data.Dataset):
    """index = group (path,q); returns 2 same-crop dists + ref + labels."""

    def __init__(self, groups, imgs, gmeans, seed=0):
        self.groups, self.imgs, self.gmeans = groups, imgs, gmeans
        self.epoch_seed = seed

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, gi):
        rng = np.random.default_rng((self.epoch_seed * 1_000_003 + gi) % 2**31)
        rows = self.groups[gi]
        i1, i2 = rng.choice(len(rows), 2, replace=False)
        r1, r2 = rows[i1], rows[i2]
        img = self.imgs[r1["path"]].astype(np.float32)
        H, W = img.shape[:2]
        oy = 16 * rng.integers(0, (H - CROP) // 16 + 1)
        ox = 16 * rng.integers(0, (W - CROP) // 16 + 1)
        ref = img[oy:oy + CROP, ox:ox + CROP]
        gm = self.gmeans[r1["path"]]
        d1 = make_dist_crop(ref, r1["kind"], r1["q"], r1["s"], r1["contrast"], gm)
        d2 = make_dist_crop(ref, r2["kind"], r2["q"], r2["s"], r2["contrast"], gm)
        if rng.random() < 0.5:  # flip aug (same for all three)
            ref, d1, d2 = ref[:, ::-1], d1[:, ::-1], d2[:, ::-1]
        to = lambda a: torch.from_numpy(np.ascontiguousarray(
            a.transpose(2, 0, 1))) / 255.0
        return (to(ref), to(d1), to(d2),
                torch.tensor([r1["vmaf_neg"] / 100.0, r2["vmaf_neg"] / 100.0],
                             dtype=torch.float32))


class Net(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        ch = [int(c * width) for c in (32, 64, 128, 256, 256)]
        layers, cin = [], 6
        for c in ch:
            layers += [nn.Conv2d(cin, c, 3, 2, 1), nn.GroupNorm(8, c), nn.ReLU()]
            cin = c
        self.feat = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(cin, 128), nn.ReLU(),
                                  nn.Linear(128, 1))

    def forward(self, ref, dist):
        h = self.feat(torch.cat([ref, dist], 1)).mean((2, 3))
        return torch.sigmoid(self.head(h)).squeeze(-1)


def load_groups(holdout_frac=0.1):
    rows = []
    for path in DATA_FILES:
        if os.path.exists(path):
            rows += [json.loads(l) for l in open(path)]
    bykey = {}
    for r in rows:
        bykey.setdefault((r["path"], r["q"]), []).append(r)
    paths = sorted({r["path"] for r in rows})
    n_ho = max(1, int(len(paths) * holdout_frac))
    ho_paths = set(paths[-n_ho:])  # deterministic tail split
    tr = [v for (p, q), v in sorted(bykey.items())
          if p not in ho_paths and len(v) >= 2]
    ho = [v for (p, q), v in sorted(bykey.items())
          if p in ho_paths and len(v) >= 2]
    return tr, ho, paths


@torch.no_grad()
def evaluate(net, groups, imgs, gmeans, dev, grid=3, max_groups=300):
    """Tiled prediction per row; -> (pairwise rank acc, mean_s>plain acc, MAE)."""
    net.eval()
    rank_ok = rank_n = dir_ok = dir_n = es_ok = es_n = 0
    maes = []
    for rows in groups[:max_groups]:
        img = imgs[rows[0]["path"]].astype(np.float32)
        gm = gmeans[rows[0]["path"]]
        H, W = img.shape[:2]
        oys = np.linspace(0, (H - CROP) // 16, grid).astype(int) * 16
        oxs = np.linspace(0, (W - CROP) // 16, grid).astype(int) * 16
        refs, dists, ridx = [], [], []
        for ri, r in enumerate(rows):
            for oy in oys:
                for ox in oxs:
                    ref = img[oy:oy + CROP, ox:ox + CROP]
                    refs.append(ref)
                    dists.append(make_dist_crop(ref, r["kind"], r["q"], r["s"],
                                                r["contrast"], gm))
                    ridx.append(ri)
        t = lambda L: (torch.from_numpy(np.stack(L).transpose(0, 3, 1, 2))
                       .to(dev) / 255.0)
        preds_all = []
        for c0 in range(0, len(refs), 64):
            preds_all.append(net(t(refs[c0:c0+64]), t(dists[c0:c0+64])))
        preds_all = torch.cat(preds_all).cpu().numpy() * 100.0
        ridx = np.array(ridx)
        pred = np.array([preds_all[ridx == ri].mean() for ri in range(len(rows))])
        ytrue = np.array([r["vmaf_neg"] for r in rows])
        maes.append(np.abs(pred - ytrue).mean())
        kinds = [r["kind"] for r in rows]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                if abs(ytrue[a] - ytrue[b]) < 0.5:
                    continue
                rank_n += 1
                ok = int((pred[a] - pred[b]) * (ytrue[a] - ytrue[b]) > 0)
                rank_ok += ok
                if kinds[a].startswith(("es", "rand_big")) or \
                        kinds[b].startswith(("es", "rand_big")):
                    es_n += 1; es_ok += ok
        if "plain" in kinds and "mean_s" in kinds:
            ip, im_ = kinds.index("plain"), kinds.index("mean_s")
            if ytrue[im_] - ytrue[ip] > 0.3:
                dir_n += 1
                dir_ok += int(pred[im_] > pred[ip])
    net.train()
    return (rank_ok / max(rank_n, 1), dir_ok / max(dir_n, 1),
            float(np.mean(maes)),
            es_ok / es_n if es_n else float("nan"), es_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "wide", "rank"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-groups", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--init-from", default="", help="warm-start ckpt (DAgger)")
    ap.add_argument("--tag", default="", help="suffix for the saved ckpt name")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    width = 1.5 if a.variant == "wide" else 1.0
    rank_w, mse_w = (4.0, 0.25) if a.variant == "rank" else (1.0, 1.0)

    tr, ho, paths = load_groups()
    print(f"[{a.variant}] {len(tr)} train groups, {len(ho)} holdout groups, "
          f"{len(paths)} imgs", flush=True)
    imgs, gmeans = {}, {}
    t0 = time.time()
    for p in paths:
        im = load16(p)
        if min(im.shape[:2]) < CROP:   # 22/1631 imgs too small for 256-crops
            continue
        imgs[p] = np.rint(im).astype(np.uint8)
        gmeans[p] = im.mean(axis=(0, 1), keepdims=True)
    tr = [g for g in tr if g[0]["path"] in imgs]
    ho = [g for g in ho if g[0]["path"] in imgs]
    print(f"[{a.variant}] images cached uint8 in {time.time()-t0:.0f}s "
          f"({sum(v.nbytes for v in imgs.values())/2**30:.1f} GiB); "
          f"after small-img filter: {len(tr)} train / {len(ho)} holdout groups",
          flush=True)

    ds = GroupPairs(tr, imgs, gmeans)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_groups, shuffle=True, num_workers=a.workers,
        drop_last=True, persistent_workers=True, prefetch_factor=4)
    net = Net(width).to(dev)
    if a.init_from:
        ck = torch.load(a.init_from, map_location=dev)
        net.load_state_dict(ck["net"])
        print(f"[{a.variant}] warm-started from {a.init_from} "
              f"(prev best {ck['best']})", flush=True)
    print(f"[{a.variant}] params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler()
    best = {"rank": -1.0}
    step, ep = 0, 0
    losses, rlosses = [], []
    while step < a.steps:
        ds.epoch_seed = ep
        for ref, d1, d2, ys in dl:
            ref, d1, d2 = ref.to(dev), d1.to(dev), d2.to(dev)
            y1, y2 = ys[:, 0].to(dev), ys[:, 1].to(dev)
            with torch.autocast("cuda"):
                p1 = net(ref, d1); p2 = net(ref, d2)
                l_mse = 0.5 * (torch.mean((p1 - y1) ** 2) + torch.mean((p2 - y2) ** 2))
                dy = y1 - y2
                margin = torch.clamp(dy.abs(), max=0.05)
                valid = (dy.abs() > 0.003).float()
                l_rank = torch.mean(valid * torch.relu(
                    margin - (p1 - p2) * torch.sign(dy)))
                loss = mse_w * l_mse + rank_w * l_rank
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(l_mse)); rlosses.append(float(l_rank))
            step += 1
            if step % 500 == 0:
                print(f"[{a.variant} s{step}] mse {np.mean(losses):.5f} "
                      f"rank {np.mean(rlosses):.5f} lr {sched.get_last_lr()[0]:.2e}",
                      flush=True)
                losses, rlosses = [], []
            if step % 2500 == 0 or step == a.steps:
                ra, da, mae, es_ra, es_n = evaluate(net, ho, imgs, gmeans, dev)
                sel = (ra + es_ra) / 2 if es_n else ra
                print(f"[{a.variant} s{step}] HOLDOUT rank_acc {ra:.4f} "
                      f"es_rank_acc {es_ra:.4f} (n={es_n}) "
                      f"dir_acc(mean_s>plain) {da:.4f} MAE {mae:.2f}", flush=True)
                if sel > best["rank"]:
                    best = {"rank": sel, "rank_all": ra, "es_rank": es_ra,
                            "dir": da, "mae": mae, "step": step}
                    torch.save({"net": net.state_dict(), "width": width,
                                "best": best},
                               os.path.join(RUNS, f"surrogate_{a.variant}{a.tag}.pt"))
                    print(f"[{a.variant}] saved (rank_acc {ra:.4f})", flush=True)
            if step >= a.steps:
                break
        ep += 1
    print(f"[{a.variant}] DONE best {best}", flush=True)


if __name__ == "__main__":
    main()
