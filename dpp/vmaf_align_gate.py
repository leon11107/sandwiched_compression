"""(f) validate-first: which DIFFERENTIABLE FR objective is VMAF-aligned AND
non-gameable? Build a gaming-relevant variant pool (jpeg sweep + enhanced + the
DPP-GAMED preproc outputs that fooled NR metrics) and measure SRCC of each candidate
(vif=VMAF core component, dists, topiq_fr) vs REAL VMAF_NEG. Key: does it rate the
gamed outputs LOW (like VMAF) or HIGH (like the NR metrics that failed)?"""
import glob, os, sys
import numpy as np
from PIL import Image, ImageFilter
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from distortion import vmaf_metric as vm
from torch_port.codec import encode_decode_with_jpeg
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb
import pyiqa
from scipy.stats import spearmanr

dev = "cuda"
def u8(a): return np.rint(np.clip(a, 0, 255)).astype(np.uint8)
def jpeg(a, q): d, _ = encode_decode_with_jpeg(u8(a)[None].astype(np.float32), q, False, False); return u8(d[0])
def restore_np(dec, orig):
    d = tf_rgb_to_yuv(torch.from_numpy(dec[None].astype(np.float32)))
    o = tf_rgb_to_yuv(torch.from_numpy(orig[None].astype(np.float32)))
    return u8(tf_yuv_to_rgb(torch.cat([d[..., 0:1], o[..., 1:3]], -1))[0].numpy())

def load256(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size; s = min(w, h, 256)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256), Image.LANCZOS)
    return np.asarray(im, np.float32)
imgs = [load256(p) for p in sorted(glob.glob("dpp/data/val/*.png"))[:12]]

# DPP-gamed preproc (nima_g05_fix games VMAF hard) — its outputs fooled NR metrics
from dpp.model import DPPModel
gm = DPPModel(ch=64, codec_forward_mode="proxy", device=dev)
gm.load_state_dict(torch.load("dpp/runs/nima_g05_fix/model.pt", map_location=dev)); gm.eval()
def gamed(a, q):
    with torch.no_grad():
        pre = np.clip(gm.preproc(torch.from_numpy(a[None]).float().to(dev))[0].cpu().numpy(), 0, 255)
    return restore_np(jpeg(pre, q), a)

# variant pool per image
pool = []
for i, im in enumerate(imgs):
    for q in [12, 20, 32, 48, 72]:
        pool.append((i, restore_np(jpeg(im, q), im)))
    pim = Image.fromarray(u8(im))
    pool.append((i, restore_np(jpeg(np.asarray(pim.filter(ImageFilter.UnsharpMask(2, 160)), np.float32), 32), im)))  # enhanced
    pool.append((i, gamed(im, 32)))  # DPP-gamed (NR said good, VMAF said bad)
    pool.append((i, gamed(im, 20)))

refs = [u8(imgs[i]) for i, _ in pool]; dists = [d for _, d in pool]
vneg = np.array([s["vmaf_neg"] for s in vm.vmaf_scores(refs, dists)])

def t(arr): return torch.from_numpy(np.stack(arr).astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)
ref_t = t([imgs[i] for i, _ in pool]); dis_t = t(dists)
print(f"{len(pool)} variants ({len(imgs)} imgs). Candidate FR metric SRCC vs VMAF_NEG:")
for name in ["vif", "dists", "topiq_fr", "lpips-vgg"]:
    try:
        m = pyiqa.create_metric(name, as_loss=False, device=dev)  # per-sample values (no grad needed to validate)
        with torch.no_grad():
            s = np.concatenate([m(dis_t[k:k+8], ref_t[k:k+8]).detach().cpu().numpy().reshape(-1)
                                for k in range(0, dis_t.shape[0], 8)])
        assert s.shape[0] == dis_t.shape[0], f"{name} returned {s.shape} not per-sample"
        s = -s if m.lower_better else s  # higher=better
        srcc = float(spearmanr(s, vneg).correlation)
        # per-image
        pim = [float(spearmanr(s[[k for k,(ii,_) in enumerate(pool) if ii==i]],
                               vneg[[k for k,(ii,_) in enumerate(pool) if ii==i]]).correlation) for i in range(len(imgs))]
        # gamed rating: avg metric percentile of the gamed variants (low => correctly penalized)
        print(f"  {name:10s} SRCC_pool={srcc:+.3f} SRCC_perimg={np.nanmean(pim):+.3f}")
    except Exception as e:
        print(f"  {name:10s} ERR {str(e)[:50]}")
print("(VMAF_NEG range: %.1f..%.1f)" % (vneg.min(), vneg.max()))
