"""Joint per-image oracle: does ANY preprocessed input X' beat baseline JPEG on
BOTH VMAF_NEG and MS-SSIM at iso-rate (fixed resolution)? Gradient ascent on
X' itself through a real-JPEG STE, with FAITHFUL differentiable metrics:
  - NEG: vmaf-torch NEG=True (fidelity gate: MAE 0.034, rank_acc 1.0 vs binary)
  - MS-SSIM: torch_port.ssim_multiscale_tf filter_size=11 == eval metric
Objective J = beta * NEG/100 + (1 - beta) * MS-SSIM (no rate term; the RD point
just moves and iso-rate deltas vs the per-image baseline ladder account for
rate; bits drift is logged). Final scoring uses the REAL vmaf binary.

Per (img, q, beta): ~steps x [PIL encode + NEG fwd/bwd @1080p-area]. Output
rows -> runs/joint_oracle.jsonl (resume-safe); summary prints iso-rate deltas
and the BOTH-win verdict per row.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, tempfile, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/sandwiched_compression")
from distortion import vmaf_metric as vm
from dpp.oracle_vmafneg import jpeg_rt, load16, up1080, vmaf_neg_batch, TARGET_AREA
from dpp.eval_v2 import msssim_luma
from dpp.oracle_vmafneg import apply_s, block_dct, luma
from dpp.spred import SPredictor
from torch_port.preproc import tf_rgb_to_yuv
from torch_port.losses import ssim_multiscale_tf
from dpp.vmafneg_torch_check import rgb_to_vmaf_luma

VAL_DIR = "/workspace/sandwiched_compression/dpp/data/val50"
RUNS = "/workspace/sandwiched_compression/dpp/runs"
QGRID = [3, 5, 8, 12, 16, 20, 26, 32, 40]  # baseline ladder for iso interp


def up_size(h, w):
    sc = np.sqrt(TARGET_AREA / (w * h))
    if sc <= 1.0:
        return h, w
    return int(round(h * sc / 2)) * 2, int(round(w * sc / 2)) * 2


# ---- differentiable JPEG rate proxy ----------------------------------------
ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
                   12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
                   28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30,
                   37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
                   54, 47, 55, 62, 63])


def pil_qtables(q):
    """libjpeg luma/chroma quant tables (natural order 8x8) at PIL quality q."""
    import io
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (16, 16)).save(buf, format="jpeg", quality=int(q),
                                       subsampling="4:2:0")
    tabs = PILImage.open(buf).quantization
    out = []
    for k in (0, 1):
        t = np.zeros(64)
        t[ZIGZAG] = np.array(tabs[k], float)
        out.append(t.reshape(8, 8))
    return out


_DCT8 = None


def dct8(dev):
    global _DCT8
    if _DCT8 is None or _DCT8.device != torch.device(dev):
        k = np.arange(8)
        D = np.cos((2 * k[None] + 1) * k[:, None] * np.pi / 16) * np.sqrt(0.25)
        D[0] /= np.sqrt(2)
        _DCT8 = torch.from_numpy(D).float().to(dev)
    return _DCT8


def blocks8(x):
    """[1,1,H,W] -> [L,8,8] non-overlapping blocks (H,W multiples of 8 assumed
    after padding)."""
    B, C, H, W = x.shape
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    u = F.unfold(x, kernel_size=8, stride=8)  # [1,64,L]
    return u[0].T.reshape(-1, 8, 8)


def rate_proxy(rgb_hw3, qt_l, qt_c, dev):
    """Differentiable JPEG-bit proxy: sum log2(1+|DCT/Q|) over Y + 4:2:0 CbCr."""
    x = rgb_hw3.permute(2, 0, 1)[None]  # [1,3,H,W]
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    yy = 0.299 * r + 0.587 * g + 0.114 * b - 128.0
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    D = dct8(dev)
    tot = 0.0
    for plane, qt, sub in ((yy, qt_l, False), (cb, qt_c, True), (cr, qt_c, True)):
        if sub:
            plane = F.avg_pool2d(plane, 2)
        bl = blocks8(plane)
        c = D @ bl @ D.T
        tot = tot + torch.log2(1.0 + (c / qt).abs()).sum()
    return tot


def luma_lim_t(rgb_bchw):
    """[B,3,H,W] float 0..255 -> [B,1,H,W] BT.601 limited-range Y (unrounded)."""
    r, g, b = rgb_bchw[:, 0:1], rgb_bchw[:, 1:2], rgb_bchw[:, 2:3]
    return 16.0 + (0.299 * r + 0.587 * g + 0.114 * b) * (219.0 / 255.0)


def msssim_t(orig_bhwc, dec_bhwc):
    ry = tf_rgb_to_yuv(orig_bhwc)[..., 0:1]
    dy = tf_rgb_to_yuv(dec_bhwc)[..., 0:1]
    return ssim_multiscale_tf(ry, dy, max_val=255.0, filter_size=11)[0]


def optimize(im, q, beta, vt, dev, steps, lr, log_pref, kn, km, dither=True,
             init_delta=None):
    """Ascent on X' of the per-image iso-rate Lagrangian
        J = beta*(NEG/100 - kn/100 * r%) + (1-beta)*(MSSSIM - km * r%),
    i.e. a first-order estimate of the weighted iso-rate deltas, with kn/km =
    real per-image baseline slopes (units per +1% bpp). Rate gradient comes
    from a differentiable JPEG bit proxy calibrated to real bits at step 0;
    clean (undithered) evals every 10 steps use REAL bits and drive best-J
    snapshotting (result >= baseline by construction on faithful metrics)."""
    H, W = im.shape[:2]
    nh, nw = up_size(H, W)
    im_t = torch.from_numpy(im).float().to(dev)
    ref_up = up1080(np.rint(im).astype(np.uint8)).astype(np.float32)
    ref_l = torch.from_numpy(rgb_to_vmaf_luma(ref_up)[None, None]).float().to(dev)
    qt_l, qt_c = [torch.from_numpy(t).float().to(dev) for t in pil_qtables(q)]
    delta = (torch.zeros_like(im_t) if init_delta is None
             else torch.from_numpy(init_delta).float().to(dev)).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    _, bits0 = jpeg_rt(im, q)
    with torch.no_grad():
        a_cal = bits0 / float(rate_proxy(im_t, qt_l, qt_c, dev))

    def metrics(x_in):
        dec_np, bits = jpeg_rt(x_in.detach().cpu().numpy(), q)
        dec_t = x_in + (torch.from_numpy(dec_np).to(dev) - x_in).detach()  # STE
        mss = msssim_t(im_t[None], dec_t[None])
        d_up = F.interpolate(dec_t.permute(2, 0, 1)[None], size=(nh, nw),
                             mode="bicubic", align_corners=False).clamp(0, 255)
        neg = vt(ref_l, luma_lim_t(d_up)).squeeze()
        return neg, mss, bits

    def lagr(neg, mss, r_pct):
        return beta * (neg / 100.0 - kn / 100.0 * r_pct) + \
            (1 - beta) * (mss - km * r_pct)

    best = {"J": -1e9, "delta": None}
    t0 = time.time()
    for s in range(steps):
        xp = torch.clamp(im_t + delta, 0, 255)
        x_in = xp + (torch.rand_like(im_t) - 0.5) if dither else xp
        neg, mss, _ = metrics(x_in)
        r_pct = (a_cal * rate_proxy(x_in, qt_l, qt_c, dev) - bits0) / bits0 * 100.0
        J = lagr(neg, mss, r_pct)
        opt.zero_grad(); (-J).backward(); opt.step()
        if s % 10 == 0 or s == steps - 1:
            with torch.no_grad():
                xp_c = torch.clamp(im_t + delta, 0, 255)
                negc, mssc, bits_c = metrics(xp_c)
                rb = (bits_c - bits0) / bits0 * 100.0
                Jc = float(lagr(negc, mssc, rb))
            if Jc > best["J"]:
                best = {"J": Jc, "delta": delta.detach().clone()}
            if s % 50 == 0 or s == steps - 1:
                print(f"{log_pref} s{s} Jclean {Jc:.5f} (best {best['J']:.5f}) "
                      f"negD {float(negc):.2f} mssD {float(mssc):.5f} "
                      f"dbits {rb:+.1f}% ({time.time()-t0:.0f}s)", flush=True)
    d = best["delta"] if best["delta"] is not None else delta.detach()
    xp = torch.clamp(im_t + d, 0, 255).detach().cpu().numpy()
    dec_np, bits = jpeg_rt(xp, q)
    return xp, dec_np, bits, bits0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-imgs", type=int, default=12)
    ap.add_argument("--qs", default="8,20,32")
    ap.add_argument("--betas", default="0.4,0.7")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--init", choices=["id", "C_l05", "b30s"], default="id",
                    help="C_l05: warm-start from the MS-SSIM smoother; b30s: "
                         "0.3*C_l05 blend + spred band pre-emphasis (the "
                         "candidate family with 28%% both-win at q5)")
    ap.add_argument("--out", default=os.path.join(RUNS, "joint_oracle.jsonl"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    from vmaf_torch import VMAF
    vt = VMAF(NEG=True).to(dev).eval()
    qs = [int(x) for x in a.qs.split(",")]
    betas = [float(x) for x in a.betas.split(",")]
    si, sn = [int(x) for x in a.shard.split("/")]
    paths = sorted(glob.glob(os.path.join(VAL_DIR, "*.png")))
    paths = paths[:: max(1, len(paths) // a.n_imgs)][: a.n_imgs][si::sn]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r["path"], r["q"], r["beta"]))
            except json.JSONDecodeError:
                pass

    with open(a.out, "a") as f:
        for p in paths:
            im = load16(p)
            npx = im.shape[0] * im.shape[1]
            todo = [(q, b) for q in qs for b in betas
                    if (p, q, b) not in done]
            if not todo:
                continue
            # per-image baseline ladder; slopes from the FAITHFUL torch metrics
            # (binary re-scores the ladder at the end for the stored base_curve)
            base_decs, base_bpp = [], []
            for q in QGRID:
                d, bits = jpeg_rt(im, q)
                base_decs.append(d); base_bpp.append(bits / npx)
            nh, nw = up_size(*im.shape[:2])
            ref_up_t = torch.from_numpy(rgb_to_vmaf_luma(
                up1080(np.rint(im).astype(np.uint8)).astype(np.float32)
            )[None, None]).float().to(dev)
            im_t = torch.from_numpy(im).float().to(dev)
            lad_neg, lad_mss = [], []
            with torch.no_grad():
                for d in base_decs:
                    dt = torch.from_numpy(d).float().to(dev)
                    du = F.interpolate(dt.permute(2, 0, 1)[None], size=(nh, nw),
                                       mode="bicubic", align_corners=False
                                       ).clamp(0, 255)
                    lad_neg.append(float(vt(ref_up_t, luma_lim_t(du))))
                    lad_mss.append(float(msssim_t(im_t[None], dt[None])))

            def slopes(q):
                qi = QGRID.index(q)
                lo, hi = max(0, qi - 1), min(len(QGRID) - 1, qi + 1)
                dr = (base_bpp[hi] - base_bpp[lo]) / base_bpp[qi] * 100.0
                return ((lad_neg[hi] - lad_neg[lo]) / dr,
                        (lad_mss[hi] - lad_mss[lo]) / dr)

            c_out = None
            if a.init in ("C_l05", "b30s"):
                from dpp.model import DPPModel
                m = DPPModel(ch=64, codec_forward_mode="proxy", device=dev)
                ck = torch.load(os.path.join(RUNS, "v2_C_l05", "model.pt"),
                                map_location=dev)
                (m.preproc.load_state_dict(ck["preproc"])
                 if isinstance(ck, dict) and "preproc" in ck
                 else m.load_state_dict(ck))
                m.eval()
                with torch.no_grad():
                    c_out = np.clip(m.preproc(torch.from_numpy(im[None]).float()
                                              .to(dev))[0].cpu().numpy(), 0, 255)
                del m
            sp = SPredictor.load(os.path.join(RUNS, "spred_model.npz")) \
                if a.init == "b30s" else None

            def init_fn(q):
                if a.init == "id":
                    return None
                if a.init == "C_l05":
                    return (c_out - im).astype(np.float32)
                mixed = 0.3 * c_out + 0.7 * im
                ym = luma(mixed); cm = block_dct(ym)
                src = apply_s(mixed, cm, ym, sp.predict(mixed, 8 if q <= 8 else 20))
                return (src - im).astype(np.float32)

            cands = []
            for q, b in todo:
                kn, km = slopes(q)
                xp, dec, bits, bits0 = optimize(
                    im, q, b, vt, dev, a.steps, a.lr,
                    f"[{os.path.basename(p)} q{q} b{b}]", kn=kn, km=km,
                    init_delta=init_fn(q))
                cands.append((q, b, dec, bits / npx))
            # one REAL NEG batch: baseline ladder + candidates
            decs_up = [up1080(np.rint(d).astype(np.uint8))
                       for d in base_decs + [c[2] for c in cands]]
            ref_up = up1080(np.rint(im).astype(np.uint8))
            with tempfile.TemporaryDirectory() as td:
                rdir = os.path.join(td, "r"); os.makedirs(rdir)
                ry4m = os.path.join(td, "ref.y4m")
                vm._png_seq_to_y4m([ref_up] * len(decs_up), rdir, ry4m)
                negs = vmaf_neg_batch(ry4m, decs_up, td, threads=4)
            msss = [msssim_luma(im, d, dev) for d in base_decs] + \
                   [msssim_luma(im, c[2], dev) for c in cands]
            bn = np.array(negs[: len(QGRID)]); bm = np.array(msss[: len(QGRID)])
            bb = np.array(base_bpp)
            o = np.argsort(bb)
            for k, (q, b, dec, bpp) in enumerate(cands):
                neg_r = negs[len(QGRID) + k]; mss_r = msss[len(QGRID) + k]
                iso_n = neg_r - float(np.interp(bpp, bb[o], bn[o]))
                iso_m = mss_r - float(np.interp(bpp, bb[o], bm[o]))
                row = {"path": p, "q": q, "beta": b, "bpp": bpp,
                       "dbpp_pct": (bpp - base_bpp[QGRID.index(q)]) /
                                   base_bpp[QGRID.index(q)] * 100.0,
                       "neg": neg_r, "iso_neg": iso_n,
                       "msssim": mss_r, "iso_msssim": iso_m,
                       "base_curve": {"q": QGRID, "bpp": base_bpp,
                                      "neg": bn.tolist(), "msssim": bm.tolist()}}
                f.write(json.dumps(row) + "\n"); f.flush()
                both = iso_n > 0 and iso_m > 0
                print(f"== [{os.path.basename(p)} q{q} b{b}] REAL iso_NEG "
                      f"{iso_n:+.2f} iso_MSSSIM {iso_m:+.5f} dbpp "
                      f"{row['dbpp_pct']:+.1f}% -> {'BOTH WIN' if both else 'no'}",
                      flush=True)
    print("SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
