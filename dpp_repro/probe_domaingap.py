"""Part 1 domain-gap probe (NO training). Measures whether the proxy
(virtual codec) ranks precoded EDITS the way real x264 does on the iso-rate
VMAF_NEG axis — the thing the precoder's gradient actually uses.

For each clip x candidate (identity + gamma-family precoders + blur/sharpen),
all-intra:
  real  = x264 all-intra (cand-luma + orig chroma) over a CRF sweep -> bpp,
          VMAF_NEG (vmaf binary vs original)
  proxy = intra_pred(k=8) + trained VirtualCodec over a QP sweep -> proxy bpp,
          VMAF_NEG (same vmaf binary on p_hat luma + orig chroma)
Both sides scored by the SAME binary => isolates the CODEC gap (rate +
reconstruction), not a metric mismatch.

Per candidate, per side: iso = dNEG_vs_identity - kappa*dbpp%_vs_identity
(kappa = that side's local NEG-per-1%bpp slope from identity). Rank candidates
by iso on each side; report Spearman + pairwise sign-agreement (proxy vs real).
High rho => proxy already RD-ranks like x264 => proxy tweaks won't help.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp_repro.y4m import read_y4m, y4m_header
from dpp_repro.virtual_codec import intra_pred, VirtualCodec, qstep_of_qp
from dpp_repro.s1_train import Precoder

WORK = "/dev/shm/dppv"
RUNS = "/workspace/sandwiched_compression/dpp/runs"
MODEL_NEG = "/workspace/sandwiched_compression/reference/vmaf_models/vmaf_v0.6.1neg.json"
VMAF_LD = "/usr/local/lib/x86_64-linux-gnu"
HELD = ["aspen_1080p", "red_kayak_1080p", "west_wind_easy_1080p", "controlled_burn_1080p"]


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, **kw)


def write_y4m(path, header, frames):
    """frames: list of (y,u,v) uint8 planes."""
    with open(path, "wb") as f:
        f.write(header)
        for y, u, v in frames:
            f.write(b"FRAME\n"); f.write(y.tobytes()); f.write(u.tobytes()); f.write(v.tobytes())


def vmaf_neg(ref_y4m, dist_y4m):
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        oj = os.path.join(td, "v.json")
        env = dict(os.environ); env["LD_LIBRARY_PATH"] = VMAF_LD
        sh(["vmaf", "--reference", ref_y4m, "--distorted", dist_y4m, "--model",
            "path=" + MODEL_NEG, "--threads", "4", "--output", oj, "--json"], env=env)
        pm = json.loads(open(oj).read())["pooled_metrics"]
    return pm[next(k for k in pm if k.startswith("vmaf"))]["mean"]


def make_candidate(name, lumas, dev, precoders):
    """-> list of edited luma frames (float32). Names:
    identity | blur | sharpen | <model> | <model>x<scale> (scaled edit, fills
    the mild operating regime: edit=model_out-identity, out=identity+s*edit)."""
    import torch.nn.functional as F
    if name == "identity":
        return [l.astype(np.float32) for l in lumas]
    if name in ("blur", "sharpen"):
        k = torch.tensor([[1, 4, 6, 4, 1]], dtype=torch.float32, device=dev)
        k = (k.T @ k); k = (k / k.sum()).view(1, 1, 5, 5)
        out = []
        for l in lumas:
            t = torch.from_numpy(l.astype(np.float32)).to(dev)[None, None]
            blur = F.conv2d(t, k, padding=2)
            t2 = blur if name == "blur" else (t + 1.2 * (t - blur))
            out.append(t2.clamp(0, 255)[0, 0].cpu().numpy())
        return out
    mname, scale = (name.split("x") + ["1"])[:2] if "x" in name else (name, "1")
    pre = precoders[mname]; s = float(scale)
    out = []
    with torch.no_grad():
        for l in lumas:
            t = torch.from_numpy(l.astype(np.float32)).to(dev)[None, None]
            edit = pre(t) - t
            out.append((t + s * edit).clamp(0, 255)[0, 0].cpu().numpy())
    return out


def real_rd(ref_y4m, header, cand_lumas, chroma, crf, npix_frames):
    """x264 all-intra at one CRF -> (bpp, VMAF_NEG)."""
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        src = os.path.join(td, "s.y4m"); mp4 = os.path.join(td, "e.mp4")
        dec = os.path.join(td, "d.y4m")
        frames = [(np.rint(np.clip(l, 0, 255)).astype(np.uint8), u, v)
                  for l, (u, v) in zip(cand_lumas, chroma)]
        write_y4m(src, header, frames)
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
            "-preset", "slow", "-crf", str(crf), "-g", "1", "-keyint_min", "1",
            "-tune", "ssim", "-x264opts", "ssim=1", mp4])
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                            "-show_entries", "packet=size", "-of", "csv=p=0", mp4],
                           capture_output=True, text=True, check=True)
        bits = sum(int(x) for x in r.stdout.split() if x) * 8
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-pix_fmt",
            "yuv420p", dec])
        neg = vmaf_neg(ref_y4m, dec)
    return bits / npix_frames, neg


def proxy_rd(ref_y4m, header, cand_lumas, chroma, qp, vc, dev):
    """intra_pred + VirtualCodec at one QP -> (proxy_bpp, VMAF_NEG on p_hat)."""
    qs = qstep_of_qp(qp)
    recon, bpps = [], []
    with torch.no_grad():
        for l in cand_lumas:
            p = torch.from_numpy(l.astype(np.float32)).to(dev)[None, None]
            pred = intra_pred(p, k=8, m=24, tau=1.0)
            r_hat, rate = vc(p - pred, qs)
            recon.append(torch.clamp(pred + r_hat, 0, 255)[0, 0].cpu().numpy())
            bpps.append(float(rate.mean()))
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        dist = os.path.join(td, "p.y4m")
        frames = [(np.rint(np.clip(rl, 0, 255)).astype(np.uint8), u, v)
                  for rl, (u, v) in zip(recon, chroma)]
        write_y4m(dist, header, frames)
        neg = vmaf_neg(ref_y4m, dist)
    return float(np.mean(bpps)), neg


def iso_rank(rows, qps):
    """rows[name] = list of (bpp, neg) over the op sweep. -> {name: iso_gain}."""
    id_b = np.array([b for b, _ in rows["identity"]])
    id_n = np.array([n for _, n in rows["identity"]])
    o = np.argsort(id_b)
    id_b, id_n = id_b[o], id_n[o]
    # local slope kappa: NEG per 1% bpp around the middle op
    kappa = (id_n[-1] - id_n[0]) / max((id_b[-1] - id_b[0]) / id_b[0] * 100.0, 1e-6)
    mid = len(qps) // 2
    out = {}
    for name, rd in rows.items():
        b, n = rd[mid]
        ib = float(np.interp(b, id_b, id_n))  # identity NEG at this bpp
        dbpp = (b - id_b[mid]) / id_b[mid] * 100.0
        out[name] = (n - id_n[mid]) - kappa * dbpp  # iso gain vs identity op
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-ckpt", default=os.path.join(RUNS, "s2_lam0.01_g2/model.pt"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--clips", default=",".join(HELD))
    ap.add_argument("--qps", default="20,28,36")
    ap.add_argument("--crfs", default="22,30,38")
    ap.add_argument("--out", default=os.path.join(RUNS, "probe_domaingap.json"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    qps = [int(x) for x in a.qps.split(",")]
    crfs = [int(x) for x in a.crfs.split(",")]
    vc = VirtualCodec().to(dev).eval()
    vc.load_state_dict(torch.load(a.proxy_ckpt, map_location=dev)["vc"])
    precoders = {}
    for g in ("g0", "g0.5", "g2", "g5"):
        pre = Precoder().to(dev).eval()
        pre.load_state_dict(torch.load(
            os.path.join(RUNS, f"s2_lam0.01_{g}/model.pt"), map_location=dev)["pre"])
        precoders[g] = pre
    # fine mild-regime coverage (g2 at 0.5x/1x/1.5x/2x edit, g0 denoiser,
    # g5 sharper) + two extreme anchors (blur/sharpen)
    CANDS = ["identity", "blur", "g0", "g2x0.5", "g2", "g2x1.5", "g2x2",
             "g5", "sharpen"]

    per_clip = {}
    for clip in a.clips.split(","):
        src = f"{WORK}/src/{clip}.y4m"
        if not os.path.exists(src):
            print(f"skip {clip} (not in shm)", flush=True); continue
        header = y4m_header(src)
        frs = list(read_y4m(src))[: a.frames]
        lumas = [y for y, _, _ in frs]
        chroma = [(u, v) for _, u, v in frs]
        npix = sum(y.size for y in lumas)
        ref = f"{WORK}/probe_ref_{clip}.y4m"
        write_y4m(ref, header, frs)
        real_rows, proxy_rows = {}, {}
        for name in CANDS:
            cl = make_candidate(name, lumas, dev, precoders)
            real_rows[name] = [real_rd(ref, header, cl, chroma, c, npix) for c in crfs]
            proxy_rows[name] = [proxy_rd(ref, header, cl, chroma, q, vc, dev) for q in qps]
            mi = len(qps) // 2
            rb, rn = real_rows[name][mi]; pb, pn = proxy_rows[name][mi]
            print(f"[{clip} {name}] real bpp{rb:.3f} NEG{rn:.1f} | "
                  f"proxy bpp{pb:.3f} NEG{pn:.1f}", flush=True)
        os.remove(ref)
        ri = iso_rank(real_rows, crfs); pi = iso_rank(proxy_rows, qps)
        names = [n for n in CANDS]
        rv = np.array([ri[n] for n in names]); pv = np.array([pi[n] for n in names])
        rho = spearman(pv, rv)
        # pairwise sign agreement (exclude identity=0)
        ok = tot = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(rv[i] - rv[j]) < 1e-6:
                    continue
                tot += 1; ok += int((pv[i] - pv[j]) * (rv[i] - rv[j]) > 0)
        per_clip[clip] = {"rho": rho, "pair_acc": ok / max(tot, 1),
                          "real_iso": {n: float(ri[n]) for n in names},
                          "proxy_iso": {n: float(pi[n]) for n in names}}
        print(f"=== {clip}: Spearman {rho:+.2f}  pairwise {ok/max(tot,1):.2f} ===",
              flush=True)

    rhos = [v["rho"] for v in per_clip.values()]
    accs = [v["pair_acc"] for v in per_clip.values()]
    print(f"\nDOMAIN-GAP PROBE ({len(per_clip)} clips): mean Spearman "
          f"{np.nanmean(rhos):+.2f}  mean pairwise {np.mean(accs):.2f}", flush=True)
    print("(high rho => proxy RD-ranks edits like x264 => proxy tweaks won't help)",
          flush=True)
    json.dump({"per_clip": per_clip, "mean_rho": float(np.nanmean(rhos)),
               "mean_pair_acc": float(np.mean(accs)), "cands": CANDS,
               "proxy_ckpt": a.proxy_ckpt}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
