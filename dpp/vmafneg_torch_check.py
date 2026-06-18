"""Fidelity gate for the FAITHFUL differentiable VMAF_NEG (vmaf-torch, NEG=True)
vs the real vmaf binary, on rows of vmafneg_data.jsonl (real NEG labels over
22 diverse distortions/image). This is what a LEARNED surrogate could not pass
under argmax pressure; an analytic reimplementation must match everywhere.

Luma path replicated from distortion/vmaf_metric.py: RGB -> ffmpeg yuv420p
(BT.601 limited range) -> Y plane. Torch: Y = 16 + (.299R+.587G+.114B)*219/255,
rounded to uint8 grid (binary sees 8-bit planes).

Gate: within-(img,q)-group pairwise rank_acc >= 0.97 (|dNEG|>0.5 pairs) and
MAE <= ~1.5. Report bias and worst rows for diagnosis.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from dpp.oracle_vmafneg import block_dct, load16, luma, up1080
from dpp.vmafneg_data import make_dist

DATA = "/workspace/sandwiched_compression/dpp/runs/vmafneg_data.jsonl"


def rgb_to_vmaf_luma(rgb):
    """float RGB [0,255] -> BT.601 limited-range Y, rounded to the 8-bit grid."""
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.rint(16.0 + y * (219.0 / 255.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-imgs", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/workspace/sandwiched_compression/dpp/runs/"
                                     "vmafneg_torch_check.json")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = f"cuda:{a.gpu}"
    from vmaf_torch import VMAF
    vt = VMAF(NEG=True).to(dev).eval()

    byimg = {}
    for line in open(DATA):
        r = json.loads(line)
        byimg.setdefault(r["path"], []).append(r)
    paths = sorted(byimg)
    paths = paths[:: max(1, len(paths) // a.n_imgs)][: a.n_imgs]

    rows_out = []
    for pi, p in enumerate(paths):
        rgb = load16(p)
        y = luma(rgb); coeffs = block_dct(y)
        ref_up = up1080(np.rint(rgb).astype(np.uint8)).astype(np.float32)
        ref_t = torch.from_numpy(rgb_to_vmaf_luma(ref_up)[None, None]
                                 ).float().to(dev)
        for r in byimg[p]:
            dec, _ = make_dist(rgb, y, coeffs, r["kind"], r["q"],
                               np.asarray(r["s"]) if r["s"] is not None else None,
                               tuple(r["contrast"]) if r["contrast"] else None)
            dist_up = up1080(np.rint(dec).astype(np.uint8)).astype(np.float32)
            dist_t = torch.from_numpy(rgb_to_vmaf_luma(dist_up)[None, None]
                                      ).float().to(dev)
            with torch.no_grad():
                pred = float(vt(ref_t, dist_t))
            rows_out.append({"path": p, "kind": r["kind"], "q": r["q"],
                             "real": r["vmaf_neg"], "pred": pred})
        if (pi + 1) % 5 == 0:
            err = np.array([abs(x["pred"] - x["real"]) for x in rows_out])
            print(f"{pi+1}/{len(paths)} imgs, running MAE {err.mean():.3f}",
                  flush=True)

    real = np.array([x["real"] for x in rows_out])
    pred = np.array([x["pred"] for x in rows_out])
    err = pred - real
    print(f"\nn={len(rows_out)}  MAE {np.abs(err).mean():.3f}  bias {err.mean():+.3f}"
          f"  p95|err| {np.percentile(np.abs(err),95):.3f}  max|err| "
          f"{np.abs(err).max():.3f}", flush=True)

    groups = {}
    for x in rows_out:
        groups.setdefault((x["path"], x["q"]), []).append(x)
    ok = n = 0
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if abs(g[i]["real"] - g[j]["real"]) < 0.5:
                    continue
                n += 1
                ok += int((g[i]["pred"] - g[j]["pred"]) *
                          (g[i]["real"] - g[j]["real"]) > 0)
    print(f"within-(img,q) pairwise rank_acc {ok/max(n,1):.4f} (n={n})", flush=True)

    worst = sorted(rows_out, key=lambda x: -abs(x["pred"] - x["real"]))[:8]
    for w in worst:
        print(f"  worst: {os.path.basename(w['path'])} {w['kind']} q{w['q']} "
              f"real {w['real']:.2f} pred {w['pred']:.2f}", flush=True)
    json.dump({"rows": rows_out, "mae": float(np.abs(err).mean()),
               "bias": float(err.mean()), "rank_acc": ok / max(n, 1)},
              open(a.out, "w"))
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
