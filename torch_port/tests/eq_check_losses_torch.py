"""Torch-side fidelity-loss equivalence checker (torch-env)."""
import sys, numpy as np, torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port import losses as L

Z = np.load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/eq/losses_ref.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
fails = []

def t(a): return torch.from_numpy(np.asarray(a)).float().to(DEV)

def cmp(name, o, ref, atol, rtol=1e-3):
    o = o.detach().cpu().numpy().astype(np.float64); r = np.asarray(ref, np.float64)
    mabs = float(np.max(np.abs(o - r)))
    srel = mabs / (float(np.max(np.abs(r))) + 1e-12)
    ok = (mabs <= atol) or (srel <= rtol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} max_abs={mabs:.3e} signal_rel={srel:.3e}")
    if not ok: fails.append(name)

cases = sorted({k.split("__")[1] for k in Z.files if k.startswith("gt__")})
for k in cases:
    gt, pred = t(Z[f"gt__{k}"]), t(Z[f"pred__{k}"])
    cmp(f"msssim[{k}]", L.ssim_multiscale_tf(gt, pred, max_val=255.0, filter_size=7),
        Z[f"msssim__{k}"], atol=1e-4, rtol=2e-3)
    cmp(f"mse01[{k}]", L.distortion_mse01(gt, pred), Z[f"mse01__{k}"], atol=1e-6, rtol=1e-4)
    cmp(f"mae01[{k}]", L.distortion_mae01(gt, pred), Z[f"mae01__{k}"], atol=1e-6, rtol=1e-4)
    cmp(f"l1ms[{k}]", L.distortion_l1_msssim(gt, pred, 1.0), Z[f"l1ms__{k}"], atol=5.0, rtol=2e-3)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
