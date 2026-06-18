"""TF reference dumper for fidelity losses + ssim_multiscale (sandwich-env)."""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/workspace/sandwiched_compression")
import numpy as np, tensorflow as tf
from distortion import perceptual_losses as pl

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eq/losses_ref.npz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
rng = np.random.default_rng(20260606)
D = {}

def pair(b, s):
    gt = rng.uniform(0, 255, (b, s, s, 3)).astype(np.float32)
    pred = np.clip(gt + rng.normal(0, 12, gt.shape), 0, 255).astype(np.float32)  # realistic distortion
    return gt, pred

cases = {"r2x128": pair(2, 128), "r1x256": pair(1, 256)}
# corner: pred == gt (perfect), and pred = heavily distorted
g, _ = pair(2, 128)
cases["identical_2x128"] = (g, g.copy())
cases["heavy_1x128"] = (pair(1, 128)[0], np.clip(pair(1, 128)[0] + rng.normal(0, 60, (1, 128, 128, 3)), 0, 255).astype(np.float32))

for k, (gt, pred) in cases.items():
    D[f"gt__{k}"] = gt; D[f"pred__{k}"] = pred
    D[f"msssim__{k}"] = tf.image.ssim_multiscale(tf.constant(gt), tf.constant(pred),
                                                 max_val=255.0, filter_size=7).numpy()
    gtd = {"image": tf.constant(gt)}; od = {"prediction": tf.constant(pred)}
    D[f"mse01__{k}"] = pl.distortion_mse01(gtd, od).numpy()
    D[f"mae01__{k}"] = pl.distortion_mae01(gtd, od).numpy()
    D[f"l1ms__{k}"] = pl.distortion_l1_msssim(gtd, od, lambda_l2=1.0).numpy()

np.savez(OUT, **D)
print(f"wrote {OUT}; cases={list(cases)}")
