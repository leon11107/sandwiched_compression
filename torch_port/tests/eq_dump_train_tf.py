"""TF reference: full training-step forward (loss components) + backward (grads).
Deterministic config: straight_through quantizer + real_ste forward (sandwich-env)."""
import os, sys
from functools import partial
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["UNET_SCALER_TRAINABLE"] = "1"; os.environ["UNET_SCALER_INIT"] = "0"
sys.path.insert(0, "/workspace/sandwiched_compression")
import numpy as np, tensorflow as tf
import experiments.m2_lowres_repro.m2_common as common
from compress_intra_model import _distortion_rate_loss
from distortion.perceptual_losses import distortion_l1_msssim

CKPT = "/workspace/sandwiched_compression/experiments/m2_lowres_repro/runs/dpp_faithful_full/p0p1/run/model.weights.h5"
GAMMA = 0.005
m = common.create_preproc_only_codec_model(
    GAMMA, pre_post_arch="paper_unet", train_qstep=False, qstep_init_override=32.0,
    quantizer_mode="straight_through", codec_forward_mode="real_ste",
    convert_to_yuv=True, preproc_luma_only=True, codec_luma_only=True)
_ = m({"image": tf.zeros([1, 64, 64, 3])}, training=True); m.load_weights(CKPT)

rng = np.random.default_rng(20260606)
x = rng.uniform(0, 255, (2, 128, 128, 3)).astype(np.float32)
batch = {"image": tf.constant(x)}
dist_fn = partial(distortion_l1_msssim, lambda_l2=1.0)

# grads w.r.t. the preproc unet weights (38, in get_weights order) + scaler
gvars = list(m._unet_preprocessor.trainable_weights) + [m._unet_preprocessor_scaler]
with tf.GradientTape() as tape:
    out = m(batch, training=True)
    loss_values = _distortion_rate_loss(batch, out, GAMMA, dist_fn)  # [B]
    total = tf.reduce_mean(loss_values)
grads = tape.gradient(total, gvars)

norm = float(out["prediction"].shape[0] / np.prod(out["prediction"].shape))
fid = float((tf.reduce_mean(dist_fn(batch, out)) * norm).numpy())
rate = float((GAMMA * tf.reduce_mean(out["rate"]) * norm).numpy())
D = {"in": x, "total": np.float32(float(total.numpy())), "fid": np.float32(fid),
     "rate": np.float32(rate), "scaler": np.float32(float(m._unet_preprocessor_scaler.numpy())),
     "ng": np.int32(len(gvars))}
# preproc weights (for torch porting) + grads (aligned to gvars order)
w = m._unet_preprocessor.get_weights()
D["nw"] = np.int32(len(w))
for i, a in enumerate(w):
    D[f"w{i}"] = a.astype(np.float32)
for i, g in enumerate(grads):
    D[f"g{i}"] = (np.zeros_like(gvars[i].numpy()) if g is None else g.numpy()).astype(np.float32)
np.savez("/tmp/eq/train_ref.npz", **D)
print(f"total={float(total):.6f} fid={fid:.6f} rate={rate:.6f} scaler={float(m._unet_preprocessor_scaler.numpy()):.5f}")
print(f"dumped {len(grads)} grads ({len(w)} unet weights + 1 scaler)")
