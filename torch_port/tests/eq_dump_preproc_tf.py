"""TF reference dumper for the paper_unet preprocessor (run in sandwich-env).

Builds the real preproc-only model, loads a TRAINED checkpoint (nonzero scaler +
real UNet weights, so the residual/UNet path is genuinely exercised), runs on
seeded random in-range inputs + a corner case, and dumps: inputs, raw UNet output
(isolates conv/pool/bilinear-upsample), full run_preprocessor output (residual +
luma-only), the 38 UNet weight arrays, and the scaler.
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["UNET_SCALER_TRAINABLE"] = "1"
os.environ["UNET_SCALER_INIT"] = "0"
sys.path.insert(0, "/workspace/sandwiched_compression")
import numpy as np
import tensorflow as tf
import experiments.m2_lowres_repro.m2_common as common

CKPT = "/workspace/sandwiched_compression/experiments/m2_lowres_repro/runs/dpp_faithful_full/p0p1/run/model.weights.h5"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eq/preproc_ref.npz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
rng = np.random.default_rng(20260606)

m = common.create_preproc_only_codec_model(
    0.005, pre_post_arch="paper_unet", train_qstep=False, qstep_init_override=32.0,
    quantizer_mode="noise_injection", codec_forward_mode="real_ste",
    convert_to_yuv=True, preproc_luma_only=True, codec_luma_only=True)
d = {"image": tf.zeros([1, 64, 64, 3])}
_ = m(d, training=True); _ = m(d, training=False)
m.load_weights(CKPT)
scaler = float(m._unet_preprocessor_scaler.numpy())
print(f"loaded {CKPT}; scaler={scaler:.5f} (nonzero => UNet exercised)")

inputs = {
    "rand_2x128": rng.uniform(0, 255, (2, 128, 128, 3)).astype(np.float32),
    "rand_1x256": rng.uniform(0, 255, (1, 256, 256, 3)).astype(np.float32),
    "edges_1x64": (rng.integers(0, 2, (1, 64, 64, 3)) * 255.0).astype(np.float32),
}
D = {"scaler": np.float32(scaler), "mean_adjust": np.float32(128.0),
     "scale_adjust": np.float32(255.0)}
w = m._unet_preprocessor.get_weights()
D["nw"] = np.int32(len(w))
for i, a in enumerate(w):
    D[f"w{i}"] = a.astype(np.float32)

for k, v in inputs.items():
    D[f"in__{k}"] = v
    adj = (tf.constant(v) - 128.0) / 255.0
    D[f"unet__{k}"] = m._unet_preprocessor(adj, training=False).numpy()
    D[f"preproc__{k}"] = m.run_preprocessor(tf.constant(v), training=False).numpy()

# tf.image yuv kernel parity sample
xs = tf.constant(rng.uniform(0, 255, (1, 4, 4, 3)).astype(np.float32))
D["yuv_in"] = xs.numpy()
D["yuv_out"] = tf.image.rgb_to_yuv(xs).numpy()
D["yuv_rt"] = tf.image.yuv_to_rgb(tf.image.rgb_to_yuv(xs)).numpy()

np.savez(OUT, **D)
print(f"wrote {OUT}: {len(w)} weight arrays, inputs={list(inputs)}")
