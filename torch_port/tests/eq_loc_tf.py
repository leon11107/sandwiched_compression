"""Localize the UNet port mismatch: dump TF per-stage intermediates (sandwich-env)."""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["UNET_SCALER_TRAINABLE"] = "1"; os.environ["UNET_SCALER_INIT"] = "0"
sys.path.insert(0, "/workspace/sandwiched_compression")
import numpy as np, tensorflow as tf
import experiments.m2_lowres_repro.m2_common as common
CKPT = "/workspace/sandwiched_compression/experiments/m2_lowres_repro/runs/dpp_faithful_full/p0p1/run/model.weights.h5"
m = common.create_preproc_only_codec_model(0.005, pre_post_arch="paper_unet", train_qstep=False,
    qstep_init_override=32.0, quantizer_mode="noise_injection", codec_forward_mode="real_ste",
    convert_to_yuv=True, preproc_luma_only=True, codec_luma_only=True)
_ = m({"image": tf.zeros([1, 64, 64, 3])}, training=True); m.load_weights(CKPT)
rng = np.random.default_rng(20260606)
x = rng.uniform(0, 255, (2, 128, 128, 3)).astype(np.float32)
adj = (tf.constant(x) - 128.0) / 255.0
unet = m._unet_preprocessor

D = {"adj": adj.numpy()}
cur = adj
sk = []
for i, enc in enumerate(unet._encoder_blocks):
    cur, s = enc(cur)
    D[f"enc{i}_pooled"] = cur.numpy()
    D[f"enc{i}_skip"] = s.numpy()
    sk.append(s)
sk.append(None)
n = len(sk)
for i, dec in enumerate(unet._decoder_blocks):
    cur = dec(cur, sk[n - 1 - i])
    D[f"dec{i}"] = cur.numpy()
D["final"] = unet._output_layer(cur).numpy()
np.savez("/tmp/eq/loc_ref.npz", **D)
print("dumped stages:", [k for k in D])
