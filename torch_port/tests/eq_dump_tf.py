"""TF reference dumper for codec equivalence (run in /venv/sandwich-env).

Generates seeded random in-range inputs + corner cases, runs the TF reference
ops (JpegProxy primitives + EncodeDecodeIntra full path incl gradients), and
saves EVERYTHING (inputs + outputs) to an npz. The torch checker loads the SAME
input arrays and compares. Covers: op-level (DCT fwd/inv, YCbCr), full codec
(proxy + real_ste, log_nonzero rate), gradient, and corner cases.
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/workspace/sandwiched_compression")
import numpy as np
import tensorflow as tf
from PIL import ImageFile
ImageFile.MAXBLOCK = 2 ** 27  # avoid "broken data stream" on highly-compressible corner cases
from image_compression import jpeg_proxy
from image_compression import encode_decode_intra_lib as edi
from compress_intra_model import differentiable_round

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eq/codec_ref.npz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
rng = np.random.default_rng(20260606)
D = {}

def rnd(b, h, w, c=3):
    return rng.uniform(0.0, 255.0, (b, h, w, c)).astype(np.float32)

# ---- input set: random in-range + corner cases -----------------------------
inputs = {
    "rand_2x16": rnd(2, 16, 16),
    "rand_1x8": rnd(1, 8, 8),            # single 8x8 block
    "rand_3x128": rnd(3, 128, 128),      # realistic crop size
    "rand_1x256": rnd(1, 256, 256),      # eval size
    "zeros_2x16": np.zeros((2, 16, 16, 3), np.float32),     # corner: all 0
    "max_2x16": np.full((2, 16, 16, 3), 255.0, np.float32),  # corner: all 255
    "edges_1x16": (rng.integers(0, 2, (1, 16, 16, 3)) * 255.0).astype(np.float32),  # 0/255 only
}
for k, v in inputs.items():
    D[f"in__{k}"] = v

ones8 = np.ones((8, 8), np.float32)
jp = jpeg_proxy.JpegProxy(downsample_chroma=False, luma_quantization_table=ones8,
                          chroma_quantization_table=ones8, convert_to_yuv=True,
                          clip_to_image_max=True)

# ---- op-level: DCT fwd/inv (on channel 0), YCbCr round-trip ----------------
for k, v in inputs.items():
    ch = tf.constant(v[..., 0:1])
    fwd = jp._forward_dct_2d(ch).numpy()
    D[f"dctfwd__{k}"] = fwd
    D[f"dctinv__{k}"] = jp._inverse_dct_2d(tf.constant(fwd)).numpy()
    D[f"rgb2yuv__{k}"] = jp._rgb_to_yuv(tf.constant(v)).numpy()
    D[f"yuv2rgb__{k}"] = jp._yuv_to_rgb(jp._rgb_to_yuv(tf.constant(v))).numpy()

# ---- full EncodeDecodeIntra: proxy + real_ste, log_nonzero rate ------------
def build(mode, qstep):
    return edi.EncodeDecodeIntra(
        rounding_fn=differentiable_round, use_jpeg_rate_model=True,
        qstep_init=float(qstep), train_qstep=False, min_qstep=1.0,
        convert_to_yuv=True, downsample_chroma=False, rate_proxy_mode="log_nonzero",
        codec_forward_mode=mode, output_clip_mode="hard")

for q in [1, 16, 32, 64, 255]:
    m = build("proxy", q)
    for k, v in inputs.items():
        dec, rate = m(tf.constant(v))
        D[f"proxydec__{k}__q{q}"] = dec.numpy()
        D[f"proxyrate__{k}__q{q}"] = rate.numpy()

mste = build("real_ste", 32)
for k, v in inputs.items():
    dec, rate = mste(tf.constant(v))
    D[f"stedec__{k}__q32"] = dec.numpy()
    D[f"sterate__{k}__q32"] = rate.numpy()

# ---- gradient: dL/dx for L=sum(decoded)+sum(rate), proxy straight_through ---
mg = build("proxy", 32)
for k in ["rand_2x16", "rand_3x128"]:
    xv = tf.Variable(inputs[k])
    with tf.GradientTape() as t:
        dec, rate = mg(xv)
        L = tf.reduce_sum(dec) + tf.reduce_sum(rate)
    D[f"grad__{k}__q32"] = t.gradient(L, xv).numpy()
    D[f"gradL__{k}__q32"] = np.float32(L.numpy())

np.savez(OUT, **D)
print(f"wrote {OUT} with {len(D)} arrays; inputs={list(inputs)}")
print("qsteps proxy: 1,16,32,64,255 | real_ste q32 | grad on rand_2x16,rand_3x128")
