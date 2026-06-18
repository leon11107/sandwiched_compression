"""TF eval-RD reference (sandwich-env): build+load p0p1, run the dpp_rd eval logic
(baseline + model) over a fixed eval set; dump images, per-qstep RD (bpp, rgb_psnr,
y_psnr), and preproc weights for torch porting. Only the preprocessor differs
TF-vs-torch; everything else (PIL jpeg, restore_chroma, psnr) is lib-agnostic."""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["UNET_SCALER_TRAINABLE"] = "1"; os.environ["UNET_SCALER_INIT"] = "0"
sys.path.insert(0, "/workspace/sandwiched_compression")
import numpy as np, tensorflow as tf
import tensorflow_datasets as tfds
import experiments.m2_lowres_repro.m2_common as common

CKPT = "/workspace/sandwiched_compression/experiments/m2_lowres_repro/runs/dpp_faithful_full/p0p1/run/model.weights.h5"
QSTEPS = [16., 24., 32., 48., 64.]
SPLIT, COUNT, SEED, ESIZE = "validation", 16, 20260520, 256

def load_half_res_images(split, count, seed, eval_size):
    ds = tfds.load("clic", split=split, shuffle_files=False, download=False, try_gcs=True)
    min_src = max(2 * eval_size, 64) if eval_size else 64
    def ok(ex):
        s = tf.shape(ex["image"])
        return tf.logical_and(tf.equal(s[2], 3),
                              tf.logical_and(s[0] >= min_src, s[1] >= min_src))
    ds = ds.filter(ok).take(count)
    out = []
    for ex in ds:
        img = tf.cast(ex["image"], tf.float32)
        h = (tf.shape(img)[0] // 2) * 2; w = (tf.shape(img)[1] // 2) * 2
        img = img[:h, :w, :]
        img = tf.image.resize(img, (h // 2, w // 2), method=tf.image.ResizeMethod.LANCZOS3, antialias=True)
        img = tf.clip_by_value(img, 0.0, 255.0)
        if eval_size:
            hh, ww = img.shape[0], img.shape[1]
            top = max((hh - eval_size) // 2, 0); left = max((ww - eval_size) // 2, 0)
            img = img[top:top + eval_size, left:left + eval_size, :]
        out.append(img.numpy())
    return out

def restore_chroma(dec, orig):
    d = tf.image.rgb_to_yuv(tf.constant(dec[None].astype(np.float32)))
    o = tf.image.rgb_to_yuv(tf.constant(orig[None].astype(np.float32)))
    rgb = tf.image.yuv_to_rgb(tf.concat([d[..., 0:1], o[..., 1:3]], axis=-1))
    return np.clip(rgb[0].numpy(), 0.0, 255.0)

def rgb_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)

m = common.create_preproc_only_codec_model(
    0.005, pre_post_arch="paper_unet", train_qstep=False, qstep_init_override=32.0,
    quantizer_mode="noise_injection", codec_forward_mode="real_ste",
    convert_to_yuv=True, preproc_luma_only=True, codec_luma_only=True)
_ = m({"image": tf.zeros([1, 64, 64, 3])}, training=True); m.load_weights(CKPT)
images = load_half_res_images(SPLIT, COUNT, SEED, ESIZE)
images = [np.asarray(im, np.float32) for im in images]
print(f"{len(images)} eval images @ {images[0].shape}")

D = {"qsteps": np.array(QSTEPS, np.float32), "n_img": np.int32(len(images)),
     "scaler": np.float32(float(m._unet_preprocessor_scaler.numpy()))}
for i, im in enumerate(images):
    D[f"img{i}"] = im
w = m._unet_preprocessor.get_weights()
D["nw"] = np.int32(len(w))
for i, a in enumerate(w):
    D[f"w{i}"] = a.astype(np.float32)

for q in QSTEPS:
    b_bpp, b_psnr, m_bpp, m_psnr = [], [], [], []
    for im in images:
        db, bb = common.jpeg_rgb_roundtrip(im, q, subsampling="4:4:4")
        db = restore_chroma(db, im)
        b_bpp.append(bb / (im.shape[0] * im.shape[1])); b_psnr.append(rgb_psnr(db, im))
        pre = np.clip(m.run_preprocessor(im[None].astype(np.float32), training=False)[0].numpy(), 0, 255)
        dm, bm = common.jpeg_rgb_roundtrip(pre, q, subsampling="4:4:4")
        dm = restore_chroma(dm, im)
        m_bpp.append(bm / (im.shape[0] * im.shape[1])); m_psnr.append(rgb_psnr(dm, im))
    D[f"base_bpp_q{int(q)}"] = np.float32(np.mean(b_bpp))
    D[f"base_psnr_q{int(q)}"] = np.float32(np.mean(b_psnr))
    D[f"model_bpp_q{int(q)}"] = np.float32(np.mean(m_bpp))
    D[f"model_psnr_q{int(q)}"] = np.float32(np.mean(m_psnr))
    print(f"q={q:g} base bpp={np.mean(b_bpp):.4f} psnr={np.mean(b_psnr):.3f} | "
          f"model bpp={np.mean(m_bpp):.4f} psnr={np.mean(m_psnr):.3f}")

np.savez("/tmp/eq/eval_ref.npz", **D)
print(f"wrote /tmp/eq/eval_ref.npz ({len(w)} weights)")
