"""Training script — faithful copy of sandwich_image_compression_grayscale_codec.ipynb.

Additions over notebook:
  - CLI args: --seed, --gpu, --max-epochs (for benchmarking)
  - Checkpoint / resume support (model + optimizer + epoch)
  - Efficiency: tf.data prefetch pipeline
  - Comprehensive debug logging to file + stdout
  - Validation every 100 epochs with Y/U/V PSNR + image dumps
  - All artefacts stored under OUTPUT_DIR/<seed>

Usage:
  python train_grayscale_codec.py --seed 1 --gpu 0
  python train_grayscale_codec.py --seed 2 --gpu 1
  python train_grayscale_codec.py --seed 1 --gpu 0 --max-epochs 5  # benchmark
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image

import compress_intra_model
import datasets

# ──────────────────────────────────────────────────────────────────────────────
# Config — matches notebook exactly
# ──────────────────────────────────────────────────────────────────────────────
NUM_EPOCHS = 800
GAMMA = 50
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 1
TAKE_COUNT = 100
TARGET_SIZE = 256
DATASET_NAME = "clic"
LEARNING_RATE = 1e-3

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "instances", "5070ti", "instance_id_36694773",
)

CKPT_INTERVAL = 10   # save every N epochs
EVAL_INTERVAL = 100  # validate every N epochs
EVAL_SHOW_COUNT = 10  # number of eval images (matches notebook cell-8)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Sandwiched Compression training")
    p.add_argument("--seed", type=int, required=True, help="Random seed (1, 2, ...)")
    p.add_argument("--gpu", type=int, required=True, help="GPU index (0 or 1)")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override NUM_EPOCHS (for benchmarking)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# GPU isolation
# ──────────────────────────────────────────────────────────────────────────────
def _setup_gpu(gpu_id, log):
    gpus = tf.config.list_physical_devices("GPU")
    log.info("Physical GPUs found: %d", len(gpus))
    for i, g in enumerate(gpus):
        d = tf.config.experimental.get_device_details(g)
        log.info("  GPU:%d  %s  %s", i, g.name, d)

    if gpu_id >= len(gpus):
        log.error("Requested GPU:%d but only %d GPUs available", gpu_id, len(gpus))
        sys.exit(1)

    tf.config.set_visible_devices([gpus[gpu_id]], "GPU")
    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    log.info("Pinned to GPU:%d  (%s)", gpu_id,
             tf.config.experimental.get_device_details(gpus[gpu_id]).get("device_name", "?"))


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def _setup_logging(output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    fmt = f"[%(asctime)s] [seed={seed}] %(levelname)s %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    logging.getLogger("absl").setLevel(logging.WARNING)
    return logging.getLogger("train")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  (notebook cell 4, optimised: GPU-resident crop pool)
# ──────────────────────────────────────────────────────────────────────────────
_INITIAL_CROP = 512
POOL_REFRESH_INTERVAL = 50  # re-crop raw images every N epochs for diversity


class ImagePool:
    """Pre-cropped, GPU-resident image pool. Eliminates tf.data overhead."""

    def __init__(self, split_name, log=None):
        import tensorflow_datasets as tfds
        self._split = split_name
        self._pool = None

        # only count images to report, don't hold them in RAM
        t0 = time.time()
        raw = tfds.load(DATASET_NAME, split=split_name, shuffle_files=True,
                        download=True, try_gcs=True)
        def _ok(ex):
            s = tf.shape(ex["image"])
            return s[2] == 3 and s[0] >= _INITIAL_CROP and s[1] >= _INITIAL_CROP
        self._ds_filtered = raw.filter(_ok)
        self._count = sum(1 for _ in self._ds_filtered)
        if log:
            log.info("ImagePool(%s): %d images (%.1fs)",
                     split_name, self._count, time.time() - t0)

    def refresh(self, log=None):
        """Read from disk, random-crop, batch-resize to 256x256 in chunks."""
        t0 = time.time()
        CHUNK = 200
        resized_chunks = []
        buf = []
        for ex in self._ds_filtered:
            with tf.device("/CPU:0"):
                c = tf.image.random_crop(ex["image"], [_INITIAL_CROP, _INITIAL_CROP, 3])
                buf.append(tf.cast(c, tf.float32))
            if len(buf) >= CHUNK:
                with tf.device("/CPU:0"):
                    stacked = tf.stack(buf)
                resized_chunks.append(tf.image.resize(
                    stacked, (TARGET_SIZE, TARGET_SIZE),
                    method=tf.image.ResizeMethod.LANCZOS3, antialias=True))
                del stacked
                buf = []
        if buf:
            with tf.device("/CPU:0"):
                stacked = tf.stack(buf)
            resized_chunks.append(tf.image.resize(
                stacked, (TARGET_SIZE, TARGET_SIZE),
                method=tf.image.ResizeMethod.LANCZOS3, antialias=True))
            del stacked, buf
        self._pool = tf.concat(resized_chunks, axis=0)
        del resized_chunks
        if log:
            log.info("ImagePool refreshed: %s (%.2f GB, %.1fs)",
                     self._pool.shape,
                     self._pool.numpy().nbytes / 1e9,
                     time.time() - t0)

    def sample_epoch(self, batch_size, take_count):
        """Return list of {image: [B,H,W,3]} dicts for one epoch."""
        n_images = take_count * batch_size
        idx = tf.random.shuffle(tf.range(self._pool.shape[0]))[:n_images]
        data = tf.gather(self._pool, idx)
        batches = tf.reshape(data, [take_count, batch_size, TARGET_SIZE, TARGET_SIZE, 3])
        return [{"image": batches[i]} for i in range(take_count)]


def eval_dataset_fn(cached_base_pool):
    """For validation: return tf.data style iterator (batch_size=1)."""
    if cached_base_pool._pool is None:
        cached_base_pool.refresh()
    ds = tf.data.Dataset.from_tensor_slices({"image": cached_base_pool._pool})
    ds = ds.batch(EVAL_BATCH_SIZE)
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Model  (notebook cell 5 — identical)
# ──────────────────────────────────────────────────────────────────────────────
def create_grayscale_codec_model(gamma):
    return compress_intra_model.create_basic_model(
        model_keys=["image"],
        bottleneck_channels=1,
        output_channels=3,
        num_mlp_layers=2,
        use_jpeg_rate_model=True,
        downsample_factor=1,
        num_truncate_bits=0,
        gamma=gamma,
        loop_filter_folder=None,
        use_unet_preprocessor=True,
        use_unet_postprocessor=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def _build_checkpoint(model, optimizer, epoch_var):
    return tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch_var)


def _save_checkpoint(ckpt, manager, epoch, log):
    path = manager.save()
    log.info("Checkpoint saved: %s  (epoch %d)", path, epoch)


def _restore_checkpoint(ckpt, manager, ckpt_dir, log):
    latest = manager.latest_checkpoint
    if latest is None:
        log.info("No checkpoint in %s — starting from scratch.", ckpt_dir)
        return 0
    status = ckpt.restore(latest)
    status.expect_partial()
    restored_epoch = int(ckpt.epoch.numpy())
    log.info("Restored %s — resuming after epoch %d", latest, restored_epoch)
    return restored_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Metrics persistence
# ──────────────────────────────────────────────────────────────────────────────
def _append_metrics(metrics_file, epoch, epoch_loss, epoch_time, extra=None):
    record = {"epoch": int(epoch), "loss": float(epoch_loss),
              "epoch_sec": round(epoch_time, 2)}
    if extra:
        record.update(extra)
    with open(metrics_file, "a") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Debug helpers
# ──────────────────────────────────────────────────────────────────────────────
def _log_grad_stats(gradients, variables, log):
    total_norm_sq = 0.0
    zero_grads, nan_grads = [], []
    for g, v in zip(gradients, variables):
        if g is None:
            zero_grads.append(v.name); continue
        g_np = g.numpy() if hasattr(g, "numpy") else g.values.numpy()
        total_norm_sq += float(np.sum(g_np ** 2))
        if np.any(np.isnan(g_np)):
            nan_grads.append(v.name)
    log.debug("  grad_norm=%.6f", np.sqrt(total_norm_sq))
    if zero_grads:
        log.warning("  None grads: %s", zero_grads)
    if nan_grads:
        log.warning("  NaN grads: %s", nan_grads)


def _log_model_summary(model, log):
    param_count = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    log.info("Trainable parameters: %s", f"{param_count:,}")
    log.info("Trainable variable count: %d", len(model.trainable_variables))
    for v in model.trainable_variables:
        log.debug("  var %-50s shape=%s dtype=%s", v.name, v.shape, v.dtype)


def _log_batch_debug(step, loss, out, log):
    log.debug(
        "  step=%d  loss=%.4f  rate=%.4f  pred=[%.2f,%.2f]  bn=[%.2f,%.2f]",
        step,
        float(tf.reduce_mean(loss)),
        float(tf.reduce_mean(out["rate"])),
        float(tf.reduce_min(out["prediction"])),
        float(tf.reduce_max(out["prediction"])),
        float(tf.reduce_min(out["bottleneck"])),
        float(tf.reduce_max(out["bottleneck"])),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validation — Y/U/V PSNR + image dumps  (mirrors notebook cell 8)
# ──────────────────────────────────────────────────────────────────────────────
def _rgb_to_yuv(img_f32):
    r, g, b = img_f32[..., 0], img_f32[..., 1], img_f32[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0
    return np.stack([y, u, v], axis=-1)


def _psnr_per_channel(a, b, peak=255.0):
    a = np.clip(a, 0, 255).astype(np.float64)
    b = np.clip(b, 0, 255).astype(np.float64)
    results = {}
    for ch, name in enumerate(["Y", "U", "V"]):
        mse = np.mean((a[..., ch] - b[..., ch]) ** 2)
        results[name] = 10.0 * np.log10(peak ** 2 / max(mse, 1e-10))
    mse_all = np.mean((a - b) ** 2)
    results["YUV_avg"] = 10.0 * np.log10(peak ** 2 / max(mse_all, 1e-10))
    return results


def _save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def run_validation(model, eval_dataset, epoch, output_dir, log):
    val_dir = os.path.join(output_dir, "val", f"epoch_{epoch:04d}")
    os.makedirs(val_dir, exist_ok=True)
    all_psnr = []
    val_t0 = time.time()

    log.info("Validation @ epoch %d — %d images ...", epoch, EVAL_SHOW_COUNT)

    for idx, sample in enumerate(eval_dataset.as_numpy_iterator()):
        if idx >= EVAL_SHOW_COUNT:
            break

        output = model(sample, training=False)
        src = sample["image"][0]
        pred = output["prediction"].numpy()[0]
        bn = output["bottleneck"].numpy()[0]
        cbn = output["compressed_bottleneck"].numpy()[0]
        rate = float(tf.reduce_mean(output["rate"]))

        src_yuv = _rgb_to_yuv(src)
        pred_yuv = _rgb_to_yuv(pred)
        psnr = _psnr_per_channel(src_yuv, pred_yuv)
        psnr["rate"] = rate
        psnr["idx"] = idx
        all_psnr.append(psnr)

        log.info("  [%02d] PSNR  Y=%.2f  U=%.2f  V=%.2f  avg=%.2f  rate=%.1f",
                 idx, psnr["Y"], psnr["U"], psnr["V"], psnr["YUV_avg"], rate)

        prefix = os.path.join(val_dir, f"img_{idx:02d}")
        _save_image(src, f"{prefix}_src.png")
        _save_image(pred, f"{prefix}_pred.png")
        _save_image(bn, f"{prefix}_bottleneck.png")
        _save_image(cbn, f"{prefix}_compressed_bn.png")

    if all_psnr:
        avg = {k: float(np.mean([p[k] for p in all_psnr]))
               for k in ["Y", "U", "V", "YUV_avg", "rate"]}
        log.info("  MEAN  PSNR  Y=%.2f  U=%.2f  V=%.2f  avg=%.2f  rate=%.1f",
                 avg["Y"], avg["U"], avg["V"], avg["YUV_avg"], avg["rate"])
        with open(os.path.join(val_dir, "psnr_summary.json"), "w") as f:
            json.dump({"epoch": epoch, "mean_psnr": avg, "per_image": all_psnr}, f, indent=2)
        log.info("  Saved to %s (%.1fs)", val_dir, time.time() - val_t0)
        return avg
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    seed = args.seed
    gpu_id = args.gpu
    num_epochs = args.max_epochs if args.max_epochs else NUM_EPOCHS

    # ── paths per seed ───────────────────────────────────────────────────
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    metrics_file = os.path.join(output_dir, "metrics.jsonl")

    log = _setup_logging(output_dir, seed)
    log.info("=" * 60)
    log.info("Sandwiched Compression — Grayscale Codec Training")
    log.info("=" * 60)
    log.info("TensorFlow %s  |  seed=%d  |  gpu=%d", tf.__version__, seed, gpu_id)
    log.info("Output dir: %s", output_dir)
    log.info("Config: epochs=%d gamma=%s lr=%s batch=%d target=%d take=%d",
             num_epochs, GAMMA, LEARNING_RATE, TRAIN_BATCH_SIZE, TARGET_SIZE, TAKE_COUNT)

    # ── GPU ──────────────────────────────────────────────────────────────
    _setup_gpu(gpu_id, log)

    # ── seed ─────────────────────────────────────────────────────────────
    tf.random.set_seed(seed)
    np.random.seed(seed)
    log.info("Random seed set: tf=%d np=%d", seed, seed)

    # ── datasets (cell 6) ────────────────────────────────────────────────
    log.info("Building image pools...")
    train_pool = ImagePool("train", log=log)
    eval_pool = ImagePool("test", log=log)

    train_pool.refresh(log=log)
    eval_pool.refresh(log=log)

    eval_dataset = eval_dataset_fn(eval_pool)

    sample_batch = train_pool.sample_epoch(TRAIN_BATCH_SIZE, 1)[0]
    log.info("Train sample shape: %s  dtype=%s  range=[%.1f, %.1f]",
             sample_batch["image"].shape, sample_batch["image"].dtype,
             float(tf.reduce_min(sample_batch["image"])),
             float(tf.reduce_max(sample_batch["image"])))

    # ── model (cell 5) ───────────────────────────────────────────────────
    log.info("Building model...")
    base_model = create_grayscale_codec_model(GAMMA)
    log.info("Model built: %s", base_model.name)

    for sample in [sample_batch]:
        _ = base_model(sample, training=False)
    _log_model_summary(base_model, log)

    # ── optimizer & loss (cell 7) ────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = compress_intra_model.create_basic_loss(gamma=GAMMA)
    epoch_stat = tf.keras.metrics.Mean()

    # ── checkpoint setup ─────────────────────────────────────────────────
    os.makedirs(ckpt_dir, exist_ok=True)
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
    ckpt = _build_checkpoint(base_model, optimizer, epoch_var)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    start_epoch = _restore_checkpoint(ckpt, manager, ckpt_dir, log)

    qstep = base_model.intra_compression_layer.get_qstep()
    pre_s, post_s = base_model.get_pre_post_scalers()
    log.info("Initial state: qstep=%.4f  pre_scaler=%.4f  post_scaler=%.4f",
             float(qstep), float(pre_s), float(post_s))

    # ── compiled training step ─────────────────────────────────────────
    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            out = base_model(batch)
            loss = loss_fn(batch, out)
            gradients = tape.gradient(loss, base_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))
            epoch_stat(loss)
        return loss, out

    # trace once to compile
    log.info("Tracing @tf.function (first call — may take ~20s) ...")
    trace_t0 = time.time()
    _loss, _out = train_step(sample_batch)
    log.info("@tf.function traced in %.1fs", time.time() - trace_t0)

    # ── baseline validation ──────────────────────────────────────────────
    if start_epoch == 0:
        log.info("Running baseline validation (epoch 0, before training) ...")
        run_validation(base_model, eval_dataset, epoch=0, output_dir=output_dir, log=log)

    # ── training loop (cell 7) ───────────────────────────────────────────
    log.info("Starting training from epoch %d to %d ...", start_epoch, num_epochs - 1)
    total_t0 = time.time()
    epoch_times = []

    for i in range(start_epoch, num_epochs):
        # refresh crop pool periodically for diversity
        if i > start_epoch and i % POOL_REFRESH_INTERVAL == 0:
            train_pool.refresh(log=log)

        epoch_t0 = time.time()
        step = 0
        epoch_stat.reset_state()

        epoch_batches = train_pool.sample_epoch(TRAIN_BATCH_SIZE, TAKE_COUNT)
        for train_batch in epoch_batches:
            loss, out = train_step(train_batch)

            if step % 20 == 0 or step < 3:
                _log_batch_debug(step, loss, out, log)
            step += 1

        epoch_time = time.time() - epoch_t0
        epoch_times.append(epoch_time)
        epoch_loss = float(epoch_stat.result())

        qstep_val = float(base_model.intra_compression_layer.get_qstep())
        pre_s, post_s = base_model.get_pre_post_scalers()
        extra = {
            "seed": seed, "gpu": gpu_id,
            "qstep": round(qstep_val, 4),
            "pre_scaler": round(float(pre_s), 6),
            "post_scaler": round(float(post_s), 6),
            "steps": step,
            "lr": float(optimizer.learning_rate),
        }

        log.info("Epoch %4d/%4d  Loss: %.4f  qstep=%.4f  time=%.1fs  steps=%d",
                 i, num_epochs, epoch_loss, qstep_val, epoch_time, step)

        _append_metrics(metrics_file, i, epoch_loss, epoch_time, extra)

        # checkpoint
        if (i + 1) % CKPT_INTERVAL == 0 or i == num_epochs - 1:
            epoch_var.assign(i + 1)
            _save_checkpoint(ckpt, manager, i, log)

        # validation
        if (i + 1) % EVAL_INTERVAL == 0 or i == num_epochs - 1:
            run_validation(base_model, eval_dataset, epoch=i + 1,
                           output_dir=output_dir, log=log)

    total_time = time.time() - total_t0
    trained_epochs = num_epochs - start_epoch

    log.info("=" * 60)
    log.info("Training complete. %d epochs in %.1fs (%.2fs/epoch)",
             trained_epochs, total_time,
             total_time / max(trained_epochs, 1))

    # ── time estimate for full 800 epochs ────────────────────────────────
    if epoch_times:
        # skip first epoch (JIT warmup) for fair estimate
        warm = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
        avg_sec = float(np.mean(warm))
        remaining = NUM_EPOCHS - num_epochs
        if remaining > 0:
            eta_sec = remaining * avg_sec
            log.info("Avg epoch (excl warmup): %.2fs  |  Remaining %d epochs  |  ETA: %.1f min (%.1f hr)",
                     avg_sec, remaining, eta_sec / 60, eta_sec / 3600)
        else:
            log.info("Avg epoch (excl warmup): %.2fs", avg_sec)

    log.info("Artefacts: %s", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
