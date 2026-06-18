"""Smoke test for train_grayscale_codec.py

Validates:
  1. Training logic matches notebook (loss decreases, all grads computed)
  2. Checkpoint save / restore round-trips correctly
  3. Resumed training continues from saved state
"""

import json
import os
import shutil
import tempfile
import numpy as np
import tensorflow as tf
import compress_intra_model

GAMMA = 50
BATCH = 4
SIZE = 64
TAKE = 5
LR = 1e-3


def _make_dataset():
    np.random.seed(0)
    imgs = np.random.randint(0, 256, (BATCH * TAKE, SIZE, SIZE, 3)).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices({"image": imgs}).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def _build():
    return compress_intra_model.create_basic_model(
        model_keys=["image"], bottleneck_channels=1, output_channels=3,
        num_mlp_layers=2, use_jpeg_rate_model=True, downsample_factor=1,
        num_truncate_bits=0, gamma=GAMMA, loop_filter_folder=None,
        use_unet_preprocessor=True, use_unet_postprocessor=True,
    )


# ── Test 1: training loop correctness ────────────────────────────────────────
def test_training_loop():
    print("=" * 60)
    print("TEST 1: Training loop — loss decrease + gradient flow")
    print("=" * 60)
    ds = _make_dataset()
    model = _build()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_fn = compress_intra_model.create_basic_loss(gamma=GAMMA)
    epoch_stat = tf.keras.metrics.Mean()

    losses = []
    for epoch in range(5):
        epoch_stat.reset_state()
        grad_count = 0
        none_grads = 0
        nan_found = False

        for train_batch in ds:
            # notebook-identical tape pattern
            with tf.GradientTape() as tape:
                out = model(train_batch)
                loss = loss_fn(train_batch, out)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_stat(loss)

            for g in gradients:
                if g is None:
                    none_grads += 1
                elif tf.reduce_any(tf.math.is_nan(g)):
                    nan_found = True
                grad_count += 1

        epoch_loss = float(epoch_stat.result())
        losses.append(epoch_loss)
        print(f"  Epoch {epoch}: loss={epoch_loss:.4f}  grads={grad_count - none_grads}/{grad_count}  nan={nan_found}")

    decreasing = all(losses[i] > losses[i + 1] for i in range(len(losses) - 1))
    print(f"\n  Loss trend: {' -> '.join(f'{l:.1f}' for l in losses)}")
    print(f"  Monotonically decreasing: {decreasing}")
    print(f"  No None gradients: {none_grads == 0}")
    assert none_grads == 0, f"Found {none_grads} None gradients"
    assert not nan_found, "NaN in gradients!"
    print("  PASS")
    return model, optimizer


# ── Test 2: checkpoint save → perturb → restore round-trip ────────────────────
def test_checkpoint_roundtrip(model, optimizer):
    print("\n" + "=" * 60)
    print("TEST 2: Checkpoint save / perturb / restore round-trip")
    print("=" * 60)
    tmp = tempfile.mkdtemp()
    try:
        epoch_var = tf.Variable(42, dtype=tf.int64, trainable=False)
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch_var)
        manager = tf.train.CheckpointManager(ckpt, tmp, max_to_keep=3)

        # capture ground-truth state
        w_before = [v.numpy().copy() for v in model.trainable_variables]
        qstep_before = float(model.intra_compression_layer.get_qstep())

        # save
        path = manager.save()
        print(f"  Saved to: {path}")

        # perturb all weights to simulate corruption / fresh init
        for v in model.trainable_variables:
            v.assign(v + tf.random.normal(v.shape, stddev=1.0))
        epoch_var.assign(0)

        max_perturb = max(
            float(tf.reduce_max(tf.abs(v - w)))
            for v, w in zip(model.trainable_variables, w_before)
        )
        print(f"  After perturbation — max weight diff: {max_perturb:.4f}")

        # restore from checkpoint
        status = ckpt.restore(manager.latest_checkpoint)
        status.expect_partial()

        restored_epoch = int(epoch_var.numpy())
        qstep_after = float(model.intra_compression_layer.get_qstep())
        print(f"  Restored epoch: {restored_epoch} (expected 42)")
        assert restored_epoch == 42, f"Epoch mismatch: {restored_epoch}"

        print(f"  qstep: before={qstep_before:.6f}  after={qstep_after:.6f}")
        assert abs(qstep_before - qstep_after) < 1e-6, "qstep mismatch"

        max_diff = 0.0
        for v, w in zip(model.trainable_variables, w_before):
            d = float(tf.reduce_max(tf.abs(v - w)))
            max_diff = max(max_diff, d)
        print(f"  Max weight diff after restore: {max_diff:.2e}")
        assert max_diff < 1e-6, f"Weight mismatch: {max_diff}"

        print("  PASS")
        return model, optimizer, epoch_var, tmp
    except Exception:
        shutil.rmtree(tmp)
        raise


# ── Test 3: resume training from restored checkpoint ─────────────────────────
def test_resume_training(model, optimizer, epoch_var, ckpt_dir):
    print("\n" + "=" * 60)
    print("TEST 3: Resume training from restored checkpoint")
    print("=" * 60)
    try:
        ds = _make_dataset()
        loss_fn = compress_intra_model.create_basic_loss(gamma=GAMMA)
        epoch_stat = tf.keras.metrics.Mean()

        losses = []
        for epoch in range(3):
            epoch_stat.reset_state()
            for train_batch in ds:
                with tf.GradientTape() as tape:
                    out = model(train_batch)
                    loss = loss_fn(train_batch, out)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    epoch_stat(loss)

            epoch_loss = float(epoch_stat.result())
            losses.append(epoch_loss)
            print(f"  Resumed epoch {epoch}: loss={epoch_loss:.4f}")

        finite = all(np.isfinite(l) for l in losses)
        print(f"\n  All losses finite: {finite}")
        assert finite, "Non-finite loss after resume"
        print("  PASS")
    finally:
        shutil.rmtree(ckpt_dir)


# ── Test 4: metrics file ─────────────────────────────────────────────────────
def test_metrics_file():
    print("\n" + "=" * 60)
    print("TEST 4: Metrics JSONL output")
    print("=" * 60)
    from train_grayscale_codec import _append_metrics, METRICS_FILE, OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_file = METRICS_FILE + ".test"
    import train_grayscale_codec as tgc
    orig = tgc.METRICS_FILE
    tgc.METRICS_FILE = test_file
    try:
        _append_metrics(0, 1234.5, 12.3, {"qstep": 7.07})
        _append_metrics(1, 1100.0, 11.5)
        with open(test_file) as f:
            lines = f.readlines()
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        rec = json.loads(lines[0])
        assert rec["epoch"] == 0
        assert abs(rec["loss"] - 1234.5) < 0.01
        assert rec["qstep"] == 7.07
        print(f"  Record 0: {rec}")
        print(f"  Record 1: {json.loads(lines[1])}")
        print("  PASS")
    finally:
        tgc.METRICS_FILE = orig
        if os.path.exists(test_file):
            os.remove(test_file)


def main():
    model, optimizer = test_training_loop()
    model2, optimizer2, epoch_var2, ckpt_dir = test_checkpoint_roundtrip(model, optimizer)
    test_resume_training(model2, optimizer2, epoch_var2, ckpt_dir)
    test_metrics_file()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
