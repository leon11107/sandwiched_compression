"""Smoke test: build model, forward pass, one training step with synthetic data."""

import numpy as np
import tensorflow as tf
import compress_intra_model

print("=== Smoke Test: Sandwiched Compression Training ===\n")

# 1. Create synthetic dataset (random 64x64 RGB images, small for speed)
print("[1/5] Creating synthetic dataset...")
batch_size = 2
img_size = 64
num_batches = 3

def make_synthetic_dataset():
  images = np.random.randint(0, 256, (batch_size * num_batches, img_size, img_size, 3)).astype(np.float32)
  ds = tf.data.Dataset.from_tensor_slices({'image': images})
  ds = ds.batch(batch_size)
  return ds

train_dataset = make_synthetic_dataset()
print(f"  Dataset: {num_batches} batches x {batch_size} images of {img_size}x{img_size}x3")

# 2. Create model (grayscale codec scenario, same as notebook)
print("\n[2/5] Creating model...")
gamma = 50
model = compress_intra_model.create_basic_model(
    model_keys=['image'],
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
print(f"  Model: {model.name}")

# 3. Forward pass
print("\n[3/5] Running forward pass...")
sample_batch = next(iter(train_dataset))
output = model(sample_batch, training=False)
print(f"  Input shape:  {sample_batch['image'].shape}")
print(f"  Output keys:  {list(output.keys())}")
print(f"  Prediction shape: {output['prediction'].shape}")
print(f"  Bottleneck shape: {output['bottleneck'].shape}")
print(f"  Rate: {np.mean(output['rate'].numpy()):.4f}")
param_count = sum(p.numpy().size for p in model.trainable_variables)
print(f"  Trainable parameters: {param_count:,}")

# 4. Training step with gradient
print("\n[4/5] Running training steps (1 epoch, {0} batches)...".format(num_batches))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = compress_intra_model.create_basic_loss(gamma=gamma)
epoch_stat = tf.keras.metrics.Mean()

for step, train_batch in enumerate(train_dataset):
  with tf.GradientTape() as tape:
    out = model(train_batch, training=True)
    loss = loss_fn(train_batch, out)

  gradients = tape.gradient(loss, model.trainable_variables)
  non_none_grads = sum(1 for g in gradients if g is not None)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  epoch_stat(loss)
  print(f"  Step {step}: loss={float(tf.reduce_mean(loss)):.4f}, grads={non_none_grads}/{len(gradients)}")

print(f"  Epoch mean loss: {float(epoch_stat.result()):.4f}")

# 5. Verify loss decreased with a second pass
print("\n[5/5] Verifying loss decreases over 3 more epochs...")
losses = []
for epoch in range(3):
  epoch_stat.reset_state()
  for train_batch in train_dataset:
    with tf.GradientTape() as tape:
      out = model(train_batch, training=True)
      loss = loss_fn(train_batch, out)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    epoch_stat(loss)
  epoch_loss = float(epoch_stat.result())
  losses.append(epoch_loss)
  print(f"  Epoch {epoch}: loss={epoch_loss:.4f}")

print(f"\n{'='*50}")
if losses[-1] < losses[0]:
  print("PASS: Loss is decreasing — training loop works correctly.")
else:
  print("WARN: Loss did not decrease (may be normal with random data/few steps).")
  print(f"  First: {losses[0]:.4f}, Last: {losses[-1]:.4f}")

print("\nSmoke test complete.")
