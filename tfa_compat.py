"""Drop-in replacement for tfa_image.resampler using pure TensorFlow ops."""

import tensorflow as tf


def resampler(data: tf.Tensor, warp: tf.Tensor) -> tf.Tensor:
  """Bilinear resampler, equivalent to tensorflow_addons.image.resampler.

  Args:
    data: [batch, height, width, channels] input tensor.
    warp: [batch, h, w, 2] tensor of (x, y) sampling coordinates.

  Returns:
    Resampled tensor of shape [batch, h, w, channels].
  """
  batch_size, data_height, data_width, num_channels = (
      tf.shape(data)[0],
      tf.shape(data)[1],
      tf.shape(data)[2],
      tf.shape(data)[3],
  )

  warp_x = warp[..., 0]
  warp_y = warp[..., 1]

  floor_x = tf.floor(warp_x)
  floor_y = tf.floor(warp_y)
  ceil_x = floor_x + 1.0
  ceil_y = floor_y + 1.0

  alpha_x = warp_x - floor_x
  alpha_y = warp_y - floor_y

  floor_x_int = tf.cast(floor_x, tf.int32)
  floor_y_int = tf.cast(floor_y, tf.int32)
  ceil_x_int = tf.cast(ceil_x, tf.int32)
  ceil_y_int = tf.cast(ceil_y, tf.int32)

  valid_floor_x = tf.logical_and(floor_x_int >= 0, floor_x_int < data_width)
  valid_floor_y = tf.logical_and(floor_y_int >= 0, floor_y_int < data_height)
  valid_ceil_x = tf.logical_and(ceil_x_int >= 0, ceil_x_int < data_width)
  valid_ceil_y = tf.logical_and(ceil_y_int >= 0, ceil_y_int < data_height)

  safe_floor_x = tf.clip_by_value(floor_x_int, 0, data_width - 1)
  safe_floor_y = tf.clip_by_value(floor_y_int, 0, data_height - 1)
  safe_ceil_x = tf.clip_by_value(ceil_x_int, 0, data_width - 1)
  safe_ceil_y = tf.clip_by_value(ceil_y_int, 0, data_height - 1)

  warp_shape = tf.shape(warp_x)
  batch_idx = tf.broadcast_to(
      tf.reshape(tf.range(batch_size), [-1] + [1] * (len(warp_x.shape) - 1)),
      warp_shape,
  )

  def _gather(y_idx, x_idx):
    indices = tf.stack([batch_idx, y_idx, x_idx], axis=-1)
    return tf.gather_nd(data, indices)

  val_ff = _gather(safe_floor_y, safe_floor_x)
  val_fc = _gather(safe_floor_y, safe_ceil_x)
  val_cf = _gather(safe_ceil_y, safe_floor_x)
  val_cc = _gather(safe_ceil_y, safe_ceil_x)

  mask_ff = tf.cast(valid_floor_y & valid_floor_x, data.dtype)[..., tf.newaxis]
  mask_fc = tf.cast(valid_floor_y & valid_ceil_x, data.dtype)[..., tf.newaxis]
  mask_cf = tf.cast(valid_ceil_y & valid_floor_x, data.dtype)[..., tf.newaxis]
  mask_cc = tf.cast(valid_ceil_y & valid_ceil_x, data.dtype)[..., tf.newaxis]

  val_ff = val_ff * mask_ff
  val_fc = val_fc * mask_fc
  val_cf = val_cf * mask_cf
  val_cc = val_cc * mask_cc

  ax = alpha_x[..., tf.newaxis]
  ay = alpha_y[..., tf.newaxis]
  ax = tf.cast(ax, data.dtype)
  ay = tf.cast(ay, data.dtype)

  top = val_ff * (1.0 - ax) + val_fc * ax
  bottom = val_cf * (1.0 - ax) + val_cc * ax
  result = top * (1.0 - ay) + bottom * ay

  return result
