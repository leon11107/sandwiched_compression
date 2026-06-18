"""Quantization-aware training (QAT) primitives.

QATConv2D: Conv2D subclass with W8A8 fake-quant.
  - weights: per-output-channel symmetric int8 (scale = max|w|/127 per filter)
  - activations: per-tensor symmetric int8 with non-trainable scale (set via
    calibration before training; fixed during fine-tune)
  - backward: STE via x + stop_gradient(q - x) (no custom_gradient needed)

Use:
  m_qat = build_qat_model(...)
  transfer_fp32_to_qat(m_fp32, m_qat)   # copy kernel+bias
  calibrate_act_scales(m_qat, images)   # set per-layer act_scale
  # Now m_qat is ready for QAT fine-tuning. fake_quant active in forward; STE.
"""
from __future__ import annotations

import tensorflow as tf


def fake_quant_ste(x, scale, qmax: float = 127.0):
  """Symmetric int8 fake-quant with STE backward.
  forward = clip(round(x/scale), ±qmax) * scale; backward = identity through quant."""
  scale_safe = tf.maximum(scale, tf.constant(1e-9, dtype=x.dtype))
  q = tf.clip_by_value(tf.round(x / scale_safe), -qmax, qmax) * scale_safe
  return x + tf.stop_gradient(q - x)


def lsq_fake_quant(x, scale_tensor, qmax: float = 127.0):
  """LSQ activation fake-quant with PROPER gradients to both x and scale.

  Forward: clip(round(x/s), ±qmax) * s.
  Backward:
    - inside [-qmax*s, qmax*s]:  d/dx = dy,  d/ds = (round(x/s) - x/s) * dy
    - outside:                    d/dx = 0,   d/ds = sign(x) * qmax * dy

  scale_tensor must be a Tensor (not Variable) — use Variable.read_value()
  to convert at the call site to avoid @tf.custom_gradient Variable-detection.
  """
  @tf.custom_gradient
  def _lsq(x, s):
    qmax_t = tf.constant(qmax, dtype=x.dtype)
    s_safe = tf.maximum(s, tf.constant(1e-9, dtype=x.dtype))
    x_scaled = x / s_safe
    inside_mask = tf.cast(tf.abs(x_scaled) <= qmax_t, x.dtype)
    rounded = tf.round(x_scaled)
    y = tf.clip_by_value(rounded, -qmax_t, qmax_t) * s_safe

    def grad(dy):
      dx = dy * inside_mask
      # d/ds inside = round(x/s) - x/s
      ds_inside = (rounded - x_scaled) * inside_mask
      # d/ds outside = sign(x_scaled) * qmax
      ds_outside = tf.sign(x_scaled) * qmax_t * (1.0 - inside_mask)
      ds_per_elem = (ds_inside + ds_outside) * dy
      # LSQ paper rec: scale gradient by 1/sqrt(N*qmax) so scale updates are stable
      n = tf.cast(tf.size(x), x.dtype)
      g_scale = 1.0 / tf.sqrt(n * qmax_t)
      ds = tf.reduce_sum(ds_per_elem) * g_scale
      return dx, ds
    return y, grad
  return _lsq(x, scale_tensor)


class LSQConv2D(tf.keras.layers.Conv2D):
  """Conv2D with W8A8 LSQ: per-channel STE weights + per-tensor LSQ activation
  (learnable scale)."""

  def build(self, input_shape):
    super().build(input_shape)
    self.act_scale = self.add_weight(
        name='act_scale', shape=(),
        initializer=tf.constant_initializer(1.0),
        trainable=True,
    )
    self._quant_enabled = True

  def call(self, inputs):
    if not self._quant_enabled:
      return super().call(inputs)
    max_abs = tf.reduce_max(tf.abs(self.kernel), axis=[0, 1, 2], keepdims=True)
    w_scale = tf.maximum(max_abs / 127.0, tf.constant(1e-9, dtype=self.kernel.dtype))
    w_q = fake_quant_ste(self.kernel, w_scale)
    padding = self.padding.upper() if isinstance(self.padding, str) else 'SAME'
    y = tf.nn.conv2d(
        inputs, w_q,
        strides=[1, self.strides[0], self.strides[1], 1],
        padding=padding, data_format='NHWC',
    )
    if self.use_bias:
      y = tf.nn.bias_add(y, self.bias)
    if self.activation is not None:
      y = self.activation(y)
    # Convert Variable to Tensor to avoid custom_gradient Variable-detection.
    # In Keras 3, Variable.read_value() is removed; use tf.identity() to extract value as Tensor.
    return lsq_fake_quant(y, tf.identity(self.act_scale))


class QATConv2D(tf.keras.layers.Conv2D):
  """Conv2D with W8A8 fake-quant (per-channel weights, per-tensor non-trainable activation scale)."""

  def build(self, input_shape):
    super().build(input_shape)
    # Non-trainable activation scale — set via calibrate_act_scales(), then fixed.
    self.act_scale = self.add_weight(
        name='act_scale', shape=(),
        initializer=tf.constant_initializer(1.0),
        trainable=False,
    )
    # Python flag to bypass quant during calibration sweep.
    self._quant_enabled = True

  def call(self, inputs):
    if not self._quant_enabled:
      # Plain fp32 forward (used during calibration)
      return super().call(inputs)
    # Weight per-channel quant
    max_abs = tf.reduce_max(tf.abs(self.kernel), axis=[0, 1, 2], keepdims=True)
    w_scale = tf.maximum(max_abs / 127.0, tf.constant(1e-9, dtype=self.kernel.dtype))
    w_q = fake_quant_ste(self.kernel, w_scale)
    padding = self.padding.upper() if isinstance(self.padding, str) else 'SAME'
    y = tf.nn.conv2d(
        inputs, w_q,
        strides=[1, self.strides[0], self.strides[1], 1],
        padding=padding, data_format='NHWC',
    )
    if self.use_bias:
      y = tf.nn.bias_add(y, self.bias)
    if self.activation is not None:
      y = self.activation(y)
    return fake_quant_ste(y, self.act_scale)


def _walk_conv2d(model, depth=0):
  if depth > 6: return
  for sub in (model.layers if hasattr(model, 'layers') else []):
    if isinstance(sub, tf.keras.layers.Conv2D):
      yield sub
    else:
      yield from _walk_conv2d(sub, depth + 1)


def transfer_fp32_to_qat(fp32_model, qat_model) -> int:
  """Copy trainable variables from fp32 to QAT/LSQ.
  Skips the LSQ extra act_scale variables (which fp32 doesn't have).
  Walker order matches except for act_scale insertion in LSQ.
  Uses NAME-SUFFIX matching for non-conv variables; CONV2D weights by walker
  order (matched across same arch layout)."""
  # Copy Conv2D kernel + bias by walker order
  fp_convs = list(_walk_conv2d(fp32_model))
  qat_convs = list(_walk_conv2d(qat_model))
  if len(fp_convs) != len(qat_convs):
    raise ValueError(f"conv count mismatch: fp32={len(fp_convs)} qat={len(qat_convs)}")
  n = 0
  for cf, cq in zip(fp_convs, qat_convs):
    cq.kernel.assign(cf.kernel.numpy()); n += 1
    if cq.use_bias and cf.use_bias:
      cq.bias.assign(cf.bias.numpy()); n += 1
  # Copy non-conv vars (preprocessor_scaler, postprocessor_scaler, qstep, etc.)
  # by name suffix match. Skip act_scale (QAT/LSQ-specific).
  src_all = {v.name.split('/')[-1]: v
             for v in (list(fp32_model.trainable_variables) +
                       list(fp32_model.non_trainable_variables))}
  for dv in (list(qat_model.trainable_variables) +
             list(qat_model.non_trainable_variables)):
    suffix = dv.name.split('/')[-1]
    if 'act_scale' in suffix: continue  # LSQ-specific
    if suffix in src_all and src_all[suffix].shape == dv.shape:
      # Find if it's a conv kernel/bias (already done by walker)
      if suffix in ('kernel', 'bias'): continue
      dv.assign(src_all[suffix].numpy())
  return n


def _is_quant_conv(c):
  return isinstance(c, (QATConv2D, LSQConv2D))


def set_quant_enabled(qat_model, enabled: bool):
  """Toggle fake-quant on all QAT/LSQ conv layers."""
  for c in _walk_conv2d(qat_model):
    if _is_quant_conv(c):
      c._quant_enabled = enabled


def calibrate_act_scales(qat_model, calib_callable, n_calib: int,
                          percentile: float = 99.9):
  """Run calibration in fp32 mode; capture max|act| per QAT/LSQ conv output;
  set act_scale (both QAT and LSQ — LSQ benefits from good init, even though
  scale is trainable)."""
  import numpy as np
  set_quant_enabled(qat_model, False)
  observations = {}
  hooks = []
  for c in _walk_conv2d(qat_model):
    if not _is_quant_conv(c): continue
    orig_call = c.call
    def make_hook(layer, oc):
      def wrapped(x, *args, **kwargs):
        y = oc(x, *args, **kwargs)
        if hasattr(y, 'numpy'):
          observations.setdefault(id(layer), []).append(np.abs(y.numpy()).reshape(-1))
        return y
      return wrapped
    c.call = make_hook(c, orig_call)
    hooks.append((c, orig_call))

  for i in range(n_calib):
    calib_callable(i)

  for c, oc in hooks:
    c.call = oc
  for c in _walk_conv2d(qat_model):
    if not _is_quant_conv(c): continue
    vals = observations.get(id(c))
    if not vals: continue
    arr = np.concatenate(vals)
    p_val = float(np.percentile(arr, percentile))
    new_scale = max(p_val / 127.0, 1e-9)
    c.act_scale.assign(new_scale)

  set_quant_enabled(qat_model, True)
  return len(observations)
