"""TF port of differentiable rounding / clipping primitives from
"Differentiable JPEG: The Devil is in the Details" (Reich et al., WACV 2024).

Reference impl: reference/Diff-JPEG/diff_jpeg/{rounding.py, clipping.py}.

Four primitives that the paper proposes for differentiable JPEG:

  * polynomial_round       : forward and backward both smooth (cubic surrogate)
                             forward NOT equal to true round.
  * ste_polynomial_round   : forward = true tf.round, backward = polynomial
                             gradient 3*(x - round(x))^2 (the paper's preferred
                             STE — outperforms constant-gradient STE).
  * soft_clip              : forward leaky-linear outside [lo, hi] with slope
                             `leak`; pure functional (no custom gradient).
  * ste_clip               : forward = exact tf.clip_by_value, backward = 1
                             inside [lo, hi] and `leak` outside.

Intended use: drop-in replacements for tf.round / tf.clip_by_value in our
existing codec proxy so we can ablate which paper "fix" actually helps the
sandwich pipeline (round, clip, both, neither).
"""

from __future__ import annotations

import tensorflow as tf


def polynomial_round(x: tf.Tensor) -> tf.Tensor:
  """Smooth differentiable rounding: round(x) + (x - round(x))^3.

  Forward is NOT exactly round (it lies in [round(x) - 0.125, round(x) + 0.125]
  when x lies in [round(x) - 0.5, round(x) + 0.5]). The backward gradient is
  the autograd of the cubic term: 3 * (x - round(x))^2 in [0, 0.75].
  Matches reference impl `differentiable_polynomial_rounding`.
  """
  r = tf.round(x)
  return r + tf.pow(x - r, 3)


def polynomial_floor(x: tf.Tensor) -> tf.Tensor:
  """Smooth differentiable floor via the identity floor(x) = round(x - 0.5)."""
  return polynomial_round(x - 0.5)


@tf.custom_gradient
def ste_polynomial_round(x: tf.Tensor):
  """STE rounding: forward = exact tf.round, backward = polynomial surrogate.

  This is the paper's "ours STE" rounding — it uses the *derivative of the
  polynomial surrogate* as the backward, NOT constant 1 (which is what plain
  STE does). Table 7 in the paper shows this matters (IFGSM top-1 7.1% vs
  25.3% for constant-gradient STE).
  """
  y = tf.round(x)
  def grad(dy):
    return 3.0 * tf.pow(x - y, 2) * dy
  return y, grad


@tf.custom_gradient
def ste_polynomial_floor(x: tf.Tensor):
  """STE floor: forward = exact tf.floor, backward = polynomial surrogate at x-0.5."""
  y = tf.floor(x)
  def grad(dy):
    shifted = x - 0.5
    return 3.0 * tf.pow(shifted - tf.round(shifted), 2) * dy
  return y, grad


def soft_clip(x: tf.Tensor, lo: float = 0.0, hi: float = 255.0,
              leak: float = 1e-3) -> tf.Tensor:
  """Leaky differentiable clip. Linear inside [lo, hi]; slope=`leak` outside.

  Forward differs slightly from hard clip outside the bounds (by `leak*(x-bound)`).
  Backward gradient is `leak` outside, 1 inside.
  Matches reference impl `differentiable_clipping`.
  """
  below = lo + leak * (x - lo)
  above = hi + leak * (x - hi)
  return tf.where(x < lo, below, tf.where(x > hi, above, x))


def ste_clip(x: tf.Tensor, lo: float = 0.0, hi: float = 255.0,
             leak: float = 1e-3) -> tf.Tensor:
  """STE clip: forward = exact tf.clip_by_value, backward = 1 inside, leak outside.

  Matches `_DifferentiableClippingSTE` from rounding.py reference (uses STE
  trick: forward returns the clip, backward returns a smooth weight).
  """
  lo_f = tf.cast(lo, x.dtype)
  hi_f = tf.cast(hi, x.dtype)
  leak_f = tf.cast(leak, x.dtype)

  @tf.custom_gradient
  def _inner(z):
    y = tf.clip_by_value(z, lo_f, hi_f)
    def grad(dy):
      inside = tf.cast((z >= lo_f) & (z <= hi_f), z.dtype)
      scale = inside + (1.0 - inside) * leak_f
      return scale * dy
    return y, grad
  return _inner(x)
