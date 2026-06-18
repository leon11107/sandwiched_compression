# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Encode-Decode library of functions that emulate intra compression scenarios."""
import io
from typing import Callable, Dict, List, Optional, Tuple

from image_compression import jpeg_proxy
import logging
import numpy as np
from PIL import Image
import tensorflow as tf

def _encode_decode_with_jpeg(
    input_images: np.ndarray,
    qstep: np.float32,
    one_channel_at_a_time: bool = False,
    use_420: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
  """Compress-decompress with actual jpeg with fixed qstep.

  Args:
    input_images: Array of shape [b, n, m, c] where b is batch size, n x m is
      the image size, and c is the number of channels.
    qstep: float that determines the step-size of the scalar quantizer.
    one_channel_at_a_time: True if each channel should be encoded independently
      as a grayscale image.
    use_420: True when desired subsmapling is 4:2:0. False when 4:4:4.

  Returns:
    decoded: Array of same size as input_images containing the
      quantized-dequantized version of the input_images.
    rate: Array of size b that contains the total number of bits needed to
      encode the input_images into decoded.
  """

  assert input_images.ndim == 4
  decoded = np.zeros_like(input_images)
  rate = np.zeros(input_images.shape[0])
  # Jpeg needs byte qsteps
  jpeg_qstep = np.clip(np.rint(qstep).astype(int), 0, 255)
  qtable = [jpeg_qstep] * 64

  def run_jpeg(input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.fromarray(
        np.rint(np.clip(input_image, 0, 255)).astype(np.uint8))
    buf = io.BytesIO()
    img.save(
        buf,
        format='jpeg',
        optimize=True,
        qtables=[qtable, qtable, qtable],
        subsampling='4:2:0' if use_420 else '4:4:4',
    )
    decoded = np.array(Image.open(buf))
    rate = np.array(8 * len(buf.getbuffer()))
    return decoded, rate

  for index in range(input_images.shape[0]):
    if not one_channel_at_a_time:
      decoded[index], rate[index] = run_jpeg(input_images[index])
    else:
      # Run each channel separately through jpeg as a grayscale image
      # (Image.mode = 'L'.) Useful when RGB <-> YUV conversions need to be
      # skipped.
      for channel in range(input_images.shape[-1]):
        decoded[index, ...,
                channel], channel_rate = run_jpeg(input_images[index, ...,
                                                               channel])
        rate[index] += channel_rate

  return decoded.astype(np.float32), rate.astype(np.float32)


def convert_420_to_444(
    inputs: tf.Tensor,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.LANCZOS3,
) -> tf.Tensor:
  """Converts a YUV420 tensor to YUV444.

  Args:
    inputs: Tensor of size [b, n, m, c] or [b, f, n, m, c] where b is batch
      size, f is the number of frames in a video clip, [n, m] is the image/frame
      shape, and c is the number of channels. When input rank is 4 inputs is a
      batch of images. When input rank is 5 inputs is a batch of video clips.
    method: Desired chroma resizing method. Using bilinear will result in the
      center pixel.

  Returns:
    outputs: Tensor of the same size as inputs where UV channels have been
    upsampled.
  """
  outputs_chroma = tf.reshape(inputs, [-1, *inputs.shape[-3:]])[..., 1:]
  outputs_chroma = outputs_chroma[
      :, 0 : outputs_chroma.shape[1] // 2, 0 : outputs_chroma.shape[2] // 2, :
  ]
  new_size = [outputs_chroma.shape[1] * 2, outputs_chroma.shape[2] * 2]
  # Upsample chroma.
  outputs_chroma = tf.image.resize(
      outputs_chroma, new_size, method=method, antialias=True
  )
  num_chroma_channels = inputs.shape[-1] - 1
  outputs_chroma = tf.reshape(
      outputs_chroma, [*inputs.shape[:-1], num_chroma_channels]
  )
  return tf.concat([inputs[..., 0:1], outputs_chroma], axis=-1)


def convert_444_to_420(
    inputs: tf.Tensor,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
) -> tf.Tensor:
  """Converts a 444 tensor to 420 by downsampling the chroma channels.

  Args:
    inputs: Tensor of size [b, n, m, c] or [b, f, n, m, c] where b is batch
      size, f is the number of frames in a video clip, [n, m] is the image/frame
      shape, and c is the number of channels. When input rank is 4 inputs is a
      batch of images. When input rank is 5 inputs is a batch of video clips.
    method: Desired chroma resizing method. Using bilinear will result in the
      center pixel.

  Returns:
    outputs: Tensor of the same size as inputs where chroma channels have been
    downsampled.
  """
  outputs_chroma = tf.reshape(inputs, [-1, *inputs.shape[-3:]])[..., 1:]
  new_size = [outputs_chroma.shape[1] // 2, outputs_chroma.shape[2] // 2]
  # Downsample chroma.
  outputs_chroma = tf.image.resize(
      outputs_chroma, new_size, method=method, antialias=True
  )
  num_chroma_channels = inputs.shape[-1] - 1
  outputs_chroma = tf.reshape(
      tf.pad(  # Zero-pad so that a single tensor is returned.
          outputs_chroma,
          [[0, 0], [0, new_size[0]], [0, new_size[1]], [0, 0]],
      ),
      [*inputs.shape[:-1], num_chroma_channels],
  )
  return tf.concat([inputs[..., 0:1], outputs_chroma], axis=-1)


class EncodeDecodeIntra(tf.keras.Model):
  """A class with methods for basic intra compression emulation."""

  def __init__(
      self,
      rounding_fn: Callable[[tf.Tensor], tf.Tensor] = tf.round,
      use_jpeg_rate_model: bool = True,
      qstep_init: float = 1.0,
      train_qstep: bool = True,
      min_qstep: float = 0.0,
      jpeg_clip_to_image_max: bool = True,
      convert_to_yuv: bool = False,
      downsample_chroma: bool = False,
      rate_proxy_mode: str = "log_nonzero",
      rate_proxy_grad_scale: float = 1.0,
      codec_forward_mode: str = "proxy",
      post_jpeg_int_round: bool = False,
      output_clip_mode: str = "hard",
  ):
    """Constructor.

    Args:
      rounding_fn: Callable that is used to round transform coefficients for
        JPEG during quantization.
      use_jpeg_rate_model: True for JPEG-specific rate model, False for
        Gaussian-distribution-based rate model.
      qstep_init: float that determines initial value for the step-size of the
        scalar quantizer.
      train_qstep: Whether qstep should be trained. When False the class will
        use qstep_init or any qsteps provided in the call(). The latter is
        useful when the same module is used in video with different qsteps for
        INTRA and INTER.
      min_qstep: Minimum value which qstep should be greater than. Set to 1 to
        reflect for some practical codecs that cannot go below integer values.
      jpeg_clip_to_image_max: True if jpeg proxy should clip the final output to
        [0, image_max]. Set to False when handling INTER frames.
      convert_to_yuv: True if color conversion should be applied during
        compression.
      downsample_chroma: Whether chroma planes should be downsampled during
        compression.
    """
    super().__init__(name='EncodeDecodeIntra')

    self.train_qstep = train_qstep
    if self.train_qstep:
      self.qstep = self.add_weight(
          shape=(),
          initializer=tf.constant_initializer(qstep_init),
          trainable=True,
          name='qstep',
          dtype=tf.float32)
    else:
      self.qstep = tf.cast(qstep_init, tf.float32)

    self.min_qstep = self.add_weight(
        shape=(),
        initializer=tf.constant_initializer(min_qstep),
        trainable=False,
        name='min_qstep',
        dtype=tf.float32)

    # output_clip_mode controls how the final pixel-domain clip to [0, image_max]
    # is implemented:
    #   "hard"      : tf.clip_by_value (zero gradient outside) — historical default.
    #                 Implemented inside JpegProxy via clip_to_image_max=True.
    #   "soft"      : differentiable leaky clip (γ=1e-3) from Reich et al. WACV 2024.
    #   "ste_leaky" : STE clip — forward exact clip, backward 1 inside, γ outside.
    #   "none"      : no output clip at all.
    # For non-"hard" modes the JpegProxy's internal clip is disabled and the
    # configured clip is applied here in _encode_decode_jpeg.
    if output_clip_mode not in ("hard", "soft", "ste_leaky", "none"):
      raise ValueError(f"unknown output_clip_mode {output_clip_mode!r}")
    self._output_clip_mode = output_clip_mode
    if output_clip_mode == "hard":
      self.clip_to_image_max = jpeg_clip_to_image_max
    else:
      self.clip_to_image_max = False

    def _quantizer_fn(x: tf.Tensor) -> tf.Tensor:
      """Implements quantize-dequantize with the trainable qstep."""
      positive_qstep = self._positive_qstep()
      return rounding_fn(x / positive_qstep) * positive_qstep

    self._jpeg_quantizer_fn = _quantizer_fn
    self._rounding_fn = rounding_fn
    if rate_proxy_mode not in (
        "log_nonzero",
        "multifeature",
        "jpeg_symbol_nonneg",
        "jpeg_symbol_banded_nonneg",
    ):
      raise ValueError(f"unknown rate_proxy_mode {rate_proxy_mode!r}")
    self._rate_proxy_mode = rate_proxy_mode
    self._rate_proxy_grad_scale = float(rate_proxy_grad_scale)
    if codec_forward_mode not in ("proxy", "real_ste"):
      raise ValueError(f"unknown codec_forward_mode {codec_forward_mode!r}")
    self._codec_forward_mode = codec_forward_mode
    # Match inference: real JPEG decoder outputs uint8 pixels.
    # When True, round dequantized pixels to integers with STE so postprocessor
    # sees inference-faithful integer-valued floats during training.
    self._post_jpeg_int_round = post_jpeg_int_round

    def add_variable_conditionally(
        variable_name: str, condition: Optional[bool] = None
    ) -> tf.Tensor:
      if condition is not None:
        return self.add_weight(
            initializer=tf.constant_initializer(condition),
            trainable=False,
            name=variable_name,
            dtype=tf.bool,
        )
      else:
        return self.add_weight(
            trainable=False, name=variable_name, dtype=tf.bool
        )

    self.use_jpeg_rate_model = add_variable_conditionally(
        'use_jpeg_rate_model', use_jpeg_rate_model
    )

    self._init_jpeg_layer(convert_to_yuv, downsample_chroma)

    # Actual jpeg is run on processed data for rate estimates. All color
    # conversion and chroma downsampling is done during differentiable
    # processing. We hence need actual jpeg to encode single channel data
    # without any downsampling unless convert_to_yuv is True. convert_to_yuv is
    # only useful in cases where the sandwiched codec is hard-coded to convert
    # to YUV.
    self.run_jpeg_one_channel_at_a_time = add_variable_conditionally(
        'run_jpeg_one_channel_at_a_time', False if convert_to_yuv else True
    )
    self.run_jpeg_with_downsampled_chroma = add_variable_conditionally(
        'run_jpeg_with_downsampled_chroma', downsample_chroma
    )

    logging.info(
        'EncodeDecodeIntra configured with %s',
        'jpeg-rate' if use_jpeg_rate_model else 'gaussian-rate',
    )

    logging.info(
        'EncodeDecodeIntra running %s',
        '420' if downsample_chroma else '444',
    )

    logging.info(
        'EncodeDecodeIntra yuv conversion is %s',
        'on' if convert_to_yuv else 'off',
    )

    # Workaround thread-unsafe PIL library by calling init in main thread.
    Image.init()

  def _positive_qstep(self):
    return tf.keras.activations.elu(self.qstep, alpha=0.01) + self.min_qstep

  def get_qstep(self) -> tf.Tensor:
    return self._positive_qstep()

  def _init_jpeg_layer(self, convert_to_yuv: bool, downsample_chroma: bool):
    # Configure the JPEG layer to use the defined quantize-dequantize function,
    # _quantizer_fn, so that trained value of qstep gets used: Use a fixed
    # quantizer step-size of 1 for all DCT coefficients and update quantizer
    # stepsize through _quantizer_fn.
    quantization_table = np.full((8, 8), 1.0, dtype=np.float32)
    self._jpeg_layer = jpeg_proxy.JpegProxy(
        downsample_chroma=downsample_chroma,
        luma_quantization_table=quantization_table,
        chroma_quantization_table=quantization_table,
        convert_to_yuv=convert_to_yuv,
        clip_to_image_max=self.clip_to_image_max,
    )

  def _rate_proxy_gaussian(self, inputs: tf.Tensor,
                           axis: List[int]) -> tf.Tensor:
    """Calculates entropy assuming a Gaussian distribution and high-res quantization.

    Args:
      inputs: Tensor of shape [b, n1, ...].
      axis: Axis of random variable realizations, e.g., with inputs b x n1 x n2
        and axis=[1] then there are n2 Gaussian variables with potentially
        different distributions, each with samples along axis=[1].

    Returns:
      rate: Tensor of shape [b] that estimates the total number of bits needed
        to represent the values quantized with self.qstep.
    """
    assert inputs.shape.rank >= np.max(np.abs(axis))
    deviations = tf.math.reduce_std(inputs, axis=axis)
    assert deviations.shape[0] == inputs.shape[0]

    hires_entropy = tf.nn.relu(
        tf.math.log(deviations / self._positive_qstep() + np.finfo(float).eps) +
        .5 * np.log(2 * np.pi * np.exp(1)))

    # Sum the entropies for total rate
    return tf.reduce_sum(
        tf.reshape(hires_entropy, [tf.shape(inputs)[0], -1]),
        axis=1) * tf.reduce_prod(
            tf.gather(tf.cast(tf.shape(inputs), dtype=tf.float32),
                      axis)) / np.log(2)

  def _rate_proxy_jpeg(
      self, three_channel_inputs: tf.Tensor,
      dequantized_dct_coeffs: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Calculates a rate proxy based on a Jpeg-specific rate model."""

    def calculate_non_zeros(dct_coeffs: Dict[str, tf.Tensor],
                            qstep: tf.float32) -> tf.Tensor:
      num_nonzeros = tf.zeros(tf.shape(three_channel_inputs)[0])
      for k in dct_coeffs:
        num_nonzeros += tf.math.reduce_sum(
            tf.reshape(
                tf.math.log(1 + tf.math.abs(dct_coeffs[k] / qstep)
                           ),  # Divide to get the quantized values
                [tf.shape(three_channel_inputs)[0], -1]),
            axis=1)
      return tf.cast(num_nonzeros, dtype=tf.float32)

    def encode_decode_inputs_with_jpeg() -> Tuple[tf.Tensor, tf.Tensor]:
      """Encodes then decodes the three_channel_inputs using actual jpeg."""
      use_420 = tf.cond(
          self.run_jpeg_one_channel_at_a_time,
          # 420 is a meaningless option for the jpeg binary here.
          lambda: tf.convert_to_tensor(False, dtype=tf.bool),
          lambda: self.run_jpeg_with_downsampled_chroma
      )
      jpeg_decoded, jpeg_rate = tf.numpy_function(
          _encode_decode_with_jpeg,
          inp=[
              three_channel_inputs,
              self._positive_qstep(),
              self.run_jpeg_one_channel_at_a_time,
              use_420,
          ],
          Tout=[tf.float32, tf.float32],
      )
      jpeg_decoded.set_shape(three_channel_inputs.shape)
      jpeg_rate.set_shape(three_channel_inputs.shape[0])
      return jpeg_decoded, jpeg_rate

    def per_sample_sum(value_fn):
      """Sum value_fn(coef) over all DCT coefficients per sample."""
      total = tf.zeros(tf.shape(three_channel_inputs)[0])
      for k in dequantized_dct_coeffs:
        total += tf.math.reduce_sum(
            tf.reshape(
                value_fn(dequantized_dct_coeffs[k]),
                [tf.shape(three_channel_inputs)[0], -1],
            ),
            axis=1,
        )
      return tf.cast(total, dtype=tf.float32)

    def jpeg_symbol_cost() -> tf.Tensor:
      """Non-negative differentiable proxy for JPEG entropy-symbol cost.

      Forward rate is still corrected to the exact PIL JPEG bit count below.
      This cost only controls the backward direction. It approximates JPEG
      entropy coding with soft nonzero decisions, soft magnitude categories,
      and a mild frequency-dependent nonzero cost for AC run-length coding.
      """
      thresholds = tf.constant(
          [0.5, 1.5, 3.5, 7.5, 15.5, 31.5, 63.5, 127.5],
          dtype=tf.float32,
      )
      thresholds = tf.reshape(thresholds, [1, 1, 1, 1, -1])
      freq = np.asarray(
          [(idx // 8 + idx % 8) / 14.0 for idx in range(64)],
          dtype=np.float32,
      )
      freq = tf.reshape(tf.constant(freq, dtype=tf.float32), [1, 1, 1, 64])
      # Keep this finite: very large slopes turn into a hard threshold and lose
      # the useful gradient exactly where real JPEG changes bitstream symbols.
      slope = tf.constant(4.0, dtype=tf.float32)

      def symbol_value(coef: tf.Tensor) -> tf.Tensor:
        q_abs = tf.math.abs(coef / qstep_pos)
        soft_nonzero = tf.sigmoid(slope * (q_abs - 0.5))
        # JPEG magnitude category ~= floor(log2(abs(q_index))) + 1 for nonzero
        # coefficients. Summed soft thresholds approximate this integer count.
        soft_category_bits = tf.reduce_sum(
            tf.sigmoid(slope * (tf.expand_dims(q_abs, axis=-1) - thresholds)),
            axis=-1,
        )
        # High-frequency nonzeros tend to be more expensive because they break
        # longer AC zero runs. The factor is deliberately mild and non-negative.
        run_cost = soft_nonzero * (1.0 + 0.75 * freq)
        magnitude_cost = soft_category_bits * (1.0 + 0.25 * freq)
        return run_cost + magnitude_cost

      return per_sample_sum(symbol_value) + 1.0

    def jpeg_symbol_banded_features() -> tf.Tensor:
      """Block/frequency-banded JPEG symbol features.

      Returns a non-negative [batch, features] matrix whose columns approximate
      separate JPEG entropy costs: DC category, AC nonzero and magnitude costs
      in low/mid/high frequency bands, plus a soft AC run-break cost.
      """
      thresholds = tf.constant(
          [0.5, 1.5, 3.5, 7.5, 15.5, 31.5, 63.5, 127.5],
          dtype=tf.float32,
      )
      thresholds = tf.reshape(thresholds, [1, 1, 1, 1, -1])
      zigzag = tf.constant(
          [
              0, 1, 8, 16, 9, 2, 3, 10,
              17, 24, 32, 25, 18, 11, 4, 5,
              12, 19, 26, 33, 40, 48, 41, 34,
              27, 20, 13, 6, 7, 14, 21, 28,
              35, 42, 49, 56, 57, 50, 43, 36,
              29, 22, 15, 23, 30, 37, 44, 51,
              58, 59, 52, 45, 38, 31, 39, 46,
              53, 60, 61, 54, 47, 55, 62, 63,
          ],
          dtype=tf.int32,
      )
      freq_sum = np.asarray(
          [idx // 8 + idx % 8 for idx in range(64)], dtype=np.float32
      )
      freq_sum_z = tf.gather(tf.constant(freq_sum, dtype=tf.float32), zigzag)
      ac_freq = freq_sum_z[1:]
      low_mask = tf.reshape(
          tf.cast(tf.logical_and(ac_freq <= 2.0, ac_freq > 0.0), tf.float32),
          [1, 1, 1, 63],
      )
      mid_mask = tf.reshape(
          tf.cast(tf.logical_and(ac_freq > 2.0, ac_freq <= 5.0), tf.float32),
          [1, 1, 1, 63],
      )
      high_mask = tf.reshape(tf.cast(ac_freq > 5.0, tf.float32), [1, 1, 1, 63])
      slope = tf.constant(4.0, dtype=tf.float32)

      total = None
      for k in dequantized_dct_coeffs:
        q_abs = tf.math.abs(dequantized_dct_coeffs[k] / qstep_pos)
        q_abs_z = tf.gather(q_abs, zigzag, axis=-1)
        soft_nonzero = tf.sigmoid(slope * (q_abs_z - 0.5))
        soft_category_bits = tf.reduce_sum(
            tf.sigmoid(slope * (tf.expand_dims(q_abs_z, axis=-1) - thresholds)),
            axis=-1,
        )
        dc_category = soft_category_bits[..., 0]
        ac_nonzero = soft_nonzero[..., 1:]
        ac_category = soft_category_bits[..., 1:]
        preceding_soft_zero = tf.cumsum(
            1.0 - ac_nonzero, axis=-1, exclusive=True
        ) / 63.0
        run_break = ac_nonzero * preceding_soft_zero

        def sample_sum(value: tf.Tensor) -> tf.Tensor:
          return tf.reduce_sum(tf.reshape(value, [tf.shape(value)[0], -1]), axis=1)

        channel_features = tf.stack(
            [
                sample_sum(dc_category),
                sample_sum(ac_nonzero * low_mask),
                sample_sum(ac_nonzero * mid_mask),
                sample_sum(ac_nonzero * high_mask),
                sample_sum(ac_category * low_mask),
                sample_sum(ac_category * mid_mask),
                sample_sum(ac_category * high_mask),
                sample_sum(run_break),
            ],
            axis=1,
        )
        total = channel_features if total is None else total + channel_features
      return tf.cast(total + 1e-6, dtype=tf.float32)

    def forward_corrected_rate(predicted: tf.Tensor) -> tf.Tensor:
      """Forward exact real JPEG bits, backward scaled surrogate gradient."""
      scaled_predicted = self._rate_proxy_grad_scale * predicted
      correction = tf.stop_gradient(jpeg_rate - scaled_predicted)
      return scaled_predicted + correction

    def nnls_forward_corrected_rate(features: tf.Tensor) -> tf.Tensor:
      """Non-negative batch fit with exact-real-rate forward correction."""
      target = tf.reshape(jpeg_rate, [-1, 1])
      feature_scales = tf.reduce_mean(features, axis=0, keepdims=True) + 1.0
      f = features / feature_scales
      num_features = tf.shape(f)[1]
      target_mean = tf.reduce_mean(target) + 1.0
      w = tf.ones([num_features, 1], dtype=tf.float32) * (
          target_mean / tf.cast(num_features, tf.float32)
      )
      ft = tf.transpose(f)
      # Multiplicative-update NNLS: keeps weights non-negative and avoids the
      # negative-gradient failures we observed with unconstrained least squares.
      for _ in range(12):
        numerator = tf.matmul(ft, target) + 1e-3
        denominator = tf.matmul(tf.matmul(ft, f), w) + 1e-3
        w = w * numerator / denominator
      w = tf.stop_gradient(w)
      predicted = tf.reshape(tf.matmul(f, w), [-1])
      return forward_corrected_rate(predicted)

    ###########################################################################
    # Jpeg-specific model fits rate using number of nonzero dct coefficients.
    # For details see:
    #  Z. He and S. K. Mitra, "A unified rate-distortion analysis framework for
    #  transform coding," in IEEE Transactions on Circuits and Systems for Video
    #  Technology, vol. 11, no. 12, pp. 1221-1236, Dec. 2001.
    #
    # Generate (rate, num_nonzero) pairs, fit a weight as
    # rate ~= weight * num_nonzero, return rate approximation as weight *
    # num_nonzero.
    ###########################################################################
    _, jpeg_rate = encode_decode_inputs_with_jpeg()
    qstep_pos = self._positive_qstep()

    if self._rate_proxy_mode == "log_nonzero":
      # Single-feature per-sample linear fit (original behaviour).
      num_nonzero_dct_coeffs = calculate_non_zeros(
          dequantized_dct_coeffs, qstep_pos
      )
      nonzero_times_rate = tf.math.multiply(num_nonzero_dct_coeffs, jpeg_rate)
      nonzero_times_nonzero = tf.math.multiply(
          num_nonzero_dct_coeffs, num_nonzero_dct_coeffs
      )
      line_weights = tf.stop_gradient(
          tf.math.divide(nonzero_times_rate, nonzero_times_nonzero + 1)
      )
      return tf.math.multiply(num_nonzero_dct_coeffs, line_weights)

    if self._rate_proxy_mode == "jpeg_symbol_nonneg":
      # Per-sample non-negative scalar calibration. Forward is exact real JPEG
      # rate via stop-gradient correction; backward follows jpeg_symbol_cost().
      cost = jpeg_symbol_cost()
      scale = tf.stop_gradient(tf.math.divide(jpeg_rate, cost + 1.0))
      predicted = cost * scale
      return forward_corrected_rate(predicted)

    if self._rate_proxy_mode == "jpeg_symbol_banded_nonneg":
      return nnls_forward_corrected_rate(jpeg_symbol_banded_features())

    # rate_proxy_mode == "multifeature":
    # Three sample-level features capturing different magnitude scales:
    #   f1 = Σ log(1 + |c|/q)         (original; ~ count + log-magnitude)
    #   f2 = Σ |c|/q                  (linear magnitude — Huffman value bits)
    #   f3 = Σ sqrt(|c|/q)            (intermediate scale; sub-linear)
    # Per-BATCH least-squares fit  jpeg_rate ≈ F @ w  with w stopped-gradient,
    # then forward-correct so per-sample forward = real jpeg_rate exactly while
    # gradient is the differentiable surrogate F @ w.
    f1 = per_sample_sum(lambda c: tf.math.log(1.0 + tf.math.abs(c / qstep_pos)))
    f2 = per_sample_sum(lambda c: tf.math.abs(c / qstep_pos))
    f3 = per_sample_sum(lambda c: tf.math.sqrt(tf.math.abs(c / qstep_pos) + 1e-8))
    features = tf.stack([f1, f2, f3], axis=1)  # [B, 3]
    target = tf.reshape(jpeg_rate, [-1, 1])    # [B, 1]
    # Normalize columns to avoid scale-driven ill-conditioning, then solve
    # with Tikhonov regularization. `fast=False` (orthogonal decomposition)
    # is numerically more robust than the default Cholesky-based path when
    # the system happens to be near-singular for very small batches.
    feature_scales = tf.reduce_mean(features, axis=0, keepdims=True) + 1.0
    features_normalized = features / feature_scales
    weights_normalized = tf.linalg.lstsq(
        features_normalized, target, l2_regularizer=1e-3, fast=False
    )  # [3, 1]
    weights_normalized = tf.stop_gradient(weights_normalized)
    predicted = tf.reshape(features_normalized @ weights_normalized, [-1])  # [B]
    # Forward correction so per-sample forward equals real jpeg_rate while
    # gradient comes from `predicted` (which depends on the three features).
    return forward_corrected_rate(predicted)

  def is_codec_proxy_420(self) -> tf.Tensor:
    return self.run_jpeg_with_downsampled_chroma

  def _encode_decode_jpeg(
      self, inputs: tf.Tensor, image_max: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encodes then decodes the input using JPEG.

    Args:
      inputs: Tensor of shape [b, n, m, c] where b is batch size, n x m is the
        image size, and c is the number of channels (c <= 3).
      image_max: Maximum possible value of the image.

    Returns:
      outputs: Tensor of same shape as inputs containing the
        quantized-dequantized version of the inputs.
      rate: Tensor of shape [b] that estimates the total number of bits needed
        to encode the input into output.
    """
    if inputs.shape.rank != 4:
      raise ValueError('inputs must have rank 4.')
    if inputs.shape[-1] > 3:
      raise ValueError('jpeg layer can handle up to 3 channels.')

    # Zero-pad to three channels as needed for the jpeg layer.
    pad_dim = 3 - inputs.shape[-1]
    if pad_dim:
      paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, pad_dim]],
                             dtype=tf.int32)
      three_channel_inputs = tf.pad(
          inputs, paddings, mode='CONSTANT', constant_values=0)
    else:
      three_channel_inputs = inputs

    # Emulate integer inputs needed when using an actual intra codec.
    three_channel_inputs = self._rounding_fn(three_channel_inputs)

    # JPEG quantize-dequantize.
    dequantized_three_channels, dequantized_dct_coeffs = self._jpeg_layer(
        three_channel_inputs, self._jpeg_quantizer_fn, image_max=image_max
    )

    # Match inference behaviour: real JPEG decoder returns uint8 pixels.
    # Without this, training postprocessor sees continuous floats while
    # inference postprocessor receives integer floats — a train/inference
    # mismatch. Use STE round so backward gradient still flows.
    if self._post_jpeg_int_round:
      rounded = tf.round(dequantized_three_channels)
      dequantized_three_channels = (
          dequantized_three_channels
          + tf.stop_gradient(rounded - dequantized_three_channels)
      )

    # Differentiable output clip to [0, image_max] when not using the
    # JpegProxy's internal hard clip. soft = leaky linear (γ=1e-3); ste_leaky
    # = forward exact clip / backward 1 inside, γ outside (Reich et al.).
    if self._output_clip_mode in ("soft", "ste_leaky"):
      from image_compression import diff_jpeg_tf  # avoid import at module load
      hi = tf.cast(image_max, dequantized_three_channels.dtype)
      if self._output_clip_mode == "soft":
        dequantized_three_channels = diff_jpeg_tf.soft_clip(
            dequantized_three_channels, lo=0.0, hi=hi, leak=1e-3
        )
      else:  # ste_leaky
        dequantized_three_channels = diff_jpeg_tf.ste_clip(
            dequantized_three_channels, lo=0.0, hi=hi, leak=1e-3
        )

    # Codec-level STE: forward replaces the proxy-decoded output with the
    # actual PIL JPEG-decoded output, while backward still flows through the
    # differentiable proxy path. This makes the wrapper see inference-faithful
    # quantization noise during training while keeping a usable gradient.
    if self._codec_forward_mode == "real_ste":
      use_420 = tf.cond(
          self.run_jpeg_one_channel_at_a_time,
          lambda: tf.convert_to_tensor(False, dtype=tf.bool),
          lambda: self.run_jpeg_with_downsampled_chroma,
      )
      real_decoded, _real_bits = tf.numpy_function(
          _encode_decode_with_jpeg,
          inp=[
              three_channel_inputs,
              self._positive_qstep(),
              self.run_jpeg_one_channel_at_a_time,
              use_420,
          ],
          Tout=[tf.float32, tf.float32],
      )
      real_decoded.set_shape(three_channel_inputs.shape)
      dequantized_three_channels = (
          dequantized_three_channels
          + tf.stop_gradient(real_decoded - dequantized_three_channels)
      )

    # Remove padding.
    if pad_dim:
      dequantized = tf.slice(dequantized_three_channels, [0, 0, 0, 0],
                             tf.shape(inputs))
    else:
      dequantized = dequantized_three_channels

    def gaussian_rate():
      gauss_rate = tf.zeros(tf.shape(inputs)[0])
      for k in dequantized_dct_coeffs:
        gauss_rate += self._rate_proxy_gaussian(
            dequantized_dct_coeffs[k], axis=[1])
      return gauss_rate

    def jpeg_rate():
      # When running the rate proxy one channel at a time with downsampled
      # chroma (420 without color conversion case), have to explicitly
      # downsample the chroma since the jpeg binary cannot handle this case.
      conversion_to_420_needed = tf.math.logical_and(
          self.run_jpeg_one_channel_at_a_time,
          self.run_jpeg_with_downsampled_chroma
      )
      def _with_420():
        ri = jpeg_proxy.pad_spatially_to_multiple_of_bsize(
            three_channel_inputs, bsize=2, mode='SYMMETRIC'
        )
        return convert_444_to_420(ri)

      rate_inputs = tf.cond(
          conversion_to_420_needed,
          _with_420,
          lambda: three_channel_inputs,
      )

      # Scale to 8-bit range so that the rate proxy can be used with any image
      # max.
      scale = tf.math.divide(255.0, image_max)
      rate_inputs = tf.math.multiply(scale, rate_inputs)
      return self._rate_proxy_jpeg(rate_inputs, dequantized_dct_coeffs)

    rate = tf.cond(self.use_jpeg_rate_model, jpeg_rate, gaussian_rate)

    return dequantized, rate

  def __call__(
      self,
      inputs: tf.Tensor,
      input_qstep: Optional[tf.Tensor] = None,
      image_max: Optional[tf.Tensor] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encodes then decodes the input.

    Args:
      inputs: Tensor of shape [b, n, m, c] where b is batch size, n x m is the
        image size, and c is the number of channels.
      input_qstep: qstep to use when self.qstep is not trained.
      image_max: Maximum possible value of the image.

    Returns:
      outputs: Tensor of same size as inputs containing the
        quantized-dequantized version of the inputs.
      rate: Tensor of size b that estimates the total number of bits needed to
        encode the input into output.
    """
    if inputs.shape.rank != 4:
      raise ValueError('inputs must have rank 4.')

    if not self.train_qstep and input_qstep is not None:
      self.qstep = input_qstep

    if image_max is None:
      image_max = tf.constant(255.0, dtype=tf.float32)

    def run_jpeg():
      if inputs.shape[-1] <= 3:
        return self._encode_decode_jpeg(inputs, image_max)

      # JPEG layer handles at most three channels. Run three channels at a time.
      # (i) Run first three channels to initialize the return tensors.
      size = np.array(inputs.shape, dtype=np.int32)
      limit = size[-1]
      size[-1] = 3
      begin = np.zeros_like(size, dtype=np.int32)
      dequantized, rate = self._encode_decode_jpeg(
          tf.slice(inputs, begin, size), image_max
      )

      # (ii) Run three channels at a time and update.
      for _ in range(3, limit, 3):
        begin[-1] += 3
        size[-1] = np.minimum(limit - begin[-1], 3)
        dequantized_loop, rate_loop = self._encode_decode_jpeg(
            tf.slice(inputs, begin, size), image_max
        )

        # Update the return variables
        dequantized = tf.concat([dequantized, dequantized_loop], axis=3)
        rate += rate_loop
      return dequantized, rate

    return run_jpeg()
