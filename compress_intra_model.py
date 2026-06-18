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
"""A model that uses pre and postprocessing around existing intra codecs to better compress data."""

# Sandwiched Compression: Repurposing Standard Codecs with Neural Network Wrappers
# https://arxiv.org/abs/2402.05887
import dataclasses
import enum
import functools
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import logging
import tensorflow as tf

from image_compression import encode_decode_intra_lib
from distortion import distortion_fns
from pre_post_models import unet as simple_unet
from utilities import serialization

SAMPLING_MODES = (
    'resize',
    'cnn_stride_transpose',
    'cnn_stride_resize_conv',
)


def downsample_tensor(
    inputs: tf.Tensor,
    factor: int,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
) -> tf.Tensor:
  """Downsamples by the input factor."""
  new_size = tf.math.floordiv(inputs.shape[1:3], factor)
  down_sampled = tf.image.resize(
      inputs, size=new_size, method=method, antialias=True
  )
  return down_sampled


def upsample_tensor(
    inputs: tf.Tensor,
    factor: int,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
) -> tf.Tensor:
  """Upsamples by the input factor."""
  new_size = tf.math.multiply(inputs.shape[1:3], factor)
  upsampled = tf.image.resize(
      inputs, size=new_size, method=method, antialias=True
  )
  return upsampled


class ChannelwiseBoxInitializer(tf.keras.initializers.Initializer):
  """Initializes a conv kernel as per-channel box filtering."""

  def __call__(self, shape, dtype=None):
    dtype = dtype or tf.float32
    kernel_h, kernel_w, input_channels, output_channels = shape
    values = [[[[0.0 for _ in range(output_channels)]
                for _ in range(input_channels)]
               for _ in range(kernel_w)]
              for _ in range(kernel_h)]
    scale = 1.0 / float(kernel_h * kernel_w)
    for output_channel in range(output_channels):
      input_channel = min(output_channel, input_channels - 1)
      for row in range(kernel_h):
        for col in range(kernel_w):
          values[row][col][input_channel][output_channel] = scale
    return tf.constant(values, dtype=dtype)


class ChannelwiseIdentityInitializer(tf.keras.initializers.Initializer):
  """Initializes a same-channel conv kernel as identity."""

  def __call__(self, shape, dtype=None):
    dtype = dtype or tf.float32
    kernel_h, kernel_w, input_channels, output_channels = shape
    values = [[[[0.0 for _ in range(output_channels)]
                for _ in range(input_channels)]
               for _ in range(kernel_w)]
              for _ in range(kernel_h)]
    center_h = kernel_h // 2
    center_w = kernel_w // 2
    for channel in range(min(input_channels, output_channels)):
      values[center_h][center_w][channel][channel] = 1.0
    return tf.constant(values, dtype=dtype)


class BilinearTransposeInitializer(tf.keras.initializers.Initializer):
  """Initializes Conv2DTranspose as per-channel bilinear upsampling."""

  def __call__(self, shape, dtype=None):
    dtype = dtype or tf.float32
    kernel_h, kernel_w, output_channels, input_channels = shape
    values = [[[[0.0 for _ in range(input_channels)]
                for _ in range(output_channels)]
               for _ in range(kernel_w)]
              for _ in range(kernel_h)]

    def axis_weight(index: int, size: int) -> float:
      factor = (size + 1) // 2
      center = factor - 1 if size % 2 == 1 else factor - 0.5
      return 1.0 - abs(index - center) / float(factor)

    for output_channel in range(output_channels):
      input_channel = min(output_channel, input_channels - 1)
      for row in range(kernel_h):
        for col in range(kernel_w):
          values[row][col][output_channel][input_channel] = (
              axis_weight(row, kernel_h) * axis_weight(col, kernel_w)
          )
    return tf.constant(values, dtype=dtype)


def create_learned_downsampler(
    output_channels: int,
    factor: int,
    sampling_mode: str,
) -> tf.keras.Model:
  """Returns a trainable spatial downsampler for bottleneck tensors."""
  if sampling_mode not in SAMPLING_MODES:
    raise ValueError(f'Unknown sampling_mode {sampling_mode!r}.')
  if sampling_mode == 'resize':
    raise ValueError('resize mode does not use a learned downsampler.')
  kernel_size = max(2 * factor, 2)
  return tf.keras.Sequential(
      [
          tf.keras.layers.Conv2D(
              filters=output_channels,
              kernel_size=kernel_size,
              strides=factor,
              padding='same',
              use_bias=False,
              kernel_initializer=ChannelwiseBoxInitializer(),
              name='learned_downsample_conv',
          )
      ],
      name='LearnedDownsampler',
  )


def create_learned_upsampler(
    output_channels: int,
    factor: int,
    sampling_mode: str,
) -> tf.keras.Model:
  """Returns a trainable spatial upsampler for bottleneck tensors."""
  if sampling_mode == 'cnn_stride_transpose':
    kernel_size = max(2 * factor, 2)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2DTranspose(
                filters=output_channels,
                kernel_size=kernel_size,
                strides=factor,
                padding='same',
                use_bias=False,
                kernel_initializer=BilinearTransposeInitializer(),
                name='learned_upsample_deconv',
            )
        ],
        name='LearnedUpsampler',
    )
  if sampling_mode == 'cnn_stride_resize_conv':
    return tf.keras.Sequential(
        [
            tf.keras.layers.UpSampling2D(
                size=(factor, factor),
                interpolation='bilinear',
                name='resize_upsample_for_cnn',
            ),
            tf.keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=3,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer=ChannelwiseIdentityInitializer(),
                name='learned_upsample_refine_conv',
            ),
        ],
        name='LearnedUpsampler',
    )
  raise ValueError(f'Unknown learned upsampler mode {sampling_mode!r}.')


def create_mlp_model(
    num_layers: int,
    num_channels: int,
    output_channels: int,
    activation: Callable[[tf.Tensor], tf.Tensor],
    name: str,
) -> tf.keras.Model:
  """Returns 1x1 models for linear/nonlinear channel conversions."""
  kernel_size = 1
  model = tf.keras.Sequential(name=name)

  import os as _os
  _orth = _os.environ.get("ORTH_INIT", "0") == "1"
  _init = 'orthogonal' if _orth else 'glorot_uniform'

  # Add hidden layers.
  for _ in range(num_layers - 1):
    model.add(
        tf.keras.layers.Conv2D(
            strides=1,
            padding='same',
            kernel_size=kernel_size,
            filters=num_channels,
            activation=activation,
            kernel_initializer=_init))

  # Add final linear layer.
  model.add(
      tf.keras.layers.Conv2D(
          strides=1,
          padding='same',
          kernel_size=kernel_size,
          filters=output_channels,
          activation=None,
          kernel_initializer=_init))
  return model


@enum.unique
class ModelConfiguration(enum.IntEnum):
  """Configurations that help construct better optimized models."""
  MLPS_ONLY = enum.auto()
  MLPS_AND_UNET_POST = enum.auto()
  FULL_MODEL = enum.auto()


# Epoch ranges at which relevant parts of the model should start being trained.
# Thresholds are formulated as a fraction of the total number of epochs, e.g.,
# mlps_only = .1 ==> epochs in the range [0, total_epochs * .1)
#   0 <= mlps_only <= mlps_and_unet_post <= 1.
@dataclasses.dataclass(frozen=True)
class ModelConfigurationThresholds:
  # Model configuration is set to MLPS_ONLY below mlps_only.
  mlps_only: float = 0
  # Model configuration is set to MLPS_AND_UNET_POST between mlps_only
  # and this mlps_and_unet_post. Model configuration is set to FULL_MODEL above
  # mlps_and_unet_post.
  mlps_and_unet_post: float = 0


class PreprocessCompressPostprocess(tf.keras.Model):
  """Builds a preprocess -> compress -> postprocess network.

  This class constructs a network by chaining together two autoencoder models
  (e.g., two UNets), as a preprocessor and a postprocessor around basic intra
  compression.
  """

  def __init__(
      self,
      model_keys: Tuple[str, ...] = ('image',),
      preprocessor_layer: Optional[tf.keras.Model] = None,
      postprocessor_layer: Optional[tf.keras.Model] = None,
      intra_compression_layer: Optional[tf.keras.Model] = None,
      loop_filter_layer: Optional[tf.keras.Model] = None,
      downsample_factor: int = 1,
      num_truncate_bits: int = 0,
      gamma: Optional[float] = None,
      bottleneck_channels: int = 1,
      output_channels: int = 3,
      num_mlp_layers: int = 2,
      num_mlp_nodes: int = 16,
      sampling_mode: str = 'resize',
      model_config_thresholds: ModelConfigurationThresholds = ModelConfigurationThresholds(),
      name: str = 'PreprocessCompressPostprocess',
  ):
    """Initializes the PreprocessCompressPostprocess model.

    Args:
      model_keys: Keys used to extract model relevant tensors from the model
        dictionary input.
      preprocessor_layer: tf.keras.Model that implements pre-processing.
      postprocessor_layer: tf.keras.Model that implements post-processing.
      intra_compression_layer: Layer that implements intra compression
        emulation.
      loop_filter_layer: Pre-trained layer that implements loop filtering.
      downsample_factor: Amount by which bottleneck channels should be spatially
        downsampled for wrapping around LR standard codecs to transport HR
        content (downsample_factor = 1 for no downsampling).
      num_truncate_bits: Number of bits to truncate from the bottleneck if
        wrapping around LDR codecs to transport HDR content.
      gamma: float that establishes the Lagrange multiplier in the optimized
        function "distortion + gamma * rate".
      bottleneck_channels: Number of channels in the bottleneck that undergo
        compression.
      output_channels: Number of channels at the output of the network.
      num_mlp_layers: Number of layers in mlp pre and postprocessors
        (num_mlp_layers=0 for identity, num_mlp_layers=1 for linear layers.)
      num_mlp_nodes: Number of nodes in the mlp hidden layer. Ignored for
        identity or linear layers.
      sampling_mode: Spatial down/up sampling mode. "resize" preserves the
        original fixed bicubic/Lanczos3 path. "cnn_stride_transpose" and
        "cnn_stride_resize_conv" replace the bottleneck resize operations with
        trainable layers.
      model_config_thresholds: Fraction of epochs over which different
        configurations of the model should be trained. Useful in changing model
        configuration during training in order to find a better minimum.
      name: string specifying a name for the model.
    """
    super().__init__(name=name)

    self.model_keys = model_keys

    # Useful in changing model configuration: For the first
    #   self._mlps_only * 100
    # percent of epochs train the mlps, then until
    #   self._mlps_and_unet_post * 100
    # percent of epochs train the mlps and unet_postprocessor, and thereafter
    # train the full model. This finds better local minima in some cases. Set
    # to zero to turn the functionality off.
    self._mlps_only = model_config_thresholds.mlps_only
    self._mlps_and_unet_post = model_config_thresholds.mlps_and_unet_post

    if not 0 <= self._mlps_only <= 1:
      raise ValueError(f'Got _mlps_only {self._mlps_only}. '
                       'Need 0 <= _mlps_only <= 1.')

    if not self._mlps_only <= self._mlps_and_unet_post <= 1:
      raise ValueError(f'Got _mlps_and_unet_post {self._mlps_and_unet_post}. '
                       f'Need {self._mlps_only}'
                       ' <= _mlps_and_unet_post <= 1.')

    if num_truncate_bits < 0:
      raise ValueError('num_truncate_bits must be non-negative.')

    if downsample_factor <= 0 or not isinstance(downsample_factor, int):
      raise ValueError('downsample_factor must be a positive integer.')

    if sampling_mode not in SAMPLING_MODES:
      raise ValueError(
          f'Got sampling_mode {sampling_mode!r}. Expected one of {SAMPLING_MODES}.'
      )

    self.downsample_factor = downsample_factor
    self.sampling_mode = sampling_mode
    self.num_truncate_bits = num_truncate_bits
    self.hdr_simul_qstep = tf.cast(1 << num_truncate_bits, dtype=tf.float32)

    if preprocessor_layer is None:
      self._unet_preprocessor = lambda x, *args, **kwargs: tf.zeros_like(x)
    else:
      self._unet_preprocessor = preprocessor_layer

    if postprocessor_layer is None:
      self._unet_postprocessor = lambda x, *args, **kwargs: tf.zeros_like(x)
    else:
      self._unet_postprocessor = postprocessor_layer

    self.num_mlp_layers = tf.Variable(
        initial_value=num_mlp_layers,
        trainable=False,
        name='num_mlp_layers',
        dtype=tf.int32)

    unet_scalers_are_trainable = True
    # Allow override via env var (diagnostic — paper doesn't say init=0.0)
    import os as _os
    _override_init = _os.environ.get("UNET_SCALER_INIT")
    unet_scalers_initializer = float(_override_init) if _override_init else 0.0
    _override_trainable = _os.environ.get("UNET_SCALER_TRAINABLE")
    if _override_trainable is not None:
        unet_scalers_are_trainable = _override_trainable != "0"
        logging.info("UNET_SCALER_TRAINABLE override: %s", unet_scalers_are_trainable)
    if self.num_mlp_layers > 0:
      self._mlp_preprocessor = create_mlp_model(
          num_layers=self.num_mlp_layers.numpy(),
          num_channels=num_mlp_nodes,
          output_channels=bottleneck_channels,
          activation=tf.math.sin,
          name='MLPPreprocessor')
      self._mlp_postprocessor = create_mlp_model(
          num_layers=self.num_mlp_layers.numpy(),
          num_channels=num_mlp_nodes,
          output_channels=output_channels,
          activation=tf.math.sin,
          name='MLPPostprocessor')
    else:
      self._mlp_preprocessor = lambda x, *a, **kw: x
      self._mlp_postprocessor = lambda x, *a, **kw: x
      if _override_trainable is None:
        unet_scalers_are_trainable = False
      if _override_init is None:
        unet_scalers_initializer = 1.0

    # Scales the preprocessors so that
    # bottleneck = self._unet_preprocessor_scaler * self._unet_preprocessor(...)
    #    + self._mlp_preprocessor(...)
    # The scaling value is trained.
    self._unet_preprocessor_scaler = self.add_weight(
        initializer=tf.constant_initializer(unet_scalers_initializer),
        trainable=unet_scalers_are_trainable,
        name='preprocessor_scaler',
        dtype=tf.float32)

    # Same as above but for the posptrocessors and final output of the model.
    self._unet_postprocessor_scaler = self.add_weight(
        initializer=tf.constant_initializer(unet_scalers_initializer),
        trainable=unet_scalers_are_trainable,
        name='postprocessor_scaler',
        dtype=tf.float32)

    # On/off switching of the unet preprocessor and postprocessors. Useful
    # when shifting training regiments.
    self._unet_preprocessor_switch = self.add_weight(
        initializer=tf.constant_initializer(1.0),
        trainable=False,
        name='unet_preprocessor_switch',
        dtype=tf.float32,
    )

    self._unet_postprocessor_switch = self.add_weight(
        initializer=tf.constant_initializer(1.0),
        trainable=False,
        name='unet_postprocessor_switch',
        dtype=tf.float32,
    )

    if intra_compression_layer is None:
      intra_compression_layer = encode_decode_intra_lib.EncodeDecodeIntra(
          rounding_fn=differentiable_round,
          use_jpeg_rate_model=True,
          min_qstep=1,
      )
    self.intra_compression_layer = intra_compression_layer

    if loop_filter_layer is None:
      self._loop_filter_layer = tf.zeros_like
    else:
      self._loop_filter_layer = loop_filter_layer

    if gamma is not None:
      self.gamma = self.add_weight(
          initializer=tf.constant_initializer(gamma),
          trainable=False,
          name='gamma',
          dtype=tf.float32)
    else:
      self.gamma = self.add_weight(
          trainable=False, name='gamma', dtype=tf.float32)

    self._configuration = self.add_weight(
        initializer=tf.constant_initializer(ModelConfiguration.FULL_MODEL),
        trainable=False,
        name='configuration',
        dtype=tf.int32)

    if self.downsample_factor > 1 and self.sampling_mode != 'resize':
      self._learned_downsampler = create_learned_downsampler(
          output_channels=bottleneck_channels,
          factor=self.downsample_factor,
          sampling_mode=self.sampling_mode,
      )
      self._learned_upsampler = create_learned_upsampler(
          output_channels=bottleneck_channels,
          factor=self.downsample_factor,
          sampling_mode=self.sampling_mode,
      )
    else:
      self._learned_downsampler = None
      self._learned_upsampler = None

    # Adjust mean/scale to improve training. Adjustment are made at the input of
    # networks and inverted at the output.
    self.mean_adjust = 128 * (1 << self.num_truncate_bits)
    self.scale_adjust = 255 * (1 << self.num_truncate_bits)

  def dict_to_model_inputs(self, input_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Returns a tensor formed using model_keys from input_dict."""
    return tf.concat(
        [tf.cast(input_dict[key], dtype=tf.float32) for key in self.model_keys],
        axis=-1)

  def model_predictions_to_dict(
      self, predictions: tf.Tensor,
      input_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Returns a dict of tensors by slicing predictions."""
    example = {}
    begin_channel = 0
    for key in self.model_keys:
      end_channel = begin_channel + input_dict[key].shape[-1]
      example[key] = predictions[..., begin_channel:end_channel]
      begin_channel = end_channel
    return example

  def _configure_model(self):
    if self._configuration == ModelConfiguration.MLPS_ONLY:
      self._unet_preprocessor_switch.assign(0)
      self._unet_postprocessor_switch.assign(0)
    elif self._configuration == ModelConfiguration.MLPS_AND_UNET_POST:
      self._unet_preprocessor_switch.assign(0)
      self._unet_postprocessor_switch.assign(1)
    else:
      self._unet_preprocessor_switch.assign(1)
      self._unet_postprocessor_switch.assign(1)

  def get_loop_filter_layer(self) -> Any:
    """Returns the loop filter (lf). Useful when independently training a lf."""
    return self._loop_filter_layer

  def get_gamma(self) -> tf.float32:
    """Returns the value of gamma used in the optimization of this model."""
    return self.gamma

  def set_gamma(self, gamma: float):
    """Sets the gamma so that the model can be evaluated at different gamma."""
    self.gamma.assign(gamma)

  def get_qstep(self) -> tf.Tensor:
    """Returns the qstep used by the codec proxy. Useful for logging."""
    return tf.convert_to_tensor(self.intra_compression_layer.get_qstep())

  def get_pre_post_scalers(self) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns the scalers used for the networks. Useful for logging."""
    pre = self._unet_preprocessor_switch * self._unet_preprocessor_scaler
    post = self._unet_postprocessor_switch * self._unet_postprocessor_scaler
    return pre, post

  def set_configuration(self, epoch: int, num_epochs: int) -> bool:
    """Sets the training sequence to postprocessor first then both."""
    if epoch < 0 or num_epochs < epoch:
      raise ValueError(f'Got epoch {epoch}, num_epochs {num_epochs}. '
                       'Need 0 <= epoch <= num_epochs.')
    training_state = epoch / num_epochs if num_epochs else 1

    def is_config_changed(next_config: ModelConfiguration) -> bool:
      return True if self._configuration != next_config else False

    if training_state < self._mlps_only:
      # Configure to run/train only the mlps.
      config_changed = is_config_changed(ModelConfiguration.MLPS_ONLY)
      self._configuration.assign(ModelConfiguration.MLPS_ONLY)
    elif training_state < self._mlps_and_unet_post:
      # Configure to run/train the mlps and the unet postprocessor.
      config_changed = is_config_changed(ModelConfiguration.MLPS_AND_UNET_POST)
      self._configuration.assign(ModelConfiguration.MLPS_AND_UNET_POST)
    else:
      # Configure to run/train all.
      config_changed = is_config_changed(ModelConfiguration.FULL_MODEL)
      self._configuration.assign(ModelConfiguration.FULL_MODEL)

    if config_changed:
      self._configure_model()
    return config_changed

  def run_preprocessor(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    """Runs the preprocessor and generates the bottleneck."""
    adjusted_inputs = (inputs - self.mean_adjust) / self.scale_adjust
    scale = self._unet_preprocessor_switch * self._unet_preprocessor_scaler
    output = self.scale_adjust * (
        self._mlp_preprocessor(adjusted_inputs, training=training) + scale *
        self._unet_preprocessor(adjusted_inputs, training=training)) + self.mean_adjust

    # DPP paradigm: preprocessor modifies LUMA only. Keep the original chroma (U,V)
    # untouched; only the Y channel carries the learned residual. (BT.601 rgb<->yuv is
    # linear so this works directly on [0,255] and is differentiable.)
    if getattr(self, 'preproc_luma_only', False) and output.shape[-1] == 3:
      y_pre = tf.image.rgb_to_yuv(output)[..., 0:1]      # preprocessed luma
      uv_in = tf.image.rgb_to_yuv(inputs)[..., 1:3]      # ORIGINAL chroma (untouched)
      output = tf.image.yuv_to_rgb(tf.concat([y_pre, uv_in], axis=-1))

    if self.downsample_factor > 1:
      if self.sampling_mode == 'resize':
        output = downsample_tensor(
            output, self.downsample_factor, method=tf.image.ResizeMethod.BICUBIC
        )
      else:
        output = self._learned_downsampler(output, training=training)

    if self.num_truncate_bits:
      # Truncate last self.num_truncate_bits.
      output = _differentiable_truncate(output / self.hdr_simul_qstep)

    return output

  def run_postprocessor(self, compressed_bottleneck: tf.Tensor,
                        training: bool) -> tf.Tensor:
    """Runs the postprocessor on the compressed bottleneck."""

    if self.num_truncate_bits:
      # Pull back to original bit range.
      compressed_bottleneck *= self.hdr_simul_qstep
      compressed_bottleneck += self.hdr_simul_qstep / 2

    if self.downsample_factor > 1:
      if self.sampling_mode == 'resize':
        compressed_bottleneck = upsample_tensor(
            compressed_bottleneck,
            self.downsample_factor,
            method=tf.image.ResizeMethod.LANCZOS3,
        )
      else:
        compressed_bottleneck = self._learned_upsampler(
            compressed_bottleneck, training=training
        )

    adjusted_inputs = (compressed_bottleneck -
                       self.mean_adjust) / self.scale_adjust
    scale = self._unet_postprocessor_switch * self._unet_postprocessor_scaler
    return self.scale_adjust * (
        self._mlp_postprocessor(adjusted_inputs, training=training) + scale *
        self._unet_postprocessor(adjusted_inputs, training=training)) + self.mean_adjust

  def apply_loop_filter(self, compressed_bottleneck: tf.Tensor) -> tf.Tensor:
    """Runs the loop-filter proxy on the compressed bottleneck."""
    filtered = tf.concat(
        [
            self._loop_filter_layer(compressed_bottleneck[..., ch : ch + 1])
            for ch in range(compressed_bottleneck.shape[-1])
        ],
        axis=-1,
    )

    return compressed_bottleneck + filtered

  def call(self,
           input_dict: Dict[str, tf.Tensor],
           training: Optional[bool] = None) -> Dict[str, Any]:
    """Forward-pass of the PreprocessCompressPostprocess model.

    Args:
      input_dict: Dictionary containing tensors that will be used to derive a
        model input tensor of shape [b,n,m,c] where b is batch size, [n,m] is
        the image shape, and c is the number of channels.
      training: bool that defines whether the call should be executed as a
        training or an inference call.

    Returns:
      outputs: Dictionary containing the prediction, rate, and bottleneck
        tensors.
    """
    # Toggle the global quantizer training phase so that noise_injection_round
    # (if selected) uses additive uniform noise during training and hard
    # round() at inference. No-op for straight_through.
    set_quantizer_training_phase(bool(training))

    inputs = self.dict_to_model_inputs(input_dict)

    bottleneck = self.run_preprocessor(inputs, training)

    compressed_bottleneck, rate = self.intra_compression_layer(
        tf.clip_by_value(bottleneck, 0., 255.))

    # DPP paradigm: codec compresses LUMA only; chroma (U,V) is carried LOSSLESS from
    # the ORIGINAL input. Restore original chroma onto the decoded luma. Cascade effects:
    #  - perceptual loss + fidelity then see decoded-Y + lossless-UV (DPP's p_hat^RGB);
    #  - fidelity becomes effectively luma-only (chroma error = 0);
    #  - the chroma part of `rate` is constant w.r.t. the (luma-only) preprocessor, so
    #    the optimized rate gradient is effectively Y-only (DPP codes Y bits only).
    if getattr(self, 'codec_luma_only', False) and compressed_bottleneck.shape[-1] == 3:
      yuv_dec = tf.image.rgb_to_yuv(compressed_bottleneck)
      yuv_in = tf.image.rgb_to_yuv(inputs)
      compressed_bottleneck = tf.image.yuv_to_rgb(
          tf.concat([yuv_dec[..., 0:1], yuv_in[..., 1:3]], axis=-1))

    compressed_bottleneck = self.apply_loop_filter(compressed_bottleneck)

    prediction = self.run_postprocessor(compressed_bottleneck, training)

    output_dict = self.model_predictions_to_dict(
        predictions=prediction, input_dict=input_dict)
    output_dict.update({
        'prediction': prediction,
        'rate': rate,
        'bottleneck': bottleneck,
        'compressed_bottleneck': compressed_bottleneck
    })
    return output_dict


def differentiable_round(x: tf.Tensor) -> tf.Tensor:
  """Straight-through rounding: forward = round, backward = identity."""
  return x + tf.stop_gradient(tf.round(x) - x)


def _differentiable_truncate(x: tf.Tensor) -> tf.Tensor:
  """Differentiable truncation."""
  return x + tf.stop_gradient(tf.math.floordiv(x, 1) - x)


# Module-level flag (tf.Variable) that the parent model toggles per call.
# 1.0 during training (model called with training=True), 0.0 otherwise.
# Used by noise_injection_round to decide forward path. This is a singleton
# because there is at most one PreprocessCompressPostprocess instance per
# Python process in our pipelines; concurrent multi-model training would need
# a per-model variable instead.
_GLOBAL_QUANTIZER_TRAINING_PHASE = tf.Variable(
    0.0, trainable=False, dtype=tf.float32,
    name="quantizer_training_phase",
)


def set_quantizer_training_phase(training: bool) -> None:
  _GLOBAL_QUANTIZER_TRAINING_PHASE.assign(1.0 if training else 0.0)


def noise_injection_round(x: tf.Tensor) -> tf.Tensor:
  """Theis/Ballé-style soft rounding for compression training.

  Training forward: x + U(-0.5, 0.5) (noise distribution matches expected
  quantization error of round). Inference forward: tf.round(x). Backward
  gradient is the identity (additive noise has unit Jacobian).
  """
  noisy = x + tf.random.uniform(tf.shape(x), minval=-0.5, maxval=0.5)
  hard = tf.round(x)
  return tf.cond(
      _GLOBAL_QUANTIZER_TRAINING_PHASE > 0.5,
      lambda: noisy,
      lambda: hard,
  )


def _diff_jpeg_polynomial_round(x: tf.Tensor) -> tf.Tensor:
  """Polynomial-surrogate rounding from Reich et al. WACV 2024.
  Forward: round(x) + (x - round(x))^3 (smooth, not exactly round).
  Backward: autograd of cubic = 3*(x - round(x))^2.
  """
  from image_compression import diff_jpeg_tf
  return diff_jpeg_tf.polynomial_round(x)


def _diff_jpeg_ste_polynomial_round(x: tf.Tensor) -> tf.Tensor:
  """STE rounding with paper's polynomial backward.
  Forward: exact tf.round.
  Backward: 3*(x - round(x))^2 (not constant 1).
  """
  from image_compression import diff_jpeg_tf
  return diff_jpeg_tf.ste_polynomial_round(x)


_QUANTIZER_FNS = {
    "straight_through": differentiable_round,
    "noise_injection": noise_injection_round,
    "polynomial": _diff_jpeg_polynomial_round,
    "ste_polynomial": _diff_jpeg_ste_polynomial_round,
}


def get_quantizer_fn(name: str):
  if name not in _QUANTIZER_FNS:
    raise ValueError(
        f"unknown quantizer mode {name!r}; expected one of {list(_QUANTIZER_FNS)}"
    )
  return _QUANTIZER_FNS[name]


def _distortion_rate_loss(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    gamma: float,
    distortion_fn: Callable[[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
                            tf.Tensor],
    add_valid_bottleneck_pixels_penalty=False) -> tf.Tensor:
  """Implements a distortion + gamma * rate loss function."""

  def valid_pixel_range_penalty(inputs, min_pixel=0, max_pixel=255, scaler=255):
    actual_min = min(min_pixel, max_pixel)
    actual_max = max(min_pixel, max_pixel)
    reshaped = tf.keras.backend.reshape(inputs, (inputs.shape[0], -1))
    spread = (actual_max - actual_min) * .1 + 1
    return scaler * tf.keras.backend.sum(
        # No penalty within [min_pixel, max_pixel].
        tf.keras.backend.abs(
            tf.keras.backend.relu(reshaped - actual_max) +
            tf.keras.backend.relu(actual_min - reshaped)) / spread,
        axis=1)

  # Normalize so that per-sample numbers are displayed.
  normalization = tf.keras.backend.cast(
      outputs['prediction'].shape[0] /
      tf.keras.backend.prod(outputs['prediction'].shape)
      if outputs['prediction'].shape[0] > 0 else 0, tf.float32)

  distortion = distortion_fn(ground_truth, outputs) * normalization

  def loss():
    return distortion + gamma * outputs['rate'] * normalization

  # Penalty for bottlenecks going beyond the valid pixel range.
  def loss_plus_penalty():
    return loss() + valid_pixel_range_penalty(
        outputs['bottleneck']) * normalization

  return loss_plus_penalty() if add_valid_bottleneck_pixels_penalty else loss()


class DistortionRateMetric(tf.keras.metrics.Metric):
  """Implements a distortion + gamma * rate metric."""

  def __init__(
      self,
      distortion_fn: Callable[[Dict[str, tf.Tensor], Dict[str, tf.Tensor]],
                              tf.Tensor],
      gamma: float = 1,
      name: str = 'distortion_rate',
      **kwargs):
    super().__init__(name=name, **kwargs)
    self.distortion_rate = self.add_weight(
        name='d_plus_gamma_r', initializer='zeros')

    # Multidimensional state to keep track of distortion per channel.
    self.distortion = self.add_weight(name='d', initializer='zeros')
    self.rate = self.add_weight(name='r', initializer='zeros')
    self.total_added = self.add_weight(name='ta', initializer='zeros')
    self.gamma = gamma
    self.distortion_fn = distortion_fn

  def update_gamma(self, gamma: float):
    self.gamma = gamma

  def update_state(self,
                   ground_truth: Dict[str, tf.Tensor],
                   outputs: Dict[str, tf.Tensor],
                   sample_weight: Optional[tf.Tensor] = None):
    normalization = tf.cast(
        outputs['prediction'].shape[0] /
        tf.keras.backend.prod(outputs['prediction'].shape)
        if outputs['prediction'].shape[0] > 0 else 0, tf.float32)
    r_values = tf.cast(outputs['rate'], self.dtype) * normalization
    d_values = tf.cast(self.distortion_fn(ground_truth, outputs),
                       self.dtype) * normalization
    dr_values = d_values + self.gamma * r_values

    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      dr_values = tf.multiply(dr_values,
                              tf.broadcast_weights(sample_weight, dr_values))
      d_values = tf.multiply(d_values,
                             tf.broadcast_weights(sample_weight, d_values))
      r_values = tf.multiply(r_values,
                             tf.broadcast_weights(sample_weight, r_values))

    self.distortion_rate.assign_add(tf.reduce_mean(dr_values))
    self.distortion.assign_add(tf.reduce_mean(d_values))
    self.rate.assign_add(tf.reduce_mean(r_values))
    self.total_added.assign_add(1)

  def result(self) -> tf.Tensor:
    return (self.distortion_rate * 1.0 /
            self.total_added) if self.total_added > 0 else tf.constant(
                0, dtype=tf.float32)

  def distortion_result(self) -> tf.Tensor:
    return (self.distortion * 1.0 /
            self.total_added) if self.total_added > 0 else tf.constant(
                0, dtype=tf.float32)

  def rate_result(self) -> tf.Tensor:
    return (self.rate * 1.0 /
            self.total_added) if self.total_added > 0 else tf.constant(
                0, dtype=tf.float32)

  # Version of reset_states() to accommodate multidimensional states.
  def reset(self):
    tf.keras.backend.batch_set_value([
        (v, tf.zeros_like(v)) for v in self.variables
    ])


def basic_preprocessor_layer(
    output_channels: int,
    encoder_filters_sequence: Sequence[int] = (32, 64, 128, 256),
    decoder_filters_sequence: Sequence[int] = (512, 256, 128, 64, 32),
    output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> tf.keras.Model:
  """Returns a unet."""
  logging.info(
      'preprocessor encoder filters: ' + ' %d' * len(encoder_filters_sequence),
      *encoder_filters_sequence)
  logging.info(
      'preprocessor decoder filters: ' + ' %d' * len(decoder_filters_sequence),
      *decoder_filters_sequence)
  unet_model = simple_unet.UNet

  return unet_model(
      name='Preprocessor',
      output_channels=output_channels,
      encoder_filters_sequence=encoder_filters_sequence,
      decoder_filters_sequence=decoder_filters_sequence,
      output_activation=output_activation,
  )


def basic_postprocessor_layer(
    output_channels: int,
    encoder_filters_sequence: Sequence[int] = (32, 64, 128, 256),
    decoder_filters_sequence: Sequence[int] = (512, 256, 128, 64, 32),
    output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> tf.keras.Model:
  """Returns a unet."""
  logging.info(
      'postprocessor encoder filters: ' + ' %d' * len(encoder_filters_sequence),
      *encoder_filters_sequence)
  logging.info(
      'postprocessor decoder filters: ' + ' %d' * len(decoder_filters_sequence),
      *decoder_filters_sequence)
  unet_model = simple_unet.UNet

  return unet_model(
      name='Postprocessor',
      output_channels=output_channels,
      encoder_filters_sequence=encoder_filters_sequence,
      decoder_filters_sequence=decoder_filters_sequence,
      output_activation=output_activation,
  )


def create_loop_filter_model(
    model_keys: Tuple[str, ...] = ('image',),
    bottleneck_channels: int = 3,
    output_channels: int = 3,
    encoder_filters_sequence: Sequence[int] = (8,),
    decoder_filters_sequence: Sequence[int] = (8, 8),
    gamma: float = 1,
    base_model: Callable[..., tf.keras.Model] = PreprocessCompressPostprocess,
    **ignored_args,
) -> tf.keras.Model:
  """Returns a unet for single-channel loop filtering.

  When emulating standard codec loop filters it is advisable to keep
  encoder/decoder filter parameters conservative as standard codecs tend to
  have simple loop filters.

  Args:
    model_keys: Keys used to extract model relevant tensors from the model
      dictionary input.
    bottleneck_channels: Number of channels in the bottleneck that undergo
      compression.
    output_channels: Number of channels at the output of the network.
    encoder_filters_sequence: Unet encoder filters.
    decoder_filters_sequence: Unet decoder filters.
    gamma: float that establishes the Lagrange multiplier in the optimized
      function "distortion + gamma * rate".
    base_model: Image/Video sandwich model to use when wrapping the loop filter.
    **ignored_args: Ignored arguments to maintain gin compatibility with
      create_basic_model()

  Returns:
    Constructed model.
  """
  # unused args for compatibility with create_basic_model when using gin.
  _ = ignored_args
  logging.info(
      'loop_filter encoder filters: ' + ' %d' * len(encoder_filters_sequence),
      *encoder_filters_sequence,
  )
  logging.info(
      'loop_filter decoder filters: ' + ' %d' * len(decoder_filters_sequence),
      *decoder_filters_sequence,
  )
  unet_model = simple_unet.UNet
  return base_model(
      model_keys=model_keys,
      preprocessor_layer=None,
      postprocessor_layer=None,
      loop_filter_layer=unet_model(
          name='LoopFilter',
          output_channels=1,  # single channel processing
          encoder_filters_sequence=encoder_filters_sequence,
          decoder_filters_sequence=decoder_filters_sequence,
      ),
      num_mlp_layers=0,  # identity
      bottleneck_channels=bottleneck_channels,
      output_channels=output_channels,
      gamma=gamma,
  )


def create_basic_model(
    model_keys: Tuple[str, ...] = ('image',),
    bottleneck_channels: int = 1,
    output_channels: int = 3,
    num_mlp_layers: int = 2,
    use_jpeg_rate_model: bool = False,
    downsample_factor: int = 1,
    num_truncate_bits: int = 0,
    gamma: float = 1,
    model_config_thresholds: ModelConfigurationThresholds = ModelConfigurationThresholds(
        mlps_only=0, mlps_and_unet_post=0
    ),
    loop_filter_folder: Optional[str] = None,
    use_unet_preprocessor: bool = True,
    use_unet_postprocessor: bool = True,
    sampling_mode: str = 'resize',
    quantizer_mode: str = 'straight_through',
    rate_proxy_mode: str = 'log_nonzero',
    rate_proxy_grad_scale: float = 1.0,
    codec_forward_mode: str = 'proxy',
    post_jpeg_int_round: bool = False,
    output_clip_mode: str = 'hard',
    convert_to_yuv: bool = False,
    downsample_chroma: bool = False,
    qstep_init_override: Optional[float] = None,
    train_qstep: bool = True,
    preprocessor_factory: Optional[Callable[..., tf.keras.Model]] = None,
    postprocessor_factory: Optional[Callable[..., tf.keras.Model]] = None,
) -> tf.keras.Model:
  """Constructs the Keras model for PreprocessCompressPostprocess.

  Args:
    model_keys: Keys used to extract model relevant tensors from the model
      dictionary input.
    bottleneck_channels: Number of channels in the bottleneck that undergo
      compression.
    output_channels: Number of channels at the output of the network.
    num_mlp_layers: Number of layers in mlp pre and postprocessors
      (num_mlp_layers=0 for identity, num_mlp_layers=1 for linear layers.)
    use_jpeg_rate_model: True for JPEG rate model, False for Gaussian rate
      model.
    downsample_factor: Amount by which bottleneck channels should be spatially
      downsampled for wrapping around LR standard codecs to transport HR content
      (downsample_factor = 1 for no downsampling).
    num_truncate_bits: Number of bits to truncate from the bottleneck if
      wrapping around LDR codecs to transport HDR content.
    gamma: float that establishes the Lagrange multiplier in the optimized
      function "distortion + gamma * rate".
    model_config_thresholds: Fraction of epochs over which different
      configurations of the model should be trained. Useful in changing model
      configuration during training in order to find a better minimum.
    loop_filter_folder: Folder containing the model for loop filtering proxy.
    use_unet_preprocessor: Whether preprocessor unet is on.
    use_unet_postprocessor: Whether postprocessor unet is on.
    sampling_mode: Spatial down/up sampling mode for LR wrapping.

  Returns:
    Constructed Keras model.
  """

  if qstep_init_override is not None:
    qstep_init = float(qstep_init_override)
  elif gamma is not None:
    qstep_init = max(math.sqrt(abs(gamma)) / (1 << num_truncate_bits), 1.0)
  else:
    qstep_init = 1.0

  if loop_filter_folder:
    loop_filter_compound_model = create_loop_filter_model(
        base_model=PreprocessCompressPostprocess, model_keys=model_keys
    )
    logging.info('Loading loop filter filter from: %s', loop_filter_folder)
    model = serialization.load_checkpoint(
        loop_filter_folder, loop_filter_compound_model
    )
    loop_filter_layer = model.get_loop_filter_layer()
    loop_filter_layer.trainable = False
    logging.info(
        'Loop filter trainable weights: %d',
        len(loop_filter_layer.trainable_weights),
    )
  else:
    loop_filter_layer = None

  logging.info('Preprocessor unet: %d', use_unet_preprocessor)
  logging.info('Postprocessor unet: %d', use_unet_postprocessor)
  if use_unet_preprocessor:
    if preprocessor_factory is not None:
      preprocessor_layer = preprocessor_factory(
          output_channels=bottleneck_channels, name='Preprocessor')
    else:
      preprocessor_layer = basic_preprocessor_layer(
          output_channels=bottleneck_channels,
      )
  else:
    preprocessor_layer = None

  if use_unet_postprocessor:
    if postprocessor_factory is not None:
      postprocessor_layer = postprocessor_factory(
          output_channels=output_channels, name='Postprocessor')
    else:
      postprocessor_layer = basic_postprocessor_layer(
          output_channels=output_channels,
      )
  else:
    postprocessor_layer = None

  # UNet for pre and postprocessing.
  selected_rounding_fn = get_quantizer_fn(quantizer_mode)
  return PreprocessCompressPostprocess(
      model_keys=model_keys,
      preprocessor_layer=preprocessor_layer,
      postprocessor_layer=postprocessor_layer,
      intra_compression_layer=encode_decode_intra_lib.EncodeDecodeIntra(
          rounding_fn=selected_rounding_fn,
          use_jpeg_rate_model=use_jpeg_rate_model,
          qstep_init=qstep_init,
          train_qstep=train_qstep,
          min_qstep=1,
          rate_proxy_mode=rate_proxy_mode,
          rate_proxy_grad_scale=rate_proxy_grad_scale,
          codec_forward_mode=codec_forward_mode,
          post_jpeg_int_round=post_jpeg_int_round,
          output_clip_mode=output_clip_mode,
          convert_to_yuv=convert_to_yuv,
          downsample_chroma=downsample_chroma,
      ),  # PIL jpeg does not seem to distinguish qsteps 0 and 1.
      loop_filter_layer=loop_filter_layer,
      downsample_factor=downsample_factor,
      num_truncate_bits=num_truncate_bits,
      gamma=gamma,
      bottleneck_channels=bottleneck_channels,
      output_channels=output_channels,
      num_mlp_layers=num_mlp_layers,
      sampling_mode=sampling_mode,
      model_config_thresholds=model_config_thresholds,
  )


def create_basic_loss(
    gamma: float = 1,
    distortion_fn: Optional[
        Callable[[Dict[str, tf.Tensor], Dict[str, tf.Tensor]], tf.Tensor]
    ] = distortion_fns.distortion_l2norm,
    add_valid_bottleneck_pixels_penalty: bool = True,
) -> Callable[[Any], Any]:
  """Constructs the basic "distortion + gamma * rate" loss.

  Args:
    gamma: float that establishes the Lagrange multiplier in the optimized
      function. Distortion and rate are calculated using ground truths and
      network outputs with gamma weighing the importance of rate helping
      establish a rate constraint.
    distortion_fn: Callable that calculates a distortion between ground-truth
      and model outputs.
    add_valid_bottleneck_pixels_penalty: When True bottlenecks are penalized for
      being out of the [0, 255] range.

  Returns:
    Constructed loss.
  """

  return functools.partial(
      _distortion_rate_loss,
      distortion_fn=distortion_fn,
      gamma=gamma,
      add_valid_bottleneck_pixels_penalty=add_valid_bottleneck_pixels_penalty)


def create_basic_metric(
    gamma: float = 1,
    distortion_fn: Optional[
        Callable[[Dict[str, tf.Tensor], Dict[str, tf.Tensor]], tf.Tensor]
    ] = distortion_fns.distortion_l2norm,
) -> tf.keras.metrics.Metric:
  """Constructs the basic "distortion + gamma * rate" metric.

  Args:
    gamma: float that establishes the Lagrange multiplier in the optimized
      function. Distortion and rate are calculated using ground truths and
      network outputs with gamma weighing the importance of rate helping
      establish a rate constraint.
    distortion_fn: Callable that calculates a distortion between ground-truth
      and model outputs.

  Returns:
    Constructed metric.
  """

  return DistortionRateMetric(gamma=gamma, distortion_fn=distortion_fn)
