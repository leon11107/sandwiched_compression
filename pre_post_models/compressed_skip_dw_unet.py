"""Compressed skip U-Net with depthwise separable convolutions."""

from typing import Optional, Sequence, Callable
import tensorflow as tf


class DWSep(tf.keras.layers.Layer):
  def __init__(self, filters, name='dw_sep'):
    super().__init__(name=name)
    self._dw = tf.keras.layers.DepthwiseConv2D(3, padding='same', name='dw')
    self._pw = tf.keras.layers.Conv2D(filters, 1, padding='same',
                                       activation='relu', name='pw')

  def call(self, x):
    return self._pw(self._dw(x))


class CompressedSkipDWUNet(tf.keras.Model):
  """Compressed-skip U-Net with depthwise separable convolutions."""

  def __init__(
      self,
      encoder_filters_sequence: Sequence[int] = (16,),
      decoder_filters_sequence: Sequence[int] = (16, 16),
      skip_channels: int = 4,
      output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      output_channels: int = 3,
      name: str = 'cs_dw_unet',
  ):
    super().__init__(name=name)

    num_enc = len(encoder_filters_sequence)
    self._encoder_convs = []
    self._skip_compressors = []
    for i in range(num_enc):
      f = encoder_filters_sequence[i]
      self._encoder_convs.append([
          DWSep(f, name=f'enc_{i}_dw_0'),
          DWSep(f, name=f'enc_{i}_dw_1'),
      ])
      if i < num_enc - 1:
        self._skip_compressors.append(
            tf.keras.layers.Conv2D(skip_channels, 1, padding='same',
                                   name=f'skip_compress_{i}'))

    num_dec = num_enc + 1
    assert num_dec == len(decoder_filters_sequence)
    self._decoder_convs = []
    self._upsample_layers = []
    for i in range(num_dec):
      f = decoder_filters_sequence[i]
      self._decoder_convs.append([
          DWSep(f, name=f'dec_{i}_dw_0'),
          DWSep(f, name=f'dec_{i}_dw_1'),
      ])
      if i < num_dec - 1:
        self._upsample_layers.append(
            tf.keras.layers.UpSampling2D(2, interpolation='bilinear',
                                         name=f'dec_{i}_up'))

    self._output_layer = tf.keras.layers.Conv2D(
        output_channels, 1, padding='same', activation=output_activation,
        name='output')

  def _pool(self, x):
    s = tf.shape(x)
    return x[:, 1:s[1]:2, 1:s[2]:2, :]

  def call(self, inputs, *ignored_args):
    x = inputs
    skips = []
    for i, conv_pair in enumerate(self._encoder_convs):
      for conv in conv_pair:
        x = conv(x)
      if i < len(self._skip_compressors):
        skips.append(self._skip_compressors[i](x))
        x = self._pool(x)

    skips.append(None)
    num_skips = len(skips)

    for i, conv_pair in enumerate(self._decoder_convs):
      for conv in conv_pair:
        x = conv(x)
      skip = skips[num_skips - 1 - i]
      if skip is not None:
        x = self._upsample_layers[i](x)
        x = tf.concat([x, skip], axis=-1)

    return self._output_layer(x)
