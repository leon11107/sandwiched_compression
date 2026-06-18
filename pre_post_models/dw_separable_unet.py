"""Plan D: Slim U-Net with depthwise separable convolutions."""

from typing import Optional, Sequence, Callable
import tensorflow as tf


class DWSeparableConv(tf.keras.layers.Layer):
  """Depthwise separable convolution: DW 3x3 + PW 1x1."""

  def __init__(self, filters: int, activation='relu', name: str = 'dw_sep'):
    super().__init__(name=name)
    self._dw = tf.keras.layers.DepthwiseConv2D(
        3, padding='same', name='dw')
    self._pw = tf.keras.layers.Conv2D(
        filters, 1, padding='same', activation=activation, name='pw')

  def call(self, x):
    return self._pw(self._dw(x))


class DWSeparableUNet(tf.keras.Model):
  """U-Net with all standard convolutions replaced by depthwise separable."""

  def __init__(
      self,
      encoder_filters_sequence: Sequence[int] = (32,),
      decoder_filters_sequence: Sequence[int] = (32, 32),
      output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      output_channels: int = 3,
      name: str = 'dw_separable_unet',
  ):
    super().__init__(name=name)

    num_encoder_blocks = len(encoder_filters_sequence)

    self._encoder_convs = []
    for i in range(num_encoder_blocks):
      f = encoder_filters_sequence[i]
      self._encoder_convs.append([
          DWSeparableConv(f, name=f'enc_{i}_dw_0'),
          DWSeparableConv(f, name=f'enc_{i}_dw_1'),
      ])

    num_decoder_blocks = num_encoder_blocks + 1
    assert num_decoder_blocks == len(decoder_filters_sequence)

    self._decoder_convs = []
    self._upsample_layers = []
    for i in range(num_decoder_blocks):
      f = decoder_filters_sequence[i]
      self._decoder_convs.append([
          DWSeparableConv(f, name=f'dec_{i}_dw_0'),
          DWSeparableConv(f, name=f'dec_{i}_dw_1'),
      ])
      if i < num_decoder_blocks - 1:
        self._upsample_layers.append(
            tf.keras.layers.UpSampling2D(2, interpolation='bilinear',
                                         name=f'dec_{i}_up'))

    self._concat = tf.keras.layers.Concatenate(axis=-1, name='concat')
    self._output_layer = tf.keras.layers.Conv2D(
        output_channels, 3, padding='same', activation=output_activation,
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
      if i < len(self._encoder_convs) - 1:
        skips.append(x)
        x = self._pool(x)

    skips.append(None)
    num_skips = len(skips)

    for i, conv_pair in enumerate(self._decoder_convs):
      for conv in conv_pair:
        x = conv(x)
      skip = skips[num_skips - 1 - i]
      if skip is not None:
        x = self._upsample_layers[i](x)
        x = self._concat([x, skip])

    return self._output_layer(x)
