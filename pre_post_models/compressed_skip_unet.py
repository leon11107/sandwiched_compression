"""Plan B: Slim U-Net with compressed skip connections via 1x1 conv."""

from typing import Optional, Sequence, Callable
import tensorflow as tf


class CompressedSkipUNet(tf.keras.Model):
  """U-Net where skip connections are compressed through 1x1 convolutions."""

  def __init__(
      self,
      encoder_filters_sequence: Sequence[int] = (32,),
      decoder_filters_sequence: Sequence[int] = (32, 32),
      skip_channels: int = 4,
      output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      output_channels: int = 3,
      name: str = 'compressed_skip_unet',
      conv_class=None,
  ):
    super().__init__(name=name)
    ConvCls = conv_class or tf.keras.layers.Conv2D

    num_encoder_blocks = len(encoder_filters_sequence)

    self._encoder_convs = []
    self._skip_compressors = []
    for i in range(num_encoder_blocks):
      f = encoder_filters_sequence[i]
      self._encoder_convs.append([
          ConvCls(f, 3, padding='same', activation='relu',
                  name=f'enc_{i}_conv_0'),
          ConvCls(f, 3, padding='same', activation='relu',
                  name=f'enc_{i}_conv_1'),
      ])
      if i < num_encoder_blocks - 1:
        self._skip_compressors.append(
            ConvCls(skip_channels, 1, padding='same',
                    name=f'skip_compress_{i}'))

    num_decoder_blocks = len(decoder_filters_sequence)
    assert num_decoder_blocks in (num_encoder_blocks, num_encoder_blocks + 1)

    self._decoder_convs = []
    self._upsample_layers = []
    for i in range(num_decoder_blocks):
      f = decoder_filters_sequence[i]
      self._decoder_convs.append([
          ConvCls(f, 3, padding='same', activation='relu',
                  name=f'dec_{i}_conv_0'),
          ConvCls(f, 3, padding='same', activation='relu',
                  name=f'dec_{i}_conv_1'),
      ])
      if i < num_decoder_blocks - 1:
        self._upsample_layers.append(
            tf.keras.layers.UpSampling2D(2, interpolation='bilinear',
                                         name=f'dec_{i}_up'))

    self._output_layer = ConvCls(
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
