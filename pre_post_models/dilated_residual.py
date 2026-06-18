"""Plan C: Single-scale dilated residual network. Zero skip connections."""

from typing import Optional, Sequence, Callable
import tensorflow as tf


class DilatedResidualBlock(tf.keras.layers.Layer):
  """Residual block with dilated first convolution."""

  def __init__(self, filters: int, dilation_rate: int, name: str = 'dil_res'):
    super().__init__(name=name)
    self._conv1 = tf.keras.layers.Conv2D(
        filters, 3, padding='same', dilation_rate=dilation_rate,
        activation='relu', name='conv_0')
    self._conv2 = tf.keras.layers.Conv2D(
        filters, 3, padding='same', name='conv_1')

  def call(self, x):
    return tf.nn.relu(self._conv2(self._conv1(x)) + x)


class DilatedResidualNet(tf.keras.Model):
  """Single-scale network using dilated convolutions for receptive field."""

  def __init__(
      self,
      filters: int = 32,
      dilations: Sequence[int] = (1, 2, 4, 8, 1, 2, 4, 8),
      output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      output_channels: int = 3,
      name: str = 'dilated_residual',
  ):
    super().__init__(name=name)

    self._input_conv = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu', name='input_conv')

    self._blocks = []
    for i, d in enumerate(dilations):
      self._blocks.append(
          DilatedResidualBlock(filters, d, name=f'block_{i}'))

    self._output_layer = tf.keras.layers.Conv2D(
        output_channels, 3, padding='same', activation=output_activation,
        name='output')

  def call(self, inputs, *ignored_args):
    x = self._input_conv(inputs)
    for block in self._blocks:
      x = block(x)
    return self._output_layer(x)
