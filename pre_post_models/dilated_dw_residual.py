"""Plan E: Single-scale dilated residual with depthwise separable convolutions."""

from typing import Optional, Sequence, Callable
import tensorflow as tf


class DilatedDWResidualBlock(tf.keras.layers.Layer):
  """Residual block: dilated DW 3x3 + PW 1x1 + residual."""

  def __init__(self, filters: int, dilation_rate: int, name: str = 'dil_dw_res'):
    super().__init__(name=name)
    self._dw = tf.keras.layers.DepthwiseConv2D(
        3, padding='same', dilation_rate=dilation_rate, name='dw')
    self._pw = tf.keras.layers.Conv2D(
        filters, 1, padding='same', activation='relu', name='pw')

  def call(self, x):
    return self._pw(self._dw(x)) + x


class DilatedDWResidualNet(tf.keras.Model):
  """Single-scale dilated DW separable residual network. Zero skip connections."""

  def __init__(
      self,
      filters: int = 32,
      dilations: Sequence[int] = (1, 2, 4, 8, 1, 2, 4, 8),
      output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      output_channels: int = 3,
      name: str = 'dilated_dw_residual',
  ):
    super().__init__(name=name)

    self._input_proj = tf.keras.layers.Conv2D(
        filters, 1, padding='same', activation='relu', name='input_proj')

    self._blocks = []
    for i, d in enumerate(dilations):
      self._blocks.append(
          DilatedDWResidualBlock(filters, d, name=f'block_{i}'))

    self._output_layer = tf.keras.layers.Conv2D(
        output_channels, 1, padding='same', activation=output_activation,
        name='output')

  def call(self, inputs, *ignored_args):
    x = self._input_proj(inputs)
    for block in self._blocks:
      x = block(x)
    return self._output_layer(x)
