"""RepVGG-style block + flat network.

During training: each block is `conv3x3(x) + conv1x1(x) + identity(x)` (3-way).
At inference, the three branches can be FUSED into a single 3x3 conv
(parameters merged, no shortcut). For our experiment we keep multi-branch
forward at both train and eval — without BN the multi-branch forward is
mathematically equivalent to the fused single conv, so eval PSNR matches a
deployed fused model exactly.

Reference: Ding et al. "RepVGG: Making VGG-style ConvNets Great Again" (CVPR 2021).
"""

from __future__ import annotations

from typing import Optional, Sequence, Callable
import tensorflow as tf


class RepVGGBlock(tf.keras.layers.Layer):
  """RepVGG conv block: 3×3 + 1×1 + identity at training; equivalent to a
  single fused 3×3 conv at inference.

  No BatchNorm — keeps forward math identical between multi-branch and fused.
  Activation (ReLU) is applied AFTER the branch sum.
  """

  def __init__(self, filters: int, name: str = "repvgg_block"):
    super().__init__(name=name)
    self.filters = filters
    self.conv3x3 = tf.keras.layers.Conv2D(
        filters, 3, padding='same', use_bias=True, name=name+'_conv3x3')
    self.conv1x1 = tf.keras.layers.Conv2D(
        filters, 1, padding='same', use_bias=True, name=name+'_conv1x1')

  def build(self, input_shape):
    self._has_identity = (input_shape[-1] == self.filters)
    super().build(input_shape)

  def call(self, x):
    y = self.conv3x3(x) + self.conv1x1(x)
    if self._has_identity:
      y = y + x
    return tf.nn.relu(y)


class RepVGGFlat(tf.keras.Model):
  """Flat single-scale RepVGG network — no skip connections, no spatial pooling.

  Mirrors the structure of CompressedSkipUNet with 1-encoder-block (which
  collapses to a flat chain), but each conv is replaced by a RepVGGBlock.

  encoder_filters_sequence: list of (only the first is used; multi-block
  encoders not supported in this flat variant — kept for arg uniformity).
  decoder_filters_sequence: 2 entries used (mimics 1-encoder/2-decoder
  collapsed-flat structure).
  """

  def __init__(
      self,
      encoder_filters_sequence: Sequence[int] = (8,),
      decoder_filters_sequence: Sequence[int] = (8, 8),
      output_channels: int = 3,
      output_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      name: str = 'repvgg_flat',
  ):
    super().__init__(name=name)
    assert len(encoder_filters_sequence) == 1, "flat variant uses 1 encoder block"
    assert len(decoder_filters_sequence) == 2, "flat variant uses 2 decoder blocks"

    enc_f = encoder_filters_sequence[0]
    self._enc_block = [
        RepVGGBlock(enc_f, name='enc_repvgg_0'),
        RepVGGBlock(enc_f, name='enc_repvgg_1'),
    ]
    self._dec_blocks = []
    for i, f in enumerate(decoder_filters_sequence):
      self._dec_blocks.append([
          RepVGGBlock(f, name=f'dec{i}_repvgg_0'),
          RepVGGBlock(f, name=f'dec{i}_repvgg_1'),
      ])

    self._output_layer = tf.keras.layers.Conv2D(
        output_channels, 3, padding='same', activation=output_activation,
        name='output')

  def call(self, inputs, *ignored):
    x = inputs
    for blk in self._enc_block:
      x = blk(x)
    for blk_pair in self._dec_blocks:
      for blk in blk_pair:
        x = blk(x)
    return self._output_layer(x)
