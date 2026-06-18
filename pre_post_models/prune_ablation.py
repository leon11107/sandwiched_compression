"""Pruning-method ablation pre/post architectures.

All target ~3K params per side (3x3 chain at 1 enc block + 2 dec blocks),
varying ONLY the conv block style. Compare:
  A: flat_4 (channel cut) — CompressedSkipUNet(4-ch)
  B: repvgg_4 (multi-branch)
  C: DwsFlat — DW separable per layer (same 1+2 block layout)
  D: BtnFlat — bottleneck blocks (1x1 down → 3x3 → 1x1 up)
  E: MsTiny3K — multi-scale + skip, very small channels
  F: GrpFlat — group conv

All use 1 encoder block (no internal pool), 2 decoder blocks, no skip (since
single encoder block produces no skip), output 3-ch.
"""
from __future__ import annotations

from typing import Optional, Sequence, Callable
import tensorflow as tf


class DwsFlat(tf.keras.Model):
  """Flat 1-encoder DW separable: each 3x3 conv → DW3x3 + PW1x1."""

  def __init__(self, main_ch=6, output_channels=3, name='dws_flat'):
    super().__init__(name=name)
    self.c = main_ch
    def dws_block(out_ch, n):
      return [
          tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='relu',
                                           name=f'{n}_dw'),
          tf.keras.layers.Conv2D(out_ch, 1, padding='same', activation='relu',
                                  name=f'{n}_pw'),
      ]
    # 2 enc convs + 4 dec convs + output = 7 stages
    self.enc = dws_block(main_ch, 'enc_0_a') + dws_block(main_ch, 'enc_0_b')
    self.dec0 = dws_block(main_ch, 'dec_0_a') + dws_block(main_ch, 'dec_0_b')
    self.dec1 = dws_block(main_ch, 'dec_1_a') + dws_block(main_ch, 'dec_1_b')
    self.out_conv = tf.keras.layers.Conv2D(output_channels, 3, padding='same',
                                            name='output')

  def call(self, x, *ignored):
    for l in self.enc + self.dec0 + self.dec1:
      x = l(x)
    return self.out_conv(x)


class BtnFlat(tf.keras.Model):
  """Flat 1-encoder bottleneck: each "conv" replaced by 1x1 down→3x3→1x1 up."""

  def __init__(self, main_ch=6, btm_ch=3, output_channels=3, name='btn_flat'):
    super().__init__(name=name)
    def btn_block(out_ch, n):
      return [
          tf.keras.layers.Conv2D(btm_ch, 1, padding='same', activation='relu',
                                  name=f'{n}_in'),
          tf.keras.layers.Conv2D(btm_ch, 3, padding='same', activation='relu',
                                  name=f'{n}_mid'),
          tf.keras.layers.Conv2D(out_ch, 1, padding='same', activation='relu',
                                  name=f'{n}_out'),
      ]
    self.enc = btn_block(main_ch, 'enc_0_a') + btn_block(main_ch, 'enc_0_b')
    self.dec0 = btn_block(main_ch, 'dec_0_a') + btn_block(main_ch, 'dec_0_b')
    self.dec1 = btn_block(main_ch, 'dec_1_a') + btn_block(main_ch, 'dec_1_b')
    self.out_conv = tf.keras.layers.Conv2D(output_channels, 3, padding='same',
                                            name='output')

  def call(self, x, *ignored):
    for l in self.enc + self.dec0 + self.dec1:
      x = l(x)
    return self.out_conv(x)


class GrpFlat(tf.keras.Model):
  """Flat 1-encoder group conv: each 3x3 conv uses group convs."""

  def __init__(self, main_ch=6, groups=2, output_channels=3, name='grp_flat'):
    super().__init__(name=name)
    def gconv(out_ch, n):
      return tf.keras.layers.Conv2D(out_ch, 3, padding='same', activation='relu',
                                     groups=groups, name=n)
    # First conv: groups=1 (input 3 channels not divisible)
    self.enc_in = tf.keras.layers.Conv2D(main_ch, 3, padding='same',
                                          activation='relu', name='enc_in')
    self.enc_b = gconv(main_ch, 'enc_b')
    self.dec0_a = gconv(main_ch, 'dec_0_a')
    self.dec0_b = gconv(main_ch, 'dec_0_b')
    self.dec1_a = gconv(main_ch, 'dec_1_a')
    self.dec1_b = gconv(main_ch, 'dec_1_b')
    self.out_conv = tf.keras.layers.Conv2D(output_channels, 3, padding='same',
                                            name='output')

  def call(self, x, *ignored):
    x = self.enc_in(x)
    x = self.enc_b(x)
    x = self.dec0_a(x)
    x = self.dec0_b(x)
    x = self.dec1_a(x)
    x = self.dec1_b(x)
    return self.out_conv(x)


class FlatStd(tf.keras.Model):
  """Pure channel-cut baseline: 1-block flat CompressedSkipUNet at main_ch=6,
  re-implemented here for clarity. Same as flat_4 structure but 6 channels."""

  def __init__(self, main_ch=6, output_channels=3, name='flat_std'):
    super().__init__(name=name)
    self.c = main_ch
    self.enc_a = tf.keras.layers.Conv2D(main_ch, 3, padding='same', activation='relu', name='enc_0_a')
    self.enc_b = tf.keras.layers.Conv2D(main_ch, 3, padding='same', activation='relu', name='enc_0_b')
    self.dec0_a = tf.keras.layers.Conv2D(main_ch, 3, padding='same', activation='relu', name='dec_0_a')
    self.dec0_b = tf.keras.layers.Conv2D(main_ch, 3, padding='same', activation='relu', name='dec_0_b')
    self.dec1_a = tf.keras.layers.Conv2D(main_ch, 3, padding='same', activation='relu', name='dec_1_a')
    self.dec1_b = tf.keras.layers.Conv2D(main_ch, 3, padding='same', activation='relu', name='dec_1_b')
    self.out_conv = tf.keras.layers.Conv2D(output_channels, 3, padding='same', name='output')

  def call(self, x, *ignored):
    x = self.enc_a(x); x = self.enc_b(x)
    x = self.dec0_a(x); x = self.dec0_b(x)
    x = self.dec1_a(x); x = self.dec1_b(x)
    return self.out_conv(x)


class MsCompressed(tf.keras.Model):
  """Multi-scale with compressed skip to keep top-level peak SRAM low.
  Encoder: 4-ch top → pool → 8-ch mid → pool → 16-ch btm
  Decoder: upsample + 1x1 compressed skip (1 channel) per level
  Peak SRAM (64x64): max(64*64*4, 32*32*8, 16*16*16, concat 64*64*(4+1)=5) = 20 KB
  """

  def __init__(self, top_ch=4, mid_ch=8, btm_ch=16, skip_ch=1,
               output_channels=3, name='ms_csk'):
    super().__init__(name=name)
    # Encoder
    self.enc_top_a = tf.keras.layers.Conv2D(top_ch, 3, padding='same', activation='relu', name='enc_top_a')
    self.enc_top_b = tf.keras.layers.Conv2D(top_ch, 3, padding='same', activation='relu', name='enc_top_b')
    self.skip_top = tf.keras.layers.Conv2D(skip_ch, 1, padding='same', name='skip_top')
    self.pool1 = tf.keras.layers.AveragePooling2D(2, name='pool1')
    self.enc_mid_a = tf.keras.layers.Conv2D(mid_ch, 3, padding='same', activation='relu', name='enc_mid_a')
    self.enc_mid_b = tf.keras.layers.Conv2D(mid_ch, 3, padding='same', activation='relu', name='enc_mid_b')
    self.skip_mid = tf.keras.layers.Conv2D(skip_ch, 1, padding='same', name='skip_mid')
    self.pool2 = tf.keras.layers.AveragePooling2D(2, name='pool2')
    # Bottleneck
    self.btm_a = tf.keras.layers.Conv2D(btm_ch, 3, padding='same', activation='relu', name='btm_a')
    self.btm_b = tf.keras.layers.Conv2D(btm_ch, 3, padding='same', activation='relu', name='btm_b')
    # Decoder
    self.up1 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name='up1')
    self.dec_mid_a = tf.keras.layers.Conv2D(mid_ch, 3, padding='same', activation='relu', name='dec_mid_a')
    self.dec_mid_b = tf.keras.layers.Conv2D(mid_ch, 3, padding='same', activation='relu', name='dec_mid_b')
    self.up2 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name='up2')
    self.dec_top_a = tf.keras.layers.Conv2D(top_ch, 3, padding='same', activation='relu', name='dec_top_a')
    self.dec_top_b = tf.keras.layers.Conv2D(top_ch, 3, padding='same', activation='relu', name='dec_top_b')
    self.out_conv = tf.keras.layers.Conv2D(output_channels, 3, padding='same', name='output')

  def call(self, x, *ignored):
    e_top = self.enc_top_b(self.enc_top_a(x))
    s_top = self.skip_top(e_top)
    e_mid = self.pool1(e_top)
    e_mid = self.enc_mid_b(self.enc_mid_a(e_mid))
    s_mid = self.skip_mid(e_mid)
    b = self.pool2(e_mid)
    b = self.btm_b(self.btm_a(b))
    u = self.up1(b)
    u = tf.concat([u, s_mid], axis=-1)
    u = self.dec_mid_b(self.dec_mid_a(u))
    u = self.up2(u)
    u = tf.concat([u, s_top], axis=-1)
    u = self.dec_top_b(self.dec_top_a(u))
    return self.out_conv(u)


class MsTiny3K(tf.keras.Model):
  """Multi-scale + compressed skip, sized to ~3K params/side."""

  def __init__(self, enc_filters=(2, 4), dec_filters=(8, 4, 2),
               output_channels=3, name='ms_tiny_3k'):
    super().__init__(name=name)
    assert len(dec_filters) == len(enc_filters) + 1
    self._enc_convs = []
    for i, f in enumerate(enc_filters):
      self._enc_convs.append([
          tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu', name=f'enc_{i}_a'),
          tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu', name=f'enc_{i}_b'),
      ])
    self._dec_convs = []
    self._ups = []
    for i, f in enumerate(dec_filters):
      self._dec_convs.append([
          tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu', name=f'dec_{i}_a'),
          tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu', name=f'dec_{i}_b'),
      ])
      if i < len(dec_filters) - 1:
        self._ups.append(tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name=f'dec_{i}_up'))
    self._pool = tf.keras.layers.AveragePooling2D(2, name='pool')
    self._out = tf.keras.layers.Conv2D(output_channels, 3, padding='same', name='output')

  def call(self, x, *ignored):
    skips = []
    for layers in self._enc_convs:
      for l in layers: x = l(x)
      skips.append(x)
      x = self._pool(x)
    skips.append(None)
    n = len(skips)
    for i, layers in enumerate(self._dec_convs):
      for l in layers: x = l(x)
      skip = skips[n - 1 - i]
      if skip is not None:
        x = self._ups[i - 1](x)
        x = tf.concat([x, skip], axis=-1)
    return self._out(x)
