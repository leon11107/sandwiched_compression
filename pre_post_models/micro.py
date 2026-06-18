"""Micro pre/post architectures aimed at <1KB-params + <1KB-SRAM + ~1K MACs/px
at 64x64 input.

Variants:
  MicroDwsMs   : depthwise-separable + multi-scale (2 stride-2 pool stages)
  MicroBtn     : bottleneck blocks (1x1 down → 3x3 → 1x1 up)
  MicroStd     : pure standard 3x3 conv + multi-scale (small channels)
  MicroGrp     : group conv (multiple parallel groups)

All use ReLU activation, stride-2 downsample (no separate pool), bilinear
upsample, optional 1x1 compressed skip.
"""
from __future__ import annotations

from typing import Optional, Sequence, Callable
import tensorflow as tf


def _conv(out_ch, k=3, stride=1, name=None, use_bias=True):
  return tf.keras.layers.Conv2D(out_ch, k, strides=stride, padding='same',
                                 activation=None, use_bias=use_bias, name=name)


def _dw_separable(out_ch, stride=1, name=None):
  """DW 3x3 + PW 1x1 (output activation only on PW)."""
  return tf.keras.Sequential([
      tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same',
                                       activation='relu',
                                       depthwise_initializer='glorot_uniform',
                                       use_bias=True,
                                       name=(name or 'block') + '_dw'),
      tf.keras.layers.Conv2D(out_ch, 1, padding='same', activation='relu',
                              use_bias=True,
                              name=(name or 'block') + '_pw'),
  ], name=name)


class MicroDwsMs(tf.keras.Model):
  """Multi-scale + DW-separable. Best for <1KB SRAM + <1KB params budget."""

  def __init__(self, ch_enc0=4, ch_enc1=8, ch_dec=4, skip_ch=1,
               output_channels=3, name='micro_dws_ms'):
    super().__init__(name=name)
    # Encoder
    self.enc0_blk = _dw_separable(ch_enc0, stride=1, name='enc0')
    self.skip_compress = _conv(skip_ch, k=1, name='skip_compress')
    # Down stride-2 to half resolution
    self.down = _dw_separable(ch_enc1, stride=2, name='down')
    # Bottleneck
    self.btm = _dw_separable(ch_enc1, stride=1, name='btm')
    # Upsample + skip
    self.up = tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name='up')
    self.dec_blk = _dw_separable(ch_dec, stride=1, name='dec')
    # Output
    self.out_conv = _conv(output_channels, k=3, name='output')

  def call(self, x, *ignored):
    e0 = self.enc0_blk(x)
    skip = self.skip_compress(e0)
    d = self.down(e0)
    b = self.btm(d)
    u = self.up(b)
    cat = tf.concat([u, skip], axis=-1)
    dec = self.dec_blk(cat)
    return self.out_conv(dec)


class MicroBtn(tf.keras.Model):
  """Bottleneck blocks: each block = 1x1 down → 3x3 → 1x1 up."""

  def __init__(self, width=4, btm_width=2, skip_ch=1,
               output_channels=3, name='micro_btn'):
    super().__init__(name=name)
    def btn_block(out_ch, n):
      return tf.keras.Sequential([
          tf.keras.layers.Conv2D(btm_width, 1, padding='same', activation='relu', name=f'{n}_in'),
          tf.keras.layers.Conv2D(btm_width, 3, padding='same', activation='relu', name=f'{n}_mid'),
          tf.keras.layers.Conv2D(out_ch, 1, padding='same', activation='relu', name=f'{n}_out'),
      ], name=n)
    self.enc0 = btn_block(width, 'enc0_btn')
    self.skip = _conv(skip_ch, k=1, name='skip')
    self.down = _conv(width * 2, k=3, stride=2, name='down')
    self.btm = btn_block(width * 2, 'btm')
    self.up = tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name='up')
    self.dec = btn_block(width, 'dec_btn')
    self.out_conv = _conv(output_channels, k=3, name='output')

  def call(self, x, *ignored):
    e0 = self.enc0(x)
    sk = self.skip(e0)
    d = self.down(e0)
    b = self.btm(d)
    u = self.up(b)
    cat = tf.concat([u, sk], axis=-1)
    dec = self.dec(cat)
    return self.out_conv(dec)


class MicroStd(tf.keras.Model):
  """Standard 3x3 multi-scale, tiny channels."""

  def __init__(self, ch_enc0=4, ch_enc1=8, ch_dec=4, skip_ch=1,
               output_channels=3, name='micro_std'):
    super().__init__(name=name)
    self.enc0_a = _conv(ch_enc0, k=3, name='enc0_a')
    self.enc0_act = tf.keras.layers.ReLU(name='enc0_relu')
    self.skip_compress = _conv(skip_ch, k=1, name='skip_compress')
    self.down = _conv(ch_enc1, k=3, stride=2, name='down')
    self.down_act = tf.keras.layers.ReLU(name='down_relu')
    self.btm = _conv(ch_enc1, k=3, name='btm')
    self.btm_act = tf.keras.layers.ReLU(name='btm_relu')
    self.up = tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name='up')
    self.dec = _conv(ch_dec, k=3, name='dec')
    self.dec_act = tf.keras.layers.ReLU(name='dec_relu')
    self.out_conv = _conv(output_channels, k=3, name='output')

  def call(self, x, *ignored):
    e0 = self.enc0_act(self.enc0_a(x))
    sk = self.skip_compress(e0)
    d = self.down_act(self.down(e0))
    b = self.btm_act(self.btm(d))
    u = self.up(b)
    cat = tf.concat([u, sk], axis=-1)
    dec = self.dec_act(self.dec(cat))
    return self.out_conv(dec)


class MicroDwsMs2(tf.keras.Model):
  """DW separable + multi-scale, smaller btm channels to fit ≤25KB peak.
  Default: ch=4,4,4 — peak = 64*64*5 (concat) = 20KB."""

  def __init__(self, ch_enc0=4, ch_btm=4, ch_dec=4, skip_ch=1,
               output_channels=3, name='micro_dws_ms_v2'):
    super().__init__(name=name)
    self.enc0_blk = _dw_separable(ch_enc0, stride=1, name='enc0')
    self.skip_compress = _conv(skip_ch, k=1, name='skip_compress')
    self.down = _dw_separable(ch_btm, stride=2, name='down')
    self.btm = _dw_separable(ch_btm, stride=1, name='btm')
    self.up = tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name='up')
    self.dec_blk = _dw_separable(ch_dec, stride=1, name='dec')
    self.out_conv = _conv(output_channels, k=3, name='output')

  def call(self, x, *ignored):
    e0 = self.enc0_blk(x)
    skip = self.skip_compress(e0)
    d = self.down(e0)
    b = self.btm(d)
    u = self.up(b)
    cat = tf.concat([u, skip], axis=-1)
    dec = self.dec_blk(cat)
    return self.out_conv(dec)


class SlimMSTiny(tf.keras.Model):
  """Paper-style small multi-scale U-Net with full skips.
  Default: enc=(4, 8), dec=(16, 8, 4) — peak controlled by max channels at top level."""

  def __init__(self, enc_filters=(4, 8), dec_filters=(16, 8, 4),
               output_channels=3, name='slim_ms_tiny'):
    super().__init__(name=name)
    assert len(dec_filters) == len(enc_filters) + 1
    self._enc_convs = []
    for i, f in enumerate(enc_filters):
      self._enc_convs.append([
          _conv(f, k=3, name=f'enc_{i}_conv_0'),
          tf.keras.layers.ReLU(name=f'enc_{i}_relu_0'),
          _conv(f, k=3, name=f'enc_{i}_conv_1'),
          tf.keras.layers.ReLU(name=f'enc_{i}_relu_1'),
      ])
    self._dec_convs = []
    self._ups = []
    for i, f in enumerate(dec_filters):
      self._dec_convs.append([
          _conv(f, k=3, name=f'dec_{i}_conv_0'),
          tf.keras.layers.ReLU(name=f'dec_{i}_relu_0'),
          _conv(f, k=3, name=f'dec_{i}_conv_1'),
          tf.keras.layers.ReLU(name=f'dec_{i}_relu_1'),
      ])
      if i < len(dec_filters) - 1:
        self._ups.append(tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name=f'dec_{i}_up'))
    self._pool = tf.keras.layers.AveragePooling2D(2, name='pool')  # avg pool stride 2
    self._output_layer = _conv(output_channels, k=3, name='output')

  def call(self, x, *ignored):
    skips = []
    for i, layers in enumerate(self._enc_convs):
      for l in layers:
        x = l(x)
      skips.append(x)
      x = self._pool(x)
    skips.append(None)
    num_skips = len(skips)
    for i, layers in enumerate(self._dec_convs):
      for l in layers:
        x = l(x)
      skip = skips[num_skips - 1 - i]
      if skip is not None:
        # Upsample index: i-th decoder block's upsample is self._ups[i-1] since
        # _ups is built for i < num_dec - 1 BUT applied after first dec block.
        # Map: dec block i ≥ 1 uses up[i-1]. dec block 0 has no upsample (no skip).
        x = self._ups[i - 1](x)
        x = tf.concat([x, skip], axis=-1)
    return self._output_layer(x)


class MicroGrp(tf.keras.Model):
  """Group conv: split channels into groups, conv each group independently."""

  def __init__(self, ch_enc0=4, ch_enc1=8, ch_dec=4, groups=2, skip_ch=1,
               output_channels=3, name='micro_grp'):
    super().__init__(name=name)
    # Standard first conv (input 3 channels, can't group easily)
    self.enc0 = _conv(ch_enc0, k=3, name='enc0')
    self.enc0_act = tf.keras.layers.ReLU(name='enc0_relu')
    self.skip = _conv(skip_ch, k=1, name='skip')
    self.down = tf.keras.layers.Conv2D(ch_enc1, 3, strides=2, padding='same',
                                        groups=groups, activation='relu', name='down_grp')
    self.btm = tf.keras.layers.Conv2D(ch_enc1, 3, padding='same', groups=groups,
                                       activation='relu', name='btm_grp')
    self.up = tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name='up')
    self.dec = _conv(ch_dec, k=3, name='dec')
    self.dec_act = tf.keras.layers.ReLU(name='dec_relu')
    self.out_conv = _conv(output_channels, k=3, name='output')

  def call(self, x, *ignored):
    e0 = self.enc0_act(self.enc0(x))
    sk = self.skip(e0)
    d = self.down(e0)
    b = self.btm(d)
    u = self.up(b)
    cat = tf.concat([u, sk], axis=-1)
    dec = self.dec_act(self.dec(cat))
    return self.out_conv(dec)
