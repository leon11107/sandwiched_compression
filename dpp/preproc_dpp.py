"""DPP dilated-conv LUMA-only preprocessor (torch).

Per DPP paper (CVPR2021 Sec 3.2 + 4.1): pixel-to-pixel CNN F(x;Θ) on the LUMA (Y)
channel, input scaled [0,1], dilated convolutions with varying dilation rate per
layer, each conv followed by a PReLU. Sandwich pre_post_models/dilated_residual.py
used as the concrete structural reference (input conv -> dilated residual blocks ->
output conv). Residual + ReZero identity-init for stable training.

Public interface: channels-LAST [B,H,W,3] in [0,255] (matches torch_port codec).
Luma-only: Y = F(Y_in), chroma carried lossless (tf.image BT.601), so the codec
compresses the modified Y and the chroma is preserved (DPP-faithful).
"""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb


class DilatedResidualBlock(nn.Module):
    """x + PReLU(conv2(PReLU(conv1_dilated(x)))) — PReLU per conv (DPP 4.1)."""
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.act1 = nn.PReLU(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act2 = nn.PReLU(ch)

    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.act2(self.conv2(h))
        return x + h


class DilatedLumaNet(nn.Module):
    """Y(1ch) -> luma residual(1ch). Dilated residual CNN, PReLU activations."""
    def __init__(self, ch: int = 64, dilations: Sequence[int] = (1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.in_conv = nn.Conv2d(1, ch, 3, padding=1)
        self.in_act = nn.PReLU(ch)
        self.blocks = nn.ModuleList(DilatedResidualBlock(ch, d) for d in dilations)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)
        # NB: do NOT zero-init out_conv. Identity at init comes from the ReZero
        # scaler=0; if out_conv were also zeroed, net(y)=0 -> scaler grad EXACTLY 0
        # -> preprocessor frozen forever (the zero-gradient trap; see TF [[pytorch-rewrite]]).

    def forward(self, y):  # y [B,1,H,W] in [0,1]
        h = self.in_act(self.in_conv(y))
        for b in self.blocks:
            h = b(h)
        return self.out_conv(h)  # luma residual (same scale as y)


class DPPPreproc(nn.Module):
    """DPP luma-only preprocessor. Input/return channels-LAST [B,H,W,3] in [0,255]."""
    def __init__(self, ch: int = 64, dilations=(1, 2, 4, 8, 1, 2, 4, 8), scaler_init: float = 0.0):
        super().__init__()
        self.net = DilatedLumaNet(ch, dilations)
        self.scaler = nn.Parameter(torch.tensor(float(scaler_init)))  # ReZero identity-init

    def forward(self, inputs):  # [B,H,W,3] in [0,255]
        yuv = tf_rgb_to_yuv(inputs)                  # tf.image BT.601 ([0,1]-style on [0,255])
        y = yuv[..., 0:1]                            # Y in [0,255]
        yn = (y / 255.0).permute(0, 3, 1, 2).contiguous()  # [B,1,H,W] in [0,1]
        res = self.net(yn)
        y_pre = (yn + self.scaler * res).permute(0, 2, 3, 1).contiguous() * 255.0  # [B,H,W,1]
        out_yuv = torch.cat([y_pre, yuv[..., 1:3]], dim=-1)  # modified Y + ORIGINAL chroma
        return tf_yuv_to_rgb(out_yuv)                # [B,H,W,3] in [0,255]
