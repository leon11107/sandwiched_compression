"""Torch full preproc-only model + loss (mirrors compress_intra_model.call +
_distortion_rate_loss, preproc-only / codec_luma_only path).

forward: bottleneck = preproc(x); (dec, rate) = codec(bottleneck);
         if codec_luma_only: dec = yuv_to_rgb([Y(dec), UV(x)]) (tf.image BT.601);
         postproc + loop filter are identity in preproc-only.
loss (per-sample [B]): distortion_fn(gt,pred)*norm + gamma*rate*norm, norm=1/(H*W*C).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from torch_port.preproc import PreprocOnlyTorch, tf_rgb_to_yuv, tf_yuv_to_rgb
from torch_port.codec import EncodeDecodeIntraTorch
from torch_port import losses as L


class PreprocOnlyCodecModel(nn.Module):
    def __init__(self, gamma=0.005, qstep_init=32.0, quantizer_mode="straight_through",
                 codec_forward_mode="real_ste", convert_to_yuv=True,
                 preproc_luma_only=True, codec_luma_only=True, scaler_init=0.0,
                 mean_adjust=128.0, scale_adjust=255.0, device="cpu", dtype=torch.float32):
        super().__init__()
        self.gamma = float(gamma)
        self.codec_luma_only = codec_luma_only
        self.preproc = PreprocOnlyTorch(mean_adjust, scale_adjust, preproc_luma_only, scaler_init)
        self.codec = EncodeDecodeIntraTorch(
            qstep_init=qstep_init, train_qstep=False, min_qstep=1.0,
            quantizer_mode=quantizer_mode, rate_proxy_mode="log_nonzero",
            codec_forward_mode=codec_forward_mode, output_clip_mode="hard",
            convert_to_yuv=convert_to_yuv, device=device, dtype=dtype)
        self.to(device)

    def forward(self, inputs, input_qstep=None, generator=None):
        bottleneck = self.preproc(inputs)
        dec, rate = self.codec(bottleneck, input_qstep=input_qstep, generator=generator)
        if self.codec_luma_only and dec.shape[-1] == 3:
            yuv_dec = tf_rgb_to_yuv(dec)
            yuv_in = tf_rgb_to_yuv(inputs)
            dec = tf_yuv_to_rgb(torch.cat([yuv_dec[..., 0:1], yuv_in[..., 1:3]], dim=-1))
        return {"prediction": dec, "rate": rate, "bottleneck": bottleneck}


def distortion_rate_loss(gt, out, gamma, anchor="l1_msssim", lambda_l2=1.0):
    """Per-sample [B] = distortion*norm + gamma*rate*norm (mirrors _distortion_rate_loss)."""
    pred = out["prediction"]
    norm = pred.shape[0] / float(np.prod(pred.shape))  # 1/(H*W*C)
    if anchor == "l1_msssim":
        dist = L.distortion_l1_msssim(gt, pred, lambda_l2)
    elif anchor == "l2_mean01":
        dist = L.distortion_mse01(gt, pred, lambda_l2)
    elif anchor == "l1_mean01":
        dist = L.distortion_mae01(gt, pred, lambda_l2)
    else:
        raise ValueError(anchor)
    return dist * norm + gamma * out["rate"] * norm
