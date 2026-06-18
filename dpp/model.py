"""DPP full model: DPP dilated-conv luma preprocessor -> validated torch_port codec
-> codec_luma_only (lossless chroma) -> prediction + rate. Reuses the TF-equivalence
-validated torch_port codec as the JPEG substrate."""
from __future__ import annotations
import torch
import torch.nn as nn

from dpp.preproc_dpp import DPPPreproc
from torch_port.codec import EncodeDecodeIntraTorch
from torch_port.preproc import tf_rgb_to_yuv, tf_yuv_to_rgb


class DPPModel(nn.Module):
    def __init__(self, ch=64, dilations=(1, 2, 4, 8, 1, 2, 4, 8), scaler_init=0.0,
                 qstep_init=32.0, quantizer_mode="noise_injection",
                 codec_forward_mode="real_ste", device="cpu", dtype=torch.float32):
        super().__init__()
        self.preproc = DPPPreproc(ch=ch, dilations=dilations, scaler_init=scaler_init)
        self.codec = EncodeDecodeIntraTorch(
            qstep_init=qstep_init, train_qstep=False, min_qstep=1.0,
            quantizer_mode=quantizer_mode, rate_proxy_mode="log_nonzero",
            codec_forward_mode=codec_forward_mode, output_clip_mode="hard",
            convert_to_yuv=True, device=device, dtype=dtype)
        self.to(device)

    def forward(self, inputs, input_qstep=None, generator=None):  # inputs [B,H,W,3] 0..255
        bottleneck = self.preproc(inputs)                 # luma-modified, lossless chroma
        dec, rate = self.codec(bottleneck, input_qstep=input_qstep, generator=generator)
        # codec_luma_only: keep decoded Y, restore ORIGINAL chroma (DPP lossless UV)
        yuv_dec = tf_rgb_to_yuv(dec); yuv_in = tf_rgb_to_yuv(inputs)
        pred = tf_yuv_to_rgb(torch.cat([yuv_dec[..., 0:1], yuv_in[..., 1:3]], dim=-1))
        return {"prediction": pred, "rate": rate, "bottleneck": bottleneck}
