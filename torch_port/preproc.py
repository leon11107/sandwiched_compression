"""PyTorch port of the paper_unet preprocessor (pre_post_models/unet.py +
compress_intra_model.run_preprocessor, preproc-only / num_mlp_layers=0 path).

residual form (mlp = identity):
    adj = (x - 128) / 255
    out = 255 * (adj + scaler * unet(adj)) + 128
    if preproc_luma_only: keep unet'd Y, original chroma (tf.image BT.601, [0,1]-style)

Internals run channels-FIRST (B,C,H,W) for torch convs; the public interface is
channels-LAST (B,H,W,3) to match the codec port. Weights are ported from the TF
checkpoint (Conv2D kernel [kh,kw,in,out] -> torch [out,in,kh,kw]).
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# tf.image.rgb_to_yuv / yuv_to_rgb kernels (ITU-R BT.601, U/V centered at 0).
_RGB2YUV = torch.tensor(
    [[0.299, -0.14714119, 0.61497538],
     [0.587, -0.28886916, -0.51496512],
     [0.114, 0.43601035, -0.10001026]], dtype=torch.float32)
_YUV2RGB = torch.tensor(
    [[1.0, 1.0, 1.0],
     [0.0, -0.394642334, 2.03206185],
     [1.13988303, -0.58062185, 0.0]], dtype=torch.float32)


def tf_rgb_to_yuv(x):  # x [...,3]
    return torch.matmul(x, _RGB2YUV.to(x.device, x.dtype))


def tf_yuv_to_rgb(x):
    return torch.matmul(x, _YUV2RGB.to(x.device, x.dtype))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, num_convs, num_filters):
        super().__init__()
        convs = []
        c = in_ch
        for _ in range(num_convs):
            convs.append(nn.Conv2d(c, num_filters, 3, 1, padding=1))
            c = num_filters
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        for cv in self.convs:
            x = F.relu(cv(x))
        pooled = x[:, :, 1::2, 1::2]  # tf: inputs[:,1:h:2,1:w:2,:]
        return pooled, x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, num_convs, num_filters):
        super().__init__()
        convs = []
        c = in_ch
        for _ in range(num_convs):
            convs.append(nn.Conv2d(c, num_filters, 3, 1, padding=1))
            c = num_filters
        self.convs = nn.ModuleList(convs)

    def forward(self, x, skip):
        for cv in self.convs:
            x = F.relu(cv(x))
        if skip is not None:
            # tf.keras UpSampling2D(bilinear) == tf.image.resize half-pixel == torch
            # interpolate align_corners=False (validated in eq harness).
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return x


class UNetTorch(nn.Module):
    def __init__(self, enc_filters=(32, 64, 128, 256),
                 dec_filters=(512, 256, 128, 64, 32), convs_per_block=2,
                 out_channels=3, in_channels=3):
        super().__init__()
        self.encoders = nn.ModuleList()
        c = in_channels
        for f in enc_filters:
            self.encoders.append(EncoderBlock(c, convs_per_block, f))
            c = f  # pooled has f channels
        # decoder input channels: dec0 takes pooled last encoder (enc_filters[-1]);
        # subsequent decoders take previous decoder output (= 2*dec_filters[i-1] after
        # concat, except dec0 which has no concat).
        self.decoders = nn.ModuleList()
        din = enc_filters[-1]  # 256 into dec0
        skip_ch = list(enc_filters)  # [32,64,128,256], used reversed for concat
        for i, f in enumerate(dec_filters):
            self.decoders.append(DecoderBlock(din, convs_per_block, f))
            if i == 0:
                din = f  # dec0: no upsample/concat -> output f
            else:
                # upsample(f) concat skip(enc_filters[-(i)] ) -> f + skip
                din = f + skip_ch[len(enc_filters) - i]
        self.out = nn.Conv2d(din, out_channels, 3, 1, padding=1)

    def forward(self, x):  # x [B,C,H,W]
        skips = []
        out = x
        for enc in self.encoders:
            out, s = enc(out)
            skips.append(s)
        skips.append(None)
        n = len(skips)
        for i, dec in enumerate(self.decoders):
            out = dec(out, skips[n - 1 - i])
        return self.out(out)


class PreprocOnlyTorch(nn.Module):
    """Residual paper_unet preprocessor (channels-last public interface)."""
    def __init__(self, mean_adjust=128.0, scale_adjust=255.0, preproc_luma_only=True,
                 scaler_init=0.0):
        super().__init__()
        self.unet = UNetTorch()
        self.scaler = nn.Parameter(torch.tensor(float(scaler_init)))
        self.mean_adjust = float(mean_adjust)
        self.scale_adjust = float(scale_adjust)
        self.preproc_luma_only = preproc_luma_only

    def forward(self, inputs):  # inputs [B,H,W,3] in [0,255]
        adj = (inputs - self.mean_adjust) / self.scale_adjust
        adj_cf = adj.permute(0, 3, 1, 2).contiguous()
        u = self.unet(adj_cf).permute(0, 2, 3, 1).contiguous()  # [B,H,W,3]
        out = self.scale_adjust * (adj + self.scaler * u) + self.mean_adjust
        if self.preproc_luma_only and out.shape[-1] == 3:
            y_pre = tf_rgb_to_yuv(out)[..., 0:1]
            uv_in = tf_rgb_to_yuv(inputs)[..., 1:3]
            out = tf_yuv_to_rgb(torch.cat([y_pre, uv_in], dim=-1))
        return out


# -----------------------------------------------------------------------------
# Weight porter: TF get_weights() ordered list -> torch state_dict.
# TF preprocessor weight order (from probe): 19 (kernel,bias) conv pairs in
# enc0..enc3 (2 each), dec0..dec4 (2 each), output. plus the scaler (separate).
# -----------------------------------------------------------------------------
def load_unet_weights(unet: UNetTorch, tf_weights: List):
    """tf_weights: list of np arrays [k0,b0,k1,b1,...,kout,bout] (38 arrays)."""
    convs = []
    for enc in unet.encoders:
        convs.extend(list(enc.convs))
    for dec in unet.decoders:
        convs.extend(list(dec.convs))
    convs.append(unet.out)
    assert len(tf_weights) == 2 * len(convs), \
        f"got {len(tf_weights)} tf arrays for {len(convs)} convs"
    with torch.no_grad():
        for i, cv in enumerate(convs):
            k = torch.from_numpy(tf_weights[2 * i])      # [kh,kw,in,out]
            b = torch.from_numpy(tf_weights[2 * i + 1])  # [out]
            cv.weight.copy_(k.permute(3, 2, 0, 1).contiguous())
            cv.bias.copy_(b)
