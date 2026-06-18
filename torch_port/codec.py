"""PyTorch port of image_compression/{jpeg_proxy,encode_decode_intra_lib}.py.

Channels-LAST (B,H,W,C) internally to mirror the TF reference op-for-op (the DCT
patch ordering, BT.601 offset-128 color transform, and rate proxy must match
exactly). Ported config path: convert_to_yuv, 4:4:4 (no chroma downsample),
rate_proxy_mode='log_nonzero' with use_jpeg_rate_model, codec_forward_mode in
{'proxy','real_ste'}, quantizer modes {straight_through, noise_injection}.

TF parity notes:
- DCT-II unit-norm basis, identical construction; forward subtracts 128, inverse
  adds 128 (JpegProxy._construct_dct_2d / _forward_dct_2d / _inverse_dct_2d).
- 8x8 non-overlapping patches flattened ROW-MAJOR (pixel (r,c) -> index r*8+c),
  matching tf.image.extract_patches with size=stride=8, VALID.
- real_ste / real-bit rate use the verbatim PIL codec (lib-agnostic, bit-exact).
"""
from __future__ import annotations
import io
import itertools
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile

Image.init()
ImageFile.MAXBLOCK = 2 ** 27  # match TF-side: avoid "broken data stream" on corner cases


# -----------------------------------------------------------------------------
# Verbatim PIL real codec (numpy/PIL; identical to encode_decode_intra_lib).
# -----------------------------------------------------------------------------
def encode_decode_with_jpeg(
    input_images: np.ndarray,
    qstep: float,
    one_channel_at_a_time: bool = False,
    use_420: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    assert input_images.ndim == 4
    decoded = np.zeros_like(input_images)
    rate = np.zeros(input_images.shape[0])
    jpeg_qstep = int(np.clip(np.rint(qstep).astype(int), 0, 255))
    qtable = [jpeg_qstep] * 64

    def run_jpeg(input_image: np.ndarray):
        img = Image.fromarray(np.rint(np.clip(input_image, 0, 255)).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="jpeg", optimize=True,
                 qtables=[qtable, qtable, qtable],
                 subsampling="4:2:0" if use_420 else "4:4:4")
        return np.array(Image.open(buf)), np.array(8 * len(buf.getbuffer()))

    for i in range(input_images.shape[0]):
        if not one_channel_at_a_time:
            decoded[i], rate[i] = run_jpeg(input_images[i])
        else:
            for ch in range(input_images.shape[-1]):
                decoded[i, ..., ch], cr = run_jpeg(input_images[i, ..., ch])
                rate[i] += cr
    return decoded.astype(np.float32), rate.astype(np.float32)


# -----------------------------------------------------------------------------
# Quantizer rounding fns (mirror compress_intra_model.{differentiable_round,
# noise_injection_round}). quantizer applied as rounding_fn(x/qstep)*qstep.
# -----------------------------------------------------------------------------
def ste_round(x: torch.Tensor) -> torch.Tensor:
    """straight_through: forward round, backward identity."""
    return x + (torch.round(x) - x).detach()


def polynomial_round(x: torch.Tensor) -> torch.Tensor:
    """diff-JPEG: round(x) + (x-round(x))^3. fwd~round (|err|<=.125), grad 3(x-round)^2."""
    r = torch.round(x)
    return r + (x - r) ** 3


def ste_polynomial_round(x: torch.Tensor) -> torch.Tensor:
    """forward = exact round, backward = polynomial surrogate grad 3(x-round)^2."""
    poly = polynomial_round(x)
    return torch.round(x) + (poly - poly.detach())


def make_noise_injection_round(training: bool, generator: Optional[torch.Generator] = None):
    """noise_injection: train forward x+U(-.5,.5) (grad 1), infer forward round."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        if training:
            noise = (torch.rand(x.shape, dtype=x.dtype, device=x.device,
                                generator=generator) - 0.5)
            return x + noise
        return torch.round(x)
    return fn


_ROUNDERS = {"straight_through": ste_round}


# -----------------------------------------------------------------------------
# JpegProxy (DCT block codec) — torch port of jpeg_proxy.JpegProxy.
# -----------------------------------------------------------------------------
class JpegProxyTorch:
    def __init__(self, convert_to_yuv: bool, clip_to_image_max: bool,
                 dct_size: int = 8, device="cpu", dtype=torch.float32):
        self.convert_to_yuv = convert_to_yuv
        self.clip_to_image_max = clip_to_image_max
        self.dct_size = dct_size
        self.device, self.dtype = device, dtype
        # BT.601 full-range matrices (identical constants to TF).
        self.rgb_from_yuv = torch.tensor(
            [[1.0, 1.0, 1.0], [0, -0.344136, 1.772], [1.402, -0.714136, 0]],
            dtype=dtype, device=device)
        self.yuv_from_rgb = torch.tensor(
            [[0.299, -0.168736, 0.5], [0.587, -0.331264, -0.418688],
             [0.114, 0.5, -0.081312]], dtype=dtype, device=device)
        self.dct_2d = self._construct_dct_2d(dct_size).to(device=device, dtype=dtype)

    def _construct_dct_2d(self, n: int) -> torch.Tensor:
        m = np.zeros((n, n), dtype=np.float32)
        for i, j in itertools.product(range(n), repeat=2):
            m[i, j] = np.cos((2 * i + 1) * j * np.pi / (2 * n))
        m *= np.sqrt(2 / n)
        m[:, 0] *= 1 / np.sqrt(2)
        bs = n * n
        d2 = np.zeros((bs, bs), dtype=np.float32)
        for i in range(bs):
            d2[:, i] = np.reshape(np.outer(m[:, i // n], m[:, i % n]), [-1])
        return torch.from_numpy(d2)

    def _rgb_to_yuv(self, rgb):  # [...,3]
        off = torch.tensor([0, 128, 128], dtype=self.dtype, device=rgb.device)
        return torch.matmul(rgb, self.yuv_from_rgb) + off

    def _yuv_to_rgb(self, yuv):
        off = torch.tensor([0, 128, 128], dtype=self.dtype, device=yuv.device)
        return torch.matmul(yuv - off, self.rgb_from_yuv)

    def _forward_dct_2d(self, ch):  # ch: [B,H,W,1] -> [B,H/8,W/8,64]
        B, H, W, _ = ch.shape
        n = self.dct_size
        # non-overlapping 8x8 patches, flattened ROW-MAJOR (r*8+c) to match TF.
        p = ch.reshape(B, H // n, n, W // n, n).permute(0, 1, 3, 2, 4).reshape(
            B, H // n, W // n, n * n)
        return torch.matmul(p - 128.0, self.dct_2d)

    def _inverse_dct_2d(self, coeffs):  # [B,H/8,W/8,64] -> [B,H,W,1]
        B, Hb, Wb, _ = coeffs.shape
        n = self.dct_size
        ch = torch.matmul(coeffs, self.dct_2d.t()) + 128.0  # [B,Hb,Wb,64]
        ch = ch.reshape(B, Hb, Wb, n, n).permute(0, 1, 3, 2, 4).reshape(
            B, Hb * n, Wb * n, 1)
        return ch

    def __call__(self, image, rounding_fn: Callable = torch.round, image_max=255.0):
        """image: [B,H,W,3] in [0,255]. Returns (decoded[B,H,W,3], coeffs dict)."""
        assert image.shape[-1] == 3
        H, W = image.shape[1:3]
        assert H % self.dct_size == 0 and W % self.dct_size == 0, \
            "torch port assumes multiple-of-8 (our crops are 128/256); pad path TODO"
        if self.convert_to_yuv:
            image = self._rgb_to_yuv(image)
        keys = ["y", "u", "v"]
        coeffs = {}
        dec = []
        for ch in range(3):
            c = image[..., ch:ch + 1]
            co = self._forward_dct_2d(c)            # qtable == 1 (flat)
            coeffs[keys[ch]] = rounding_fn(co)      # quantize (qtable=1)
            deq = coeffs[keys[ch]]                  # dequantize (*1)
            dec.append(self._inverse_dct_2d(deq))
        decoded = torch.cat(dec, dim=-1)
        if self.convert_to_yuv:
            decoded = self._yuv_to_rgb(decoded)
        decoded = decoded[:, 0:H, 0:W, :]
        if self.clip_to_image_max:
            decoded = torch.clamp(decoded, 0.0, image_max)
        return decoded, coeffs


# -----------------------------------------------------------------------------
# EncodeDecodeIntra — torch port (config path).
# -----------------------------------------------------------------------------
class EncodeDecodeIntraTorch(nn.Module):
    def __init__(self, qstep_init=32.0, train_qstep=False, min_qstep=1.0,
                 quantizer_mode="straight_through", rate_proxy_mode="log_nonzero",
                 rate_proxy_grad_scale=1.0, codec_forward_mode="real_ste",
                 output_clip_mode="hard", convert_to_yuv=True,
                 downsample_chroma=False, jpeg_clip_to_image_max=True,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        assert not downsample_chroma, "4:4:4 only in this port"
        assert rate_proxy_mode == "log_nonzero", "ported rate mode"
        assert codec_forward_mode in ("proxy", "real_ste")
        self.train_qstep = train_qstep
        if train_qstep:
            self.qstep = nn.Parameter(torch.tensor(float(qstep_init), dtype=dtype))
        else:
            self.register_buffer("qstep", torch.tensor(float(qstep_init), dtype=dtype))
        self.min_qstep = float(min_qstep)
        self.quantizer_mode = quantizer_mode
        self.rate_proxy_grad_scale = float(rate_proxy_grad_scale)
        self.codec_forward_mode = codec_forward_mode
        self.output_clip_mode = output_clip_mode
        self.convert_to_yuv = convert_to_yuv
        # convert_to_yuv -> run real jpeg as one 3ch image; else one channel at a time
        self.run_jpeg_one_channel_at_a_time = not convert_to_yuv
        clip = jpeg_clip_to_image_max if output_clip_mode == "hard" else False
        self.proxy = JpegProxyTorch(convert_to_yuv, clip, device=device, dtype=dtype)

    def positive_qstep(self) -> torch.Tensor:
        # tf.keras.activations.elu(qstep, alpha=0.01) + min_qstep
        q = self.qstep
        elu = torch.where(q > 0, q, 0.01 * (torch.expm1(q)))
        return elu + self.min_qstep

    def _rounding_fn(self, generator=None):
        if self.quantizer_mode == "straight_through":
            return ste_round
        if self.quantizer_mode == "noise_injection":
            return make_noise_injection_round(self.training, generator)
        if self.quantizer_mode == "polynomial":
            return polynomial_round
        if self.quantizer_mode == "ste_polynomial":
            return ste_polynomial_round
        raise ValueError(self.quantizer_mode)

    def _quantizer_fn(self, generator=None):
        qpos = self.positive_qstep()
        r = self._rounding_fn(generator)
        return lambda x: r(x / qpos) * qpos

    def forward(self, inputs, input_qstep=None, image_max=255.0, generator=None):
        """inputs [B,H,W,3] in [0,255]. Returns (decoded[B,H,W,3], rate[B])."""
        if (not self.train_qstep) and (input_qstep is not None):
            self.qstep = torch.as_tensor(float(input_qstep), dtype=self.qstep.dtype,
                                         device=self.qstep.device)
        qpos = self.positive_qstep()
        # Emulate integer pixel inputs (TF call: three_channel_inputs = _rounding_fn(...)).
        # Bare pixel rounding (NOT the /qstep quantizer) feeds BOTH proxy DCT and real codec.
        inputs = self._rounding_fn(generator)(inputs)
        # proxy DCT codec (differentiable). coeffs are dequantized-scale values.
        dequantized, coeffs = self.proxy(inputs, self._quantizer_fn(generator), image_max)

        # Real PIL codec once (forward value + exact bits); reused by real_ste + rate.
        real, jpeg_rate = self._real_codec(inputs, qpos, image_max)

        # real_ste: forward replaces proxy decode with real PIL JPEG, backward via proxy.
        if self.codec_forward_mode == "real_ste":
            dequantized = dequantized + (real - dequantized).detach()

        # log_nonzero rate: forward = exact real bits (via per-sample linear fit),
        # backward via num_nonzero = sum log(1+|round(coeff/qstep)|).
        num_nonzero = self._calc_nonzeros(coeffs, qpos)           # [B]
        nz_rate = num_nonzero * jpeg_rate
        nz_nz = num_nonzero * num_nonzero
        line_w = (nz_rate / (nz_nz + 1.0)).detach()
        rate = num_nonzero * line_w
        return dequantized, rate

    def _calc_nonzeros(self, coeffs: Dict[str, torch.Tensor], qstep) -> torch.Tensor:
        B = next(iter(coeffs.values())).shape[0]
        total = torch.zeros(B, dtype=self.proxy.dtype,
                            device=next(iter(coeffs.values())).device)
        for k in coeffs:
            total = total + torch.log(1.0 + torch.abs(coeffs[k] / qstep)).reshape(
                B, -1).sum(dim=1)
        return total

    def _real_codec(self, inputs, qpos, image_max):
        """Real PIL jpeg (forward value + bits), scaled to 8-bit like the TF path."""
        scale = 255.0 / float(image_max)
        x = (inputs * scale).detach().cpu().numpy().astype(np.float32)
        q = float(qpos.detach().cpu()) if torch.is_tensor(qpos) else float(qpos)
        dec, bits = encode_decode_with_jpeg(
            x, q, one_channel_at_a_time=self.run_jpeg_one_channel_at_a_time,
            use_420=False)
        dec = torch.from_numpy(dec / scale).to(inputs.device, inputs.dtype)
        bits = torch.from_numpy(bits).to(inputs.device, inputs.dtype)
        return dec, bits
