# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""LPIPS-style perceptual loss using a frozen ImageNet VGG16 (pure TensorFlow).

Standard LPIPS needs torch + the AlexNet/VGG lpips weights; this env has torch
CPU-only and no lpips package, so the GPU-trainable substitute is a VGG16
feature distance: extract features at several VGG layers, channel-normalize
each (LPIPS' unit-normalization trick), and average the squared feature
difference. This is "VGG-LPIPS" — same spirit as LPIPS, different backbone /
no learned linear calibration. Used both as the TRAINING perceptual term and
the EVAL perceptual metric so the two are consistent.

Inputs are RGB in [0,255] (the repo's working range). vgg16.preprocess_input
handles the RGB->BGR + ImageNet-mean shift. The VGG conv stack is fully
convolutional so any spatial size works (no resize to 224 needed).
"""
import os as _os
from typing import Dict, Sequence

import tensorflow as tf

# Repo root, for locating the cloned LPIPS TF port (this file is <repo>/distortion/).
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# LPIPS-style multi-scale layer set (early..mid VGG blocks).
_DEFAULT_LAYERS = (
    "block1_conv2",
    "block2_conv2",
    "block3_conv3",
    "block4_conv3",
)


class VGGPerceptualLoss(tf.Module):
    """Frozen VGG16 feature distance, per-sample, LPIPS-style.

    __call__(a, b) -> [B] tensor of perceptual distance between image batches a
    and b (both [B,H,W,3] float32 in [0,255]). 0 for identical inputs, larger
    for more perceptually different. Differentiable wrt both inputs.
    """

    def __init__(self, layers: Sequence[str] = _DEFAULT_LAYERS, name: str = "vgg_lpips"):
        super().__init__(name=name)
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        base.trainable = False
        outputs = [base.get_layer(n).output for n in layers]
        self._extractor = tf.keras.Model(inputs=base.input, outputs=outputs,
                                         name="vgg16_features")
        self._extractor.trainable = False
        self._layers = tuple(layers)

    @staticmethod
    def _unit_normalize(feat: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
        # LPIPS normalizes features to unit length across the channel axis at
        # each spatial location before comparing — removes magnitude, keeps
        # direction (what the perceptual difference is really about).
        norm = tf.sqrt(tf.reduce_sum(tf.square(feat), axis=-1, keepdims=True) + eps)
        return feat / norm

    def __call__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        a = tf.keras.applications.vgg16.preprocess_input(tf.identity(a))
        b = tf.keras.applications.vgg16.preprocess_input(tf.identity(b))
        fa = self._extractor(a)
        fb = self._extractor(b)
        if not isinstance(fa, (list, tuple)):
            fa, fb = [fa], [fb]
        per_layer = []
        for x, y in zip(fa, fb):
            x = self._unit_normalize(x)
            y = self._unit_normalize(y)
            # mean over spatial + channel => per-sample [B]; mean over layers.
            d = tf.reduce_mean(tf.square(x - y), axis=[1, 2, 3])
            per_layer.append(d)
        return tf.add_n(per_layer) / float(len(per_layer))


def distortion_l2_255(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    lambda_l2: float = 1.0,
    image_key: str = "image",
) -> tf.Tensor:
    """Per-sample SUM of squared error in [0,255] (the sandwich's distortion_l2norm
    scale), x lambda_l2. STRONG fidelity anchor: this is ~1e4-1e5 per sample, so
    destroying the image to save rate incurs a huge penalty the tiny rate term
    cannot offset (fixes the [0,1]-MSE collapse where loss fell while PSNR -> 13dB).
    The keras metric reduces over the batch and adds gamma*rate; the VGG perceptual
    term is added separately in the GradientTape and MUST be rescaled UP (lambda_vgg
    ~ thousands) to stay comparable to this strong L2 — calibrate empirically."""
    gt = ground_truth[image_key]
    pred = outputs["prediction"]
    err = tf.reshape(gt - pred, (tf.shape(gt)[0], -1))
    return lambda_l2 * tf.reduce_sum(tf.square(err), axis=1)


def distortion_l1_255(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    lambda_l2: float = 1.0,
    image_key: str = "image",
) -> tf.Tensor:
    """Per-sample SUM of ABSOLUTE error in [0,255], x lambda_l2. L1 sibling of
    distortion_l2_255: same [0,255] scale and per-sample-sum reduction, but L1 is
    more outlier/texture tolerant than L2 (DPP prefers an L1-style fidelity anchor),
    so for equal weight it permits larger sparse pixel deviations the perceptual
    term wants while still penalising global drift. lambda_l2 keeps the flag name
    shared with the L2 path; calibrate lambda_vgg per anchor (the L1 vs L2 gradient
    scale differs)."""
    gt = ground_truth[image_key]
    pred = outputs["prediction"]
    err = tf.reshape(gt - pred, (tf.shape(gt)[0], -1))
    return lambda_l2 * tf.reduce_sum(tf.abs(err), axis=1)


def distortion_mse01(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    lambda_l2: float = 1.0,
    image_key: str = "image",
) -> tf.Tensor:
    """Per-sample MSE on images normalized to [0,1], scaled by lambda_l2.

    This is the DPP fidelity term in a form SAFE to plug into the keras
    DistortionRateMetric (pure tensor ops, no nested keras Model — the VGG term
    must NOT go in the metric, it triggers a graph-mode None<int shape-inference
    bug; add VGG separately in the training GradientTape). Returns [B]; the
    metric reduces over the batch and adds gamma*rate. Scale: MSE01 ~ 1e-3..1e-2
    so lambda_vgg ~ 0.1-0.3 on the VGG term balances as DPP intends (vs the stock
    [0,255] sum-of-squares L2 which would dwarf the perceptual term).
    """
    gt = ground_truth[image_key]
    pred = outputs["prediction"]
    mse01 = tf.reduce_mean(tf.square((gt - pred) / 255.0), axis=[1, 2, 3])
    return lambda_l2 * mse01


def distortion_mae01(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    lambda_l2: float = 1.0,
    image_key: str = "image",
) -> tf.Tensor:
    """Per-sample MEAN absolute error on images normalized to [0,1], x lambda_l2.

    L1 sibling of distortion_mse01: same reduce_MEAN + [0,1] normalization, so it is
    the same O(1e-3) scale as the unit-normalized VGG term — this is what makes the
    fidelity-DOMINANT objective work (lambda_fid=1.0 genuinely dominates a small
    lambda_perc, instead of the old [0,255]-SUM anchor ~1e5 that forced lambda_vgg
    ~1e9 and let VGG drag PSNR to -10 dB = the collapse). L1 is more outlier/texture
    tolerant than L2 (DPP uses an L1-style fidelity term), permitting sparse pixel
    edits the perceptual term wants while still pinning global tone to the source.
    Returns [B]."""
    gt = ground_truth[image_key]
    pred = outputs["prediction"]
    mae01 = tf.reduce_mean(tf.abs((gt - pred) / 255.0), axis=[1, 2, 3])
    return lambda_l2 * mae01


def distortion_l1_msssim(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    lambda_l2: float = 1.0,
    image_key: str = "image",
    alpha: float = 0.2,
    beta: float = 0.8,
) -> tf.Tensor:
    """DPP-faithful fidelity anchor: lambda_l2 * (alpha*L1_[0,1] + beta*(1-MS-SSIM)).

    This is the exact fidelity loss DPP (Chadha & Andreopoulos, CVPR 2021, Eq.6) uses:
    an L1 luminance term (alpha=0.2) plus a structural (1 - MS-SSIM) term (beta=0.8).
    MS-SSIM tolerates perceptually-invisible deviations but PUNISHES texture/contrast
    distortion, which is what caps the preprocessor from 'hacking' a perceptual metric
    (the failure mode of our pure-L2/mse01 anchor: only distort or pre-emphasis).

    SCALE: _distortion_rate_loss multiplies the returned per-sample value by
    normalization = 1/(H*W*C). To land L_F at its NATURAL magnitude (~1e-2..1e-1) in
    the optimizer loss (so it is comparable to gamma*rate*norm ~ O(0.1-1) and a small
    perceptual term, NOT crushed to ~1e-9 like a raw mse01), we pre-multiply by H*W*C
    here so the norm cancels. lambda_l2 is then the natural fidelity weight (use ~1.0).
    MS-SSIM uses filter_size=7 so it is valid for 128px training crops (128->...->8>=7).
    Returns [B]."""
    gt = ground_truth[image_key]
    pred = outputs["prediction"]
    l1_01 = tf.reduce_mean(tf.abs((gt - pred) / 255.0), axis=[1, 2, 3])  # [B]
    msssim = tf.image.ssim_multiscale(
        gt, pred, max_val=255.0, filter_size=7)  # [B]
    lf = alpha * l1_01 + beta * (1.0 - msssim)  # [B], natural scale ~1e-2..1e-1
    hwc = tf.cast(tf.reduce_prod(tf.shape(pred)[1:]), tf.float32)  # norm cancellation
    return lambda_l2 * lf * hwc


def dpp_distortion(
    ground_truth: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    vgg_loss: VGGPerceptualLoss,
    lambda_l2: float = 1.0,
    lambda_vgg: float = 0.1,
    image_key: str = "image",
) -> tf.Tensor:
    """DPP per-sample distortion: lambda_l2 * MSE01 + lambda_vgg * VGG-LPIPS.

    MSE is computed on images normalized to [0,1] (per-pixel mean squared error,
    O(1e-3..1e-2)) so it sits at a comparable scale to the VGG term (O(0.1..0.3));
    the DPP starting ratio lambda_l2=1.0, lambda_vgg=0.1 then behaves as intended
    (fidelity-leaning, PSNR-protective). Returns per-sample [B]; the caller
    reduces over the batch. NO rate term (DPP holds rate via the codec qstep).
    """
    gt = ground_truth[image_key]
    pred = outputs["prediction"]
    mse01 = tf.reduce_mean(tf.square((gt - pred) / 255.0), axis=[1, 2, 3])
    vgg = vgg_loss(gt, pred)
    return lambda_l2 * mse01 + lambda_vgg * vgg


class LPIPSPerceptualLoss(tf.Module):
    """REAL LPIPS (Zhang et al. 2018) as a differentiable TF training loss.

    Wraps the TF port (Image-X-Institute/lpips_torch2tf) whose weights are converted
    from the ORIGINAL richzhang/PerceptualSimilarity .pth (VGG/Alex features + the
    learned human-perception linear calibration — the part the old uncalibrated
    VGGPerceptualLoss lacked). Cross-checked vs the torch original: trend/ordering
    match, identical->0, max abs diff ~1e-2 (worst on noise). The user accepted this
    ~1e-2 ONNX-conversion error for TRAINING use; EVAL still uses torch LPIPS as the
    ground truth. See memory reference_lpips_tf_port.

    Interface mirrors VGGPerceptualLoss: __call__(a, b) takes [B,H,W,3] in [0,255]
    and returns per-sample [B] LPIPS distance (0 identical, larger = more different),
    differentiable wrt both inputs (verified: grad non-None, no NaN). The port wants
    [-1,1] NHWC (pre_norm=False); we map [0,255] -> [-1,1] via x/127.5 - 1, matching
    the original LPIPS input contract.
    """

    _PORT_DIR = "experiments/m2_lowres_repro/lpips_tf_ext/dev_src"

    def __init__(self, net: str = "alex", name: str = "lpips"):
        super().__init__(name=name)
        import sys
        # original/torch nets are 'alex'/'vgg'; the TF port uses 'alex'/'vgg16'.
        self._net = "vgg16" if net in ("vgg", "vgg16") else net
        port = _os.path.join(_REPO_ROOT, self._PORT_DIR)
        if port not in sys.path:
            sys.path.insert(0, port)
        from loss_fns import lpips_base_tf  # noqa: E402
        self._model = lpips_base_tf.LPIPS(base=self._net, pre_norm=False)

    def __call__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        a = tf.cast(a, tf.float32) / 127.5 - 1.0
        b = tf.cast(b, tf.float32) / 127.5 - 1.0
        d = self._model(a, b)            # spatial-averaged -> [B,1,1,1] (or [B])
        return tf.reshape(d, (tf.shape(a)[0],))
