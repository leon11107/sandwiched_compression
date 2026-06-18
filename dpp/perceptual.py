"""DPP perceptual model: no-reference NIMA-on-Koniq MOS (torch, frozen).

DPP paper 3.6: NR NIMA trained on Koniq-10k -> ACR distribution; L_P = -E[Σ i P_i]
= -predicted MOS (maximize MOS). Frozen during DPP training. Perceptual input is the
decoded-Y + lossless-UV -> RGB frame (= the codec's prediction when codec_luma_only).

IMPL: pyiqa `nima-koniq` (NIMA, Koniq-10k MOS, no-reference) — same model family +
training data as the paper. Backbone is InceptionResNetV2 (paper used VGG-16); NOTED
as a backbone difference. torch-native => usable as a GPU training gradient (the
original TF blocker). pyiqa returns a scalar MOS (higher = better), so L_P = -MOS.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import pyiqa


class NimaMOS(nn.Module):
    def __init__(self, name: str = "nima-koniq", device: str = "cuda"):
        super().__init__()
        self.name = name
        # CRITICAL: as_loss=True enables GRADIENT through the metric. Default (False)
        # runs inference under no_grad -> output.requires_grad=False -> the perceptual
        # term contributes ZERO gradient (it silently does nothing in training).
        self.metric = pyiqa.create_metric(name, as_loss=True, device=device)
        self.lower_better = bool(self.metric.lower_better)  # nima: False (higher=better)
        for p in self.metric.parameters():
            p.requires_grad_(False)
        self.metric.eval()

    def mos(self, pred_bhwc_255: torch.Tensor) -> torch.Tensor:
        """pred [B,H,W,3] in [0,255] -> per-sample MOS (higher=better)."""
        x = (pred_bhwc_255 / 255.0).clamp(0, 1).permute(0, 3, 1, 2).contiguous()
        s = self.metric(x).reshape(-1)
        return -s if self.lower_better else s  # normalize to higher=better quality

    def loss(self, pred_bhwc_255: torch.Tensor) -> torch.Tensor:
        """L_P = -MOS (maximize MOS), per-sample [B]."""
        return -self.mos(pred_bhwc_255)
