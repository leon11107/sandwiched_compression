"""Ballé (2018) factorized-prior entropy model — DPP's rate model (paper eq 5/7).

Self-contained torch port of the non-parametric univariate density / EntropyBottleneck
(no compressai/torch_geometric dependency). Models each of C channels with a monotonic
cumulative network c(x) in [0,1]; per-coefficient likelihood = c(x+0.5)-c(x-0.5);
rate(bits) = -log2(likelihood). During training, additive uniform noise replaces
rounding (matches DPP / Ballé). Used on DCT sub-band coefficients (C = 64 sub-bands).

Reference: Ballé, Minnen, Singh, Hwang, Johnston, "Variational image compression with
a scale hyperprior", ICLR 2018, appendix 6.1 (the factorized entropy bottleneck).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedEntropy(nn.Module):
    def __init__(self, channels: int, filters=(3, 3, 3), init_scale: float = 10.0):
        super().__init__()
        self.channels = C = channels
        self.filters = tuple(int(f) for f in filters)
        dims = (1,) + self.filters + (1,)
        scale = init_scale ** (1.0 / (len(self.filters) + 1))
        self._H, self._b, self._a = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for i in range(len(self.filters) + 1):
            din, dout = dims[i], dims[i + 1]
            # H: [C, dout, din] pre-softplus so the network is monotonic non-decreasing
            h = torch.full((C, dout, din), float(np.log(np.expm1(1.0 / scale / dout))))
            self._H.append(nn.Parameter(h))
            self._b.append(nn.Parameter(torch.empty(C, dout, 1).uniform_(-0.5, 0.5)))
            if i < len(self.filters):
                a = torch.zeros(C, dout, 1)
                self._a.append(nn.Parameter(a))

    def _cumulative(self, x):
        """x: [C, 1, N] -> logits of CDF [C, 1, N] (apply sigmoid for CDF)."""
        out = x
        for i in range(len(self.filters) + 1):
            H = F.softplus(self._H[i])            # [C,dout,din] >= 0 -> monotonic
            out = torch.matmul(H, out) + self._b[i]
            if i < len(self.filters):
                a = torch.tanh(self._a[i])
                out = out + a * torch.tanh(out)
        return out                                 # logits; CDF = sigmoid(out)

    def likelihood(self, x):
        """x: [B,C,H,W]; returns per-element likelihood [B,C,H,W] (training: +U noise)."""
        B, C, H, W = x.shape
        assert C == self.channels
        if self.training:
            x = x + (torch.rand_like(x) - 0.5)
        # to [C, 1, N]
        v = x.permute(1, 0, 2, 3).reshape(C, 1, -1)
        lo = self._cumulative(v - 0.5)
        hi = self._cumulative(v + 0.5)
        # likelihood = sigmoid(hi) - sigmoid(lo), stable via sign trick
        sign = -torch.sign(lo + hi).detach()
        lik = torch.abs(torch.sigmoid(sign * hi) - torch.sigmoid(sign * lo))
        lik = lik.clamp_min(1e-9).reshape(C, B, H, W).permute(1, 0, 2, 3)
        return lik

    def bits(self, x):
        """Total bits per sample [B] = -sum log2 likelihood."""
        lik = self.likelihood(x)
        return -torch.log2(lik).reshape(x.shape[0], -1).sum(dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    eb = FactorizedEntropy(64).cuda().train()
    # validate: (a) gradient flows; (b) lower-magnitude (smoother) coeffs -> fewer bits
    x = torch.randn(4, 64, 16, 16, device="cuda", requires_grad=True) * 5
    b = eb.bits(x); b.sum().backward()
    print("grad to input:", float(x.grad.abs().sum()) > 0, "bits/sample mean=%.1f" % float(b.mean()))
    with torch.no_grad():
        big = eb.bits(torch.randn(4, 64, 16, 16, device="cuda") * 20).mean()
        small = eb.bits(torch.randn(4, 64, 16, 16, device="cuda") * 1).mean()
    print("bits: large-scale=%.1f  small-scale=%.1f  (small should be << large)" % (float(big), float(small)))
    # train the entropy model briefly on a fixed dist, check it lowers its own aux/bits
    opt = torch.optim.Adam(eb.parameters(), lr=1e-2)
    data = torch.randn(64, 64, 16, 16, device="cuda") * 5
    b0 = float(eb.bits(data).mean())
    for _ in range(200):
        opt.zero_grad(); l = eb.bits(data).mean(); l.backward(); opt.step()
    print("entropy-model fit: bits %.1f -> %.1f (should drop as model learns the dist)" % (b0, float(eb.bits(data).mean())))
