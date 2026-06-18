# Part 1: Codec-Surrogate Architecture Exploration — Design

Goal: do proxy (virtual-codec) architecture changes that shrink the
proxy↔real-codec domain gap (e.g. deblocking/SAO, better entropy model)
improve held-out real-codec BD-rate?

## Prior (from our own ablations — why we probe before retraining)

- The intra-only ablation showed inter TRAINING contributes ~0 (precoder is
  robust to that proxy detail). So "more faithful proxy ⇒ better" is a WEAK
  prior. We therefore measure each proxy change with a CHEAP no-training probe
  first, and only full-retrain variants that actually move it.

## The domain-gap probe (no training) — what we build first

Question the gradient actually cares about: **does the proxy rank precoded
EDITS the way real x264 does, on the iso-rate quality axis?** If yes, a better
proxy can't help (→ pivot to Part 2). If no, there is room.

Construction:
- **Candidates** (precoded variants spanning the edit space the precoder
  explores): identity + the gamma-family precoders {g0 denoiser, g0.5, g2,
  g5 sharpener} + two hand-crafted extremes {gaussian blur, unsharp}. ~7.
- **Clips**: the 4 held-out clips (the probe measures codec agreement, not
  generalization, so any clips are fine), 8 frames each, coded ALL-INTRA
  (intra≈inter per our ablation → isolate the spatial codec gap, no temporal
  confound).
- **Real side**: encode (candidate-luma + original chroma) with x264 all-intra
  over a CRF sweep → (real bpp, real VMAF_NEG via the vmaf binary vs original).
- **Proxy side**: intra_pred (k=8) → residual → trained VirtualCodec at a QP
  sweep → proxy bpp + reconstruction p_hat; score p_hat (luma) + original
  chroma with the SAME vmaf binary → proxy VMAF_NEG. Using the binary on BOTH
  sides isolates the CODEC gap (rate + reconstruction), not a metric mismatch.
- **Scalarization** (makes the two sides' different rate units comparable):
  per candidate, per side, iso-rate gain vs identity
  `iso = ΔNEG_vs_identity − κ · Δbpp%_vs_identity`, κ = that side's local
  NEG-per-1%bpp slope from the identity RD curve. Rank candidates by `iso` on
  each side.
- **Metric**: Spearman ρ and pairwise sign-agreement between proxy-iso and
  real-iso rankings, per clip then averaged. Also report raw rate-gap
  (proxy vs real bpp at matched op) and reconstruction agreement.

Decision gate:
- ρ high (≳0.8): proxy already RD-ranks edits like x264 → proxy improvements
  won't transfer → STOP, report, go to Part 2.
- ρ low/mid: identify WHICH axis disagrees (rate ranking vs reconstruction) →
  pick the proxy variant that targets that axis → full retrain + held-out eval.

## Proxy variants (ranked, only retrained if the probe says so)
1. Entropy model: factorized `EntropyBottleneck(16)` → hyperprior/context
   (shapes the rate gradient most directly).
2. Differentiable H.264 in-loop deblocking (shapes which HF edits survive).
3. Transform 4×4 → +8×8 (energy compaction).
4. Quantization: uniform-noise → dead-zone / soft-STE.
(SAO is HEVC-only → only relevant to x265/AV1 transfer; evaluate all 3 codecs.)

## Cost
Probe: a few hours, zero training. Each promising variant: 2 models
(γ0.5,γ2) × 19-clip/30k ≈ 6 h + held-out eval; compare to the current best
−4.00 / −11.19 / −18.98 (SSIM/AH-VMAF/VMAF) on x264/x265/AV1.

## RESULT (2026-06-14) — probe says NO meaningful gap; skip proxy surgery

At **quality-matched** operating points (proxy QP{34,38,42} ≈ real CRF{22,30,38},
both at identity NEG ~73), mean **Spearman +0.99 / pairwise 0.98** over the 4
clips (`probe_domaingap_matched.json`). The proxy already RD-ranks precoded edits
essentially identically to x264 in the deploy regime; the pre-emphasis-strength
sweep is near-identical on both sides (identity..g2x2: real 0/+1.1/+2.2/+2.9/+3.4
vs proxy 0/+1.4/+2.4/+3.1/+3.4).

METHODOLOGY NOTE (load-bearing): a naive sweep that did NOT match quality gave a
misleading +0.41 — the proxy was measured in its near-lossless QP region
(QP28→NEG93) where pre-emphasis matters less, while x264 sat at NEG~75. The
proxy's QP→NEG curve differs greatly from x264's CRF→NEG, so you MUST pick QPs
that match the deploy quality before comparing edit rankings.

Decision: ρ≈0.99 ≫ 0.8 ⇒ the gradient direction is already faithful ⇒
proxy-architecture changes (deblock/SAO/entropy hyperprior) are very unlikely to
transfer to downstream BD-rate. **Do not run the Part-1 proxy-variant retrains.**

Two residual proxy limits that are real but not the bottleneck:
- RD range floors: QP≥48 saturates at bpp 0.243 / NEG 46 — the proxy can't reach
  x264's low-bitrate regime (a calibration/range gap, not a ranking gap).
- Off-manifold edits (naive unsharp/blur) are over-penalized — outside the
  precoder's learned mild-edit family, so it doesn't affect the gradient.

Side finding (a Part-2 lever, not a proxy fix): both proxy and real reward
STRONGER pre-emphasis (g2x2 +3.4 vs the deployed g2 +2.2 ⇒ ~+1.2 NEG iso pts
left on the table), but pushing there trades SSIM (cf. the gamma dose-response) —
it's a loss-balance / edit-magnitude question, not a proxy-fidelity one.
