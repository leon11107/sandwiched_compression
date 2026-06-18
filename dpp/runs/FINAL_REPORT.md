# DPP-style JPEG Preprocessing — Final Report (2026-06-10)

> **POST-SCRIPT (same day, supersedes the VMAF_NEG verdict below).** The user
> challenged the ceiling claim; the oracle experiment built to prove it instead
> (a) exposed an eval bug and (b) REFUTED the ceiling:
> 1. **Eval bug**: eval_v2 batched 50 different-size frames into one y4m; ffmpeg
>    silently warped all frames to frame-0 dims → all earlier VMAF/VMAF_NEG
>    magnitudes biased (PSNR/MS-SSIM unaffected; directional training conclusions
>    survived re-verification, eval_v3_verified.json). Fixed: per-image VMAF
>    grouping + equal-size assert in vmaf_metric.py.
> 2. **Oracle bound (oracle_vmafneg.py)**: per-image ES directly on the real
>    VMAF_NEG binary over 64-dim DCT-band luma pre-scaling: **20/20 jobs positive,
>    iso-bpp +0.84..+2.84 (q8 mean +1.83, q20 +1.26)**. Preprocessing CAN gain
>    VMAF_NEG on JPEG intra.
> 3. **Deployable already**: the MEAN oracle s as a FIXED global preprocessor
>    (no network): **BD-VMAF_NEG −1.97% [CI −2.64,−1.32]**; per-q variant
>    **−2.35% [−3.00,−1.78]** on full val50 (fixed_s_validation.json).
> 4. **Mechanism**: mild mid/high-band pre-emphasis (s mid≈1.12–1.16,
>    high≈1.04–1.21) — the exact direction our L_F-anchored training suppressed
>    as a "failure mode". The trained-model conclusions below remain true FOR
>    THAT TRAINING FAMILY; the ceiling generalization was wrong.
> Next: image-adaptive band-shaping (oracle distillation / direct s-prediction)
> to push toward the oracle bound (≈ −4..−6% BD plausible).

Goal (user bar): MS-SSIM BD-rate ≤ −2% OR VMAF_NEG BD-rate ≤ −5% vs no-preproc
real JPEG (4:2:0, Annex-K), val50, qualities 5–32 (VMAF_NEG 57–92, paper regime).

## Verdict
- **MS-SSIM: ACHIEVED, −7.7% BD-rate** (record; bar was −2%). Best: C_l10 −7.71
  [CI −8.50,−6.90], D_v03_l20 −7.71 [−8.89,−6.35]. Deliverable operating point
  **C_l05** (λ=0.05): **MS-SSIM −7.01%, BD-PSNR +0.07% (iso-quality neutral),
  iso-bpp ΔPSNR +0.29…−0.69 dB** (≤0.5dB loss up to 0.40bpp, positive ≤0.25bpp).
- **VMAF_NEG: CEILING ≈ 0 (neutral), empirically established.** Best trained model
  E_v40 (VIF=0.4): plain-VMAF BD **−0.98 (win)** but VMAF_NEG +1.63 (iso-bpp
  −0.06…−1.0). ~25 configs across TF+torch eras (NIMA/VIF/CLIP-IQA/LPIPS ×
  rate-down/quality-up × flat/aligned codecs × gradient-valid/invalid regimes)
  all converge: nothing crosses 0.

## The two mechanisms (why the ceiling exists)
1. **Rate-targeted smoothing** (λ·honest-rate vs L_F): wins MS-SSIM at iso-bpp
   (multi-scale structure preserved, bits saved) but VMAF_NEG punishes the detail
   loss monotonically (λ dial: msssim −7→−7.7 as vmaf_neg +38→+54).
2. **Detail protection / quality-up** (VIF↑): buys exactly what plain VMAF rewards
   (E_v40 VMAF −0.98) and what VMAF_NEG's gain-limit discounts (+1.63). Asymptote
   is neutral; pushing harder degrades msssim/PSNR.
DPP paper's VMAF_NEG −7…−11% relies on video/inter (denoised frames → cheaper
motion prediction) + convex-hull-over-resolutions eval — neither exists for JPEG
intra. Its SSIM claim (−1…−3%) we EXCEED at −7.7%.

## What made the MS-SSIM record possible (the recipe)
1. **Protocol fix (Phase 0 / eval_v2.py)**: real JPEG 4:2:0 anchor at VMAF_NEG
   57–92 (old eval was flat-qtable 4:4:4 at VMAF_NEG 97.8–99.5, saturated), 1080p
   VMAF upscale, MS-SSIM added (never measured before), BD + Pareto envelope.
2. **Architecture audit (AUDIT_REPORT.md)**: training↔eval codec shift (3 layers),
   straight_through anti-aligned (−0.42) vs noise_injection (~0), loss VALUE-share
   vs GRAD-share split (rate was 90% of value but 2–3 orders below in gradient),
   pre-emphasis drift monitor (hfO/I), ReZero ±1-cosine caveat.
3. **Aligned codec (codec_aligned.py)**: real fwd byte-identical to deployment
   (Annex-K 4:2:0), per-subband-qvec luma proxy bwd (agree 43–47dB).
4. **Honest rate**: frozen FactorizedEntropy pretrained on aligned-codec stats +
   per-quality calibration k(q) → λ in real-luma-bpp units (est/real 0.4–0.8
   stable vs 4.3–8.1 drifting under co-training).
5. **Gradient-valid training zone**: q∈[40,90] (AC steps 11–71). Phase-1a trained
   at the eval regime q∈[5,32] (AC steps 91–230) and FAILED (L_F rose all runs —
   proxy distortion gradient is noise at brutal quantization, audit B). Train
   where the gradient is valid, deploy down via generalization.

## λ dial (Arm C, vif=0, q40–90 training)
| λ | BD-MSSSIM | BD-PSNR | max same-q PSNR drop |
|---|---|---|---|
| 0.05 | −7.01 | +0.07 | 1.56 dB |
| 0.075 | −7.66 | +0.74 | 1.98 dB |
| 0.10 | −7.71 | +1.87 | 2.33 dB |
| 0.20 | −6.93 | +4.87 | 3.36 dB |
| 0.40 | −1.08 | +11.8 | 4.59 dB |
| 0.80 | +22.8 (collapse) | — | 6.97 dB |

## Best operating point per 0.1-bpp bin (final models)
| bin | MS-SSIM best | VMAF_NEG best |
|---|---|---|
| [0.1,0.2) | D_v03_l20 +0.0158 | E_v40 −0.06 |
| [0.2,0.3) | C_l05 +0.0070 | E_v40 −0.28 |
| [0.3,0.4) | C_l05 +0.0056 | E_v40 −0.63 |
| [0.4,0.5) | C_l05 +0.0008 | E_v40 −0.38 |

Plots: rd_final_v2.png (final), rd_eval_v2.png (Phase 0), iso_bpp_gain.png.
Evals: eval_v2_full/1b/E/lowlam.json (per-image data incl.). Checkpoints:
dpp/runs/v2_*/model.pt. Trainer: train_v2.py (instrumented); audit tools:
proxy_real_debug*.py, audit_codec_domain.py, train_probe.py.

## If VMAF_NEG must be won (out of current scope)
The two in-scope mechanisms are exhausted. Paths that change the problem:
(a) pre+post sandwich (paper-supported, abandons preproc-only premise);
(b) video/inter codec target (where DPP's gains actually live);
(c) encoder-side optimization (qtable/trellis per-image) instead of pixel preproc.
