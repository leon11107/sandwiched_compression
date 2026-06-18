# Architecture Audit — proxy/real/eval codec alignment + training dynamics
2026-06-10. Scripts: `proxy_real_debug.py`, `proxy_real_debug2.py`, `proxy_real_grad_terms.py`
(qsteps 32/64/96), `audit_codec_domain.py`, `train_probe.py` (instrumented champion recipe, 3 ep).
Raw outputs: `runs/audit_*.out`, `runs/probe_champion.out`, `runs/probe_champion/steps.jsonl`.

## A. Proxy ↔ real codec, FORWARD path — ALIGNED (not the problem)
| q | PSNR proxy/real | bpp proxy/real | decoded agree |
|---|---|---|---|
| 32 | 33.43 / 33.61 | 0.930 / 0.955 | 38.21 dB |
| 64 | 29.66 / 29.74 | 0.514 / 0.522 | 36.82 dB |
| 96 | 27.63 / 27.68 | 0.362 / 0.365 | 37.22 dB |

dPSNR ≤ 0.18 dB, rate proxy forward = real bits (per-sample fit). The proxy faithfully
represents the *flat-table* JPEG it models. BUT both ≠ the corrected eval codec (see C).

## B. Quantizer BACKWARD (gradient) — intrinsic limit, mode choice matters
FD self-consistency control = +1.000 (measurements trustworthy).

distortion-grad cosine(proxy autodiff, real finite-diff), no preproc:
| mode | q32 | q64 | q96 |
|---|---|---|---|
| straight_through | **−0.416** | −0.167 | −0.050 |
| noise_injection (ours) | +0.039 | +0.042 | −0.027 |
| polynomial / ste_poly | 0.000 | 0.000 | 0.000 (dead grad) |

Per-term (noise_injection): distortion ⊥ (≈0), rate weakly + (+0.06..+0.23),
NIMA weakly + (+0.01..+0.14). Same conclusion as the TF-era diagnostic: the hard
block-DCT quantizer's micro-landscape is not differentiable-approximable; macro
structure (smoothing/energy moves) is what transfers — consistent with Phase 0
showing REAL MS-SSIM BD gains despite ⊥ micro-gradients.
ACTION: keep noise_injection. NEVER straight_through for codec backward (anti-aligned).

## C. TRAINING codec ↔ EVAL codec — THE dominant domain shift (3 layers)
Training arm: flat qtable, 4:4:4, lossless-chroma restore. Eval arm (eval_v2,
deployable target): Annex-K scaled tables, 4:2:0, full lossy.

1. **Rate regime**: training qstep 12–64 → bpp **0.35–1.56**; eval q 5–32 → bpp
   **0.12–0.53**. Overlap only at the extreme edge (qstep64 ≈ q20). Checkpoints
   never saw the low-rate regime they are evaluated in.
2. **Quantization SHAPE**: effective Annex-K luma steps at eval qualities:
   q5 DC=160/ACmean=230, q12 DC=67/ACmean=187, q32 DC=25/ACmean=91 — vs flat ≤64
   in training. At matched bpp, luma error spectra differ structurally:
   HF-error ratio train/eval = 0.60 (q32-match) .. 0.99 (q5-match) at equal total
   error — Annex-K concentrates error in HF, flat spreads uniformly (incl. LF
   blocking). The preproc learned trades for the WRONG artifact distribution.
   Matched-bpp decoded agreement only 23–31 dB.
3. **Chroma**: 4:2:0-full vs 4:4:4+lossless-chroma at same quality: bpp axis shifts
   ~25–40% (e.g. 0.123 vs 0.171 @q5), PSNR +1.6–2 dB offset; luma-MS-SSIM unaffected.

## D. Training dynamics (instrumented probe of champion recipe, 3 ep, live numbers)
- **Loss VALUE vs GRADIENT split**: rate term = 84–93% of loss VALUE, but on preproc
  params the weighted grad norms are |gLF| 0.03–0.68 >> |gP|w 3.5e-3–1.4e-2 >>
  |gR|w 5e-4–2.7e-3. The huge rate value is mostly an entropy-model calibration
  offset (constant wrt preproc). Optimization is driven by L_F, then NIMA (~10x the
  rate gradient); the rate gradient is nearly inert at init.
- **Rate proxy miscalibration, live**: balle/real bpp ratio = 4.3–8.1 within 3
  epochs (co-trained prior conflates model-fitting with rate change; the declining
  "BalleBpp" in old logs was mostly the prior fitting itself).
- **Pre-emphasis drift**: luma HF energy out/in climbs 1.003 → 1.17 in 300 steps;
  dScaler signs show L_F(−) and NIMA(−) jointly pull toward MORE residual while only
  the tiny rate grad (+) opposes — explains old-series 25-ep endpoints with bpp
  ABOVE baseline at q32 (0.72–0.86 vs 0.745) and the TF-era pre-emphasis failures.
- **ReZero degeneracy caveat**: with scaler ≈ 0.01, all per-term grads flow through
  the scaler direction → pairwise cosines pin to ±1; cosine conflict readout only
  becomes meaningful once scaler departs ~0. dScaler per-term scalars are the
  reliable tug-of-war readout early.
- **Transfer check**: after 3 ep, TRAIN-codec diag moved (base 34.21@0.745 → model
  33.32@0.856) while EVAL-codec diag is static (27.95→27.87 dB, ms 0.95742→0.95762)
  — training progress does not reach the eval domain at this regime/shape mismatch.

## Fix list for Phase 1 (architecture level, ordered)
1. **Align training codec to the eval target**: real_ste forward = PIL quality-scale
   Annex-K + 4:2:0 (exactly eval_v2's `jpeg_rt`); proxy quantizer gets per-subband
   qvec = scaled Annex-K table (codec, not just the rate proxy as d13 did);
   marginalize over quality U[5,32] instead of qstep U[12,64].
2. **Honest rate term**: pretrain FactorizedEntropy on the ALIGNED codec's DCT
   coefficients, FREEZE it (--entropy-ckpt path exists), verify live balle/real
   ratio ~1 in the dashboard; re-derive lambda in real-bpp units (paper-regime).
3. Keep noise_injection backward; never straight_through.
4. **Containment monitors** (now standard via train_probe dashboard): hfO/I (pre-
   emphasis drift), dScaler per term, balle/real ratio, dual-codec diag per epoch —
   any Phase 1 run must show eval-codec diag MOVING, else stop early.
5. Loss balance: with rate honest and gradients rebalanced, re-calibrate (gamma,
   lambda) so that |gR|w is within ~1 order of |gLF| (not 2–3 orders below).
