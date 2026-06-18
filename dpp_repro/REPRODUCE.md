# Reproduction Guide — DPP/DPO Video Precoding

This document lets another agent reproduce the DPP reproduction from scratch:
environment, data, architecture, training, evaluation, the bugs we hit and how
to detect them, and the expected performance curves. All numbers are measured
(real x264/x265/AV1 + libvmaf binary, per-clip convex-hull BD-rate, held-out
clips). Paper reference: Chadha et al., "Escaping the complexity-bitrate-
quality barriers of video encoders via deep perceptual optimization",
Proc. SPIE 11510, 115100C (2020).

Code lives in `dpp_repro/`. Scratch (videos, encodes) MUST live on tmpfs
(`/dev/shm/dppv`) — the root disk is ~95% full.

---

## 1. Environment

```
GPU:    4x 16 GB (CUDA 12.x); sm_120 ok
Python: /venv/torch-env/bin/python  (torch 2.8.0+cu128, CUDA available)
        system python3 has NO torch — always use the venv for torch jobs
ffmpeg: with libx264, libx265, libaom-av1 encoders
vmaf:   /usr/local/bin/vmaf (built from source, e4b93c6);
        LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu
        models in reference/vmaf_models/{vmaf_v0.6.1.json, vmaf_v0.6.1neg.json}
```

Pip installs (into the torch venv):
```
pip install -e /workspace/VMAF-torch          # github alvitrioliks/VMAF-torch
pip install yuvio                              # VMAF-torch dep (but see Bug #1)
pip install compressai                         # EntropyBottleneck (rate model)
```
WARNING: `pip install compressai` pulled numpy 1.26.4 + torch-geometric as
deps; verify afterwards that `torch`, `vmaf_torch`, and CUDA still import
cleanly (we did; if numpy is downgraded incompatibly, pin it).

Sanity check before any run:
```
python -c "import torch,vmaf_torch,compressai; print(torch.cuda.is_available())"
```

GPU hard rule: every torch entrypoint asserts `torch.cuda.is_available()`
(`[gpu-guard]`). CUDA pip-wheel libs are not always on LD_LIBRARY_PATH —
confirm GPU visibility before launching.

---

## 2. Data

XIPH derf 1080p clips, first 150 frames, transcoded to yuv420p on tmpfs:
```
ffmpeg -y -i https://media.xiph.org/video/derf/y4m/<clip>.y4m \
       -frames:v 150 -pix_fmt yuv420p /dev/shm/dppv/src/<clip>.y4m
```
Filename gotcha: many clips are `<name>_1080p.y4m` (NOT `_1080p30`); probe with
`curl -sI` and expect 404 on the wrong suffix.

- **Training corpus (19 clips)**: blue_sky, pedestrian_area, riverbed,
  rush_hour, sunflower, tractor (the original 6, all `_1080p25`) + station2_1080p25,
  dinner_1080p30, factory_1080p30, life_1080p30, crowd_run_1080p50,
  ducks_take_off_1080p50, in_to_tree_1080p50, old_town_cross_1080p50,
  park_joy_1080p50, snow_mnt_1080p, speed_bag_1080p, touchdown_pass_1080p,
  rush_field_cuts_1080p.
- **Held-out eval (4 clips, NEVER trained on)**: aspen_1080p, red_kayak_1080p,
  west_wind_easy_1080p, controlled_burn_1080p. These are hard-coded in
  `s1_train.py::load_frames()` HOLDOUT and skipped.
- Stills: CLIC `dpp/data/train_big/*.png` (luma, BT.601 limited range) used as
  extra intra-only data.

---

## 3. Architecture (`virtual_codec.py`, `s1_train.py`)

### Precoder (the trained net, ~10k params)
`Conv(3,16,d=1) ×2 -> Conv(3,16,d=2) -> Conv(3,16,d=4) -> Conv(3,16,d=8) ->
Conv(3,1,1)`, PReLU after each; **global residual skip + zero-init output conv**
(identity at init — without this the conv gets zero gradient and stays frozen;
this is the ReZero lesson). Luma-only (single channel).

### Virtual codec (differentiable, training only)
- **Soft block matching** (paper Eq.1-3): split current precoded frame into
  KxK blocks (K random in {4,8,16} per step, M=24 search window); MAE
  similarity over (M+1)^2=625 candidate patches; `softmax(-eps/tau)` relaxation
  of argmin (tau=1.0); prediction = convex combination of candidates.
  - INTER: reference = previous precoded frame (open-loop, shared precoder
    weights). 99:1 inter:intra steps.
  - INTRA: reference = current frame with a checkerboard DC self-mask (the
    queried-colour blocks are reduced to their block-mean so they cannot
    predict themselves; the opposite colour keeps detail). See Bug #5.
- **Transform**: H.264 4x4 integer core transform, orthonormal scaling
  (T = diag(1/||rows||) @ [[1,1,1,1],[2,1,-1,-2],[1,-1,-1,1],[1,-2,2,-1]]).
- **Quantization**: Qstep(QP) = base6[QP%6]*2^(QP//6),
  base6 = [.625,.6875,.8125,.875,1.0,1.125]; QP ~ U[12,42] per step
  (rate marginalization, one model serves all rates); additive uniform noise
  in training (handled by EntropyBottleneck), rounding at eval.
- **Rate**: `compressai.EntropyBottleneck(16)` over the 16 subbands of
  DCT(residual)/Qstep; rate = sum(-log2 likelihood)/pixels.

### Loss
`L = gamma*L_P + lam*L_R + L_F`
- `L_F = 0.2*L1/255 + 0.8*(1 - MS-SSIM)` (filter_size 11, torch port matching
  the eval metric).
- `L_P = (100 - VMAF_NEG(x, p_hat).clamp(max=100))/100`, VMAF_NEG = vmaf-torch
  NEG=True. **DECLARED SUBSTITUTION** for the paper's proprietary NR-IQA MOS.
- `L_R` = mean bpp from the bottleneck.
- `gamma=0.5..2` (NOT 0.1 — see §6 ablation), `lam in {0.01 (enh0m),
  0.001 (enh3m)}`. Adam 1e-4, x0.1 at half, grad-clip 1.0, batch 8 x 512^2
  luma crops, 30k steps (paper 40k). The EntropyBottleneck quantiles train on
  a separate aux optimizer (`vc.eb.loss().backward()`).

---

## 4. Training

```
TMPDIR=/dev/shm python -m dpp_repro.s1_train \
    --lam 0.01 --gamma 2 --gpu 0 --steps 30000 --inter --tag _big
```
`--inter` = S2 (99:1 inter:intra); omit for S1 (intra-only ablation). Ckpts to
`dpp/runs/s2_lam<lam>_g<gamma><tag>/model.pt`. ~0.65 s/it (=~5.5 h for 30k).
Healthy log line: NEG <= 100, MS-SSIM >= 0.93, rate decreasing
(5 bpp -> ~0.8-1.5 bpp), `|p-x|` growing slowly (0.3 -> ~1.0 grey, NOT
exploding), no NaN.

---

## 5. Evaluation (hull-vs-hull BD, `s3_eval.py`)

For each ckpt: precode each clip's luma at full 1080p (replace Y, keep UV,
write the y4m container yourself — see Bug #2), then run the S0 ladder on the
precoded clip, score vs the ORIGINAL reference, per-clip convex hull, BD-rate
vs the baseline hull. The DPO hull POOLS both model variants (g0.5 + g2) as the
paper does.

```
# baseline ladder (once):
TMPDIR=/dev/shm python dpp_repro/s0_hull.py --clips <held-out comma list>
# precoder eval:
TMPDIR=/dev/shm python -m dpp_repro.s3_eval \
    --ckpts bg05=.../s2_lam0.01_g0.5_big/model.pt,bg2=.../s2_lam0.01_g2_big/model.pt \
    --clips aspen_1080p,red_kayak_1080p,west_wind_easy_1080p,controlled_burn_1080p \
    --gpu 0 --out dpp/runs/s3_big.json
# cross-codec (x265/AV1), reuses precoded sources:
TMPDIR=/dev/shm python -m dpp_repro.s4_codecs --codec x265   # then --codec av1
```
Protocol (matches paper §4.2, verified): x264 slow `-tune ssim -refs 5
-g/-keyint 150 -sc_threshold 0`; 8 resolutions (1080..144p, Lanczos down /
bicubic up to 1080p before libvmaf); 7 CRF (18..42); VMAF + AH-VMAF
(=VMAF_NEG, the v0.6.1neg model) + SSIM (float_ssim); per-clip monotone +
convex hull; quality clamps 40<=VMAF<=96, 88<=SSIM(x100)<=99 on BOTH arms;
BD-rate per clip then averaged; aggregation does NOT matter (per-resolution
average vs cross-resolution hull differ <1pt — both checked).

---

## 6. Performance curves (expected results)

Figures generated by `make_curves.py` (`fig_*.png`). Key tables:

**Reproduction arc (held-out pooled BD-rate %)** — `fig_reproduction_arc.png`:
| stage | SSIM | AH-VMAF | VMAF |
|---|---|---|---|
| initial (6 clips, gamma=0.1) | -2.53 | -6.53 | -10.93 |
| gamma 0.5-2 (6 clips) | -3.37 | -9.27 | -15.18 |
| gamma 0.5-2 (19 clips) | **-4.00** | **-11.19** | **-18.98** |
| paper Table 1 | -4.67 | -12.27 | -25.08 |

**Gamma dose-response (single model, held-out)** — `fig_gamma_doseresponse.png`:
| gamma | 0 | 0.1 | 0.5 | 2 | 5 |
|---|---|---|---|---|---|
| AH-VMAF | -3.95 | -5.65 | -8.49 | -9.09 | -9.12 |
| VMAF | -6.71 | -8.77 | -14.02 | -14.93 | -14.48 |
| SSIM | -2.12 | -2.34 | -3.01 | -2.43 | -2.12 |

**Cross-codec (pooled, held-out, no retraining)** — `fig_cross_codec.png`:
| | SSIM | AH-VMAF | VMAF |
|---|---|---|---|
| x264 slow | -3.37 | -9.27 | -15.18 |
| x265 slow | -3.05 | -8.56 | -16.99 |
| AV1 cpu5 | -3.30 | -7.80 | -16.70 |

**S0 protocol bonus** (baseline ladder-hull vs fixed-1080p, NO precoder):
VMAF -37.6 / AH-VMAF -32.8 / SSIM -55.8.

**Intra-only ablation** (S1 vs S2, held-out): -2.45/-6.68/-11.47 vs
-2.53/-6.53/-10.93 — inter TRAINING contributes ~0; temporal leverage is at
the deploy codec.

---

## 7. Bugs we hit and how to detect them

1. **yuvio reads .y4m as headerless raw** — the 60-byte header + 6-byte
   `FRAME\n` markers enter the pixel stream and shift every frame cumulatively.
   Symptom: precoded clips score VMAF ~1.8 (garbage). Detect: byte-level diff
   of read-back vs source (dY ~23, NOT 0). Fix: own reader `y4m.py`. Re-verify
   precode is byte-exact on untouched chroma (dU == 0.0000).
2. **ffmpeg rawvideo->y4m range-converts** (+~23 mean luma, limited->full).
   Fix: write the y4m container yourself (`header + b"FRAME\n" + planes`).
3. **vmaf-torch does not clip the SVR at 100** (the binary does). Unclamped,
   `L_P` goes negative and chases meaningless >100 extrapolation. Symptom:
   training NEG logged as 110. Fix: `.clamp(max=100)`.
4. **OOM in soft block matching** — the [B,k^2,L,(m+1)^2] gather tensor (K=16
   is ~2.2 GB + its grad) OOMs at batch 8 on 16 GB. Fix: fp16 internals +
   gradient checkpointing per chunk + fp16 unfolded patches; chunk sizes
   {4:4096, 8:1024, else 256}. Peak then ~6 GB at K=16.
5. **Checkerboard intra mask inverted** (self-leak): random-noise frames
   "predict" at MAE 0.00. Detect: self-test asserts intra resid MAE on a random
   frame is high (~55, near the unpredictable DC-only baseline ~64), low on a
   smooth gradient (~0.6). The mask that DC's colour-c blocks must be used to
   predict colour-c (not the opposite).
6. **Stills luma range mismatch** — RGB->luma gives full range; y4m Y planes
   are BT.601 limited. Fix: `y = 16 + luma*(219/255)` for stills.
7. **Held-out contamination** — the gamma-ablation runs initially loaded the
   eval clips into training (effect measured ~0 for a 10k-param net, but
   improper). Fix: HOLDOUT exclusion list in `load_frames()`; the corpus-
   expansion run is clean. Always verify the "frames N pairs M" log line
   matches the intended corpus size (19 clips -> frames ~570, pairs ~1425).
8. **Two-model vmaf invocation fails** — `--model A --model B --feature
   float_ssim` in one call errored; run two separate vmaf invocations (std+ssim,
   then neg) and merge.

---

## 8. Declared deviations from the paper
1. Perceptual model = faithful diff-NEG (vmaf-torch), not the proprietary
   NR-IQA MOS. Verified vs the vmaf binary: MAE 0.016, rank-acc 1.0000 on 880
   pairs. (Run `dpp_repro/vmafneg_torch_check.py` to re-verify the fidelity gate.)
2. Corpus 19 clips x 150 frames + CLIC stills, vs the paper's large
   proprietary corpus.
3. 30k steps (paper 40k); scaled LR schedule.
4. Checkerboard DC intra self-masking (vectorizable approximation).
5. x264/x265/AV1; AV1 arm uses single-pass CRF (paper: 2-pass target-bitrate).
6. Residual not rescaled to [0,255] before DCT (folds into Qstep).

## 9. File map
- `virtual_codec.py` — soft block matching, 4x4 DCT, EntropyBottleneck codec.
- `s1_train.py` — precoder + training loop (`--inter` toggles S1/S2).
- `s0_hull.py` — baseline x264 ladder + protocol-bonus measurement.
- `s3_eval.py` — precode + hull-vs-hull BD (x264).
- `s4_codecs.py` — x265/AV1 cross-codec eval (reuses precoded sources).
- `y4m.py` — correct y4m reader/writer.
- `vmafneg_torch_check.py` — fidelity gate for diff-NEG vs the binary.
- `make_curves.py` — regenerates `fig_*.png`.
- Results: `dpp/runs/{s0_baseline_hull, s3_eval, s3_holdout, s3_eval_s1_holdout,
  s3_ablation_gamma, s3_big, s4_x265, s4_av1}.json`.
- Ckpts: `dpp/runs/s2_lam0.01_g{0.5,2}_big/model.pt` (best models).
