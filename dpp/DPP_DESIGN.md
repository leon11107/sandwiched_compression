# DPP (Deep Perceptual Preprocessing) — torch implementation

Architecture **primarily from the DPP paper** (Chadha & Andreopoulos, CVPR 2021,
reference/DPP/*.pdf), **sandwich torch_port as the reference substrate/codec**.
Built on the TF-equivalence-validated torch base (torch_port/, see [[pytorch-rewrite]]).

## Grounded paper facts (with page refs)

- **3.2 Learnable Preprocessing**: F(x;Θ) pixel-to-pixel CNN, **LUMA (Y) only**
  (Y holds structure + drives bitrate), input scaled **[0,1]**. Implemented as a CNN
  with **dilated convolutions, varying dilation rate per layer** (larger receptive
  field, lower complexity, translational equivariance), single-frame latency.
- **4.1**: "**Each convolutional layer is followed by a parametric ReLU (PReLU)**."
  Adam optimizer; curriculum + multi-scale crops; perceptual model trained then frozen.
- **3.4 Transform/Quant**: residual -> DCT (paper: 4x4 H.264 integer DCT; OURS: JPEG
  8x8, the project's target codec). Q_step randomly sampled during training
  (**QP-marginalization**; one model spans rates). Rounding non-diff -> **additive
  i.i.d. uniform noise of support width 1** during training (= our noise_injection).
- **3.5 Entropy/Rate (eq7)**: L_Rs = -E[Σ_n log2 p(z_{n,s,t}; Φ^{(s)})] per DCT
  sub-band on **divisively-normalized** coefficients z; L_R = Σ_s L_Rs (S=16 for 4x4).
  Factorized prior (Ballé). [IMPL: start with codec log_nonzero rate (validated,
  sandwich ref); factorized-prior is a DPP-faithful upgrade — modular swap.]
- **3.6 Perceptual model**: **no-reference NIMA** — VGG-16 (ImageNet) with FC removed
  -> global-avg-pool + concat intermediate-layer GAP activations -> single FC **5
  neurons + softmax** (ACR dist 1..5). Trained on **Koniq-10k** MOS, then **FROZEN**.
  Perceptual input = decoded-**Y** + **lossless U,V** of the RGB input -> YUV->RGB.
  [IMPL: NR-MOS NIMA-on-Koniq via pyiqa `nima-koniq` (Koniq MOS, NR — same model
  family/training as the paper; backbone IRv2 vs paper VGG-16, NOTED; VGG-16 NIMA is
  an exact-backbone option if needed). torch-native => usable as a GPU training grad,
  which was the original blocker.]
- **3.7 Loss (eq6,8)**:
  - L_F = α·L1(x,p̂) + β·(1 − MS-SSIM(x,p̂)) on **luma** (Zhao et al.; α=0.2, β=0.8).
  - L_P = −E[Σ_{i=1..5} i·P(p̂^RGB)_i] = **−predicted MOS** (maximize MOS).
  - **Total: L = γ·L_P + λ·L_R + L_F.**
- **Hyperparams (supp 1)**: **γ ≈ 0.01** (γ=0.1 too high → SSIM drops; γ=0.005 weak);
  **λ ∈ [0.001, 0.01]** (shifts the whole RD curve; train multiple λ). VMAF_NEG with
  gain limits = 1.0 for gaming-resistant eval.
- N/A for single-image JPEG (paper is VIDEO): inter/intra motion (3.3), 4x4 H.264 DCT
  (we use JPEG 8x8), Vimeo-90k video. We do intra/single-image only.

## Implementation map (dpp/ on torch_port substrate)
- preproc_dpp.py — dilated-conv luma-only preprocessor (input conv → dilated residual
  blocks w/ PReLU, dilations 1,2,4,8,1,2,4,8 → output conv; residual; lossless chroma).
- perceptual.py — NIMA NR-MOS (pyiqa nima-koniq), L_P = −MOS, frozen, input pred (=
  decoded-Y + lossless-UV from codec_luma_only).
- loss.py — L = γ·L_P + λ·L_R + L_F (reuse torch_port.losses.distortion_l1_msssim +
  codec rate).
- train.py — loop on torch_port codec (real_ste, noise quant, QP-marginalize), Adam.
- Reuse VALIDATED torch_port/{codec,preproc,losses,model}.py (TF-equivalent).

## Why this should work where the TF runs stalled
The TF runs were blocked by CLIP-IQA (≠ VMAF, gameable) and the inability to put a
torch quality-MOS model in the TF-GPU training graph ([[dpp-preproc-experiment]]).
torch-native NIMA-on-Koniq (quality-correlated NR-MOS) directly addresses the binding
constraint. Validate-first: adversarial gate (now torch-native, reaches the gaming
regime) picks the quality-correlated target before the full train.
