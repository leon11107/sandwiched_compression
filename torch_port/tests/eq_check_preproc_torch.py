"""Torch-side preprocessor equivalence checker (run in /venv/torch-env).

Loads the TF preproc reference, ports the 38 UNet weights + scaler into the torch
PreprocOnlyTorch, runs on the SAME inputs, and compares: raw UNet output (isolates
conv/pool/bilinear-upsample), full preproc output (residual + luma-only), and the
tf.image YUV kernel parity. PASS/FAIL table; exit !=0 on any FAIL.
"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.preproc import (PreprocOnlyTorch, load_unet_weights,
                                tf_rgb_to_yuv, tf_yuv_to_rgb)

REF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eq/preproc_ref.npz"
Z = np.load(REF)
T = torch.float32
DEV = "cuda" if torch.cuda.is_available() else "cpu"
fails = []
# Cross-framework conv float floor: TF(GPU Winograd) vs torch(cuDNN) deep conv nets
# differ at ~1e-3 relative (device-dependent; torch is internally consistent to ~8e-6).
# Equivalence is judged at this floor PLUS the decisive uint8/downstream check below.
RTOL_CONV = 6e-3

def t(a): return torch.from_numpy(np.asarray(a)).to(T).to(DEV)

def cmp(name, o, ref, atol, rtol=1e-4):
    o = o.detach().cpu().numpy().astype(np.float64); r = np.asarray(ref, np.float64)
    if o.shape != r.shape:
        print(f"  [FAIL] {name:26s} shape {o.shape} vs {r.shape}"); fails.append(name); return
    mabs = float(np.max(np.abs(o - r)))
    # relative to SIGNAL SCALE (max|ref|), not per-element (avoids near-zero blowup)
    srel = mabs / (float(np.max(np.abs(r))) + 1e-12)
    ok = (mabs <= atol) or (srel <= rtol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} max_abs={mabs:.3e} signal_rel={srel:.3e}")
    if not ok: fails.append(name)

# build + port weights
scaler = float(Z["scaler"])
m = PreprocOnlyTorch(mean_adjust=float(Z["mean_adjust"]), scale_adjust=float(Z["scale_adjust"]),
                     preproc_luma_only=True, scaler_init=scaler).to(DEV)
m.eval()
nw = int(Z["nw"])
tf_w = [Z[f"w{i}"] for i in range(nw)]
load_unet_weights(m.unet, tf_w)
print(f"ported {nw} weight arrays; scaler={scaler:.5f}; device={DEV}")

# tf.image YUV kernel parity
print("== tf.image YUV kernel parity ==")
cmp("rgb2yuv", tf_rgb_to_yuv(t(Z["yuv_in"])), Z["yuv_out"], atol=1e-3)
cmp("yuv_roundtrip", tf_yuv_to_rgb(tf_rgb_to_yuv(t(Z["yuv_in"]))), Z["yuv_rt"], atol=1e-3)

inputs = {k[4:]: Z[k] for k in Z.files if k.startswith("in__")}
print("== raw UNet output (conv/pool/upsample) — cross-framework conv float floor ==")
for k, v in inputs.items():
    adj = (t(v) - float(Z["mean_adjust"])) / float(Z["scale_adjust"])
    u = m.unet(adj.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    cmp(f"unet[{k}]", u, Z[f"unet__{k}"], atol=1e-3, rtol=RTOL_CONV)

print("== full run_preprocessor (residual + luma-only); signal_rel vs 255 ==")
for k, v in inputs.items():
    out = m(t(v))
    cmp(f"preproc[{k}]", out, Z[f"preproc__{k}"], atol=0.6, rtol=RTOL_CONV)
    # informational: uint8 codec-input diff (cross-framework conv float floor manifestation)
    o8 = np.rint(np.clip(out.detach().cpu().numpy(), 0, 255)).astype(np.int32)
    r8 = np.rint(np.clip(Z[f"preproc__{k}"], 0, 255)).astype(np.int32)
    frac = float(np.mean(o8 != r8)); maxd = int(np.max(np.abs(o8 - r8)))
    print(f"        (info) uint8 codec-input: {frac:.2%} px differ by <= {maxd} level")

# ---- DECISIVE downstream equivalence: jpeg(preproc_torch) vs jpeg(preproc_tf) -----
# What actually matters: identical preproc -> identical real-JPEG decoded PSNR + bits.
print("== DECISIVE: real-JPEG decoded PSNR + bits of torch-preproc vs tf-preproc ==")
from torch_port.codec import encode_decode_with_jpeg
def jpeg_psnr_bits(img_bhwc, q=32):
    dec, bits = encode_decode_with_jpeg(np.asarray(img_bhwc, np.float32), q, False, False)
    return dec, bits
for k, v in inputs.items():
    out_torch = m(t(v)).detach().cpu().numpy()
    out_tf = Z[f"preproc__{k}"]
    dt, bt = jpeg_psnr_bits(out_torch); df, bf = jpeg_psnr_bits(out_tf)
    # PSNR of each decoded vs the ORIGINAL input v (the real eval quantity)
    def psnr(a, b):
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return 10 * np.log10(255.0 ** 2 / mse)
    p_t = psnr(dt, v); p_f = psnr(df, v)
    dpsnr = abs(p_t - p_f); dbits = float(np.mean(np.abs(bt - bf)))
    bavg = float(np.mean(bf))
    ok = dpsnr <= 0.05 and dbits <= 0.01 * bavg + 64
    print(f"  [{'PASS' if ok else 'FAIL'}] downstream[{k}]  dPSNR={dpsnr:.4f}dB  dBits={dbits:.1f} (avg {bavg:.0f})")
    if not ok: fails.append(f"downstream[{k}]")

print("\n" + ("ALL PASS" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
