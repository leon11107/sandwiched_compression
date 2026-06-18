"""Torch-side equivalence checker (run in /venv/torch-env).

Loads the TF reference npz, runs the torch codec port on the SAME input arrays,
compares op-level + full-codec + gradient outputs with tolerances, and runs
torch-only sanity checks. Prints a PASS/FAIL table. Exit code != 0 on any FAIL.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.codec import JpegProxyTorch, EncodeDecodeIntraTorch

REF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eq/codec_ref.npz"
Z = np.load(REF)
T = torch.float64  # compare in float64 to separate algorithm vs float32 noise? -> use f32
T = torch.float32
fails = []

def t(a):
    return torch.from_numpy(np.asarray(a)).to(T)

def cmp(name, torch_out, ref_key, atol, rtol=1e-4):
    ref = Z[ref_key]
    o = torch_out.detach().cpu().numpy().astype(np.float64)
    r = ref.astype(np.float64)
    if o.shape != r.shape:
        print(f"  [FAIL] {name:34s} shape {o.shape} vs ref {r.shape}"); fails.append(name); return
    mabs = float(np.max(np.abs(o - r)))
    denom = np.maximum(np.abs(r), 1e-6)
    mrel = float(np.max(np.abs(o - r) / denom))
    ok = (mabs <= atol) or (mrel <= rtol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} max_abs={mabs:.3e} max_rel={mrel:.3e}")
    if not ok:
        fails.append(name)

inputs = {k[4:]: Z[k] for k in Z.files if k.startswith("in__")}

# ---- op-level: DCT fwd/inv, YCbCr (convert_to_yuv=True, clip on) ------------
jp = JpegProxyTorch(convert_to_yuv=True, clip_to_image_max=True, dtype=T)
print("== op-level: DCT forward / inverse / YCbCr ==")
for k, v in inputs.items():
    ch = t(v)[..., 0:1]
    fwd = jp._forward_dct_2d(ch)
    cmp(f"dctfwd[{k}]", fwd, f"dctfwd__{k}", atol=1e-2)
    cmp(f"dctinv[{k}]", jp._inverse_dct_2d(t(Z[f'dctfwd__{k}'])), f"dctinv__{k}", atol=1e-2)
    cmp(f"rgb2yuv[{k}]", jp._rgb_to_yuv(t(v)), f"rgb2yuv__{k}", atol=1e-3)
    cmp(f"yuv2rgb[{k}]", jp._yuv_to_rgb(jp._rgb_to_yuv(t(v))), f"yuv2rgb__{k}", atol=1e-3)

# ---- full EncodeDecodeIntra: proxy (log_nonzero rate) ----------------------
def build(mode, q):
    m = EncodeDecodeIntraTorch(qstep_init=float(q), train_qstep=False, min_qstep=1.0,
                               quantizer_mode="straight_through", rate_proxy_mode="log_nonzero",
                               codec_forward_mode=mode, output_clip_mode="hard",
                               convert_to_yuv=True, dtype=T)
    m.eval()
    return m

print("== full codec: proxy decoded + log_nonzero rate ==")
for q in [1, 16, 32, 64, 255]:
    m = build("proxy", q)
    for k, v in inputs.items():
        dec, rate = m(t(v))
        cmp(f"proxydec[{k},q{q}]", dec, f"proxydec__{k}__q{q}", atol=2e-2)
        cmp(f"proxyrate[{k},q{q}]", rate, f"proxyrate__{k}__q{q}", atol=1.0, rtol=1e-3)

print("== full codec: real_ste decoded (= real PIL) + rate ==")
mste = build("real_ste", 32)
for k, v in inputs.items():
    dec, rate = mste(t(v))
    cmp(f"stedec[{k}]", dec, f"stedec__{k}__q32", atol=1e-3)   # PIL bit-exact
    cmp(f"sterate[{k}]", rate, f"sterate__{k}__q32", atol=1.0, rtol=1e-3)

# ---- gradient equivalence (proxy, straight_through, L=sum(dec)+sum(rate)) ---
print("== gradient: dL/dx (proxy, straight_through) ==")
for k in ["rand_2x16", "rand_3x128"]:
    m = build("proxy", 32)
    xv = t(inputs[k]).requires_grad_(True)
    dec, rate = m(xv)
    L = dec.sum() + rate.sum()
    L.backward()
    cmp(f"grad[{k}]", xv.grad, f"grad__{k}__q32", atol=1e-2, rtol=1e-3)
    Lref = float(Z[f"gradL__{k}__q32"])
    dL = abs(float(L) - Lref)
    okL = dL <= max(1e-2, 1e-4 * abs(Lref))
    print(f"  [{'PASS' if okL else 'FAIL'}] gradL[{k}]                    |L_torch-L_tf|={dL:.3e} (L={Lref:.1f})")
    if not okL: fails.append(f"gradL[{k}]")

# ---- torch-only sanity checks ----------------------------------------------
print("== sanity (torch-only) ==")
I = jp.dct_2d @ jp.dct_2d.t()
ortho = float((I - torch.eye(64, dtype=T)).abs().max())
print(f"  [{'PASS' if ortho < 1e-4 else 'FAIL'}] dct orthogonality           max_abs={ortho:.3e}")
if ortho >= 1e-4: fails.append("dct_ortho")
x = t(inputs["rand_3x128"])
rt = jp._yuv_to_rgb(jp._rgb_to_yuv(x))
rterr = float((rt - x).abs().max())
print(f"  [{'PASS' if rterr < 1e-3 else 'FAIL'}] ycbcr round-trip identity    max_abs={rterr:.3e}")
if rterr >= 1e-3: fails.append("ycbcr_rt")
# proxy round-trip at qstep=1 reconstructs the rounded input closely
mfine = build("proxy", 1)
dec1, _ = mfine(x)
rterr2 = float((dec1 - torch.round(x).clamp(0, 255)).abs().mean())
print(f"  [{'PASS' if rterr2 < 1.0 else 'FAIL'}] proxy round-trip @q1 (mean)  mean_abs={rterr2:.3e}")
if rterr2 >= 1.0: fails.append("proxy_rt_q1")

print("\n" + ("ALL PASS" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(1 if fails else 0)
