"""Torch full training-step equivalence: forward loss components + backward grads
(torch-env). Loss values exact-ish; grads compared by cosine + relative L2 norm
(cross-framework conv float floor)."""
import sys, numpy as np, torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.model import PreprocOnlyCodecModel, distortion_rate_loss
from torch_port.preproc import load_unet_weights

Z = np.load("/tmp/eq/train_ref.npz")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = 0.005
fails = []

m = PreprocOnlyCodecModel(gamma=GAMMA, qstep_init=32.0, quantizer_mode="straight_through",
                          codec_forward_mode="real_ste", convert_to_yuv=True,
                          preproc_luma_only=True, codec_luma_only=True,
                          scaler_init=float(Z["scaler"]), device=DEV)
m.train()
load_unet_weights(m.preproc.unet, [Z[f"w{i}"] for i in range(int(Z["nw"]))])

x = torch.from_numpy(Z["in"]).float().to(DEV).requires_grad_(False)
out = m(x)
loss_values = distortion_rate_loss(x, out, GAMMA, anchor="l1_msssim")
total = loss_values.mean()

# forward loss components
pred = out["prediction"]; norm = pred.shape[0] / float(np.prod(pred.shape))
from torch_port import losses as L
fid = float((L.distortion_l1_msssim(x, pred, 1.0).mean() * norm))
rate = float((GAMMA * out["rate"].mean() * norm))
print("== forward loss components ==")
for name, o, r in [("total", float(total), float(Z["total"])), ("fid", fid, float(Z["fid"])),
                   ("rate", rate, float(Z["rate"]))]:
    d = abs(o - r); ok = d <= max(1e-3, 2e-3 * abs(r))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:6s} torch={o:.6f} tf={r:.6f} |d|={d:.3e}")
    if not ok: fails.append(name)

# backward grads
total.backward()
convs = []
for enc in m.preproc.unet.encoders: convs += list(enc.convs)
for dec in m.preproc.unet.decoders: convs += list(dec.convs)
convs.append(m.preproc.unet.out)
# gvars order in TF: [k0,b0,k1,b1,...] then scaler
torch_grads = []
for cv in convs:
    torch_grads.append(cv.weight.grad.permute(2, 3, 1, 0).contiguous())  # ->[kh,kw,in,out]
    torch_grads.append(cv.bias.grad)
torch_grads.append(m.preproc.scaler.grad)

def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

print("== backward grads ==")
ng = int(Z["ng"])
rows = []  # (i, gnorm_tf, cosine, rel_norm)
flat_t, flat_f = [], []
for i in range(ng):
    gt = torch_grads[i].detach().cpu().numpy().astype(np.float64); gf = Z[f"g{i}"].astype(np.float64)
    flat_t.append(gt.reshape(-1)); flat_f.append(gf.reshape(-1))
    nf = np.linalg.norm(gf)
    if nf == 0 and np.linalg.norm(gt) == 0:
        continue
    rows.append((i, nf, cos(gt, gf), abs(np.linalg.norm(gt) - nf) / (nf + 1e-30)))
# global gradient cosine (the training-trajectory-relevant quantity)
gcos = cos(np.concatenate(flat_t), np.concatenate(flat_f))
# weighted: split params into "significant" (grad norm >= 1% of max) vs "tiny"
maxn = max(r[1] for r in rows)
sig = [r for r in rows if r[1] >= 0.01 * maxn]
tiny = [r for r in rows if r[1] < 0.01 * maxn]
sig_mincos = min(r[2] for r in sig); sig_maxrel = max(r[3] for r in sig)
print(f"  GLOBAL grad cosine (all params concatenated) = {gcos:.6f}")
print(f"  significant params (grad>=1% max, n={len(sig)}): min_cosine={sig_mincos:.6f} max_rel_norm={sig_maxrel:.3e}")
print(f"  tiny params (grad<1% max, n={len(tiny)}): "
      f"min_cosine={min((r[2] for r in tiny), default=1):.4f} (float-noise on near-zero grads)")
worst = sorted(rows, key=lambda r: r[2])[:4]
for i, nf, c, rel in worst:
    print(f"    worst[{i}]: tf_grad_norm={nf:.3e} cosine={c:.4f} rel_norm={rel:.3e}")
sc_i = ng - 1
print(f"  scaler grad: torch={float(torch_grads[sc_i]):.6e} tf={float(Z[f'g{sc_i}']):.6e}")
# acceptance: global cosine very high + significant params well-aligned
ok = (gcos >= 0.999) and (sig_mincos >= 0.99) and (sig_maxrel <= 0.05)
print(f"  [{'PASS' if ok else 'FAIL'}] global cosine>=0.999 AND significant-param cosine>=0.99 & rel<=5%")
if not ok: fails.append("grad")

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
