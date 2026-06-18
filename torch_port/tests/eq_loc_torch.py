"""Localize the UNet port mismatch: torch per-stage compare (torch-env)."""
import sys, numpy as np, torch
sys.path.insert(0, "/workspace/sandwiched_compression")
from torch_port.preproc import PreprocOnlyTorch, load_unet_weights
L = np.load("/tmp/eq/loc_ref.npz")
W = np.load("/tmp/eq/preproc_ref.npz")
m = PreprocOnlyTorch(128, 255, True, float(W["scaler"])); m.eval()
load_unet_weights(m.unet, [W[f"w{i}"] for i in range(int(W["nw"]))])

def d(name, o, ref):
    o = o.detach().cpu().numpy().astype(np.float64); r = np.asarray(ref, np.float64)
    shp = "OK" if o.shape == r.shape else f"SHAPE {o.shape}vs{r.shape}"
    print(f"  {name:14s} max_abs={np.max(np.abs(o-r)):.3e}  {shp}")

adj = torch.from_numpy(L["adj"]).float().permute(0, 3, 1, 2).contiguous()  # BCHW
cur = adj
sk = []
u = m.unet
for i, enc in enumerate(u.encoders):
    cur, s = enc(cur)
    d(f"enc{i}_pooled", cur.permute(0, 2, 3, 1), L[f"enc{i}_pooled"])
    d(f"enc{i}_skip", s.permute(0, 2, 3, 1), L[f"enc{i}_skip"])
    sk.append(s)
sk.append(None)
n = len(sk)
for i, dec in enumerate(u.decoders):
    cur = dec(cur, sk[n - 1 - i])
    d(f"dec{i}", cur.permute(0, 2, 3, 1), L[f"dec{i}"])
d("final", u.out(cur).permute(0, 2, 3, 1), L["final"])
