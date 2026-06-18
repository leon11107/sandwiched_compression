"""ORIGINAL torch LPIPS (richzhang) as an EVAL ground-truth metric, run in a SEPARATE
process to avoid the TF<->torch same-process segfault (importing both crashes; verified).

The TF eval process calls lpips_scores(refs, dists): it dumps the image batch to a
temp .npz and spawns `python -m distortion.lpips_torch_metric <npz> <out>` (torch ONLY,
no TF import) which computes per-frame LPIPS for net=alex and net=vgg and writes JSON.
The parent reads it back. EVAL-ONLY, not differentiable (training uses the TF port).

This is the user-specified metric's ORIGINAL implementation = the ground truth the TF
training port was cross-checked against (see memory reference_lpips_tf_port).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List

import numpy as np

_VENV_PY = sys.executable  # the venv python running this


def available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("lpips") is not None and \
               importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def lpips_scores(refs: List[np.ndarray], dists: List[np.ndarray],
                 nets=("alex", "vgg")) -> List[Dict[str, float]]:
    """Per-frame {lpips_alex, lpips_vgg} for equal-shape RGB [0,255] frame lists.
    Lower = better (0 identical). Runs torch LPIPS in a clean subprocess."""
    assert len(refs) == len(dists) and refs, "need equal non-empty ref/dist lists"
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "pairs.npz")
        out = os.path.join(td, "scores.json")
        np.savez(npz,
                 refs=np.stack(refs).astype("float32"),
                 dists=np.stack(dists).astype("float32"),
                 nets=np.array(list(nets)))
        # subprocess: torch only, no TF, no CUDA needed (CPU lpips)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""          # lpips on CPU
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        r = subprocess.run(
            [_VENV_PY, "-m", "distortion.lpips_torch_metric", npz, out],
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
            env=env, capture_output=True, text=True)
        if not os.path.exists(out):
            raise RuntimeError("lpips subprocess failed:\n" + r.stderr[-2000:])
        data = json.loads(open(out).read())
    return data["per_frame"]


def _worker(npz_path: str, out_path: str) -> None:
    """Child process: torch LPIPS only."""
    import torch
    import lpips as lpips_orig
    d = np.load(npz_path, allow_pickle=True)
    refs = d["refs"].astype("float32")          # [N,H,W,3] in [0,255]
    dists = d["dists"].astype("float32")
    nets = [str(x) for x in d["nets"].tolist()]
    n = refs.shape[0]
    models = {net: lpips_orig.LPIPS(net=net, verbose=False).eval() for net in nets}
    per_frame = []
    for i in range(n):
        a = torch.from_numpy(refs[i].transpose(2, 0, 1)[None] / 127.5 - 1.0).float()
        b = torch.from_numpy(dists[i].transpose(2, 0, 1)[None] / 127.5 - 1.0).float()
        row = {}
        with torch.no_grad():
            for net in nets:
                row[f"lpips_{net}"] = float(models[net](a, b).item())
        per_frame.append(row)
    json.dump({"per_frame": per_frame}, open(out_path, "w"))


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2])
