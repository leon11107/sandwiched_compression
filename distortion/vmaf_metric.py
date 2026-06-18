"""VMAF / VMAF_NEG metric via the from-source libvmaf CLI (see memory
reference_vmaf_build). EVAL-ONLY (CPU subprocess, not differentiable).

Why VMAF_NEG: the DPP preprocessor learned to game a full-reference VGG loss by
global contrast/saturation enhancement (PSNR -10 dB but VGG "better", bpp up).
Standard VMAF rewards that enhancement; VMAF_NEG clips the local gain terms to 1.0
so it does NOT credit pure contrast/brightness boosts — verified: on a contrast-
boosted pair, VMAF=92.08 but VMAF_NEG=78.45. So VMAF_NEG is the eval guard that
exposes this gaming.

Design: batch a list of equal-size RGB frames into ONE multi-frame y4m (via ffmpeg
from PNGs — ffmpeg here canNOT compute vmaf but DOES convert), then one `vmaf` call
PER MODEL (multi-model single-call mis-reports — both come back as the std score;
verified). Returns per-frame {vmaf, vmaf_neg}.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

_VMAF_BIN = "/usr/local/bin/vmaf"
_LD = "/usr/local/lib/x86_64-linux-gnu"
_FFMPEG = "/usr/bin/ffmpeg"
_REPO = Path(__file__).resolve().parents[1]
_MODEL_STD = str(_REPO / "reference" / "vmaf_models" / "vmaf_v0.6.1.json")
_MODEL_NEG = str(_REPO / "reference" / "vmaf_models" / "vmaf_v0.6.1neg.json")


def available() -> bool:
    return (os.path.exists(_VMAF_BIN) and os.path.exists(_FFMPEG)
            and os.path.exists(_MODEL_STD) and os.path.exists(_MODEL_NEG))


def _to_uint8(a: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(a, 0, 255)).astype(np.uint8)


def _png_seq_to_y4m(frames: List[np.ndarray], png_dir: str, y4m: str) -> None:
    for i, f in enumerate(frames):
        Image.fromarray(_to_uint8(f), mode="RGB").save(
            os.path.join(png_dir, "f_%04d.png" % i))
    # -pix_fmt yuv420p matches the validated test; default BT.601 for SD content.
    cmd = [_FFMPEG, "-y", "-loglevel", "error", "-framerate", "1",
           "-i", os.path.join(png_dir, "f_%04d.png"),
           "-pix_fmt", "yuv420p", y4m]
    subprocess.run(cmd, check=True)


def _run_vmaf(ref_y4m: str, dist_y4m: str, model_path: str, out_json: str) -> List[float]:
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = _LD
    cmd = [_VMAF_BIN, "--reference", ref_y4m, "--distorted", dist_y4m,
           "--model", "path=" + model_path, "--output", out_json, "--json"]
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.loads(Path(out_json).read_text())
    # per-frame metrics live in d['frames'][i]['metrics']; key starts with 'vmaf'
    out = []
    for fr in d["frames"]:
        m = fr["metrics"]
        k = next(x for x in m if x.startswith("vmaf"))
        out.append(float(m[k]))
    return out


def vmaf_scores(refs: List[np.ndarray], dists: List[np.ndarray]) -> List[Dict[str, float]]:
    """Per-frame {vmaf, vmaf_neg} for equal-size RGB [0,255] frame lists.
    Higher = better; vmaf_neg refuses to credit pure contrast/brightness gaming."""
    assert len(refs) == len(dists) and refs, "need equal non-empty ref/dist lists"
    # y4m holds equal-size frames ONLY. Mixed sizes get SILENTLY warped by ffmpeg
    # to frame-0 dims (loglevel error hides the warning) => corrupted scores.
    # (Bug found 2026-06-10: eval_v2 batched 50 different-size val50 frames.)
    shapes = {f.shape for f in refs} | {f.shape for f in dists}
    assert len(shapes) == 1, f"vmaf_scores requires equal-size frames, got {shapes}"
    with tempfile.TemporaryDirectory() as td:
        rdir = os.path.join(td, "ref"); ddir = os.path.join(td, "dist")
        os.makedirs(rdir); os.makedirs(ddir)
        ry4m = os.path.join(td, "ref.y4m"); dy4m = os.path.join(td, "dist.y4m")
        _png_seq_to_y4m(refs, rdir, ry4m)
        _png_seq_to_y4m(dists, ddir, dy4m)
        std = _run_vmaf(ry4m, dy4m, _MODEL_STD, os.path.join(td, "std.json"))
        neg = _run_vmaf(ry4m, dy4m, _MODEL_NEG, os.path.join(td, "neg.json"))
    return [{"vmaf": s, "vmaf_neg": n} for s, n in zip(std, neg)]


if __name__ == "__main__":
    # self-test: identical pair -> ~100/100; contrast-boosted -> vmaf high, vmaf_neg lower
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(256, 256, 3)).astype(np.float32)
    # smooth it a bit so vmaf is meaningful (random noise saturates features)
    from PIL import ImageFilter
    base = np.asarray(Image.fromarray(_to_uint8(base)).filter(
        ImageFilter.GaussianBlur(3)), dtype=np.float32)
    mean = base.mean(axis=(0, 1), keepdims=True)
    enh = np.clip((base - mean) * 1.3 + mean + 8.0, 0, 255)
    print("available():", available())
    res = vmaf_scores([base, base], [base, enh])
    print("frame0 (identical):", res[0])
    print("frame1 (contrast-boosted):", res[1])
    ok = (res[0]["vmaf_neg"] > 95 and res[1]["vmaf_neg"] < res[1]["vmaf"])
    print("SELF_TEST:", "PASS" if ok else "REVIEW",
          "(identical~100; boosted vmaf_neg < vmaf => gaming penalized)")
