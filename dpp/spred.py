"""Image-adaptive s-predictor (line A): predict the 64-dim DCT-band luma
pre-scaling vector s(image, q) from image content, trained on oracle targets
(oracle_targets.jsonl).

Candidates (selected by REAL holdout iso-bpp VMAF_NEG delta, not MSE):
  - mean_s      : per-q train-mean s (non-adaptive reference)
  - ridge_aXX   : per-q multi-output ridge on standardized features, with
                  shrinkage toward the per-q mean (alpha in {0.5,0.75,1.0})
  - knn         : distance-weighted k-NN in feature space (k=5)
  - mlp         : small GPU MLP, q one-hot input, sigmoid output -> [0.3,1.5],
                  early stop on inner val split
Holdout split is BY IMAGE (20%) so no leakage across the two q rows.

Artifact: runs/spred_model.npz (+ .json report). Deploy entry point:
    from dpp.spred import SPredictor
    sp = SPredictor.load(path); s = sp.predict(rgb_f32, q)   # q in {8,20}
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, "/workspace/sandwiched_compression")
from distortion import vmaf_metric as vm
from dpp.oracle_vmafneg import (apply_s, block_dct, jpeg_rt, load16, luma,
                                up1080, vmaf_neg_batch)

TARGETS = "/workspace/sandwiched_compression/dpp/runs/oracle_targets.jsonl"
RUNS = "/workspace/sandwiched_compression/dpp/runs"
QS = (8, 20)
RC = (np.arange(64) // 8 + np.arange(64) % 8)


def features(rgb):
    """Content features from the ORIGINAL image (deploy-computable). -> [145]"""
    y = luma(rgb)
    c = block_dct(y).reshape(-1, 64)                      # [Nblk, 64]
    mean_abs = np.log1p(np.abs(c).mean(0))                # 64
    std = np.log1p(c.std(0))                              # 64
    e = (c ** 2).mean(0)
    etot = max(e.sum(), 1e-9)
    rc_frac = np.array([e[RC == k].sum() / etot for k in range(15)])  # 15
    g = np.array([y.mean() / 255.0, y.std() / 255.0])     # 2
    return np.concatenate([mean_abs, std, rc_frac, g])


class SPredictor:
    """npz-backed predictor; kind in {mean, ridge, knn, mlp}."""

    def __init__(self, d):
        self.d = d

    @classmethod
    def load(cls, path):
        return cls(dict(np.load(path, allow_pickle=False)))

    def predict(self, rgb, q):
        d = self.d
        f = (features(rgb) - d["mu"]) / d["sd"]
        kind = str(d["kind"])
        mean = d[f"mean_{q}"]
        if kind == "mean":
            s = mean
        elif kind == "ridge":
            s = mean + float(d["alpha"]) * (f @ d[f"W_{q}"] + d[f"b_{q}"] - mean)
        elif kind == "knn":
            X, Y = d[f"X_{q}"], d[f"Y_{q}"]
            dist = np.sqrt(((X - f) ** 2).sum(1))
            k = min(int(d["k"]), len(X))
            ii = np.argsort(dist)[:k]
            w = 1.0 / (dist[ii] + 1e-6)
            s = (Y[ii] * w[:, None]).sum(0) / w.sum()
        elif kind == "mlp":
            qf = np.array([1.0, 0.0] if q <= 8 else [0.0, 1.0])
            h = np.concatenate([f, qf])
            for li in range(int(d["n_layers"])):
                h = h @ d[f"l{li}_w"] + d[f"l{li}_b"]
                if li < int(d["n_layers"]) - 1:
                    h = np.maximum(h, 0)
            s = 0.3 + 1.2 / (1.0 + np.exp(-h))
        else:
            raise ValueError(kind)
        return np.clip(s, 0.3, 1.5)


def load_rows():
    rows = {}
    for line in open(TARGETS):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[(r["path"], r["q"])] = r   # dedup keep-last
    return list(rows.values())


def fit_ridge(X, Y, lam):
    """multi-output ridge with intercept on standardized X."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam  # don't penalize intercept
    W = np.linalg.solve(A, Xb.T @ Y)
    return W[:-1], W[-1]


def fit_mlp(Xtr, Qtr, Ytr, Xva, Qva, Yva, seed=0):
    import torch
    assert torch.cuda.is_available(), "[gpu-guard] CUDA required"
    dev = "cuda"
    torch.manual_seed(seed)
    qoh = lambda Q: np.stack([(Q <= 8).astype(np.float32),
                              (Q > 8).astype(np.float32)], 1)
    xtr = torch.tensor(np.concatenate([Xtr, qoh(Qtr)], 1), dtype=torch.float32, device=dev)
    ytr = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    xva = torch.tensor(np.concatenate([Xva, qoh(Qva)], 1), dtype=torch.float32, device=dev)
    yva = torch.tensor(Yva, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64))
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    act = lambda o: 0.3 + 1.2 * torch.sigmoid(o)
    best = (1e9, None)
    for ep in range(2000):
        opt.zero_grad()
        loss = torch.mean((act(net(xtr)) - ytr) ** 2)
        loss.backward(); opt.step()
        if ep % 20 == 0:
            with torch.no_grad():
                vl = float(torch.mean((act(net(xva)) - yva) ** 2))
            if vl < best[0] - 1e-6:
                best = (vl, [p.detach().cpu().numpy().copy() for p in net.parameters()])
    print(f"[mlp] best val MSE {best[0]:.5f}")
    return best[1]  # [w0,b0,w1,b1,w2,b2] with w as [out,in]


def holdout_eval(cands, ho_rows, workers=8):
    """Real-codec holdout: per (img,q) ONE vmaf NEG call over all candidates.
    Returns {cand: {q: mean iso delta}} + per-row detail."""
    names = list(cands)

    def one(r):
        rgb = load16(r["path"])
        y = luma(rgb); coeffs = block_dct(y)
        npix = rgb.shape[0] * rgb.shape[1]
        decs, bpps = [], []
        for n in names:
            s = cands[n].predict(rgb, r["q"])
            dec, bits = jpeg_rt(apply_s(rgb, coeffs, y, s), r["q"])
            decs.append(up1080(np.rint(dec).astype(np.uint8)))
            bpps.append(bits / npix)
        ref_up = up1080(np.rint(rgb).astype(np.uint8))
        with tempfile.TemporaryDirectory() as td:
            rdir = os.path.join(td, "r"); os.makedirs(rdir)
            ry4m = os.path.join(td, "ref.y4m")
            vm._png_seq_to_y4m([ref_up] * len(names), rdir, ry4m)
            vs = vmaf_neg_batch(ry4m, decs, td, threads=2)
        bc = r["base_curve"]
        bb, vb = np.array(bc["bpp"]), np.array(bc["vmaf_neg"])
        o = np.argsort(bb)
        return {n: v - float(np.interp(b, bb[o], vb[o]))
                for n, v, b in zip(names, vs, bpps)}

    with ThreadPoolExecutor(workers) as ex:
        details = list(ex.map(one, ho_rows))
    out = {n: {} for n in names}
    for q in QS:
        dq = [d for d, r in zip(details, ho_rows) if r["q"] == q]
        for n in names:
            out[n][q] = float(np.mean([d[n] for d in dq]))
    return out, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join(RUNS, "spred_model.npz"))
    a = ap.parse_args()
    rows = load_rows()
    imgs = sorted({r["path"] for r in rows})
    rng = np.random.default_rng(123)
    ho_imgs = set(rng.choice(imgs, int(len(imgs) * a.holdout_frac), replace=False))
    tr_rows = [r for r in rows if r["path"] not in ho_imgs]
    ho_rows = [r for r in rows if r["path"] in ho_imgs]
    print(f"{len(rows)} rows / {len(imgs)} imgs -> train {len(tr_rows)} rows, "
          f"holdout {len(ho_rows)} rows ({len(ho_imgs)} imgs)")

    feat_cache = {}
    for r in rows:
        if r["path"] not in feat_cache:
            feat_cache[r["path"]] = features(load16(r["path"]))
    F = np.stack([feat_cache[r["path"]] for r in tr_rows])
    mu, sd = F.mean(0), F.std(0) + 1e-6

    X = {q: np.stack([(feat_cache[r["path"]] - mu) / sd
                      for r in tr_rows if r["q"] == q]) for q in QS}
    Y = {q: np.stack([np.array(r["s"]) for r in tr_rows if r["q"] == q]) for q in QS}
    art = {"mu": mu, "sd": sd}
    for q in QS:
        art[f"mean_{q}"] = Y[q].mean(0)

    cands = {}
    cands["mean_s"] = SPredictor({**art, "kind": "mean"})

    # ridge: lambda by 5-fold CV on train (MSE), then shrinkage variants
    lams = [1.0, 10.0, 100.0, 300.0, 1000.0]
    cv_mse = {}
    for lam in lams:
        errs = []
        for q in QS:
            n = len(X[q]); fold = np.arange(n) % 5
            for k in range(5):
                if (fold == k).sum() == 0 or (fold != k).sum() < 10:
                    continue
                W, b = fit_ridge(X[q][fold != k], Y[q][fold != k], lam)
                errs.append(np.mean((X[q][fold == k] @ W + b - Y[q][fold == k]) ** 2))
        cv_mse[lam] = float(np.mean(errs))
    lam = min(cv_mse, key=cv_mse.get)
    mean_mse = float(np.mean([np.mean((Y[q] - Y[q].mean(0)) ** 2) for q in QS]))
    print(f"ridge CV MSE: {cv_mse} -> lam={lam} (predict-mean MSE {mean_mse:.5f})")
    ridge_art = dict(art)
    for q in QS:
        W, b = fit_ridge(X[q], Y[q], lam)
        ridge_art[f"W_{q}"], ridge_art[f"b_{q}"] = W, b
    for alpha in (0.5, 0.75, 1.0):
        cands[f"ridge_a{int(alpha*100)}"] = SPredictor(
            {**ridge_art, "kind": "ridge", "alpha": alpha})

    knn_art = dict(art)
    for q in QS:
        knn_art[f"X_{q}"], knn_art[f"Y_{q}"] = X[q], Y[q]
    cands["knn"] = SPredictor({**knn_art, "kind": "knn", "k": 5})

    # mlp: inner val split by image within train
    tr_imgs = sorted({r["path"] for r in tr_rows})
    iv = set(rng.choice(tr_imgs, max(1, len(tr_imgs) // 6), replace=False))
    def xqy(rs):
        Xm = np.stack([(feat_cache[r["path"]] - mu) / sd for r in rs])
        Qm = np.array([r["q"] for r in rs])
        Ym = np.stack([np.array(r["s"]) for r in rs])
        return Xm, Qm, Ym
    inner_tr = [r for r in tr_rows if r["path"] not in iv]
    inner_va = [r for r in tr_rows if r["path"] in iv]
    params = fit_mlp(*xqy(inner_tr), *xqy(inner_va))
    mlp_art = dict(art); mlp_art["kind"] = "mlp"; mlp_art["n_layers"] = 3
    for li in range(3):
        mlp_art[f"l{li}_w"] = params[2 * li].T   # torch Linear weight is [out,in]
        mlp_art[f"l{li}_b"] = params[2 * li + 1]
    cands["mlp"] = SPredictor(mlp_art)

    print(f"holdout real-codec eval: {len(ho_rows)} rows x {len(cands)} candidates")
    table, details = holdout_eval(cands, ho_rows)
    oracle_ho = {q: float(np.mean([r["iso_bpp_delta"] for r in ho_rows if r["q"] == q]))
                 for q in QS}
    print(f"{'candidate':12s}  q8 iso    q20 iso   mean")
    print(f"{'(oracle ub)':12s}  {oracle_ho[8]:+.3f}    {oracle_ho[20]:+.3f}   "
          f"{np.mean(list(oracle_ho.values())):+.3f}")
    scores = {}
    for n, t in table.items():
        scores[n] = float(np.mean([t[q] for q in QS]))
        print(f"{n:12s}  {t[8]:+.3f}    {t[20]:+.3f}   {scores[n]:+.3f}")
    bestn = max(scores, key=scores.get)
    print(f"WINNER: {bestn} (holdout mean iso {scores[bestn]:+.3f})")

    bd = cands[bestn].d
    np.savez(a.out, **{k: (np.asarray(v) if not isinstance(v, np.ndarray) else v)
                       for k, v in bd.items()})
    json.dump({"winner": bestn, "scores": scores,
               "table": {n: {str(q): t[q] for q in QS} for n, t in table.items()},
               "oracle_holdout": {str(q): oracle_ho[q] for q in QS},
               "ridge_lam": lam, "cv_mse": {str(k): v for k, v in cv_mse.items()},
               "n_train_rows": len(tr_rows), "n_holdout_rows": len(ho_rows)},
              open(os.path.join(RUNS, "spred_report.json"), "w"), indent=2)
    print(f"saved {a.out} + spred_report.json")


if __name__ == "__main__":
    main()
