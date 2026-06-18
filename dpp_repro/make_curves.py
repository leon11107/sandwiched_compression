"""Generate the performance-curve figures for the analysis/tech docs.
All numbers are the measured held-out BD-rates from this session's runs.
Outputs dpp_repro/fig_*.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/workspace/sandwiched_compression/dpp_repro"

# ---- 1. perceptual-term (gamma) dose-response, held-out single-model BD ----
gammas = [0.0, 0.1, 0.5, 2.0, 5.0]
ah = [-3.95, -5.65, -8.49, -9.09, -9.12]
vmaf = [-6.71, -8.77, -14.02, -14.93, -14.48]
ssim = [-2.12, -2.34, -3.01, -2.43, -2.12]

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.plot(gammas, ah, "o-", label="AH-VMAF (=VMAF_NEG)", lw=2)
ax.plot(gammas, vmaf, "s-", label="VMAF", lw=2)
ax.plot(gammas, ssim, "^-", label="SSIM", lw=2)
ax.axvspan(0.5, 2.0, color="green", alpha=0.08, label="usable range")
ax.set_xlabel("gamma (perceptual-loss weight)")
ax.set_ylabel("held-out BD-rate % (negative = win)")
ax.set_title("Perceptual-term dose-response (x264, single model)")
ax.grid(alpha=0.3); ax.legend(); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gamma_doseresponse.png", dpi=130)

# ---- 2. reproduction arc: how the two levers close the gap ----------------
stages = ["initial\n(6 clips, g0.1)", "g0.5-2\n(6 clips)",
          "g0.5-2\n(19 clips)", "paper\nTable 1"]
x = np.arange(len(stages))
m = {"SSIM": [-2.53, -3.37, -4.00, -4.67],
     "AH-VMAF": [-6.53, -9.27, -11.19, -12.27],
     "VMAF": [-10.93, -15.18, -18.98, -25.08]}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for k, mk in zip(m, ("o-", "s-", "^-")):
    ax.plot(x, m[k], mk, label=k, lw=2, ms=7)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("held-out BD-rate % (negative = win)")
ax.set_title("Reproduction arc: perceptual weight + corpus scale close the gap")
ax.grid(alpha=0.3); ax.legend(); ax.invert_yaxis()
for k in m:
    ax.annotate(f"{m[k][-2]:.1f}", (x[2], m[k][2]), textcoords="offset points",
                xytext=(0, -12), ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_reproduction_arc.png", dpi=130)

# ---- 3. cross-codec transfer (pooled g05+g2, held-out, no retraining) ------
codecs = ["x264\nslow", "x265\nslow", "AV1\ncpu5"]
cm = {"SSIM": [-3.37, -3.05, -3.30], "AH-VMAF": [-9.27, -8.56, -7.80],
      "VMAF": [-15.18, -16.99, -16.70]}
x = np.arange(len(codecs)); w = 0.25
fig, ax = plt.subplots(figsize=(6.6, 4.6))
for i, k in enumerate(cm):
    ax.bar(x + (i - 1) * w, [-v for v in cm[k]], w, label=k)
ax.set_xticks(x); ax.set_xticklabels(codecs)
ax.set_ylabel("held-out BD-rate gain % (higher = better)")
ax.set_title("Encoder-agnostic transfer (precoder trained on virtual H.264 only)")
ax.grid(alpha=0.3, axis="y"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_cross_codec.png", dpi=130)

# ---- 4. JPEG-line two-metric frontier (the AND-bar infeasibility) ----------
fig, ax = plt.subplots(figsize=(6.4, 5.2))
# hull-selection frontier (alpha sweep), fixed resolution
al_mss = [-6.65, -2.67, 0.98, 3.35, 4.02, 4.51, 4.57, 5.05, 5.49, 5.79]
al_neg = [40.26, 2.94, -1.75, -3.11, -3.51, -3.67, -3.85, -4.07, -4.39, -4.77]
ax.plot(al_mss, al_neg, "b-o", ms=4, label="selection frontier (alpha sweep)")
pts = {"C_l05 (MS-SSIM specialist)": (-5.27, 52.25),
       "adaptive spred (NEG specialist)": (5.12, -3.48),
       "baseline": (0, 0)}
for nm, (xx, yy) in pts.items():
    ax.plot(xx, yy, "r^", ms=8); ax.annotate(nm, (xx, yy), fontsize=8,
                                             xytext=(4, 4), textcoords="offset points")
ax.axhline(-5, color="g", ls="--", lw=1); ax.axvline(-2, color="g", ls="--", lw=1)
ax.fill_between([-8, -2], -5, -12, color="green", alpha=0.08)
ax.text(-7.5, -8, "AND-bar target\n(empty)", color="green", fontsize=9)
ax.set_xlabel("BD-rate MS-SSIM % (negative = win)")
ax.set_ylabel("BD-rate VMAF_NEG % (negative = win)")
ax.set_title("JPEG fixed-res: the two metrics anti-correlate (AND-bar infeasible)")
ax.grid(alpha=0.3); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_jpeg_frontier.png", dpi=130)

print("wrote fig_gamma_doseresponse, fig_reproduction_arc, fig_cross_codec, fig_jpeg_frontier")
