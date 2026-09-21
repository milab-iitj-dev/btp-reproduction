#!/usr/bin/env python3
"""
Depth sweep figures for the Phase 3 report.

Data are the measured TextVQA scores from the E11 and E12 sweeps, in which all visual
tokens are deleted at a given decoder layer and nothing else is changed. Graded pruning is
disabled in both, so the only variable is the depth of the deletion.

  Qwen2.5-VL-7B   jobs 421205-421218, 421231-421233, baseline 0.8624, 28 layers
  InternVL2-2B    jobs 419005-419012, 421234-421237, baseline 0.7156, 24 layers

Writes:
  figs/fig7_depth_sweep.png   two panels, absolute depth and relative depth
  figs/fig8_one_layer.png     what the shipped constant costs

Run from the reports/ directory:
  python make_depth_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

NAVY, TEAL, CRIMS, GREY = "#1F3864", "#2E8B8B", "#A3312F", "#8A8A8A"

# ---------------------------------------------------------------- measured data
qL = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 25, 26, 27]
qS = [0.0718, 0.0780, 0.0914, 0.0990, 0.1094, 0.1264, 0.1354, 0.1452,
      0.1372, 0.1418, 0.1584, 0.2344, 0.6454, 0.7632, 0.7806, 0.7820]
qB, qN = 0.8624, 28

iL = [2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20]
iS = [0.0780, 0.0766, 0.0690, 0.0808, 0.0868, 0.1036, 0.1378, 0.1576,
      0.2616, 0.3628, 0.5740, 0.7234]
iB, iN = 0.7156, 24

qP = [100 * s / qB for s in qS]
iP = [100 * s / iB for s in iS]
qD = [100 * l / qN for l in qL]
iD = [100 * l / iN for l in iL]


def half_crossing(layers, pct):
    """Linear interpolation of the layer at which the curve crosses 50 per cent."""
    for i in range(1, len(layers)):
        if pct[i - 1] < 50 <= pct[i]:
            f = (50 - pct[i - 1]) / (pct[i] - pct[i - 1])
            return layers[i - 1] + f * (layers[i] - layers[i - 1])
    return None


q_half = half_crossing(qL, qP)
i_half = half_crossing(iL, iP)
print("Qwen half-crossing      layer %.2f of %d = %.1f%% depth" % (q_half, qN, 100 * q_half / qN))
print("InternVL half-crossing  layer %.2f of %d = %.1f%% depth" % (i_half, iN, 100 * i_half / iN))

# ---------------------------------------------------------------- figure 7
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

a = axes[0]
a.plot(qL, qP, "o-", color=NAVY, lw=2, ms=4.5, label="Qwen2.5-VL-7B (28 layers)")
a.plot(iL, iP, "s-", color=TEAL, lw=2, ms=4.5, label="InternVL2-2B (24 layers)")
a.axhline(100, color=GREY, ls="--", lw=1)
a.text(28.2, 102, "no deletion", fontsize=8, color=GREY, ha="right")
a.axvline(23, color=CRIMS, ls=":", lw=1.4)
a.annotate("BTP ships\nlayer 23", xy=(23, 27.2), xytext=(14.5, 52),
           fontsize=8.5, color=CRIMS, ha="center",
           arrowprops=dict(arrowstyle="->", color=CRIMS, lw=1.1))
a.set_xlabel("Layer at which all visual tokens are deleted")
a.set_ylabel("TextVQA retained (% of baseline)")
a.set_ylim(0, 115)
a.set_xlim(0, 28.5)
a.set_title("By absolute depth", fontsize=11)
a.legend(frameon=False, fontsize=8.5, loc="upper left")

b = axes[1]
b.plot(qD, qP, "o-", color=NAVY, lw=2, ms=4.5, label="Qwen2.5-VL-7B")
b.plot(iD, iP, "s-", color=TEAL, lw=2, ms=4.5, label="InternVL2-2B")
b.axhline(100, color=GREY, ls="--", lw=1)
b.axhline(50, color=GREY, ls=":", lw=0.9)
b.text(6, 53, "half of baseline", fontsize=8, color=GREY)
b.plot([100 * i_half / iN], [50], "*", color=TEAL, ms=15, zorder=5)
b.plot([100 * q_half / qN], [50], "*", color=NAVY, ms=15, zorder=5)
# Offset the two labels vertically, the crossings are only nine points apart on this axis
b.annotate("InternVL\n%.0f%% depth" % (100 * i_half / iN),
           xy=(100 * i_half / iN, 50), xytext=(46, 62),
           fontsize=8.5, color=TEAL, ha="center", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))
b.annotate("Qwen\n%.0f%% depth" % (100 * q_half / qN),
           xy=(100 * q_half / qN, 50), xytext=(62, 33),
           fontsize=8.5, color=NAVY, ha="center", fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=NAVY, lw=1))
b.set_xlabel("Depth of the deletion (% through the decoder)")
b.set_ylabel("TextVQA retained (% of baseline)")
b.set_ylim(0, 115)
b.set_xlim(0, 100)
b.set_title("By relative depth: same shape, different threshold", fontsize=11)
b.legend(frameon=False, fontsize=8.5, loc="upper left")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_depth_sweep.png"), bbox_inches="tight", facecolor="white")
print("wrote fig7_depth_sweep.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 8
fig2, ax = plt.subplots(figsize=(4.6, 3.0))
lay = [22, 23, 24]
val = [0.1584, 0.2344, 0.6454]
pct = [100 * v / qB for v in val]
bars = ax.bar([str(l) for l in lay], pct, color=[CRIMS, CRIMS, TEAL], width=0.55)
for bb, p in zip(bars, pct):
    ax.text(bb.get_x() + bb.get_width() / 2, p + 2, "%.1f%%" % p, ha="center", fontsize=9)
ax.axhline(100, color=GREY, ls="--", lw=1)
ax.text(2.4, 102, "no deletion", fontsize=8, color=GREY, ha="right")
ax.annotate("", xy=(2, pct[2]), xytext=(1, pct[1]),
            arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.4))
ax.text(1.5, (pct[1] + pct[2]) / 2, "  +%.1f points\n  for one layer" % (pct[2] - pct[1]),
        fontsize=9, color=NAVY, ha="center")
ax.set_xlabel("Layer at which visual tokens are deleted")
ax.set_ylabel("TextVQA retained (%)")
ax.set_ylim(0, 118)
ax.set_title("What the shipped constant costs", fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig8_one_layer.png"), bbox_inches="tight", facecolor="white")
print("wrote fig8_one_layer.png")
