"""Single large threshold graph for the Phase 3 deck. Values as in make_depth_figures.py."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, CRIMS, GREY = "#1E2761", "#A3312F", "#8A8A8A"
# layers with image access (deletion placed before this layer)
qL = [e - 1 for e in [2,4,6,8,10,12,14,16,18,20,22,23,24,25,26,27]]
qS = [0.0718,0.0780,0.0914,0.0990,0.1094,0.1264,0.1354,0.1452,0.1372,0.1418,0.1584,0.2344,0.6454,0.7632,0.7806,0.7820]
iL = [2,4,6,8,10,12,14,16,17,18,19,20]
iS = [0.0780,0.0766,0.0690,0.0808,0.0868,0.1036,0.1378,0.1576,0.2616,0.3628,0.5740,0.7234]
qP = [100*s/0.8624 for s in qS]; iP = [100*s/0.7156 for s in iS]
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.plot(qL, qP, "o-", color=NAVY, lw=2.4, ms=5.5, label="Qwen2.5-VL-7B (28 layers)")
ax.plot(iL, iP, "s-", color=CRIMS, lw=2.4, ms=5.5, label="InternVL2-2B (24 layers)")
ax.axhline(50, color=GREY, ls="--", lw=1.2)
ax.text(0.8, 52.5, "50% of original performance", color=GREY, fontsize=10.5)
for x, c, t, dx, ty in [(22.48, NAVY, "Qwen threshold\n≈ 22.5 layers", 0.6, 22), (17.95, CRIMS, "InternVL threshold\n≈ 18 layers", -9.5, 36)]:
    ax.plot([x], [50], marker="*", color=c, ms=17, zorder=5)
    ax.annotate(t, xy=(x, 50), xytext=(x + dx, ty), color=c, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
ax.set_xlabel("Layer at which all remaining visual tokens are removed", fontsize=12)
ax.set_ylabel("TextVQA performance\n(% of original)", fontsize=12)
ax.set_xlim(0, 28.5); ax.set_ylim(0, 110)
ax.legend(frameon=False, loc="upper left", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_threshold_single.png"), dpi=200, facecolor="white")
print("wrote fig9_threshold_single.png")
