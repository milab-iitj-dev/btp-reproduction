#!/usr/bin/env python3
"""Investigation workflow figure for the Phase 3 report."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})

NAVY, TEAL, CRIMS, GREY = "#1F3864", "#2E8B8B", "#A3312F", "#666666"

fig, ax = plt.subplots(figsize=(7.4, 3.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

def box(x, y, w, h, title, body, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                linewidth=1.3, edgecolor=edge, facecolor=fill))
    ax.text(x + w/2, y + h - 3.2, title, ha="center", va="top",
            fontsize=9, fontweight="bold", color=edge)
    ax.text(x + w/2, y + h - 8.0, body, ha="center", va="top",
            fontsize=8, color="#1A1A1A", linespacing=1.45)

def arrow(x1, y1, x2, y2, label=None, colour=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color=colour))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 1.4, label, ha="center", fontsize=7.5,
                color=colour, style="italic")

box(1, 26, 21, 18, "Observation",
    "TextVQA 86.2 to 23.3\nunder BTP, AI2D\nalmost unchanged", NAVY, "#EDF1F8")
box(27, 26, 21, 18, "Four tests",
    "selector, similarity,\nbudget, OCR oracle\nall return 17 to 20", NAVY, "#EDF1F8")
box(53, 26, 21, 18, "Control",
    "100% retention also\nscores 0.178, so the\ncause is downstream", CRIMS, "#FCE9E7")
box(79, 26, 20, 18, "Code",
    "layer 23 deletes\nall image tokens\nunconditionally", CRIMS, "#FCE9E7")

box(79, 2, 20, 18, "Intervention",
    "one condition off,\n0.1725 to 0.7860,\nAI2D unmoved", TEAL, "#E9F2EC")
box(53, 2, 21, 18, "Threshold sweep",
    "move the deletion:\n22.5 layers in Qwen,\n18.0 in InternVL", TEAL, "#E9F2EC")
box(27, 2, 21, 18, "Second instrument",
    "forward hook agrees\nwithin a few points\non the same curve", TEAL, "#E9F2EC")
box(1, 2, 21, 18, "Decomposition",
    "pruning and deletion\ncosts separated;\nChartQA reverses", TEAL, "#E9F2EC")

arrow(22.5, 35, 26.5, 35)
arrow(48.5, 35, 52.5, 35)
arrow(74.5, 35, 78.5, 35)
arrow(89, 25.5, 89, 20.5)
arrow(78.5, 11, 74.5, 11)
arrow(52.5, 11, 48.5, 11)
arrow(26.5, 11, 22.5, 11)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig0_workflow.png"), bbox_inches="tight",
            facecolor="white", dpi=200)
print("wrote fig0_workflow.png")
