"""Figures for the top-level README.

Experiment R figures are drawn from figures/exp1_numbers.json (copied from exp1_antidote's analysis summaries).
Experiment U figures live in exp2_erased_or_hidden/analysis/figures and are produced by its analyze.py.
    python figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "exp1_numbers.json").read_text())
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# R1: harmful score after each repair, English and Russian attack
arms = ["Antidote", "Magnitude prune", "Benign-calibrated", "Norm shrink", "Random prune"]
fig, ax = plt.subplots(figsize=(7.4, 3.3), dpi=200)
w = 0.38
for i, (lang, col) in enumerate((("English", BLUE), ("Russian", AQUA))):
    xs = [j - w / 2 + i * w for j, a in enumerate(arms) if a in N["repair_hs"][lang]]
    ys = [N["repair_hs"][lang][a] for a in arms if a in N["repair_hs"][lang]]
    ax.bar(xs, ys, w * 0.92, color=col, label=f"{lang} attack (unrepaired: {N['attacked_hs'][lang]:.0f})", zorder=2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.8, f"{y:.1f}", ha="center", fontsize=7.5, color=INK)
ax.axhline(N["clean_hs"], color=INK2, lw=1, ls="--")
ax.text(2.42, N["clean_hs"] + 0.6, f"clean {N['clean_hs']}", ha="left", fontsize=7.5, color=INK2)
ax.set_xticks(range(len(arms)))
ax.set_xticklabels(arms)
ax.set_ylabel("harmful score (lower = safer)")
ax.set_ylim(0, 35)
ax.set_title("Antidote repairs the attack, but magnitude pruning with no data does the same", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(HERE / "r1_repair_vs_nulls.png")
plt.close(fig)

# R2: relearning re-attack
fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=200)
steps = N["relearn_steps"]
for name, col, lw in (("Attacked, unrepaired", GRAY, 2.5), ("Antidote", BLUE, 2), ("Random prune", ORANGE, 2), ("Norm shrink", AQUA, 2)):
    ys = N["relearn_hs"][name]
    ax.plot(steps, ys, color=col, lw=lw, marker="o", ms=5, mec="white", mew=1.5, label=name, zorder=3)
ax.axhline(N["clean_hs"], color=INK2, lw=1, ls="--")
ax.text(1, N["clean_hs"] + 1.5, "clean model", ha="left", fontsize=7.5, color=INK2)
ax.set_xticks(steps)
ax.set_xlabel("harmful fine-tuning steps after repair (64 fresh examples)")
ax.set_ylabel("harmful score")
ax.set_ylim(0, 80)
ax.set_title("Ten steps of harmful fine-tuning undo the repair", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.38))
style(ax)
fig.tight_layout()
fig.savefig(HERE / "r2_relearning.png")
plt.close(fig)
# R3: relative repair, standard attack vs an attacker who knows Antidote's pruning rule
arms3 = ["Antidote", "Magnitude prune", "Norm shrink", "Random prune"]
fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=200)
w = 0.38
for i, (cond, col) in enumerate((("Standard attack", BLUE), ("Mask-aware attacker", ORANGE))):
    xs = [j - w / 2 + i * w for j in range(len(arms3))]
    ys = [N["relative_repair"][cond][a] for a in arms3]
    ax.bar(xs, ys, w * 0.92, color=col, label=cond, zorder=2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(len(arms3)))
ax.set_xticklabels(arms3)
ax.set_ylabel("relative repair\n(1 = back to clean, 0 = no effect)")
ax.set_ylim(0, 1.12)
ax.set_title("An attacker who knows the pruning rule defeats Antidote", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper right")
style(ax)
fig.tight_layout()
fig.savefig(HERE / "r3_mask_aware.png")
plt.close(fig)
print("wrote", HERE / "r1_repair_vs_nulls.png", HERE / "r2_relearning.png", HERE / "r3_mask_aware.png")
