"""Order-k classical baseline figure (paper App., Fig. orderk_baselines.pdf).

Input: orderk_sweep_agg.csv (results/, from scripts/activation_analysis/orderk_sweep.py).
Output: orderk_baselines.pdf in the paper folder.
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
d = defaultdict(dict)
ceil = defaultdict(dict)
for r in rows:
    key = (r["process"], r["arch"])
    if r["quantity"] == "belief":
        d[key]["belief"] = (float(r["mean"]), float(r["ci95_hw"] or 0))
    else:
        k = int(r["k"])
        d[key][k] = (float(r["mean"]), float(r["ci95_hw"] or 0))
        if r["ceiling_257_over_Xk"]:
            ceil[r["process"]][k] = float(r["ceiling_257_over_Xk"])

procs = [("Bloch", "Bloch Walk", 4), ("Mess3", "Mess3", 3), ("Moon", "Moon", 3)]
archs = [("transformer", "Transformer", "#1f5fa8", "o"), ("LSTM", "LSTM", "#d08a00", "s"),
         ("GRU", "GRU", "#1a8f6a", "^"), ("vanilla_RNN", "vanilla RNN", "#c4441b", "D")]
ks = list(range(1, 9))

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for ax, (p, title, X) in zip(axes, procs):
    for a, lab, col, mk in archs:
        m = [d[(p, a)][k][0] for k in ks]
        e = [d[(p, a)][k][1] for k in ks]
        ax.plot(ks, m, marker=mk, ms=3, lw=1.1, color=col, label=lab)
        ax.fill_between(ks, [x - y for x, y in zip(m, e)], [x + y for x, y in zip(m, e)],
                        color=col, alpha=0.15, lw=0)
    b = min(d[(p, a)]["belief"][0] for a, *_ in archs)
    ax.axhline(b, color="0.35", ls="--", lw=0.8,
               label="minimal geometry (lowest arch. mean)" if p == "Bloch" else None)
    cv = [min(1.0, ceil[p].get(k, 1.0)) for k in ks]
    ax.plot(ks, cv, color="k", ls=":", lw=0.9, label=r"probe ceiling $\min(1, 257/|\mathcal{X}|^k)$" if p == "Bloch" else None)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_title(f"{title}  ($|\\mathcal{{X}}|={X}$)", fontsize=8.5)
    ax.set_xticks(ks)
    ax.set_xlabel("Markov order $k$")
    ax.set_ylim(-0.65, 1.05)
    ax.grid(alpha=0.25, lw=0.4)
axes[0].set_ylabel("weighted $R^2$ (5 seeds)")
axes[0].legend(fontsize=6, loc="lower left", frameon=False)
fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
