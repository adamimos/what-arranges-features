"""Redraw the RMSE bar column of Fig. 2 in the submitted style, with 5-seed 95% error bars,
and paste it over the old column of fig2_mainresult_fixed.png (scatter panels untouched).

Style measured from the submitted PNG (288 dpi): blue #467AEE, red #E14747, grey #818793,
bottom axis #666 1pt, gridlines #F7F7F7, tick labels #A6A6A6, Inter; bar width 0.40 (2 bars)
/ 0.26 (3 bars); axis x-span and baselines at the pixel rows of the original.
usage: fig2_rmse_bars.py metrics.csv old_fig.png out.png
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from PIL import Image

src, old, out = sys.argv[1:4]
_font_dirs = [os.environ.get("INTER_FONT_DIR", ""), "~/Library/Fonts", "~/.local/share/fonts", "/usr/share/fonts"]
for f in [g for d in _font_dirs if d for g in glob.glob(os.path.join(os.path.expanduser(d), "**", "Inter-*.otf"), recursive=True)]:
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Inter"

T975 = {4: 2.776, 3: 3.182, 2: 4.303}
d = defaultdict(list)
for r in csv.DictReader(open(src)):
    if r["converged"] == "True":
        d[(r["process"], r["arch"], r["checkpoint"].split(":")[0], r["target"])].append(float(r["rmse_cv"]))


def mci(p, a, ck, t):
    v = np.array(d[(p, a, ck, t)])
    return v.mean(), T975[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v))


BLUE, RED, GREY = "#467AEE", "#E14747", "#818793"
ROWS = [  # process, labels, bars, baseline px, top px, ylim, major, minor
    ("Mess3", ["Classical", "Random"], [("final", "belief", BLUE), ("init", "belief", GREY)], 757, 90, 0.50, 0.1, 0.05),
    ("Bloch", ["Quantum", "Classical", "Random"],
     [("final", "belief", BLUE), ("final", "markov3", RED), ("init", "belief", GREY)], 1603, 1013, 0.80, 0.2, 0.1),
    ("Moon", ["Post-Quant.", "Classical", "Random"],
     [("final", "belief", BLUE), ("final", "markov3", RED), ("init", "belief", GREY)], 2414, 1906, 0.80, 0.2, 0.1),
]
ARCHS = ["transformer", "LSTM"]

oldim = Image.open(old).convert("RGB")
W, H = oldim.size
X0 = 2786                      # left edge of the bar column in the full figure
AX_L, AX_R = 3005 - X0, 3568 - X0
cw = W - X0
DPI = 288
fig = plt.figure(figsize=(cw / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor("white")

LEG_TOP = {"Mess3": 90, "Bloch": 940, "Moon": 1815}   # legend top px in the submitted figure
for proc, labels, bars, base, top, ymax_tick, major, minor in ROWS:
    # 'top' = pixel row of the highest gridline (ymax_tick); extend the axes up to the legend top
    atop = min(top, LEG_TOP[proc])
    ymax = ymax_tick * (base - atop) / (base - top)
    ax = fig.add_axes([AX_L / cw, 1 - base / H, (AX_R - AX_L) / cw, (base - atop) / H])
    n = len(bars)
    w = 0.40 if n == 2 else 0.26
    offs = [-w / 2, w / 2] if n == 2 else [-w, 0, w]
    for (ck, t, col), off in zip(bars, offs):
        for i, a in enumerate(ARCHS):
            m, h = mci(proc, a, ck, t)
            ax.bar(i + off, m, w, color=col, edgecolor="none", zorder=2)
            ax.errorbar(i + off, m, yerr=h, color="#333333", lw=0.9, capsize=2.2, capthick=0.9, zorder=3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(True, which="both", axis="y", color="#F7F7F7", lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["bottom"].set_linewidth(1.0)
    # y tick labels sit just above their gridline, flush with the left edge (submitted style)
    ax.tick_params(axis="y", which="both", length=0, labelleft=False)
    from matplotlib.transforms import blended_transform_factory, offset_copy
    tr = offset_copy(blended_transform_factory(ax.transAxes, ax.transData), fig=fig, x=-4.5, y=1.6, units="points")
    for v in np.arange(0, ymax_tick + 1e-9, major):
        ax.text(0, v, "0" if abs(v) < 1e-12 else f"{v:g}", transform=tr, ha="left", va="bottom",
                fontsize=8.5, color="#A6A6A6")
    ax.tick_params(axis="x", length=0, labelsize=15.7, labelcolor="#333333", pad=5.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Transformer", "LSTM"])
    from matplotlib.transforms import ScaledTranslation
    lab = ax.get_xticklabels()[1]   # submitted figure has LSTM ~5pt right of its group centre
    lab.set_transform(lab.get_transform() + ScaledTranslation(5.2 / 72, 0, fig.dpi_scale_trans))
    ax.set_ylabel("RMSE", fontsize=15, color="#333333", labelpad=14)
    cols = [b[2] for b in bars]
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(cols, labels)],
              loc="upper left", bbox_to_anchor=(0.581, 1.0), frameon=False, fontsize=15.3,
              handlelength=0.6, handleheight=0.3, handletextpad=0.33, labelspacing=-0.12,
              borderpad=0.0, borderaxespad=0.0)

fig.canvas.draw()
col = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
assert col.size == (cw, H), (col.size, (cw, H))
new = oldim.copy()
new.paste(col, (X0, 0))
new.save(out, dpi=(DPI, DPI))
col.save(out.replace(".png", "_col.png"))
print("wrote", out)
for proc, *_ in ROWS:
    for a in ARCHS:
        print(proc, a, {f"{ck}-{t}": "%.4f±%.4f" % mci(proc, a, ck, t) for ck, t in [("final", "belief"), ("final", "markov3"), ("init", "belief")]})
