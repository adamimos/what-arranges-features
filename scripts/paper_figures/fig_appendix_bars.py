"""Replace the RMSE bar column of an App. G figure (belief_grid_<Arch>_extended_combined_last_fixed.pdf)
with 5-seed bars in the original FigAppendix.py style (+ 95% error bars), as a vector overlay on the
original PDF page. Scatter panels, titles, row labels are untouched.

usage: fig_appendix_bars.py metrics.csv[,frdn.csv] in.pdf ARCHKEY out.pdf
  ARCHKEY: transformer | LSTM | GRU | vanilla_RNN
Geometry (288-dpi px, measured): bar-axis bottoms at y = 686/1316/1947/2577, each 600 px tall;
x-span 1353-1720 (Transformer figure) or 1360-1719 (the RNN figures).
FRDN uses in-sample RMSE: its fit is underdetermined (256 histories, 257 features), so held-out
RMSE is not meaningful; the caption says so.
"""
import csv
import io
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from pypdf import PdfReader, PdfWriter

csvs, src, arch, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4]
plt.rcParams["font.family"] = "DejaVu Sans"
T975 = {4: 2.776, 3: 3.182, 2: 4.303}
d = defaultdict(list)
for f in csvs:
    for r in csv.DictReader(open(f)):
        if r["converged"] != "True":
            continue
        key = (r["process"], r["arch"], r["checkpoint"].split(":")[0], r["target"])
        d[key].append(float(r["rmse_in"] if r["process"] == "FRDN" else r["rmse_cv"]))

colors = {"correct": "#2563eb", "incorrect": "#dc2626", "random": "#6b7280"}
ROWS = [("Mess3", 686, ["Classical", "Random"]), ("Bloch", 1316, ["Quantum", "Classical", "Random"]),
        ("FRDN", 1947, ["Quantum", "Classical", "Random"]), ("Moon", 2577, ["Post-Quant", "Classical", "Random"])]
XL, XR = (1353, 1720) if arch == "transformer" else (1360, 1719)
COVER_X = 1290  # left edge of the white patch (bar column starts with the RMSE label at ~1300)

reader = PdfReader(src)
page = reader.pages[0]
Wpt, Hpt = float(page.mediabox.width), float(page.mediabox.height)
fig = plt.figure(figsize=(Wpt / 72, Hpt / 72), dpi=72)
fig.patch.set_alpha(0)
px = 0.25  # pt per 288-dpi pixel


def mci(p, ck, t):
    v = np.array(d[(p, arch, ck, t)])
    if len(v) == 0:
        return None
    return v.mean(), T975[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


drawn = []
for proc, base, labels in ROWS:
    keys = [("final", "belief", "correct"), ("init", "belief", "random")] if proc == "Mess3" else \
           [("final", "belief", "correct"), ("final", "markov3", "incorrect"), ("init", "belief", "random")]
    stats = [mci(proc, ck, t) for ck, t, _ in keys]
    if any(s is None for s in stats):
        continue  # no data for this row: keep the original bars
    drawn.append(proc)
    top = base - 600
    # white patch over the old column (ylabel, ticks, bars, legend), in figure coordinates
    y0 = (base + 14) * px
    y1 = (top - (0 if proc == "Mess3" else 4)) * px
    fig.patches.append(Rectangle((COVER_X * px / Wpt, 1 - y0 / Hpt), 1 - COVER_X * px / Wpt, (y0 - y1) / Hpt,
                                 transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=-5))
    ax = fig.add_axes([XL * px / Wpt, 1 - base * px / Hpt, (XR - XL) * px / Wpt, 600 * px / Hpt])
    ax.patch.set_alpha(0)
    n = len(keys)
    width = 0.4 if n == 2 else 0.26
    pos = [-width / 2, width / 2] if n == 2 else [-width, 0, width]
    tops = [m + h for m, h in stats]
    for (ck, t, c), (m, h), x in zip(keys, stats, pos):
        ax.bar(x, m, width, color=colors[c], alpha=0.85, edgecolor="none")
        ax.errorbar(x, m, yerr=h, color="#333333", lw=0.6, capsize=1.6, capthick=0.6)
        ax.text(x, m + h + max(tops) * 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=6, color="#333333")
    ax.set_ylim(0, max(tops) * 1.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#666666")
    ax.set_ylabel("RMSE", fontsize=9, color="#333333")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7, length=3, width=0.5, color="#666666")
    leg = ax.legend(handles=[Patch(facecolor=colors[k[2]], alpha=0.85, label=l) for k, l in zip(keys, labels)],
                    loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=7,
                    handlelength=1.0, handletextpad=0.4)
    ax.grid(True, axis="y", alpha=0.15, linestyle="-", linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)

buf = io.BytesIO()
fig.savefig(buf, format="pdf", transparent=True)
overlay = PdfReader(io.BytesIO(buf.getvalue())).pages[0]
page.merge_page(overlay)
w = PdfWriter()
w.add_page(page)
w.write(out)
print("wrote", out, "rows replaced:", drawn)
