#!/usr/bin/env python3
# make_topology_fig.py -- Figure 2 of the transition-current paper.
# Panel (a): the two-spin state complex, currents on the edges, coloured by
# coherence order. Panel (b): the 4x6 incidence matrix d (the topology inside Q).
# The continuity law is the CORRECTED form  du/dt = -Q^T v^c  (dissipative
# component), matching Eq. (4) and the framework figure.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 12.5, "mathtext.fontset": "dejavusans"})
CB, CR, CG, CD = "#2E5F8A", "#B5502A", "#4A7A4C", "#1E2A38"   # SQ, DQ, ZQ, dark
GREY = "#E7E7E7"

fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.8, 5.3))
fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.10, wspace=0.30)

# ============================ panel (a): state complex =====================
axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis("off")
axa.set_title("(a)  two-spin state complex: currents live on the edges",
              fontsize=13, loc="left", color=CD)

V = {r"$|\!\uparrow\uparrow\rangle$":   (0.50, 0.90),
     r"$|\!\uparrow\downarrow\rangle$": (0.13, 0.50),
     r"$|\!\downarrow\uparrow\rangle$": (0.87, 0.50),
     r"$|\!\downarrow\downarrow\rangle$": (0.50, 0.10)}
names = list(V.keys()); UU, UD, DU, DD = (V[n] for n in names)

def edge(p, q, col, rad=0.0, lw=3.2):
    axa.add_patch(FancyArrowPatch(p, q, arrowstyle="-", mutation_scale=1,
                  lw=lw, color=col, connectionstyle=f"arc3,rad={rad}", zorder=1))

def lab(x, y, txt, col):
    axa.text(x, y, txt, ha="center", va="center", fontsize=12.5, color=col,
             zorder=4, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

# single-quantum (blue): four outer edges
for p, q, lx, ly in ((UU, UD, 0.27, 0.73), (UU, DU, 0.73, 0.73),
                     (UD, DD, 0.27, 0.27), (DU, DD, 0.73, 0.27)):
    edge(p, q, CB); lab(lx, ly, r"$v_{\mathrm{sq}}$", CB)
# zero-quantum flip-flop (green): horizontal, bows up
edge(UD, DU, CG, rad=-0.30); lab(0.50, 0.635, r"$v_{\mathrm{zq}}$", CG)
# double-quantum (red): vertical, bows right; label on lower arc, clear of box
edge(UU, DD, CR, rad=0.32); lab(0.615, 0.265, r"$v_{\mathrm{dq}}$", CR)

# central continuity-law annotation (edges pass behind it)
axa.add_patch(FancyBboxPatch((0.355, 0.435), 0.29, 0.105,
              boxstyle="round,pad=0.008", fc="white", ec=CD, lw=1.2, zorder=3))
axa.text(0.5, 0.507, r"$\partial_t u=-Q^{\top}v^{c}$", ha="center",
         va="center", fontsize=15, color=CD, zorder=4)
axa.text(0.5, 0.462, "populations change only\nby current divergence",
         ha="center", va="center", fontsize=8.8, color="#555555", zorder=4)

for n, (x, y) in V.items():
    axa.add_patch(plt.Circle((x, y), 0.052, fc="white", ec=CD, lw=1.8, zorder=5))
    axa.text(x, y, n, ha="center", va="center", fontsize=12.5, zorder=6)

for i, (txt, col) in enumerate((("double-quantum", CR),
                                ("zero-quantum (flip-flop)", CG),
                                ("single-quantum (observable)", CB))):
    yy = 0.055 + 0.052*i
    axa.plot([0.01, 0.065], [yy, yy], color=col, lw=3.2)
    axa.text(0.076, yy, txt, va="center", fontsize=9.6, color="#333333")

# ============================ panel (b): incidence matrix ==================
axb.set_title(r"(b)  incidence matrix $\mathrm{d}$: the topology inside $Q$",
              fontsize=13, loc="left", color=CD)
B = np.array([[-1, -1,  0,  0,  0, -1],
              [ 1,  0, -1,  0, -1,  0],
              [ 0,  1,  0, -1,  1,  0],
              [ 0,  0,  1,  1,  0,  1]])
nr, nc = B.shape
axb.set_xlim(-2.0, nc - 0.4); axb.set_ylim(-1.35, nr - 0.30); axb.axis("off")
for i in range(nr):
    yy = nr - 1 - i                       # state i=0 (up-up) at the top
    for j in range(nc):
        v = B[i, j]
        fc = GREY if v == 0 else (CB if v > 0 else CR)
        axb.add_patch(plt.Rectangle((j-0.46, yy-0.46), 0.92, 0.92, fc=fc,
                      ec="white", lw=2.2))
        if v != 0:
            axb.text(j, yy, f"{v:+d}", ha="center", va="center",
                     color="white", fontsize=13, fontweight="bold")
rows = [r"$|\!\uparrow\uparrow\rangle$", r"$|\!\uparrow\downarrow\rangle$",
        r"$|\!\downarrow\uparrow\rangle$", r"$|\!\downarrow\downarrow\rangle$"]
cols = [r"$v_{\mathrm{sq}}^{(1)}$", r"$v_{\mathrm{sq}}^{(2)}$",
        r"$v_{\mathrm{sq}}^{(3)}$", r"$v_{\mathrm{sq}}^{(4)}$",
        r"$v_{\mathrm{zq}}$", r"$v_{\mathrm{dq}}$"]
for i, r in enumerate(rows):
    axb.text(-0.72, nr-1-i, r, ha="right", va="center", fontsize=12.5, color=CD)
for j, c in enumerate(cols):
    axb.text(j, -0.72, c, ha="center", va="center", fontsize=12, color="#333333")
axb.text(-1.55, (nr-1)/2, "spin states (vertices)", rotation=90,
         ha="center", va="center", fontsize=10.5, color="#555555")
axb.text((nc-1)/2, -1.18, "transition currents (edges)", ha="center",
         va="center", fontsize=10.5, color="#555555")

fig.savefig("fig_p3_topology.pdf"); fig.savefig("fig_p3_topology.png", dpi=300)
print("figure written: fig_p3_topology")
