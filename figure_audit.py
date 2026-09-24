#!/usr/bin/env python3
""

Run after regenerating figures. Exit status is non-zero if a problem is found.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MIN_FONT = 9.5          # points, below this is hard to read in print
TOL = 4.0               # pixels of permitted overlap (antialiasing, padding)


def bbox_overlap(a, b):
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return min(x1 - x0, y1 - y0)


def audit_figure(fig, name):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    problems = []

    items = []            # (label, bbox, fontsize, axes)
    for ax in fig.get_axes():
        for txt in ax.texts:
            if not txt.get_text().strip():
                continue
            items.append((repr(txt.get_text()[:28]),
                          txt.get_window_extent(rend),
                          txt.get_fontsize(), ax))
        leg = ax.get_legend()
        if leg is not None:
            items.append(("<legend>", leg.get_window_extent(rend),
                          MIN_FONT + 1, ax))

    # pairwise overlap within the same axes
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            li, bi, _, axi = items[i]
            lj, bj, _, axj = items[j]
            if axi is not axj:
                continue
            ov = bbox_overlap(bi, bj)
            if ov > TOL:
                problems.append(f"overlap {ov:5.1f}px: {li} vs {lj}")

    # text escaping its axes
    for label, bb, _, ax in items:
        if label == "<legend>":
            continue
        ab = ax.get_window_extent(rend)
        if bb.x0 < ab.x0 - TOL or bb.x1 > ab.x1 + TOL or \
           bb.y0 < ab.y0 - TOL or bb.y1 > ab.y1 + TOL:
            problems.append(f"text outside axes: {label}")

    # font floor
    for label, _, fs, _ in items:
        if fs < MIN_FONT:
            problems.append(f"font {fs}pt below {MIN_FONT}pt: {label}")

    print(f"\n{name}: {len(items)} text items, "
          f"{len(problems)} problem(s)")
    for p in problems:
        print("   ", p)
    return problems


def audit_png(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    b = im.point(lambda p: 255 if p < 245 else 0).getbbox()
    m = min(b[0], b[1], w - b[2], h - b[3])
    status = "ok" if m > 3 else "CLIPPED"
    print(f"    canvas {w}x{h}, margins min {m}px -> {status}")
    return [] if m > 3 else [f"{path} clipped"]


# which script produces which figure, so a missing file is self-explaining
SOURCE = {
    "fig_p3_framework.png": "make_figures_p3.py",
    "fig_p3_topology.png": "make_topology_fig.py",
    "nmrd_gd_dtpa_overlay.png": "nmrd_gd_dtpa_overlay.py",
    "fig_p3_validation.png": "make_figures_p3.py",
    "fig_p3_fit.png": "fid_fit.py         (also runs the field-series fit)",
    "fig_p3_protocol.png": "protocol_fit.py    (also runs the protocol fit)",
}

if __name__ == "__main__":
    all_problems = []
    for png in ("fig_p3_framework.png", "fig_p3_topology.png",
                "nmrd_gd_dtpa_overlay.png", "fig_p3_validation.png",
                "fig_p3_fit.png", "fig_p3_protocol.png"):
        try:
            print(f"\n=== {png} ===")
            all_problems += audit_png(png)
        except FileNotFoundError:
            print(f"    missing: {png}")
            print(f"    generate it with:  python {SOURCE[png]}")
            print( "    or regenerate everything:  python make_all_figures.py")
            all_problems.append(f"{png} missing")
    print("\n" + "=" * 60)
    print("clipping audit complete;",
          "no clipping detected" if not all_problems else "problems above")
    sys.exit(1 if all_problems else 0)
