#!/usr/bin/env python3
"""
make_all_figures.py -- regenerate every figure in the manuscript.

The figure-producing code is spread over four scripts because two of them
also perform the fits whose results they plot. This script runs all of them
in order and then audits the output.

    Figure 1  fig_p3_framework       <- make_figures_p3.py
    Figure 2  fig_p3_topology        <- make_topology_fig.py
    Figure 3  nmrd_gd_dtpa_overlay   <- nmrd_gd_dtpa_overlay.py
    Figure 4  fig_p3_validation      <- make_figures_p3.py
    Figure 5  fig_p3_fit             <- fid_fit.py        (also runs the fit)
    Figure 6  fig_p3_protocol        <- protocol_fit.py   (also runs the fit)

Usage
    python make_all_figures.py            regenerate everything, then audit
    python make_all_figures.py --quiet    suppress the scripts' own output
    python make_all_figures.py --no-audit skip the audit step

Runtime is a few minutes: fid_fit.py and protocol_fit.py evaluate exact
hierarchies on a parameter grid, which dominates the cost.
"""
import argparse
import io
import os
import runpy
import sys
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = [
    ("make_figures_p3.py", ["fig_p3_framework", "fig_p3_validation"],
     "Figures 1 and 4: framework diagram and validation panels"),
    ("make_topology_fig.py", ["fig_p3_topology"],
     "Figure 2: state-complex topology (fast)"),
    ("nmrd_gd_dtpa_overlay.py", ["nmrd_gd_dtpa_overlay"],
     "Figure 3: measured Gd-DTPA dispersion overlay (fast)"),
    ("fid_fit.py", ["fig_p3_fit"],
     "Figure 5: field-series fit (runs the fit, takes ~1-2 min)"),
    ("protocol_fit.py", ["fig_p3_protocol"],
     "Figure 6: characterization protocol and held-out echoes (~2-4 min)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="hide the individual scripts' console output")
    ap.add_argument("--no-audit", action="store_true",
                    help="skip the overlap and clipping audit")
    args = ap.parse_args()

    os.chdir(HERE)
    print("=" * 68)
    print("REGENERATING ALL MANUSCRIPT FIGURES")
    print("=" * 68)

    failed = []
    for script, figs, blurb in JOBS:
        if not os.path.exists(script):
            print(f"\n[skip] {script} not found in {HERE}")
            failed.append(script)
            continue
        print(f"\n[run ] {script}")
        print(f"       {blurb}")
        t0 = time.time()
        try:
            if args.quiet:
                with redirect_stdout(io.StringIO()):
                    runpy.run_path(script, run_name="__main__")
            else:
                runpy.run_path(script, run_name="__main__")
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL] {script}: {type(exc).__name__}: {exc}")
            failed.append(script)
            continue
        dt = time.time() - t0
        made = [f for f in figs
                if os.path.exists(f + ".pdf") and os.path.exists(f + ".png")]
        print(f"[done] {script} in {dt:.1f}s -> {', '.join(made) or 'nothing'}")
        if len(made) != len(figs):
            failed.append(script)

    print("\n" + "=" * 68)
    expected = [f for _, figs, _ in JOBS for f in figs]
    missing = [f for f in expected
               if not (os.path.exists(f + ".pdf") and os.path.exists(f + ".png"))]
    for f in expected:
        mark = "ok  " if f not in missing else "MISS"
        print(f"  [{mark}] {f}.pdf / .png")
    if missing:
        print(f"\n{len(missing)} figure(s) missing. See the messages above.")
        return 1
    print("\nAll five figures regenerated.")

    if not args.no_audit:
        print("\n" + "=" * 68)
        print("AUDIT")
        print("=" * 68)
        if os.path.exists("figure_audit.py"):
            runpy.run_path("figure_audit.py", run_name="not_main")
            import figure_audit as fa
            probs = []
            for png in [f + ".png" for f in expected]:
                print(f"\n=== {png} ===")
                probs += fa.audit_png(png)
            print("\nclipping audit:",
                  "no clipping detected" if not probs else "problems above")
        else:
            print("figure_audit.py not found, skipping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
