#!/usr/bin/env python3
"""
verify_claims.py -- bind every number in the papers to the code that makes it.

WHY
    The manuscript asserts many numerical facts. Each edit to a script, a
    parameter, or a sentence risks silent drift between what the code computes
    and what the manuscript claims. Referees find that drift. This finds it
    first.

WHAT
    A registry (CLAIMS below) maps each asserted number to
      - the script that produces it,
      - a regex that extracts it from that script's stdout,
      - the value stated in the manuscript,
      - a comparison rule.
    The verifier runs each script once, extracts, compares, and additionally
    greps the .tex to confirm the stated value literally appears there.

USAGE
    python3 verify_claims.py                 # run everything
    python3 verify_claims.py --quick         # skip slow scripts
    python3 verify_claims.py --list          # show the registry

COMPARISON RULES
    'bound'  : computed <= stated * slack        (residuals: smaller is fine)
    'close'  : |computed - stated| <= tol_rel * |stated|
    'exact'  : computed == stated (integers, counts)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLACK = 10.0        # a residual may be up to 10x the stated value
TOL_REL = 0.05      # measured quantities: 5 percent

# id, script, regex, stated, rule, tex file (or None)
CLAIMS = [
    ("pair_is_tier1", "lift_validation.py",
     r"max\|tier-1 - pair\|\s*=\s*([0-9.eE+-]+)", 4e-16, "bound",
     "dikaios_scipost.tex"),
    ("lift_stability", "lift_validation_v2.py",
     r"stability\s+max Re eig\s*=\s*([+-]?[0-9.eE+-]+)", 5e-16, "bound",
     "dikaios_scipost.tex"),
    ("lift_vacuum", "lift_validation_v2.py",
     r"vacuum\s+\|\|G\(e\^-f,0\)\|\|\s*=\s*([0-9.eE+-]+)", 2.9e-17, "bound",
     "dikaios_scipost.tex"),
    ("lift_parity", "lift_validation_v2.py",
     r"parity DB\s+\|\|EGE - G\^T\|\|\s*=\s*([0-9.eE+-]+)", 1e-15, "bound",
     None),
    ("bpp_emergence", "lift_validation_v2.py",
     r"per-edge BPP emergence \|\|Heff-Hbpp\|\|\s*=\s*([0-9.eE+-]+)", 2.8e-17,
     "bound", "dikaios_scipost.tex"),
    ("solomon_sigma", "paper3_milestones.py",
     r"sigma = ([+-][0-9.]+) vs w_DQ - w_ZQ", -0.81691, "close", None),
    ("fast_bath_solomon", "paper3_milestones.py",
     r"max\|lift - Solomon\| / \|m_I\(0\)\| = ([0-9.]+)", 6e-4, "bound",
     "dikaios_scipost.tex"),
    ("noe_onset_lift", "paper3_milestones.py",
     r"onset exponent lift ([0-9.]+) \(2 = ballistic\)", 1.87, "close",
     "dikaios_scipost.tex"),
    ("noe_curvature", "paper3_milestones.py",
     r"d2 m_S/dt2 \(0\) from generator = ([+-][0-9.]+)", 0.80, "close",
     "dikaios_scipost.tex"),
    ("spread_ballistic", "paper3_milestones.py",
     r"spreading exponent \(95% radius\): lift ([0-9.]+)", 0.82, "close",
     "dikaios_scipost.tex"),
    ("causal_leak_lift", "paper3_milestones.py",
     r"mass beyond cone\s+lift = ([0-9.eE+-]+)", 4.2e-15, "bound",
     "dikaios_scipost.tex"),
    ("l1_ball_margin", "paper3_milestones.py",
     r"min_t,i 8\*p_i = ([0-9.]+)", 0.0041, "close",
     "dikaios_scipost.tex"),
    ("positivity_worst", "positivity_domain.py",
     r"worst violation: min entry = ([+-][0-9.]+)", -0.39129, "close",
     "dikaios_scipost.tex"),
    ("fit_recovered_delta", "fid_fit.py",
     r"headline \(depth-6\): Delta = ([0-9.]+)", 1.000, "close",
     "dikaios_scipost.tex"),
    ("fit_recovered_tau", "fid_fit.py",
     r"headline \(depth-6\): Delta = [0-9.]+\s+tau = ([0-9.]+)", 1.996,
     "close", "dikaios_scipost.tex"),
    ("fit_rms_ratio", "fid_fit.py",
     r"RMS ratio Bloch/kinetic = ([0-9.]+)", 4.40, "close",
     "dikaios_scipost.tex"),
    ("fit_daic", "fid_fit.py",
     r"Delta AIC \(Bloch - kinetic\) = \+([0-9]+)", 5930, "close",
     "dikaios_scipost.tex"),
    ("battery_echo_pred", "protocol_fit.py",
     r"kinetic prediction ([0-9.]+)", 0.0310, "close",
     "dikaios_scipost.tex"),
    ("battery_ratio_t2star", "protocol_fit.py",
     r"ratios: ([0-9.]+) and", 14.1, "close",
     "dikaios_scipost.tex"),
    ("battery_ratio_2t1", "protocol_fit.py",
     r"ratios: [0-9.]+ and ([0-9.]+)", 9.8, "close",
     "dikaios_scipost.tex"),
    ("dict_T1_smallk", "bloch_dictionary_check.py",
     r"kappa=0.05 : 1/T1 kinetic [0-9.]+ vs Bloch [0-9.]+\s+rel dev ([0-9.eE+-]+)",
     3.72e-4, "bound", "dikaios_scipost.tex"),
    ("dict_scaling", "bloch_dictionary_check.py",
     r"1/T1 ratio ([0-9.]+)", 17.4, "close", "dikaios_scipost.tex"),
    ("dict_D_slope", "bloch_dictionary_check.py",
     r"slope/2 = [0-9.]+ vs input [0-9.]+\s+rel dev ([0-9.eE+-]+)",
     1.9e-13, "bound", "dikaios_scipost.tex"),
    ("dict_Dapp", "bloch_dictionary_check.py",
     r"analytic MSD formula [0-9.]+\s+dev ([0-9.eE+-]+)",
     3.1e-11, "bound", "dikaios_scipost.tex"),
    ("rigid_half_decay", "figure_concept_audit.py",
     r"half-decay ratio ([0-9.]+)x", 8.8, "close", "dikaios_scipost.tex"),
]


def run_script(name, cache={}):
    if name in cache:
        return cache[name]
    path = ROOT / name
    if not path.exists():
        cache[name] = None
        return None
    out = subprocess.run([sys.executable, str(path)], capture_output=True,
                         text=True, timeout=1800)
    cache[name] = out.stdout
    return cache[name]


def compare(computed, stated, rule):
    if rule == "bound":
        return abs(computed) <= abs(stated) * SLACK
    if rule == "close":
        return abs(computed - stated) <= TOL_REL * max(abs(stated), 1e-30)
    if rule == "exact":
        return computed == stated
    raise ValueError(rule)


def tex_mentions(texfile, stated):
    """Confirm the stated value literally appears in the manuscript, so a
    number cannot silently drift out of the prose."""
    if texfile is None:
        return None
    p = ROOT / texfile
    if not p.exists():
        return None
    body = p.read_text()
    mant = f"{abs(stated):.1e}"
    m, e = mant.split("e")
    cands = [mant, f"{abs(stated):g}", f"{abs(stated):.2f}",
             f"{m}\\times10^{{{int(e)}}}", f"{m}\\times10^{{{int(e)}}}"]
    if abs(stated) < 1:
        cands.append(f"{abs(stated):.4f}".rstrip("0"))
    return any(c in body for c in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for cid, script, _, stated, rule, tex in CLAIMS:
            print(f"  {cid:<20} {stated:>12}  {rule:<6} {script}")
        return 0

    slow = {"positivity_domain.py", "paper3_milestones.py"}
    rows, n_fail, n_skip = [], 0, 0
    for cid, script, pat, stated, rule, tex in CLAIMS:
        if args.quick and script in slow:
            rows.append((cid, "skip", "", "")); n_skip += 1
            continue
        out = run_script(script)
        if out is None:
            rows.append((cid, "NOSCRIPT", "", script)); n_fail += 1
            continue
        m = re.search(pat, out)
        if not m:
            rows.append((cid, "NOMATCH", "", script)); n_fail += 1
            continue
        val = float(m.group(1))
        ok = compare(val, stated, rule)
        intex = tex_mentions(tex, stated)
        note = ""
        if intex is False:
            note = "value not found in tex"
            ok = False
        rows.append((cid, "ok" if ok else "FAIL",
                     f"{val:.4g} vs {stated:.4g}", note))
        if not ok:
            n_fail += 1

    print("=" * 74)
    print("CLAIMS VERIFICATION")
    print("=" * 74)
    for cid, status, cmp_, note in rows:
        print(f"  [{status:>8}] {cid:<20} {cmp_:<24} {note}")
    print("-" * 74)
    print(f"  {len(rows) - n_fail - n_skip} verified, {n_fail} failed, "
          f"{n_skip} skipped")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
