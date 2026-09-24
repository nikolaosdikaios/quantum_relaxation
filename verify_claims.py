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
    checks that the stated value appears verbatim (as written in LaTeX) in the
    manuscript, so that code and prose cannot drift apart.

USAGE
    python3 verify_claims.py                 # run everything
    python3 verify_claims.py --quick         # skip slow scripts
    python3 verify_claims.py --list          # show the registry
    python3 verify_claims.py --tex other.tex # check against another file

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
TEX = "dikaios_scipost.tex"

# id, script, regex, stated value, rule, literal LaTeX string that must appear
# in the manuscript (None: no textual check). The literal is matched exactly,
# so a number cannot drift out of the prose unnoticed.
CLAIMS = [
    ("pair_is_tier1", "lift_validation.py",
     r"max\|tier-1 - pair\|\s*=\s*([0-9.eE+-]+)", 4e-16, "bound",
     r"4\times10^{-16}"),
    ("lift_stability", "lift_validation_v2.py",
     r"stability\s+max Re eig\s*=\s*([+-]?[0-9.eE+-]+)", 5e-16, "bound",
     r"5\times10^{-16}"),
    ("lift_vacuum", "lift_validation_v2.py",
     r"vacuum\s+\|\|G\(e\^-f,0\)\|\|\s*=\s*([0-9.eE+-]+)", 2.9e-17, "bound",
     r"2.9\times10^{-17}"),
    ("lift_parity", "lift_validation_v2.py",
     r"parity DB\s+\|\|EGE - G\^T\|\|\s*=\s*([0-9.eE+-]+)", 1e-15, "bound",
     None),
    ("bpp_emergence", "lift_validation_v2.py",
     r"per-edge BPP emergence \|\|Heff-Hbpp\|\|\s*=\s*([0-9.eE+-]+)", 2.8e-17,
     "bound", r"2.8\times10^{-17}"),
    ("solomon_sigma", "paper3_milestones.py",
     r"sigma = ([+-][0-9.]+) vs w_DQ - w_ZQ", -0.81691, "close", None),
    ("fast_bath_solomon", "paper3_milestones.py",
     r"max\|lift - Solomon\| / \|m_I\(0\)\| = ([0-9.]+)", 6e-4, "bound",
     r"6\times10^{-4}"),
    ("noe_onset_lift", "paper3_milestones.py",
     r"onset exponent lift ([0-9.]+) \(2 = ballistic\)", 1.87, "close",
     r"1.87"),
    ("noe_curvature", "paper3_milestones.py",
     r"d2 m_S/dt2 \(0\) from generator = ([+-][0-9.]+)", 0.80, "close",
     r"+0.80"),
    ("spread_ballistic", "paper3_milestones.py",
     r"spreading exponent \(95% radius\): lift ([0-9.]+)", 0.82, "close",
     r"0.82"),
    ("causal_leak_lift", "paper3_milestones.py",
     r"mass beyond cone\s+lift = ([0-9.eE+-]+)", 4e-15, "bound",
     r"4\times10^{-15}"),
    ("l1_ball_margin", "paper3_milestones.py",
     r"min_t,i 8\*p_i = ([0-9.]+)", 0.0041, "close", r"0.0041"),
    ("positivity_worst", "positivity_domain.py",
     r"worst violation: min entry = ([+-][0-9.]+)", -0.39129, "close",
     r"-0.39"),
    ("lowfield_pair_dev_bpp", "lift_validation.py",
     r"(?s)slow bath, low field.*?max relative deviation lift vs Bloch = ([0-9.]+)",
     0.48, "close", r"48\%"),
    ("lowfield_tier1_crossings", "lift_validation.py",
     r"tier 1 \(= pair\): zero crossings = (\d+)", 9, "exact",
     r"nine zero crossings"),
    ("lowfield_higher_crossings", "lift_validation.py",
     r"tiers 2-32: max zero crossings = (\d+)", 0, "exact",
     r"no zero crossings"),
    ("lowfield_best_exp", "lift_validation.py",
     r"best single exponential \(rate [0-9.]+\): max deviation ([0-9.]+)",
     0.17, "close", r"$0.17$ of the initial"),
    ("lowfield_bpp_exp", "lift_validation.py",
     r"Markovian BPP exponential \(rate [0-9.]+\): max deviation ([0-9.]+)",
     0.49, "close", r"$0.49$"),
    ("lowfield_pair_vs_conv", "lift_validation.py",
     r"pair \(tier 1\): max deviation from converged ([0-9.]+)", 0.51,
     "close", r"$0.51$"),
    ("fit_recovered_delta", "fid_fit.py",
     r"headline \(depth-6\): Delta = ([0-9.]+)", 1.000, "close", r"1.000"),
    ("fit_recovered_tau", "fid_fit.py",
     r"headline \(depth-6\): Delta = [0-9.]+\s+tau = ([0-9.]+)", 1.996,
     "close", r"1.996"),
    ("fit_rms_ratio", "fid_fit.py",
     r"RMS ratio Bloch/kinetic = ([0-9.]+)", 4.40, "close", r"$4.4$"),
    ("fit_daic", "fid_fit.py",
     r"Delta AIC \(Bloch - kinetic\) = \+([0-9]+)", 5930, "close", r"5930"),
    ("fit_bpp_delta", "fid_fit.py",
     r"global BPP \(Markov, 2 parameters\): Delta = ([0-9.]+)", 0.77, "close",
     r"\Delta\omega=0.77"),
    ("fit_bpp_tau", "fid_fit.py",
     r"global BPP \(Markov, 2 parameters\): Delta = [0-9.]+\s+tau = ([0-9.]+)",
     1.16, "close", r"\tau_{c}=1.16"),
    ("fit_bpp_rms", "fid_fit.py",
     r"global BPP \(Markov, 2 parameters\).*?pooled RMS ([0-9.]+)", 0.044,
     "close", r"$0.044$"),
    ("fit_bpp_daic", "fid_fit.py",
     r"Delta AIC \(BPP - kinetic\) = \+([0-9]+)", 5953, "close", r"5953"),
    ("battery_echo_pred", "protocol_fit.py",
     r"kinetic prediction ([0-9.]+)", 0.0310, "close", r"0.031"),
    ("battery_ratio_t2star", "protocol_fit.py",
     r"ratios: ([0-9.]+) and", 14.1, "close", r"14.1"),
    ("battery_ratio_2t1", "protocol_fit.py",
     r"ratios: [0-9.]+ and ([0-9.]+)", 9.8, "close", r"9.8"),
    ("dict_T1_smallk", "bloch_dictionary_check.py",
     r"kappa=0.05 : 1/T1 kinetic [0-9.]+ vs Bloch [0-9.]+\s+rel dev ([0-9.eE+-]+)",
     3.7e-4, "bound", r"3.7\times10^{-4}"),
    ("dict_scaling", "bloch_dictionary_check.py",
     r"1/T1 ratio ([0-9.]+)", 17.4, "close", r"17.4"),
    ("dict_D_slope", "bloch_dictionary_check.py",
     r"slope/2 = [0-9.]+ vs input [0-9.]+\s+rel dev ([0-9.eE+-]+)",
     1.9e-13, "bound", r"1.9\times10^{-13}"),
    ("dict_Dapp", "bloch_dictionary_check.py",
     r"analytic MSD formula [0-9.]+\s+dev ([0-9.eE+-]+)",
     3.1e-11, "bound", r"3.1\times10^{-11}"),
    ("nmrd_tau_ns", "nmrd_gd_dtpa_overlay.py",
     r"tau1 = ([0-9.]+) ns", 11.0, "close", r"\tau_{c}\approx11"),
    ("nmrd_tau_ps", "nmrd_gd_dtpa_overlay.py",
     r"= ([0-9.]+) ps if assigned", 17.0, "close", r"about $17$~ps"),
    ("nmrd_rms_one", "nmrd_gd_dtpa_overlay.py",
     r"(?s)single channel.*?RMS residual = ([0-9.]+)", 0.17, "close",
     r"0.17~\mathrm{s^{-1}mM^{-1}}"),
    ("nmrd_rms_two", "nmrd_gd_dtpa_overlay.py",
     r"(?s)two channels.*?RMS residual = ([0-9.]+)", 0.15, "close",
     r"0.15~\mathrm{s^{-1}mM^{-1}}"),
    ("krawtchouk_exact", "hierarchy_checks.py",
     r"M = 5: max\|ladder - exact joint dynamics\| = ([0-9.eE+-]+)", 2e-15,
     "bound", r"2\times10^{-15}"),
    ("noe_pair_enh", "hierarchy_checks.py",
     r"tier 1 \(\d+ dim\) extremum [+-][0-9.]+ at t = [0-9.]+ \(([+-]\d+)% vs Solomon\)",
     26, "close", r"$26\%$"),
    ("noe_gauss_enh", "hierarchy_checks.py",
     r"tier 8 \(\d+ dim\) extremum [+-][0-9.]+ at t = [0-9.]+ \(([+-]\d+)% vs Solomon\)",
     -17, "close", r"reduced by $17\%$"),
    ("noe_gauss_conv", "hierarchy_checks.py",
     r"max\|tier 8 - tier 6\| = ([0-9.eE+-]+)", 1e-3, "bound",
     r"converged to $10^{-3}$"),
    ("noe_dich_enh", "hierarchy_checks.py",
     r"dichotomous, exact \(64 blocks\)\s+extremum [+-][0-9.]+ at t = [0-9.]+ \(([+-]\d+)% vs Solomon\)",
     23, "close", r"$23\%$"),
    ("lowfield_M1_crossings", "hierarchy_checks.py",
     r"M =  1: zero crossings = (\d+)", 9, "exact", r"nine zero crossings"),
    ("lowfield_M2_crossings", "hierarchy_checks.py",
     r"M =  2: zero crossings = (\d+)", 0, "exact", r"no longer changes sign"),
    ("rigid_half_decay", "figure_concept_audit.py",
     r"half-decay ratio ([0-9.]+)x", 8.8, "close", r"8.8"),
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


def tex_mentions(texfile, literal):
    """True if the literal LaTeX string appears in the manuscript, None if
    there is nothing to check or the manuscript is not found."""
    if literal is None:
        return None
    p = ROOT / texfile
    if not p.exists():
        return None
    return literal in p.read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tex", default=TEX,
                    help="manuscript to check the stated values against")
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
        intex = tex_mentions(args.tex, tex)
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
