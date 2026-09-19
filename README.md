# Non-Markovian dynamic transition-current theory of quantum relaxation — code

This repository reproduces every figure and every quoted number in the
manuscript. All results come from these scripts; there is no hidden state and
no pre-computed data. Numbers printed to `stdout` are the numbers that appear
in the text, tables, and figure captions.

## Requirements

Python 3.9+ with

```
numpy  scipy  matplotlib  sympy
```

```
pip install numpy scipy matplotlib sympy
```

Every script is self-contained and is run directly, for example
`python3 fid_fit.py`. Figures are written to the working directory as `.pdf`
and `.png`. Runtime for the whole suite is a few minutes; the two grid-search
fits (`fid_fit.py`, `protocol_fit.py`) dominate the cost.

## Reproducing the figures

```
python3 make_all_figures.py          # regenerates all six figures, then audits
```

or individually:

| Figure | Script | Content |
|-------|--------|---------|
| 1 | `make_figures_p3.py` | framework flowchart (`fig_p3_framework`) |
| 2 | `make_topology_fig.py` | state complex and incidence matrix (`fig_p3_topology`) |
| 3 | `nmrd_gd_dtpa_overlay.py` | measured Gd-DTPA dispersion overlay (`nmrd_gd_dtpa_overlay`) |
| 4 | `make_figures_p3.py` | verified reductions and predictions (`fig_p3_validation`) |
| 5 | `fid_fit.py` | field-series fit (`fig_p3_fit`), also runs the fit |
| 6 | `protocol_fit.py` | characterization protocol, held-out echoes (`fig_p3_protocol`), also runs the fit |

`make_figures_p3.py`, `fid_fit.py`, and `protocol_fit.py` also print the
numerical results they plot.

## Reproducing the numbers

The numerical claims of Appendix B (Table 2), Appendix D (Table 3), and
Appendix E are produced by:

| Script | Produces |
|--------|----------|
| `lift_validation.py` | pair equals tier 1 of the exact hierarchy; stability, Gibbs vacuum, Onsager–Casimir parity, and emergent BPP rates on the three-spin complex |
| `lift_validation_v2.py` | the same identities with distinct per-edge memory times, plus a positivity audit |
| `paper3_milestones.py` | locality of positivity violations; Solomon cross-relaxation; the quadratic Overhauser launch, its curvature, onset exponent, and peak shift; and causal transport |
| `positivity_domain.py` | the entrywise-positivity map over `(tau, field)` and the worst propagator entry `-0.39` |
| `bloch_dictionary_check.py` | the round trip from the microscopic parameters to the Bloch constants and the `O(kappa^2)` residual scaling (Table 3) |
| `symbolic_checks.py` | an **independent symbolic** re-derivation (SymPy) of every analytic formula quoted in the text, from its stated premise |

Two verification utilities check the repository against the manuscript:

```
python3 verify_claims.py             # binds each quoted number to the script that prints it,
                                     # and confirms the value literally appears in the .tex
python3 verify_claims.py --quick     # skip the slow scripts
python3 symbolic_checks.py           # re-derive every analytic result symbolically
```

`verify_claims.py` reads `dikaios_scipost.tex`; place the manuscript source in
the same directory (or adjust the path in the registry) to enable the
literal-in-text check. Figure legibility and figure content are audited by
`figure_audit.py` and `figure_concept_audit.py`.

## File index

Figures and fits
- `make_figures_p3.py` — Figures 1 and 4
- `make_topology_fig.py` — Figure 2
- `nmrd_gd_dtpa_overlay.py` — Figure 3 (points digitized from Varga-Szemes *et al.*, PLoS ONE 11, e0149260 (2016), CC-BY; re-digitize before submission)
- `fid_fit.py` — Figure 5 and the field-series fit (Appendix E)
- `protocol_fit.py` — Figure 6 and the characterization protocol (Appendix E.1)
- `make_all_figures.py` — orchestrator

Numerical verification
- `lift_validation.py`, `lift_validation_v2.py`, `paper3_milestones.py`,
  `positivity_domain.py`, `bloch_dictionary_check.py`

Analytic verification and auditing
- `symbolic_checks.py` — SymPy re-derivations
- `verify_claims.py` — number-to-script binding
- `figure_audit.py`, `figure_concept_audit.py` — figure audits

## Notes

- Reported residuals at the `1e-12` level and below are double-precision
  consistency checks (an exact algebraic identity evaluated numerically), not
  physical accuracies.
- Time integration uses fixed-step fourth-order Runge–Kutta; spectral and
  stationary-state quantities use dense eigendecomposition; all arithmetic is
  IEEE double precision.
- The inner propagation loops in `fid_fit.py` and `protocol_fit.py` are
  vectorized over time. This is a speed-only change and leaves every printed
  value unchanged.
