# Code for "Spin relaxation with bath memory beyond the correlation function"

Two scripts, no stored data. Requirements: Python 3.9+, numpy, scipy, matplotlib.

    python3 make_figures_jcp.py   # Figures 1 and 2 and every number quoted in the text (about 20 s)
    python3 verify_jcp.py         # numerical checks of the analytical statements (about 10 s)

`numbers.txt` and `verify_output.txt` are the outputs of the two scripts.

| Script | What it covers |
|---|---|
| `make_figures_jcp.py` | Fig. 1(b) free-induction decay; Fig. 2(a) low-field recovery; Fig. 2(b) Overhauser transient; long-time decay rate for M jump processes, Eq. (13); convergence of the Gaussian hierarchy |
| `verify_jcp.py` | Eqs. (3)-(4) against the exact two-state dynamics; short-memory limit (BPP rate, transverse rate, dynamic shift); thermal state, conservation and decrease of F, Eq. (7); Eqs. (10) and (13); ladder of Eq. (12) against direct simulation; two-site exchange (Bloch-McConnell) against Eq. (5); kurtosis, Eq. (14); onset of the Overhauser transient, Eq. (15); positivity bound and its limit |
