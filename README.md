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

Every script is self-contained.

## Reproducing the figures

```
python3 make_all_figures.py          # regenerates all six figures, then audits
```


## Reproducing the numbers

The numerical claims of Appendix B (Table 2), Appendix D (Table 3), and
Appendix E are produced by: lift_validation.py`, lift_validation_v2.py, positivity_domain.py, bloch_dictionary_check.py, symbolic_checks.py.
Two verification utilities check the repository against the manuscript:

