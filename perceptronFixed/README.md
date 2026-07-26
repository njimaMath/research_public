# Ising perceptron research artifacts

This directory contains the manuscript, numerical checks, and Lean 4
formalization for *“Uniqueness of RS Saddle Point for Ising Perceptron”*.

## Contents

- [`manuscript/index.html`](manuscript/index.html): interactive HTML manuscript.
- [`numerics/`](numerics/): Python scripts and numerical reports.
- [`Lean/`](Lean/): Lean 4 files, proof blueprints, and project configuration.

## Quick start

### Read the manuscript

Open [`manuscript/index.html`](manuscript/index.html) in a browser, or read the
[hosted version](https://njimamath.github.io/research_public/perceptronFixed/manuscript/index.html).

### Run numerics

From the repository root, using Python 3.10 or later:

```bash
python -m venv .venv
# Activate the virtual environment, then run:
python -m pip install numpy scipy
python perceptronFixed/numerics/simulate_bprime_tobechecked.py --help
```

The separate `numerical_check.py` script depends on local modules such as
`normal_utils` that are not included in this repository.

### Check Lean proofs

The [`Lean/`](Lean/) directory is a Lake project pinned to Lean and mathlib
`v4.26.0`. To check the public entry point and all of its imports, run:

```bash
cd perceptronFixed/Lean
lake env lean mainresult.lean
```
