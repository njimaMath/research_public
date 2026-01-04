# research_public

This repository collects public research artifacts (manuscripts, numerics, and formal proof sketches).

## Contents

### `perceptronFixed/`

Supporting material for *“Uniqueness of RS Saddle Point for Ising Perceptron”*.

- `perceptronFixed/manuscript/index.html`: interactive HTML manuscript (open in a browser).
- `perceptronFixed/numerics/`: Python scripts + reports.
- `perceptronFixed/Lean/`: Lean 4 + mathlib files and proof blueprints.

## Quick start

### Read the manuscript

Open `perceptronFixed/manuscript/index.html` in your browser.

### Run numerics

From the repo root (Python 3.10+ recommended):

```bash
python -m venv .venv
# activate the venv, then:
python -m pip install numpy scipy
python perceptronFixed/numerics/simulate_bprime_tobechecked.py --help
```

Note: `perceptronFixed/numerics/numerical_check.py` currently imports modules (e.g. `normal_utils`) that are not included in this repository.

### Check Lean proofs

The Lean files import `Mathlib`. To compile them, place the files under a Lean 4 + mathlib (Lake) project and run, e.g.:

```bash
lake env lean perceptronFixed/Lean/Fneg/FnegLemma.lean
```
