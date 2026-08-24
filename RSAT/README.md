# RSAT: Quantitative Strict Almeida-Thouless Theorem

Formalization in Lean of quantitative results in the strict Almeida-Thouless region for the Sherrington-Kirkpatrick spin glass model.

The fixed endpoint of this package is [`Main.lean`](./Main.lean). It states a quantitative strict-AT theorem whose proof is supplied by [`Lemmas/MainResult.lean`](./Lemmas/MainResult.lean).

## Main result

For a compact parameter set `K : Set (ℝ × ℝ)` contained in the positive-field strict Almeida-Thouless region, the project proves three quantitative conclusions along the replica-symmetric smart path:

- a uniform bound on the scaled second moment `N * A path s`;
- an `O(1 / N)` bound on the replica-symmetric/free-energy discrepancy;
- convergence of the scaled replicon combination
  `N * (A path s - 2 * B path s + C path s)`
  to the strict-AT expression
  `rsA β h / (1 - s * atParameter β h)`.

The public theorem in `Main.lean` is:

```lean
theorem quantitative_strictAT {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K)
    (hKsub : K ⊆ strictATRegion) :
    QuantitativeAT (Ω := Ω) K
```

`Main.lean` is intentionally treated as immutable. Proof work belongs in its dependencies.

## Lean version

This package uses:

```text
Lean 4.32.1
```

as specified by [`lean-toolchain`](./lean-toolchain):

```text
leanprover/lean4:v4.32.1
```

## Project layout

```text
RSAT/
├── Main.lean                 # fixed endpoint and public quantitative theorem
├── Lemmas/                   # analytic and probabilistic proof development
│   ├── AT/                   # strict-AT definitions and fixed-point analysis
│   ├── MainResult.lean       # proof of the final quantitative theorem
│   ├── Cavity/               # cavity-method estimates and blueprint
│   ├── Concentration/        # concentration and transport estimates
│   ├── Gaussian/             # Gaussian estimates
│   ├── GuerraTalagrand/      # Guerra-Talagrand bounds and flatness
│   ├── Price/                # quantitative interpolation estimates
│   ├── SmartPath/            # smart-path interpolation arguments
│   └── ...
├── SpinGlass/                # SK-model and AT infrastructure
│   ├── AT/                   # AT definitions, calculus, model, and bounds
│   └── Replicas.lean
├── refs/                     # mathematical references/supporting material
├── lakefile.lean
├── lake-manifest.json
└── lean-toolchain
```

The proof dependency direction is roughly

```text
Main.lean
    ↑
Lemmas/MainResult.lean
    ↑
strict-AT quantitative estimates
    ↑
Guerra-Talagrand / cavity / concentration estimates
    ↑
interpolation, Gaussian, fixed-point, and model infrastructure
```

The Lean import graph is authoritative when this schematic picture differs from the code.

## Dependencies

The Lake package is named `LatalaMeetsAT`.

The current [`lakefile.lean`](./lakefile.lean) uses local shared dependencies:

- Mathlib from `../../.lake/packages/mathlib`;
- additional spin-glass infrastructure from `../generalizedLatala`.

Therefore the package is not configured as a completely standalone checkout. Preserve the repository/local dependency layout expected by `lakefile.lean`, or adjust those paths for your local environment.

## Build

From this directory, the primary checks are:

```bash
lake build LatalaMeetsAT
lake env lean Main.lean
```

To confirm the selected Lean toolchain:

```bash
lean --version
```

which should report Lean `4.32.1`.

## Proof integrity

The aim is a genuine Lean proof of the unchanged final theorem. Project-local proof placeholders or substitute axioms are not acceptable.

Before treating the development as complete, run:

```bash
lake build LatalaMeetsAT
lake env lean Main.lean
git diff -- Main.lean
rg '\b(sorry|admit)\b|sorryAx|^[[:space:]]*axiom\b' . \
  --glob '*.lean' \
  --glob '!.lake/**'
```

The intended completion criterion is:

- `lake build LatalaMeetsAT` succeeds;
- `lake env lean Main.lean` succeeds;
- `Main.lean` has no diff;
- every proof obligation relevant to the dependency closure of `Main.lean` is discharged by genuine Lean proofs, without `sorry`, `admit`, `sorryAx`, or project-local `axiom` declarations.

## Development guidance

When working on the formalization:

- start from `Main.lean` and trace dependencies backward through `Lemmas/MainResult.lean`;
- preserve the mathematical statements required by the final theorem;
- prefer established Mathlib results and already-proved repository lemmas;
- compile the smallest affected module after coherent changes;
- periodically rebuild `LatalaMeetsAT`;
- do not edit `Main.lean` to make downstream proof obligations easier.
