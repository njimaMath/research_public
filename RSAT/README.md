# RSAT: Quantitative Strict Almeida-Thouless Theorem

This directory contains a Lean formalization of quantitative replica-symmetric results for the Sherrington-Kirkpatrick spin glass model in the positive-field strict Almeida-Thouless region.

[`Main.lean`](./Main.lean) is the public endpoint. It defines a concrete smart-path Hamiltonian on a canonical countable product Gaussian space and states the final theorem in the notation of the reference argument. The abstract analytic proof is assembled in [`Lemmas/MainResult.lean`](./Lemmas/MainResult.lean), then connected to the concrete Gaussian model by [`Lemmas/Gaussian/CanonicalModel.lean`](./Lemmas/Gaussian/CanonicalModel.lean) and [`Lemmas/Gaussian/ConcreteModel.lean`](./Lemmas/Gaussian/ConcreteModel.lean).

## Main result

Let `K : Set (ℝ × ℝ)` be compact and contained in

```text
strictStabilityRegion = {(β, h) | 0 < β ∧ 0 < h ∧ stabilityIndex β h < 1}.
```

For the canonical overlap `q = canonicalOverlap β h` and every smart-path time `s ∈ [0, 1]`, the project proves:

- uniform overlap concentration: `N * A_s N β h s ≤ C_K`;
- a signed `O(1 / N)` free-energy correction:
  `0 ≤ replicaSymmetricFreeEnergy β h - φ_N N β h ≤ C_K / N`;
- uniform convergence of the scaled replicon susceptibility
  `N * (A_s N β h s - 2 * B_s N β h s + C_s N β h s)`
  to
  `stabilityIndex β h / (β ^ 2 * (1 - s * stabilityIndex β h))`.

Here `A_s`, `B_s`, and `C_s` are the equal-pair, shared-index, and disjoint-index centered-overlap moments. The stability index is

```text
stabilityIndex β h = β^2 E[sech^4(h + β sqrt(q) Z)].
```

The public theorem is:

```lean
theorem strictAT_main
    (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K)
    (hKsub : K ⊆ strictStabilityRegion) :
    StrictATClaim K
```

The fields of `StrictATClaim` are `overlapConcentration`, `freeEnergyCorrection`, and `repliconSusceptibility`.

## Concrete smart path

`Main.lean` defines

```text
H_{N,s}(σ) = β sqrt(s) / sqrt(2N) Σᵢⱼ gᵢⱼ σᵢ σⱼ
             + Σᵢ (h + β sqrt((1-s)q) zᵢ) σᵢ,
```

where all `gᵢⱼ` and `zᵢ` are independent standard Gaussian coordinates on `CanonicalGaussianSpace`. The theorem `H_N_s_eq_smartPath` identifies this displayed Hamiltonian with the implementation used by the proof backend.

## Proof architecture

The main dependency path is:

```text
Main.lean
  -> Lemmas/Gaussian/ConcreteModel.lean
  -> Lemmas/Gaussian/CanonicalModel.lean
  -> Lemmas/MainResult.lean
  -> cavity, concentration, Guerra-Talagrand, smart-path,
     interpolation, Gaussian, and fixed-point estimates
```

Two intermediate interfaces separate the proof layers:

- `SpinGlass.AT.QuantitativeATConclusion` is the abstract conclusion for a replica-symmetric smart-path disorder.
- `QuantitativeAT` transfers that conclusion to the concrete Gaussian-disorder interface before `strictAT_main` specializes it to canonical coordinates.

The compactness theorem `SpinGlass.AT.quantitative_strictAT_on_compact` extracts uniform numerical data from a compact subset of the strict AT region. The core theorem `SpinGlass.AT.quantitative_strictAT` consumes this data and proves the three estimates.

## Project layout

```text
RSAT/
├── Main.lean
├── Lemmas/
│   ├── AT/                   # fixed point, AT data, and scalar interpolation
│   ├── Cavity/               # cavity interpolation and remainder estimates
│   ├── Concentration/        # coupled-pressure, transport, and tail bounds
│   ├── Gaussian/             # canonical coordinates and concrete model bridge
│   ├── GuerraTalagrand/      # two-replica bounds and strict flatness
│   ├── Price/                # quantitative Gaussian interpolation estimates
│   ├── SmartPath/            # smart-path identities and endpoint estimates
│   └── MainResult.lean       # abstract quantitative theorem
├── SpinGlass/                # underlying SK and AT infrastructure
├── refs/                     # reference argument
├── lakefile.lean
├── lake-manifest.json
└── lean-toolchain
```

## Dependencies

The Lake package is named `LatalaMeetsAT`. Its [`lakefile.lean`](./lakefile.lean) uses local shared dependencies:

- Mathlib at `../../.lake/packages/mathlib`;
- shared spin-glass support at `../generalizedLatala`.

The package therefore expects the surrounding repository layout. Adjust the local paths in `lakefile.lean` if the dependencies are stored elsewhere.

## Build

The selected toolchain is Lean 4.32.1, specified by [`lean-toolchain`](./lean-toolchain).

From the `RSAT` directory, run:

```bash
lake build LatalaMeetsAT
lake env lean Main.lean
```

The first command builds the library modules. The second checks the public endpoint, which is not included in the library glob.

To inspect the selected compiler version:

```bash
lake env lean --version
```

## Proof integrity

The intended result is a kernel-checked proof with no project-local placeholders or substitute axioms. A useful final check is:

```bash
lake build LatalaMeetsAT
lake env lean Main.lean
rg -n '\b(sorry|admit)\b|sorryAx|^[[:space:]]*axiom\b' . \
  --glob '*.lean' \
  --glob '!.lake/**'
```

When changing the proof, keep the statements of `StrictATClaim` and `SpinGlass.AT.QuantitativeATConclusion` fixed, compile the smallest affected module first, and then rerun both project-level checks.
