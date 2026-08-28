# RSAT: Quantitative Strict Almeida-Thouless Theorem and Overlap CLT

This directory contains a Lean formalization of quantitative replica-symmetric
results for the Sherrington-Kirkpatrick spin glass model in the
strict Almeida-Thouless region. It proves uniform finite-size estimates on a 
compact set of parameter and a pointwise central limit theorem for the centered
overlap.

[`Main.lean`](./Main.lean) is the public endpoint. It instantiates the abstract
results on a canonical countable product Gaussian space and exports the
theorems `strictAT_main` and `strictAT_overlapCLT_weak`.

## Concrete smart path

`Main.lean` defines

```text
H_{N,s}(σ) = β sqrt(s) / sqrt(2N) Σᵢⱼ gᵢⱼ σᵢ σⱼ
             + Σᵢ (h + β sqrt((1-s)q) zᵢ) σᵢ,
```

where the `gᵢⱼ` and `zᵢ` are independent standard Gaussian coordinates on
`CanonicalGaussianSpace`. The theorem `H_N_s_eq_smartPath` identifies this
Hamiltonian with the implementation used by the abstract proof. The
definition `canonicalRSSmartPathDisorder` supplies the same concrete model to
the CLT interface.


## Quantitative strict-AT theorem

Let `K : Set (ℝ × ℝ)` be compact and contained in

```text
strictStabilityRegion = {(β, h) | 0 < β ∧ 0 < h ∧ stabilityIndex β h < 1}.
```

For the replica-symmetric overlap `q = canonicalOverlap β h` and every
smart-path time `s ∈ [0, 1]`, the main results prove:

- "overlapConcentration": uniform overlap concentration: `N * A_s N β h s ≤ C_K`;
- "freeEnergyCorrection": a signed `O(1 / N)` free-energy correction:
  `0 ≤ replicaSymmetricFreeEnergy β h - φ_N N β h ≤ C_K / N`;
- "repliconSusceptibility": uniform convergence of the scaled replicon susceptibility
  `N * (A_s N β h s - 2 * B_s N β h s + C_s N β h s)` to
  `stabilityIndex β h / (β ^ 2 * (1 - s * stabilityIndex β h))`.

Here `A_s`, `B_s`, and `C_s` are the equal-pair, shared-index, and
disjoint-index centered-overlap moments. The stability parameter is

```text
stabilityIndex β h = β^2 E[sech^4(h + β sqrt(q) Z)].
```

So, the first main claim is:

```lean
theorem strictAT_main
    (K : Set (ℝ × ℝ))
    (hKcompact : IsCompact K)
    (hKsub : K ⊆ strictStabilityRegion) :
    StrictATClaim K
```

The contents of `StrictATClaim` are `overlapConcentration`,
`freeEnergyCorrection`, and `repliconSusceptibility`.

## Overlap central limit theorem

For fixed `β > 0` and `h > 0` with `atParameter β h < 1`, set

```text
X_N = sqrt(N) * (R₁₂ - rsQ β h).
```

The theorem
`SpinGlass.AT.overlapCLT_weak` showes the weak convergence of 'X_N' to a centered
Gaussian law with strictly positive variance

```text
σ² = 3a / (1 - α)
     - 2κ / (1 - β²κ)
     - ζ / (1 - β²κ)²,
```

where `α = atParameter β h`, `a = rsA β h`,
`κ = cavityKappa (rsQ β h) (rsR β h)`, and
`ζ = cavityZeta (rsQ β h) (rsR β h)`.

The second main claim is:

```lean
theorem strictAT_overlapCLT_weak
    {β h : ℝ}
    (hβ : 0 < β)
    (hh : 0 < h)
    (hAT : SpinGlass.AT.atParameter β h < 1) :
    let σ2 : ℝ := ...
    0 < σ2 ∧
      Filter.Tendsto
        (fun N : ℕ => SpinGlass.AT.scaledOverlapLaw
          (canonicalRSSmartPathDisorder N.succ β h))
        Filter.atTop
        (nhds (SpinGlass.AT.centeredGaussianLaw σ2))
```


## Proof architecture

The quantitative theorem follows this dependency path:

```text
Main.lean
  -> Lemmas/Gaussian/ConcreteModel.lean
  -> Lemmas/Gaussian/CanonicalModel.lean
  -> Lemmas/MainResult.lean
  -> cavity, concentration, Guerra-Talagrand, smart-path,
     interpolation, Gaussian, and fixed-point estimates
```

The CLT is downstream of those estimates:

```text
Main.lean
  -> Lemmas/CLT/CLT_Main.lean
  -> Lemmas/CLT/SteinLimit.lean
  -> Lemmas/CLT/SteinSystem.lean
  -> Lemmas/CLT/SteinCavity.lean
  -> Lemmas/CLT/Basic.lean
  -> Lemmas/MainResult.lean
```

Its proof uses a cavity-Stein system for the three overlap covariance classes,
solves that system in the existing cavity-mode basis, identifies the limiting
variance, proves convergence of characteristic functions, and then invokes a
Lévy continuity theorem to obtain weak convergence.

## Project layout

```text
RSAT/
├── Main.lean
├── Lemmas/
│   ├── AT/                   # fixed point, AT data, and scalar interpolation
│   ├── Cavity/               # cavity interpolation and remainder estimates
│   ├── CLT/                  # Stein system and overlap CLT
│   ├── Concentration/        # coupled-pressure, transport, and tail bounds
│   ├── Gaussian/             # canonical coordinates and concrete model bridge
│   ├── GuerraTalagrand/      # two-replica bounds and strict flatness
│   ├── Price/                # quantitative Gaussian interpolation estimates
│   ├── SpinGlass/            # underlying SK infrastructure
│   ├── SmartPath/            # smart-path identities and endpoint estimates
│   └── MainResult.lean       # abstract quantitative strict-AT theorem
├── refs/                     # reference arguments
├── lakefile.lean
├── lake-manifest.json
└── lean-toolchain
```

## Dependencies

The Lake package is named `QuantitativeStrictAT`. Its
[`lakefile.lean`](./lakefile.lean) uses:

- Mathlib tag `v4.32.1` from the official Mathlib Git repository;
- shared spin-glass support at `../generalizedLatala`.

The local support dependency is included in the same `research_public`
repository, so no external directory layout is required.

## Maintenance policy

[`Main.lean`](./Main.lean) is the fixed public theorem file. Refactoring must preserve
`Main.lean` byte for byte. Its two project imports,
`Lemmas.Gaussian.ConcreteModel` and `Lemmas.CLT.CLT_Main`, are the roots of the
required dependency graph.

The `Lemmas` tree is kept dependency-driven: obsolete forwarding modules,
unused internal declarations, and broad imports should be removed when the
two public proof paths continue to compile. 

## Build

The selected toolchain is Lean 4.32.1, specified by
[`lean-toolchain`](./lean-toolchain). For a fresh clone, run:

```bash
git clone https://github.com/njimaMath/research_public.git
cd research_public/RSAT
lake update
lake build
lake env lean Main.lean
```

The first `lake update` fetches the pinned Mathlib release and its transitive
dependencies into the ignored local `.lake` directory. For subsequent checks
from the `RSAT` directory, run:

```bash
lake build QuantitativeStrictAT
lake env lean Main.lean
```

The first command builds all modules below `Lemmas`. The second checks the
public endpoint, which is outside the library glob. To check only the CLT
endpoint, run:

```bash
lake env lean Lemmas/CLT/CLT_Main.lean
```

Before a release or after dependency refactoring, perform a clean build:

```bash
lake clean
lake build QuantitativeStrictAT
lake env lean Main.lean
git diff -- Main.lean
```

The last command must produce no output. Because Mathlib is shared from a
local Lake package directory, `lake update` restores the pinned dependency set
after that directory is removed.

To inspect the selected compiler version, run:

```bash
lake env lean --version
```

## Proof integrity

The intended result is a kernel-checked proof with no project-local
placeholders or substitute axioms. A full verification is:

```bash
lake clean
lake build QuantitativeStrictAT
lake env lean Main.lean
git diff -- Main.lean
rg -n '\b(sorry|admit)\b|sorryAx|^[[:space:]]*axiom\b' . \
  --glob '*.lean' \
  --glob '!.lake/**'
```

When changing a proof, compile the smallest affected module first and then
rerun the project-level build, public-endpoint check, and placeholder scan.

## License

The RSAT formalization is distributed under the Apache License 2.0; see
[`LICENSE`](./LICENSE). Portions of [`Lemmas/SpinGlass/`](./Lemmas/SpinGlass/)
are borrowed or adapted from
[`or4nge19/SpinGlass`](https://github.com/or4nge19/SpinGlass) and retain the
applicable Apache-2.0 attribution; see [`NOTICE`](./NOTICE).
