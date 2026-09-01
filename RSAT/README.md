# RSAT: Quantitative Strict de Almeida-Thouless Theorem and Overlap CLT

This directory contains a Lean formalization of quantitative
replica-symmetric results for the Sherrington-Kirkpatrick spin glass model in
the strict Almeida-Thouless region. It proves uniform finite-size estimates on
compact parameter sets and a pointwise central limit theorem for the centered
overlap. It tracks [arXiv:2608.23413v2](https://arxiv.org/abs/2608.23413v2),
which adds the overlap central limit theorem.

[`Main.lean`](./Main.lean) is the public endpoint. The `Main` namespace
defines the model directly in the notation of the paper and exports these
principal results:

- `replicaSymmetricFixedPointClaim_of_pos_field`
- `strictAT_main`
- `strictAT_overlapCLT_weak`

## Correspondence with the paper

| Paper statement | Lean declaration | File |
| --- | --- | --- |
| Replica-symmetric fixed-point characterization | [`Main.replicaSymmetricFixedPointClaim_of_pos_field`](./Main.lean#L45) | `Main.lean` |
| Smart-path Hamiltonian $H_{N,s}$ | [`Main.H_N_s`](./Main.lean#L119) | `Main.lean` |
| Identification with the proof backend | [`Main.H_N_s_eq_smartPath`](./Main.lean#L127) | `Main.lean` |
| Theorem 1.1 | [`Main.strictAT_main`](./Main.lean#L332) | `Main.lean` |
| Theorem 1.2 | [`Main.strictAT_overlapCLT_weak`](./Main.lean#L534) | `Main.lean` |

## Concrete parameters and model

For a standard real Gaussian random variable $Z$, `Main.lean` defines

```text
q = E[tanh(h + beta sqrt(q) Z)^2],
r = E[tanh(h + beta sqrt(q) Z)^4],
alpha = beta^2 (1 - 2q + r).
```

The canonical value `q beta h` is defined as the infimum of the fixed points
in $[0,1]$. When $h>0$,
`replicaSymmetricFixedPointClaim_of_pos_field` proves that it is the unique
fixed point in this interval.

The strict AT region is the concrete set

```text
strictATRegion = {(beta, h) | 0 < beta and 0 < h and alpha beta h < 1}.
```

A configuration on $N$ sites is a function `Fin N -> Bool`, with Boolean
coordinates mapped to spins in ${-1,1}$. All disorder coordinates live on

```text
GaussianSpace = ((Nat x Nat) + Nat) -> Real,
```

equipped with the countable product standard Gaussian measure. The two summands
index the coordinates $g_{ij}$ and $z_i$. The smart-path Hamiltonian is

```text
H_{N,s}(sigma) = beta sqrt(s) / sqrt(2N) sum_{i,j} g_{ij} sigma_i sigma_j
                 + sum_i (h + beta sqrt((1-s)q) z_i) sigma_i.
```

The file then defines the partition function, Gibbs weights, replica families,
the Gibbs bracket, the averaged expectation $nu_s$, the overlap $R_{ab}$, and
the centered overlap $Q_{ab}=R_{ab}-q$. Bridge theorems identify each concrete
object with the corresponding implementation used by the proof library.

## Quantitative strict-AT theorem

`StrictAT_main` states that every compact
`K : Set (Real x Real)` contained in `strictATRegion` satisfies
`StrictATClaim K`:

```lean
theorem strictAT_main : StrictAT_main
```

The `quantitativeBounds` field supplies one nonnegative constant $C_K$ for both
of the following estimates, uniformly over the indicated parameters:

```text
N A_s <= C_K,
0 <= replicaSymmetricFreeEnergy - phi_N <= C_K / N.
```

Here $A_s=nu_s[Q_{12}^2]$, and the bounds hold for every positive $N$, every
$(beta,h)$ in $K$, and, for the first bound, every $s$ in $[0,1]$.

The `repliconSusceptibility` field states the uniform convergence

```text
N (A_s - 2 B_s + C_s)
  -> alpha / (beta^2 (1 - s alpha)),
```

where $B_s=nu_s[Q_{12}Q_{13}]$ and $C_s=nu_s[Q_{12}Q_{34}]$.

## Overlap central limit theorem

Define

```text
kappa = 1 - 4q + 3r,
zeta  = 2q + q^2 - 3r,

overlapVariance = 3(1 - 2q + r) / (1 - alpha)
                  - 2 kappa / (1 - beta^2 kappa)
                  - zeta / (1 - beta^2 kappa)^2.
```

For a test function $f$, `scaledOverlapExpectation N beta h f` is the quenched
expectation of

```text
f(sqrt(N) (R_12 - q))
```

at the endpoint of the smart path. The public theorem is

```lean
theorem strictAT_overlapCLT_weak (beta h : Real) : OverlapCLTClaim beta h
```

Under $beta>0$, $h>0$, and $alpha<1$, it proves that `overlapVariance` is
strictly positive and, for every bounded continuous real-valued $f$,

```text
scaledOverlapExpectation N beta h f
  -> E[f(sqrt(overlapVariance beta h) Z)].
```

Thus the scaled centered overlap converges weakly to the centered Gaussian with
the displayed variance, stated directly through bounded continuous test
functions.

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
solves that system in the cavity-mode basis, identifies the limiting variance,
proves convergence of characteristic functions, and invokes a Levy continuity
theorem to obtain weak convergence.

## Project layout

```text
RSAT/
|-- ARTIFACT.md               # standalone referee guide
|-- AFM_SUBMISSION.md         # journal-submission checklist and cover note
|-- CITATION.cff              # artifact citation metadata
|-- Main.lean                 # concrete public statements and proofs
|-- Lemmas/
|   |-- AT/                   # fixed point, AT data, and scalar interpolation
|   |-- Cavity/               # cavity interpolation and remainder estimates
|   |-- CLT/                  # Stein system and overlap CLT
|   |-- Concentration/        # coupled-pressure, transport, and tail bounds
|   |-- Gaussian/             # canonical coordinates and concrete model bridge
|   |-- GuerraTalagrand/      # two-replica bounds and strict flatness
|   |-- Price/                # quantitative Gaussian interpolation estimates
|   |-- SpinGlass/            # underlying SK infrastructure
|   |-- SmartPath/            # smart-path identities and endpoint estimates
|   `-- MainResult.lean       # abstract quantitative strict-AT theorem
|-- lakefile.lean
|-- lake-manifest.json
|-- lean-toolchain
|-- verify.ps1                # equivalent checks for Windows PowerShell
`-- verify.sh                 # build, API, and source-integrity checks
```

## Dependencies and build

The Lake package is named `QuantitativeStrictAT`. It uses Lean 4.32.1 and
Mathlib tag `v4.32.1`. A fresh checkout can be prepared and checked with:

```bash
git clone https://github.com/njimaMath/research_public.git
cd research_public/RSAT
lake update
lake build QuantitativeStrictAT
lake env lean Main.lean
```

`lake update` fetches the pinned dependencies into the ignored `.lake`
directory. It is needed for initial setup, not for every verification run.

For the complete project check, run:

```bash
./verify.sh
```

On Windows PowerShell, run:

```powershell
./verify.ps1
```

If the local Windows execution policy blocks scripts, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1
```

The script builds all modules under `Lemmas`, checks `Main.lean`, type-checks
the public `Main` theorem contracts, and scans project Lean sources for
placeholders and project-local axioms.

Before a release or after dependency refactoring, a clean verification is:

```bash
lake clean
./verify.sh
```

## Maintenance policy

`Main.lean` is the public statement layer. Changes to its definitions or theorem
contracts should be reflected in this README and in the API smoke test in
`verify.sh`. Its two project imports, `Lemmas.Gaussian.ConcreteModel` and
`Lemmas.CLT.CLT_Main`, are the roots of the required dependency graph.

The `Lemmas` tree is kept dependency-driven: obsolete forwarding modules,
unused internal declarations, and broad imports should be removed when the
public proof paths continue to compile.

## Trust boundary

The project contains no `sorry`, `admit`, `sorryAx`, or explicit project-local
`axiom` declarations. The principal theorems use Lean's standard `propext`,
`Classical.choice`, and `Quot.sound` axioms. They also depend on axioms generated
by `native_decide` for finite combinatorial computations. Consequently, these
computations additionally trust Lean's native-code evaluation mechanism. There
are 23 visible `native_decide` uses in four source files under `Lemmas/`.

See [`ARTIFACT.md`](./ARTIFACT.md) for the referee-oriented audit guide and
[`AFM_SUBMISSION.md`](./AFM_SUBMISSION.md) for the journal release checklist.

## License

The RSAT formalization is distributed under the Apache License 2.0; see
[`LICENSE`](./LICENSE). Most parts of [`Lemmas/SpinGlass/`](./Lemmas/SpinGlass/)
are borrowed or adapted from
[`or4nge19/SpinGlass`](https://github.com/or4nge19/SpinGlass) and retain the
applicable Apache-2.0 attribution; see [`NOTICE`](./NOTICE).
