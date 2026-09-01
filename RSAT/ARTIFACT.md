# RSAT artifact guide

- Artifact: RSAT: Quantitative Strict de Almeida-Thouless Theorem and Overlap CLT
- Associated paper: “A quantitative replica-symmetric bound for Sherrington--Kirkpatrick model in the entire de Almeida--Thouless region”
- arXiv: [`2608.23413v2`](https://arxiv.org/abs/2608.23413v2)
- Proof assistant: Lean `v4.32.1`
- Library: Mathlib `v4.32.1`
- License: Apache-2.0
- Public endpoint: `Main.lean`

## Referee starting point

The shortest audit path is to read `Main.lean` from top to bottom. It defines
the paper-facing model, connects each definition to the proof backend, and ends
with the two main theorems. The implementation under `Lemmas/` is organized by
proof method rather than by the order of the paper. `README.md` gives both
dependency paths.

## Correspondence with the paper

| Paper statement | Lean declaration | File |
| --- | --- | --- |
| Smart-path Hamiltonian defining $H_{N,s}$ | [`Main.H_N_s`](./Main.lean#L119) | `Main.lean` |
| Concrete-to-abstract smart-path identification | [`Main.H_N_s_eq_smartPath`](./Main.lean#L127) | `Main.lean` |
| Theorem 1.1 | [`Main.strictAT_main`](./Main.lean#L332) | `Main.lean` |
| Theorem 1.2 | [`Main.strictAT_overlapCLT_weak`](./Main.lean#L534) | `Main.lean` |

For every compact subset `K` of the strict AT region, `strictAT_main` supplies
a value of `StrictATClaim K`. Its fields `quantitativeBounds` and
`repliconSusceptibility` contain the three conclusions of Theorem 1.1. The
first field contains both the overlap-concentration estimate and the
free-energy correction with a common compact-set constant.

## Formal and informal conventions

- The paper uses spin configurations in $\{-1,1\}^N$. The artifact represents
  a configuration by `Fin N -> Bool` and the function `Main.spin` maps Boolean
  values to real spins in $\{-1,1\}$.
- The paper writes finite Gaussian families $(g_{ij})$ and $(z_i)$. The artifact
  uses one countable product Gaussian space and selects the required
  coordinates. The bridge theorem `Main.H_N_s_eq_smartPath` identifies the
  displayed paper Hamiltonian with the abstract implementation.
- The artifact defines `Main.q` as the infimum of fixed points in $[0,1]$.
  `Main.replicaSymmetricFixedPointClaim_of_pos_field` proves that for $h>0$ this
  value is the unique fixed point used in the paper.
- Theorem 1.2 is stated in Lean as convergence against every bounded continuous
  real-valued test function. This is the weak-convergence formulation of the
  distributional convergence stated in the paper.
- Lean sequences are indexed by all natural numbers. The CLT backend is first
  applied to positive sizes `N.succ`; the final proof removes this finite index
  shift, yielding the stated limit for `scaledOverlapExpectation N`.

## Fresh build

```bash
git clone https://github.com/njimaMath/research_public.git
cd research_public/RSAT
lake update
lake build QuantitativeStrictAT
lake env lean Main.lean
```

Lake obtains all dependencies within the local RSAT build environment. No
sibling repository or directory is needed. `lake-manifest.json` records the
resolved revisions of Mathlib and its transitive dependencies.

## Verification

After dependency setup, run:

```bash
./verify.sh
git diff -- Main.lean
```

On Windows PowerShell, the equivalent command is:

```powershell
./verify.ps1
git diff -- Main.lean
```

If the local Windows execution policy blocks scripts, invoke it as
`powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1`.

The script builds the `QuantitativeStrictAT` library, checks `Main.lean`, and
rejects project Lean sources containing `sorry`, `admit`, `sorryAx`, or a
project-local `axiom` declaration. For a clean rebuild, run:

```bash
lake clean
./verify.sh
```

`Main.lean` is the stable public interface of the artifact. It defines the
paper-facing objects and states the principal exported results while importing
the proof implementation from `Lemmas/`.

## Trust and automation

The source scan rejects `sorry`, `admit`, `sorryAx`, and explicit project-local
`axiom` declarations. The exported theorems depend on the standard Lean axioms
`propext`, `Classical.choice`, and `Quot.sound`. They also use generated
`native_decide` axioms for finite combinatorial identities, so those small
computations trust Lean's native evaluator in addition to the kernel. The
source contains 23 `native_decide` invocations in four files under `Lemmas/`.
This distinction is why the verification script describes its scan as a check
for placeholders and explicit project-local axiom declarations, rather than as
a claim that `#print axioms` is empty.
