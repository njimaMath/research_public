# Codex agent instructions for the two-replica Guerra-Talagrand bound

## Scope

Complete the Lean proof of the finite-volume two-replica Guerra-Talagrand bound in

```text
latalaStrictAlmeidaThouless/Lemmas/GTbound/GTBound.lean
```

The current target is the theorem named

```lean
twoReplica_GT_bound
```

The mathematical proof to formalize is the finite-step Guerra-Talagrand interpolation described in `blueprint.txt` in this directory.

Do not replace the theorem by an axiom, a stronger assumption, an opaque wrapper around an unproved result, `by sorry`, `admit`, `exact Classical.choice ...` from an unproved existence statement, or any equivalent proof placeholder.

Every new auxiliary theorem used in the dependency chain of `twoReplica_GT_bound` must also be proved without placeholders.

## Exact target theorem and hypotheses

The theorem statement is now fixed. Prove exactly this theorem:

```lean
theorem twoReplica_GT_bound
    {Ω : Type u} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    {N : ℕ} {β h q s v : ℝ}
    (path : RSSmartPathDisorder Ω N β h q)
    (lam : ℝ)
    (hN : 0 < N)
    (hβ : 0 < β)
    (hh : 0 < h)
    (hq : q ∈ Set.Ioo (0 : ℝ) 1)
    (hs : s ∈ Set.Icc (0 : ℝ) 1)
    (hv : v ∈ attainableOverlaps N) :
    expectedConstrainedFreeEnergy path s v ≤
      gtFunctional β h q s lam v := by
  ...
```

Do not add `UniformATData`, a compact parameter set `K`, `hp : (β,h) ∈ K`, or an assumption `q = rsQ β h` to this theorem. The positivity/range facts needed by the finite-volume GT interpolation are now supplied directly by `hβ`, `hh`, and `hq`.

At the beginning of the proof, record the elementary consequences explicitly rather than asking automation to rediscover them inside large expressions:

```lean
have hNr : 0 < (N : ℝ) := by exact_mod_cast hN
have hN0 : (N : ℝ) ≠ 0 := ne_of_gt hNr
have hβ0 : 0 ≤ β := hβ.le
have hh0 : 0 ≤ h := hh.le
have hqpos : 0 < q := hq.1
have hqlt : q < 1 := hq.2
have hq0 : 0 ≤ q := hqpos.le
have hq1 : q ≤ 1 := hqlt.le
have h1q0 : 0 ≤ 1 - q := sub_nonneg.mpr hq1
have h1qpos : 0 < 1 - q := sub_pos.mpr hqlt
have hs0 : 0 ≤ s := hs.1
have hs1 : s ≤ 1 := hs.2
have h1s0 : 0 ≤ 1 - s := sub_nonneg.mpr hs1
```

Use `hq0`, `h1q0`, `hs0`, and `h1s0` for square-root identities such as `Real.sq_sqrt`. Use the strict versions only where strict positivity or a nonzero denominator is actually required.

The mathematical GT interpolation itself does not essentially use `hh : 0 < h`; the paper allows arbitrary real external field. Preserve `hh` because it is part of the current public theorem statement, but do not manufacture a fake use of it. It is useful at downstream callers where `q = rsQ β h` and strict positivity of `q` must be established.

The main downstream caller in `Lemmas/weak_concentration.lean` works in a `UniformATData` context with `q = rsQ β h`. There, derive the new hypotheses before calling this theorem:

```lean
have hβpos : 0 < β := data.β_pos (β, h) hp
have hhpos : 0 < h := data.h_pos (β, h) hp
have hqIoo : rsQ β h ∈ Set.Ioo (0 : ℝ) 1 := by
  exact ⟨rsQ_pos hβpos hhpos, rsQ_lt_one hβpos hhpos⟩
```

and then call `twoReplica_GT_bound` with arguments in the exact order

```text
path, lam, hN, hβ, hh, hq, hs, hv.
```

## Mandatory files to inspect before editing

Read these files in full or inspect every relevant declaration before designing the proof.

```text
Lemmas/GTbound/GTBound.lean
Lemmas/ATDefs.lean
Lemmas/fixed_point.lean
Lemmas/Concentration_Coupled.lean
Lemmas/Concentration_Coupled_Transport.lean
Lemmas/weak_concentration.lean
Lemmas/smart_path/proof.lean
Lemmas/smart_path/IndependentGaussianAffineIBP.lean
SpinGlassAT/Defs.lean
SpinGlassAT/SKModel.lean
SpinGlassAT/GuerraBound.lean
SpinGlassAT/Mathlib/Probability/Distributions/Gaussian_IBP_Hilbert.lean
refs/latalaArgumentsStrictAlmeidaThoulessCondition.tex
```

The paper-side finite-step proof supplied with this task is the mathematical source for the interpolation. Read `blueprint.txt` before coding.

## Existing declarations that are authoritative

Reuse the existing definitions in `ATDefs.lean`. Do not create duplicate mathematical notions with slightly different normalization.

At minimum inspect and reuse:

```lean
RSSmartPathDisorder
fullPathHamiltonian
attainableOverlaps
constrainedPartition
expectedConstrainedFreeEnergy
gtCovarianceFunction
gtPathSign
signedMatrixPath
gtMassParameter
gtCovarianceMatrix
gtScalarVariance
gtTerminal
GTTwoField
gtIncrementScale
gtDiagonalStep
gtRankOneStep
gtSemigroupSolution
gtCorrection
gtFunctional
gtEnvelope
```

For the downstream `weak_concentration.lean` caller, also inspect the RS facts around:

```lean
rsQ
rsQ_mem_Icc
rsQ_pos
rsQ_lt_one
```

Use the exact theorem names present in the local checkout.

## Preferred law-transport route

The project already contains a canonical standard-Gaussian realization of the smart path in `Lemmas/Concentration_Coupled.lean`:

```lean
CoupledGaussianIndex
coupledDisorderCoefficient
coupledCoordinateHamiltonian
coupledConstrainedLogPartition
```

and a law-transport theorem in `Lemmas/Concentration_Coupled_Transport.lean`:

```lean
coupled_constrained_log_partition_vector_law
```

Prefer transporting the arbitrary `RSSmartPathDisorder` to these canonical Euclidean Gaussian coordinates before doing the GT interpolation. This avoids rebuilding Gaussian coordinates from the abstract sample space `Ω`.

If a scalar expectation identity is enough, prove a small public corollary of the vector-law theorem giving

```lean
∫ ω, Real.log (constrainedPartition (fullPathHamiltonian path s ω) v) ∂volume
  =
∫ x, coupledConstrainedLogPartition N β h q s v x
  ∂SYK.standardGaussianMeasureOnEuclidean (CoupledGaussianIndex N)
```

under the needed measurability and integrability hypotheses.

Do not duplicate the long law-transport proof.

## Useful code that currently exists only as private lemmas

`Lemmas/weak_concentration.lean` already contains proofs of several facts needed here, but some are private. Reuse their proof patterns, or move/generalize the lemmas to a nonprivate helper file if that is cleaner.

In particular inspect:

```lean
constrained_log_integrable
constrainedPartition_pos_of_attainable
attainableOverlap_mem_Icc
fullPath_eval_integrable
```

Do not copy large proofs blindly. If two files need the same fact, promote it to a shared helper with a stable name and update both callers.

## Recommended file organization

Do not put the entire proof in one theorem block.

A good layout is:

```text
Lemmas/GTbound/Basic.lean
Lemmas/GTbound/FiniteStep.lean
Lemmas/GTbound/Interpolation.lean
Lemmas/GTbound/GTBound.lean
```

`Basic.lean` should contain finite constrained-pair identities, positivity, overlap algebra, elementary covariance identities, and small Gaussian-expectation lemmas.

`FiniteStep.lean` should contain the reusable finite-step hierarchical Gaussian differentiation theorem, or a specialized three-level version if the general abstraction becomes expensive.

`Interpolation.lean` should instantiate the derivative theorem for the signed two-replica path, prove the endpoint identities and the nonpositive derivative, and expose one theorem that directly bounds the canonical constrained expectation by `gtFunctional`.

`GTBound.lean` should then be short: transport to canonical coordinates, invoke the specialized interpolation theorem, normalize by `N`, and rewrite definitions.

It is acceptable to use fewer files if the implementation stays readable and compiles independently.

## Do not overgeneralize the hierarchy

The current `gtSemigroupSolution` has only the breakpoints `q`, `|v|`, and `1`, with masses among `0`, `1/2`, and `1`.

If a completely general Parisi hierarchy causes excessive type and index overhead, prove exactly the finite three-increment theorem needed here, split into the two branches

```lean
q ≤ |v|
```

and

```lean
|v| < q.
```

The branch structure should line up with the existing definition of `gtSemigroupSolution`.

Do not introduce a large abstract hierarchy API unless it makes the specialized proof shorter and more stable.

## Sign conventions are critical

The repository uses

```lean
Z H = ∑ σ, exp (-H σ)
```

and `constrainedPartition` also contains `exp (-(H σ₁ + H σ₂))`.

The paper proof is written with a positive Hamiltonian in the exponent. Do not transfer paper formulas literally without checking the repository convention.

The canonical realization `coupledCoordinateHamiltonian` already inserts the minus sign so that `constrainedPartition` becomes a positive-field log-sum-exp in the Gaussian coordinates. Prefer this realization when proving endpoint factorization.

When handling the deterministic magnetic field, remember that `gtTerminal` is symmetric under simultaneous sign reversal of both field coordinates, and the standard Gaussian law is symmetric. Prove the exact identity needed rather than relying on informal sign cancellation.

## Lagrange multiplier normalization

For a constrained pair with overlap `v`, prove explicitly that

```lean
∑ i : Fin N,
  SpinGlass.spin N σ₁ i * SpinGlass.spin N σ₂ i = (N : ℝ) * v
```

using `hN` and the definition of `SpinGlass.overlap`.

Then prove that the Lagrange term

```lean
lam * (∑ i, spin σ₁ i * spin σ₂ i) - lam * (N : ℝ) * v
```

vanishes on the constrained set.

This is the Lean bridge behind extending the constrained sum to all pairs.

Do not lose a factor `N`, a factor `2`, or the sign of `lam`.

## Endpoint zero must match the existing `gtTerminal`

The local unrestricted two-spin sum is

```text
sum_{ε₁,ε₂=±1} exp(x₁ ε₁ + x₂ ε₂ + lam ε₁ ε₂).
```

By the definition of `gtTerminal`, this is

```text
4 * exp (gtTerminal lam x₁ x₂).
```

Prove this as an explicit Lean lemma by enumerating the four Boolean spin values. It should be a stable `[simp]`-style helper only if doing so does not create rewriting loops.

This identity is responsible for the term `2 * Real.log 2` after `N` sites are factorized and divided by `N`.

## Tensorization is a separate proof obligation

At the independent endpoint, the unconstrained partition function factors over sites. The hierarchical Gaussian recursion must then tensorize over sites.

Prove reusable lemmas for the exact operators that occur:

```lean
gtDiagonalStep
gtRankOneStep
```

For mass `m = 0`, tensorization is linearity of Gaussian expectation.

For `m > 0`, use independence and

```text
exp(m * sum_i F_i) = product_i exp(m * F_i)
```

followed by factorization of the standard Gaussian product measure, then `log_prod` and division by `m`.

Because the specialized masses are only `1/2` and `1`, it is acceptable to prove only those positive-mass cases if that is substantially simpler.

## Gaussian differentiation infrastructure

Before writing a new Gaussian integration-by-parts theorem, inspect:

```text
Lemmas/smart_path/IndependentGaussianAffineIBP.lean
SpinGlassAT/Mathlib/Probability/Distributions/Gaussian_IBP_Hilbert.lean
```

The project already has finite-dimensional Hilbert-space Gaussian IBP and moderate-growth infrastructure.

Prefer a covariance-operator proof of the interpolation derivative over coordinate-by-coordinate one-dimensional IBP if the existing API supports it cleanly.

For a finite state space, all Gibbs observables are smooth log-sum-exp functions. Prove the required `ContDiff`, derivative, Hessian, measurability, and growth facts once and reuse them.

## Interpolation derivative target

The specialized derivative should end in the exact square form

```text
-(s * β^2 / 4) *
  sum_j (m_{j+1} - m_j) *
    E_j < sum_{k,l : Fin 2}
      (R(σ^k,τ^l) - Q_j[k,l])^2 >
```

or an algebraically equivalent expression.

Every weight must be nonnegative. Every square is nonnegative. Therefore the derivative is nonpositive.

Do not stop at an opaque covariance expression. The square completion is the mathematical point of the GT bound and is needed to certify the sign.

## Endpoint one and correction

The scalar compensation variables in the paper may be eliminated from the Lean construction if convenient.

A preferred simplification is to add the deterministic compensation

```lean
t * gtCorrection β q s
```

to the normalized interpolating pressure, after proving that this is exactly what the scalar Gaussian levels contribute.

Then the endpoint at `t = 1` is

```text
expected constrained free energy + gtCorrection β q s
```

and the endpoint at `t = 0` is bounded by

```text
2 log 2 + E[gtSemigroupSolution ...] - lam * v.
```

This gives exactly `gtFunctional` after rearranging.

If keeping the scalar Gaussian variables is easier with the chosen finite-step API, that is also acceptable. In that case prove their endpoint recursion explicitly and prove that the finite sum of scalar variance increments equals `gtCorrection β q s`.

## Calculus near t = 0 and t = 1

The interpolation uses square roots. Do not claim differentiability of `sqrt t` at `t = 0` or of `sqrt (1-t)` at `t = 1`.

A safe pattern is:

```text
prove the derivative identity for t ∈ (0,1),
prove continuity on [0,1],
prove monotonicity on every [ε,1-ε],
let ε ↓ 0.
```

Another acceptable pattern is to reparameterize by an angle so the independent Gaussian components appear with `sin θ` and `cos θ`. If you do this, document the covariance derivative and endpoint conversion carefully.

## Finite sums and positivity

Use `hv : v ∈ attainableOverlaps N` immediately to obtain a constrained pair witness. This gives:

```lean
Nonempty {p : Config N × Config N // overlap N p.1 p.2 = v}
```

and strict positivity of the constrained partition function.

This positivity is needed for `Real.log_le_log`, log-sum-exp rewriting, and integrability arguments.

Prefer subtype sums over repeated `if overlap = v then ... else 0` expressions. Prove one rewriting lemma between the current `constrainedPartition` definition and the subtype/filter form.

## Compilation discipline

The project toolchain is Lean 4.32.1.

Compile each new helper file independently before using it from the next file. From the `latalaStrictAlmeidaThouless` directory use commands of the form

```text
lake env lean Lemmas/GTbound/Basic.lean
lake env lean Lemmas/GTbound/FiniteStep.lean
lake env lean Lemmas/GTbound/Interpolation.lean
lake env lean Lemmas/GTbound/GTBound.lean
lake env lean Lemmas/weak_concentration.lean
lake build
```

If a proof needs a larger heartbeat budget, first identify the expensive theorem. Prefer local simplification or a local `set_option maxHeartbeats ... in` block to raising a global budget.

Do not finish with only `GTBound.lean` compiling if a changed downstream caller fails.

## Debugging rules

When Lean reports a mismatch involving a large expression:

- use `change` to expose the intended definition;
- use small `have` statements for cast facts such as `(N : ℝ) ≠ 0`;
- normalize algebra with `ring` or `ring_nf` only after all square-root identities are rewritten;
- prove square-root identities with `Real.sq_sqrt` from explicit nonnegativity hypotheses;
- use `field_simp` only after proving every denominator nonzero;
- split `q ≤ |v|` exactly where `gtSemigroupSolution` splits;
- split `m = 0` exactly where `gtDiagonalStep` and `gtRankOneStep` split;
- do not unfold every definition at once.

For matrices indexed by `Fin 2`, use `fin_cases` when proving entrywise identities. For spin values, case-split the underlying `Bool` only in small local lemmas.

## Completion criteria

The task is complete only when:

```text
GTBound.lean contains no sorry,
all new helper files contain no sorry,
all modified callers compile,
no new axiom is introduced,
the final theorem has the intended GT functional with the existing normalization,
and the proof follows the finite-step interpolation rather than assuming its conclusion.
```

Keep comments that explain normalization, sign, covariance, and endpoint choices. Remove exploratory dead code before finishing.
